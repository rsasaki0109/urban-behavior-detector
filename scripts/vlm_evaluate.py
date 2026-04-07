#!/usr/bin/env python3
"""Evaluate detection snapshots using a local VLM (Ollama or OpenAI-compatible).

Reads event JSON, sends each snapshot to a vision-language model,
and adds the VLM's assessment to the event data.

Supports two backends:
  - ollama (default): Uses /api/generate with images field
  - openai: Uses /v1/chat/completions (works with vLLM, LiteLLM, etc.)

Usage:
    python scripts/vlm_evaluate.py outputs/demo_events/kabukicho_night_pose_events.json
    python scripts/vlm_evaluate.py events.json --model gemma4
    python scripts/vlm_evaluate.py events.json --backend openai --base-url http://localhost:8000
    python scripts/vlm_evaluate.py events.json --rate-limit 2 --timeout 60
"""

import argparse
import base64
import json
import time
from pathlib import Path

import requests

DEFAULT_MODEL = "gemma4"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OPENAI_URL = "http://localhost:8000"
MAX_RETRIES = 3
BACKOFF_BASE = 1  # seconds

PROMPT = """Look at this surveillance camera image carefully.
Is there a person who appears to be smoking while walking?
Look for these signs:
- A person holding something small (like a cigarette) near their mouth
- Hand raised to mouth level while walking
- Any visible smoke

Answer in this JSON format only:
{"smoking_detected": true/false, "confidence": "high/medium/low", "description": "brief description of what you see"}"""


def _call_ollama(image_b64: str, model: str, base_url: str, timeout: int) -> dict:
    """Send an image to Ollama VLM and get assessment."""
    url = f"{base_url.rstrip('/')}/api/generate"
    resp = requests.post(url, json={
        "model": model,
        "prompt": PROMPT,
        "images": [image_b64],
        "stream": False,
    }, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _call_openai(image_b64: str, model: str, base_url: str, timeout: int) -> dict:
    """Send an image to an OpenAI-compatible VLM endpoint."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    resp = requests.post(url, json={
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}",
                }},
            ],
        }],
        "max_tokens": 512,
    }, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _extract_text(raw: dict, backend: str) -> str:
    """Extract the text response from a backend-specific response dict."""
    if backend == "ollama":
        return raw.get("response", "")
    # openai-compatible: choices[0].message.content
    choices = raw.get("choices", [])
    if choices:
        return choices[0].get("message", {}).get("content", "")
    return ""


def evaluate_snapshot(
    image_path: str,
    model: str,
    backend: str,
    base_url: str,
    timeout: int,
) -> dict:
    """Send an image to a VLM and get assessment, with retry logic."""
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    call_fn = _call_ollama if backend == "ollama" else _call_openai
    last_err = None

    for attempt in range(MAX_RETRIES):
        try:
            raw = call_fn(image_b64, model, base_url, timeout)
            text = _extract_text(raw, backend)

            # Try to parse JSON from response
            try:
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

            return {"raw_response": text}

        except (requests.RequestException, requests.Timeout) as exc:
            last_err = exc
            if attempt < MAX_RETRIES - 1:
                wait = BACKOFF_BASE * (2 ** attempt)
                print(f"\n    Retry {attempt + 1}/{MAX_RETRIES - 1} after {wait}s ({exc})")
                time.sleep(wait)

    return {"error": f"Failed after {MAX_RETRIES} attempts: {last_err}"}


def main():
    parser = argparse.ArgumentParser(description="VLM evaluation of detection snapshots")
    parser.add_argument("json_file", help="Event JSON file to evaluate")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="VLM model name")
    parser.add_argument("--backend", choices=["ollama", "openai"], default="ollama",
                        help="VLM backend type (default: ollama)")
    parser.add_argument("--base-url", default=None,
                        help="Base URL for the VLM server (default depends on backend)")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Request timeout in seconds (default: 120)")
    parser.add_argument("--rate-limit", type=float, default=0,
                        help="Max requests per second, 0 = no limit (default: 0)")
    parser.add_argument("--snapshot-dir", default=None,
                        help="Base directory for snapshot paths")
    args = parser.parse_args()

    # Resolve base URL
    if args.base_url is None:
        args.base_url = (
            DEFAULT_OLLAMA_URL if args.backend == "ollama" else DEFAULT_OPENAI_URL
        )

    # Rate-limit interval
    min_interval = 1.0 / args.rate_limit if args.rate_limit > 0 else 0

    json_path = Path(args.json_file)
    with open(json_path) as f:
        data = json.load(f)

    events = data.get("events", [])
    if not events:
        print("No events to evaluate.")
        return

    # Determine base dir for snapshot paths
    if args.snapshot_dir:
        base_dir = Path(args.snapshot_dir)
    elif "docs/assets" in str(json_path):
        base_dir = json_path.parent.parent.parent  # docs/
    else:
        base_dir = json_path.parent

    print(f"Evaluating {len(events)} events with {args.model} ({args.backend})...")

    last_request_time = 0.0

    for i, event in enumerate(events):
        snapshot = event.get("snapshot")
        if not snapshot:
            print(f"  Event {i}: no snapshot, skipping")
            continue

        snap_path = base_dir / snapshot
        if not snap_path.exists():
            # Try relative to json dir
            snap_path = json_path.parent / snapshot.split("/")[-1].replace(
                "assets/img/events/", ""
            )
            if not snap_path.exists():
                print(f"  Event {i}: snapshot not found at {snapshot}")
                continue

        # Rate limiting
        if min_interval > 0:
            elapsed = time.monotonic() - last_request_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

        print(
            f"  Event {i} (Track #{event['track_id']}): evaluating {snap_path.name}...",
            end=" ",
            flush=True,
        )

        last_request_time = time.monotonic()
        result = evaluate_snapshot(
            str(snap_path), args.model, args.backend, args.base_url, args.timeout,
        )
        event["vlm_evaluation"] = result

        smoking = result.get("smoking_detected", "?")
        conf = result.get("confidence", "?")
        desc = result.get("description", result.get("raw_response", ""))[:80]
        print(f"smoking={smoking} conf={conf} - {desc}")

    # Save updated JSON
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved evaluations to {json_path}")

    # Summary
    evaluated = [e for e in events if "vlm_evaluation" in e]
    confirmed = [
        e for e in evaluated if e["vlm_evaluation"].get("smoking_detected") is True
    ]
    print(f"Results: {len(confirmed)}/{len(evaluated)} confirmed by VLM")


if __name__ == "__main__":
    main()
