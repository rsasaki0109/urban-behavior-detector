#!/usr/bin/env python3
"""Evaluate detection snapshots using a local VLM (via Ollama).

Reads event JSON, sends each snapshot to a vision-language model,
and adds the VLM's assessment to the event data.

Usage:
    python scripts/vlm_evaluate.py outputs/demo_events/kabukicho_night_pose_events.json
    python scripts/vlm_evaluate.py docs/assets/json/kabukicho_night_events.json --model gemma4
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import requests

DEFAULT_MODEL = "gemma4"
OLLAMA_URL = "http://localhost:11434/api/generate"

PROMPT = """Look at this surveillance camera image carefully.
Is there a person who appears to be smoking while walking?
Look for these signs:
- A person holding something small (like a cigarette) near their mouth
- Hand raised to mouth level while walking
- Any visible smoke

Answer in this JSON format only:
{"smoking_detected": true/false, "confidence": "high/medium/low", "description": "brief description of what you see"}"""


def evaluate_snapshot(image_path: str, model: str) -> dict:
    """Send an image to Ollama VLM and get assessment."""
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    resp = requests.post(OLLAMA_URL, json={
        "model": model,
        "prompt": PROMPT,
        "images": [image_b64],
        "stream": False,
    }, timeout=120)

    if resp.status_code != 200:
        return {"error": f"Ollama returned {resp.status_code}"}

    text = resp.json().get("response", "")

    # Try to parse JSON from response
    try:
        # Find JSON in response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except json.JSONDecodeError:
        pass

    return {"raw_response": text}


def main():
    parser = argparse.ArgumentParser(description="VLM evaluation of detection snapshots")
    parser.add_argument("json_file", help="Event JSON file to evaluate")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--snapshot-dir", default=None,
                        help="Base directory for snapshot paths")
    args = parser.parse_args()

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

    print(f"Evaluating {len(events)} events with {args.model}...")

    for i, event in enumerate(events):
        snapshot = event.get("snapshot")
        if not snapshot:
            print(f"  Event {i}: no snapshot, skipping")
            continue

        snap_path = base_dir / snapshot
        if not snap_path.exists():
            # Try relative to json dir
            snap_path = json_path.parent / snapshot.split("/")[-1].replace("assets/img/events/", "")
            if not snap_path.exists():
                print(f"  Event {i}: snapshot not found at {snapshot}")
                continue

        print(f"  Event {i} (Track #{event['track_id']}): evaluating {snap_path.name}...", end=" ", flush=True)
        result = evaluate_snapshot(str(snap_path), args.model)
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
    confirmed = [e for e in evaluated if e["vlm_evaluation"].get("smoking_detected") is True]
    print(f"Results: {len(confirmed)}/{len(evaluated)} confirmed by VLM")


if __name__ == "__main__":
    main()
