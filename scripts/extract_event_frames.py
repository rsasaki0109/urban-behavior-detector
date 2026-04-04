#!/usr/bin/env python3
"""Extract frame snapshots at each violation event from annotated videos.

Reads event JSONs and extracts the corresponding frames from the detected
video, saving them as images for the demo site.

Usage:
    python scripts/extract_event_frames.py
    python scripts/extract_event_frames.py --video-dir outputs/demo_videos --json-dir outputs/demo_events --output-dir docs/assets/img/events
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def extract_frames(video_path: Path, events: list[dict], output_dir: Path,
                   video_id: str) -> list[str]:
    """Extract frames at event timestamps and return saved file paths."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Warning: Cannot open {video_path}")
        return []

    saved = []
    for i, event in enumerate(events):
        frame_num = event["start_frame"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue

        filename = f"{video_id}_event_{i}.jpg"
        output_path = output_dir / filename
        cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        saved.append(filename)
        print(f"  Saved {filename} (frame {frame_num})")

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser(description="Extract event frame snapshots")
    parser.add_argument("--video-dir", default="outputs/demo_videos")
    parser.add_argument("--json-dir", default="outputs/demo_events")
    parser.add_argument("--output-dir", default="docs/assets/img/events")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    json_dir = Path(args.json_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for jf in sorted(json_dir.glob("*.json")):
        if jf.name in ("demo_index.json", "heatmap_data.json"):
            continue

        with open(jf) as f:
            data = json.load(f)

        events = data.get("events", [])
        if not events:
            continue

        video_id = data["video_id"]
        video_path = video_dir / f"{video_id}_detected.mp4"

        if not video_path.exists():
            # Try other naming patterns
            candidates = list(video_dir.glob(f"{video_id}*detected*.mp4"))
            if candidates:
                video_path = candidates[0]
            else:
                print(f"  Skipping {video_id}: no video found")
                continue

        print(f"Processing {video_id} ({len(events)} events)")
        filenames = extract_frames(video_path, events, output_dir, video_id)

        # Update JSON with snapshot paths
        for i, event in enumerate(events):
            if i < len(filenames):
                event["snapshot"] = f"assets/img/events/{filenames[i]}"

        # Save updated JSON
        with open(jf, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Also update docs copy if it exists
        docs_json = Path("docs/assets/json") / jf.name
        if docs_json.exists():
            with open(docs_json, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    print("\nDone. Re-run export_demo_assets.py to update demo_index.json")


if __name__ == "__main__":
    main()
