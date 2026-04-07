#!/usr/bin/env python3
"""Run the urban behavior detection pipeline on a video."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines.video_pipeline import VideoPipeline


def main():
    parser = argparse.ArgumentParser(description="Urban Behavior Detector")
    parser.add_argument("video", help="Input video file path")
    parser.add_argument("--config", default="configs/rules.yaml",
                        help="Config file path (default: configs/rules.yaml)")
    parser.add_argument("--output-video", "-ov", default=None,
                        help="Output annotated video path")
    parser.add_argument("--output-json", "-oj", default=None,
                        help="Output events JSON path")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip video output")
    parser.add_argument("--log-jsonl", default=None,
                        help="Path for structured JSONL event log")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    # Default output paths
    output_dir = Path("outputs")
    if args.output_video is None and not args.no_video:
        args.output_video = str(output_dir / "demo_videos" / f"{video_path.stem}_detected.mp4")
    if args.output_json is None:
        args.output_json = str(output_dir / "demo_events" / f"{video_path.stem}_events.json")

    pipeline = VideoPipeline(args.config)
    events = pipeline.process_video(
        str(video_path),
        output_video=args.output_video if not args.no_video else None,
        output_json=args.output_json,
        log_jsonl=args.log_jsonl,
    )

    print(f"\nDetected {len(events)} violation events:")
    for e in events:
        print(f"  [{e['type']}] Track #{e['track_id']} "
              f"frames {e['start_frame']}-{e['end_frame']} "
              f"(conf: {e['confidence']})")


if __name__ == "__main__":
    main()
