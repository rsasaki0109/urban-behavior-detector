#!/usr/bin/env python3
"""Run the urban behavior detection pipeline on a stream (RTSP, webcam, etc.).

Examples:
    # Webcam (device 0)
    python scripts/run_stream.py 0

    # RTSP stream
    python scripts/run_stream.py rtsp://192.168.1.100:554/stream

    # HTTP stream
    python scripts/run_stream.py http://example.com/live.m3u8
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines.video_pipeline import VideoPipeline


def main():
    parser = argparse.ArgumentParser(description="Urban Behavior Detector - Stream Mode")
    parser.add_argument("source", help="Stream source: device index (0,1,...), RTSP URL, or HTTP URL")
    parser.add_argument("--config", default="configs/rules.yaml",
                        help="Config file path (default: configs/rules.yaml)")
    parser.add_argument("--output-video", "-ov", default=None,
                        help="Output annotated video path (optional)")
    parser.add_argument("--output-json", "-oj", default=None,
                        help="Output events JSON path")
    parser.add_argument("--display", action="store_true",
                        help="Show live display window")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after N frames (0 = unlimited)")
    args = parser.parse_args()

    # Convert numeric source to int (webcam device)
    source = int(args.source) if args.source.isdigit() else args.source

    # Default JSON output
    if args.output_json is None:
        name = f"stream_{args.source}" if isinstance(source, str) else f"webcam_{source}"
        name = name.replace("/", "_").replace(":", "_")
        args.output_json = f"outputs/demo_events/{name}_events.json"

    pipeline = VideoPipeline(args.config)
    events = pipeline.process_stream(
        source=source,
        output_video=args.output_video,
        output_json=args.output_json,
        display=args.display,
        max_frames=args.max_frames,
    )

    print(f"\nDetected {len(events)} violation events:")
    for e in events:
        print(f"  [{e['type']}] Track #{e['track_id']} "
              f"frames {e['start_frame']}-{e['end_frame']} "
              f"(conf: {e['confidence']})")


if __name__ == "__main__":
    main()
