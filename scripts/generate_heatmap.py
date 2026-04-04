#!/usr/bin/env python3
"""Generate a heatmap data file from multiple detection result JSONs.

Reads all event JSON files from a directory, extracts geo-referenced
violation events, and produces a heatmap_data.json for the demo site.

Usage:
    python scripts/generate_heatmap.py [--input-dir outputs/demo_events]
    python scripts/generate_heatmap.py --output docs/assets/json/heatmap_data.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def collect_heatmap_points(input_dir: Path) -> list[dict]:
    """Collect geo-referenced violation points from event JSONs."""
    points = []

    for jf in sorted(input_dir.glob("*.json")):
        if jf.name == "demo_index.json" or jf.name == "heatmap_data.json":
            continue

        with open(jf) as f:
            data = json.load(f)

        location = data.get("location")
        if not location or not location.get("lat") or not location.get("lon"):
            continue

        for event in data.get("events", []):
            points.append({
                "lat": location["lat"],
                "lon": location["lon"],
                "type": event["type"],
                "confidence": event["confidence"],
                "video_id": data.get("video_id", ""),
                "location_name": location.get("name", ""),
            })

    return points


def main():
    parser = argparse.ArgumentParser(description="Generate heatmap data")
    parser.add_argument("--input-dir", default="outputs/demo_events",
                        help="Directory with event JSON files")
    parser.add_argument("--output", default="docs/assets/json/heatmap_data.json",
                        help="Output heatmap data file")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    points = collect_heatmap_points(input_dir)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({
            "points": points,
            "total": len(points),
        }, f, indent=2, ensure_ascii=False)

    print(f"Generated heatmap with {len(points)} points -> {output_path}")


if __name__ == "__main__":
    main()
