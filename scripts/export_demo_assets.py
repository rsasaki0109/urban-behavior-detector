#!/usr/bin/env python3
"""Export detection outputs to docs/assets/ for GitHub Pages demo."""

import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DOCS_ASSETS = PROJECT_ROOT / "docs" / "assets"


def export_videos():
    """Copy demo videos to docs/assets/videos/."""
    src = OUTPUTS_DIR / "demo_videos"
    dst = DOCS_ASSETS / "videos"
    dst.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        print("No demo videos found in outputs/demo_videos/")
        return

    for video in src.glob("*.mp4"):
        target = dst / video.name
        shutil.copy2(video, target)
        print(f"Copied {video.name} -> docs/assets/videos/")


def export_events():
    """Copy and merge event JSONs to docs/assets/json/."""
    src = OUTPUTS_DIR / "demo_events"
    dst = DOCS_ASSETS / "json"
    dst.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        print("No event files found in outputs/demo_events/")
        return

    all_results = []

    # Include existing sample JSONs in docs/assets/json/ (not from outputs)
    for jf in sorted(dst.glob("sample_*_events.json")):
        with open(jf) as f:
            data = json.load(f)
        all_results.append(data)
        print(f"Included existing {jf.name}")

    # Copy and include pipeline output JSONs
    for jf in sorted(src.glob("*.json")):
        shutil.copy2(jf, dst / jf.name)
        with open(jf) as f:
            data = json.load(f)
        all_results.append(data)
        print(f"Copied {jf.name} -> docs/assets/json/")

    # Create merged index
    index_path = dst / "demo_index.json"
    with open(index_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Created merged index with {len(all_results)} entries")


def main():
    print("Exporting demo assets to docs/assets/...")
    export_videos()
    export_events()
    print("Done.")


if __name__ == "__main__":
    main()
