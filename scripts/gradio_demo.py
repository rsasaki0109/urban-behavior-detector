#!/usr/bin/env python3
"""Gradio web demo for Urban Behavior Detector.

Upload a video or enter a stream URL to detect violations in real-time.

Usage:
    python scripts/gradio_demo.py
    python scripts/gradio_demo.py --share  # public URL
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr

from pipelines.video_pipeline import VideoPipeline


def get_config_choices():
    """List available config files."""
    config_dir = Path("configs")
    return [str(f) for f in sorted(config_dir.glob("*.yaml"))]


def process_video(video_path, config_path, max_frames):
    """Process uploaded video and return results."""
    if not video_path:
        return None, "No video uploaded", "{}"

    max_frames = int(max_frames) if max_frames else 0

    pipeline = VideoPipeline(config_path)

    with tempfile.NamedTemporaryFile(suffix="_detected.mp4", delete=False) as vf:
        output_video = vf.name
    with tempfile.NamedTemporaryFile(suffix="_events.json", delete=False) as jf:
        output_json = jf.name

    # Process
    if max_frames > 0:
        # Use stream mode for frame limiting
        events = pipeline.process_stream(
            source=video_path,
            output_video=output_video,
            output_json=output_json,
            max_frames=max_frames,
        )
    else:
        events = pipeline.process_video(
            video_path=video_path,
            output_video=output_video,
            output_json=output_json,
        )

    # Format summary
    if events:
        lines = [f"**{len(events)} violation(s) detected:**\n"]
        for e in events:
            lines.append(
                f"- **{e['type']}** Track #{e['track_id']} "
                f"({e.get('start_time', '?')}s - {e.get('end_time', '?')}s) "
                f"conf: {e['confidence']:.0%}"
            )
    else:
        lines = ["No violations detected."]

    summary = "\n".join(lines)

    # JSON output
    with open(output_json) as f:
        json_str = json.dumps(json.load(f), indent=2, ensure_ascii=False)

    return output_video, summary, json_str


# Build UI
with gr.Blocks(
    title="Urban Behavior Detector",
    theme=gr.themes.Base(primary_hue="sky"),
) as demo:
    gr.Markdown("# Urban Behavior Detector")
    gr.Markdown(
        "Upload a video to detect urban violations "
        "(walking smoking, bicycle infractions, signal violations)."
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Input Video")
            config_dropdown = gr.Dropdown(
                choices=get_config_choices(),
                value="configs/rules_cigarette.yaml",
                label="Config",
            )
            max_frames_input = gr.Number(
                value=0, label="Max frames (0 = all)", precision=0,
            )
            run_btn = gr.Button("Detect Violations", variant="primary")

        with gr.Column(scale=1):
            video_output = gr.Video(label="Annotated Output")
            summary_output = gr.Markdown(label="Results")

    json_output = gr.Code(label="JSON Output", language="json")

    run_btn.click(
        fn=process_video,
        inputs=[video_input, config_dropdown, max_frames_input],
        outputs=[video_output, summary_output, json_output],
    )

    gr.Markdown(
        "---\n"
        "[GitHub](https://github.com/rsasaki0109/urban-behavior-detector) | "
        "[Demo Site](https://rsasaki0109.github.io/urban-behavior-detector/)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
