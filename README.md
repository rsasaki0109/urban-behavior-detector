# urban-behavior-detector

**2026年4月施行「自転車青切符」対応** — 自転車の交通違反をAIで自動検知するOSS。

ながらスマホ(12,000円)・信号無視(6,000円)・傘差し運転(6,000円)・逆走(6,000円)・歩道走行(6,000円) を映像から検出。

**[Demo Site](https://rsasaki0109.github.io/urban-behavior-detector/)** | **[GitHub](https://github.com/rsasaki0109/urban-behavior-detector)**

## What it does

Processes video (live camera, RTSP stream, or file) through a multi-stage pipeline:

1. **Detection** - YOLOv8 detects persons, bicycles, vehicles, phones, umbrellas, traffic lights
2. **Tracking** - IoU-based SORT assigns persistent IDs across frames
3. **Specialized models** - YOLOv8-pose (keypoints), cigarette detector (fine-tuned YOLOv11m)
4. **Behavior analysis** - Rule-based analyzers check spatial/temporal patterns
5. **VLM verification** - Local VLM (Gemma 4) filters false positives from snapshots
6. **Output** - Structured JSON events + annotated video + event snapshots

### 青切符対象違反の検知

| 違反 | 反則金 | 検出方法 | VLM検証 |
|---|---|---|---|
| **ながらスマホ** | 12,000円 | YOLO cell phone + cyclist pairing | 1/1 confirmed |
| **信号無視** | 6,000円 | Signal ROI HSV + crossing zone | implemented |
| **傘差し運転** | 6,000円 | Umbrella overlap with cyclist | implemented |
| **逆走** | 6,000円 | Direction angle vs expected | implemented |
| **歩道走行** | 6,000円 | ROI polygon zone check | implemented |

### 付加機能

| 違反 | 検出方法 | VLM検証 |
|---|---|---|
| 歩きタバコ | Cigarette model + pose | 4/4 confirmed |
| 歩きスマホ | Phone object + pose | implemented |

## Quick start

```bash
pip install -r requirements.txt

# Run on a video file
python scripts/run_demo.py path/to/video.mp4

# With cigarette detection model
python scripts/run_demo.py video.mp4 --config configs/rules_cigarette.yaml

# JSON only (skip video generation)
python scripts/run_demo.py video.mp4 --no-video

# VLM verification of results
python scripts/vlm_evaluate.py outputs/demo_events/result.json
```

### Streaming (RTSP / Webcam)

```bash
# Webcam
python scripts/run_stream.py 0 --display

# RTSP stream
python scripts/run_stream.py rtsp://192.168.1.100:554/stream

# YouTube live camera (resolve URL first)
STREAM=$(yt-dlp --format 95 --get-url 'https://youtube.com/watch?v=VIDEO_ID')
python scripts/run_stream.py "$STREAM" --max-frames 1800
```

### Docker

```bash
docker build -t urban-behavior-detector .
docker run -v $(pwd)/samples:/app/samples -v $(pwd)/outputs:/app/outputs \
  urban-behavior-detector samples/video.mp4
```

## Configuration

All thresholds and toggles are in `configs/rules.yaml`:

```yaml
walking_smoking:
  enabled: true
  speed_threshold: 0.8
  min_duration_frames: 5
  pose_wrist_nose_ratio: 0.15
  min_oscillations: 2

signal_violation:
  enabled: true
  signal_rois: [[685, 210, 720, 240]]     # fixed signal position
  crossing_zones: [[[350,280],[550,280],[550,340],[350,340]]]

detection:
  model: "yolov8n.pt"
  use_pose: true
  pose_model: "yolov8n-pose.pt"
  cigarette_model: "models/cigarette_yolov11m.pt"
```

## Output format

```json
{
  "video_id": "smoking_street",
  "events": [
    {
      "type": "walking_smoking",
      "track_id": 1,
      "start_frame": 0,
      "end_frame": 2,
      "confidence": 0.77,
      "snapshot": "snapshots/smoking_street_event_0.jpg",
      "vlm_evaluation": {
        "smoking_detected": true,
        "confidence": "high",
        "description": "A lit cigarette is held in his right hand..."
      }
    }
  ]
}
```

## Project structure

```
urban-behavior-detector/
├── detectors/          # YOLO detection + pose + cigarette + signal
├── trackers/           # SORT tracking
├── behaviors/          # 7 violation analyzers
├── pipelines/          # Video + stream pipeline
├── configs/            # YAML configs (default, cigarette, traffic, etc.)
├── scripts/            # CLI: run_demo, run_stream, vlm_evaluate, etc.
├── tests/              # 67 unit tests (pytest)
├── models/             # Fine-tuned models (cigarette detector)
├── docs/               # GitHub Pages demo site + heatmap
├── Dockerfile
└── outputs/            # Detection results + snapshots
```

## Accuracy

### Dog fooding results (RTX 4070, 720p)

| Input | Detection | VLM verified | Speed |
|---|---|---|---|
| Stock footage (720p close-up) | Cigarette model detects smoking | **4/4 confirmed** | 22-29 fps |
| Live camera (720p Kabukicho) | Cigarette model false positives | **0/2 confirmed** | 29 fps |
| Live camera (480p Sapporo) | Signal ROI + crossing zone | 0 violations (compliant) | 29 fps |

### Key findings

- **720p close-up footage**: Cigarette detection works reliably, VLM confirms
- **Live camera (720p far angle)**: Cigarette too small (~5px), false positives from bags/masks
- **Signal violation**: Crossing zone concept eliminates false positives from nearby pedestrians
- **GPU (RTX 4070)**: Real-time processing at 29 fps

### Recommendation

For production use, install dedicated cameras (1080p, close angle, 3-5m height) near target areas. Public live cameras are too far for reliable small-object detection.

## Development

```bash
# Run tests
pytest tests/ -v

# Lint
ruff check .

# Export demo assets
python scripts/export_demo_assets.py

# Generate heatmap data
python scripts/generate_heatmap.py
```

## License

MIT
