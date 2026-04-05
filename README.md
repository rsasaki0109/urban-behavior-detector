# urban-behavior-detector

Detect urban violations from video: walking smoking, bicycle phone usage, umbrella cycling, wrong-way cycling, and more.

**[Demo Site](https://rsasaki0109.github.io/urban-behavior-detector/)**

## What it does

Processes video (live camera, RTSP stream, or file) through a pipeline:

1. **Detection** - YOLOv8 detects persons, bicycles, phones, umbrellas
2. **Tracking** - IoU-based SORT assigns persistent IDs across frames
3. **Behavior analysis** - Rule-based analyzers check spatial/temporal patterns
4. **Output** - Structured JSON events + annotated video

### Supported violations

| Violation | Method |
|---|---|
| Walking smoking | Hand-mouth proximity + movement speed + duration (or pose keypoints) |
| Bicycle + phone | Phone near cyclist's face region |
| Bicycle + umbrella | Umbrella overlap with cyclist bbox |
| Wrong-way cycling | Direction angle vs expected lane direction |
| Red light running | Traffic signal HSV color analysis + bicycle proximity |
| Sidewalk riding | ROI polygon zone check + bicycle position |

## Quick start

```bash
pip install -r requirements.txt

# Run on a video file
python scripts/run_demo.py path/to/video.mp4

# With custom output paths
python scripts/run_demo.py video.mp4 -ov output.mp4 -oj events.json

# JSON only (skip video generation)
python scripts/run_demo.py video.mp4 --no-video
```

### Streaming (RTSP / Webcam)

```bash
# Webcam
python scripts/run_stream.py 0 --display

# RTSP stream
python scripts/run_stream.py rtsp://192.168.1.100:554/stream

# With frame limit
python scripts/run_stream.py 0 --max-frames 1000 -oj results.json
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
  hand_mouth_distance: 0.12
  speed_threshold: 1.5
  min_duration_frames: 8
  pose_wrist_nose_ratio: 0.15  # for pose-based detection

bicycle_phone:
  enabled: true
  phone_near_face_threshold: 0.15
  min_duration_frames: 6

bicycle_wrong_way:
  enabled: false  # requires per-scene direction config
  expected_direction: "right"
  angle_tolerance: 45

detection:
  model: "yolov8n.pt"
  use_pose: false        # enable for pose-based smoking detection
  pose_model: "yolov8n-pose.pt"
```

## Output format

```json
{
  "video_id": "sample_01",
  "fps": 30.0,
  "events": [
    {
      "type": "walking_smoking",
      "track_id": 3,
      "start_frame": 120,
      "end_frame": 165,
      "confidence": 0.71,
      "start_time": 4.0,
      "end_time": 5.5
    }
  ]
}
```

## Project structure

```
urban-behavior-detector/
├── detectors/          # YOLO object detection + pose estimation
├── trackers/           # Object tracking (SORT)
├── behaviors/          # Violation analysis logic
├── pipelines/          # End-to-end video pipeline
├── configs/            # YAML rule configurations
├── scripts/            # CLI entry points (run_demo, run_stream, export)
├── tests/              # Unit tests (pytest)
├── docs/               # GitHub Pages demo site
├── Dockerfile          # Container support
└── outputs/            # Generated results
```

## Development

```bash
# Run tests
pytest tests/ -v

# Lint
ruff check .

# Export demo assets to GitHub Pages
python scripts/export_demo_assets.py
```

## Accuracy notes

Violation detection accuracy depends heavily on input quality:

| Condition | Walking Smoking | Walking Phone | Signal Violation |
|---|---|---|---|
| 480p live camera (far) | Not reliable | Not reliable | Requires signal ROI |
| 720p+ close angle | Cigarette model + pose | Phone object + pose | Auto or ROI |
| 1080p+ close angle | High accuracy | High accuracy | High accuracy |

All detection results are verified by local VLM (Gemma 4) to filter false positives.

## Roadmap

- [x] Core pipeline (detection, tracking, behavior analysis)
- [x] Wrong-way cycling detection
- [x] Pose-based smoking detection (keypoints)
- [x] Streaming input (RTSP, webcam)
- [x] Docker support
- [x] Traffic signal violation detection (HSV color analysis)
- [x] Sidewalk cycling detection (ROI polygon zones)
- [x] Geo-referenced violation heatmaps (Leaflet.js)

## License

MIT
