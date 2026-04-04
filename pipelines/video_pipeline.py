"""Main video processing pipeline."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from behaviors.base import ViolationEvent
from behaviors.bicycle_violation import BicycleViolationAnalyzer
from behaviors.sidewalk_riding import SidewalkRidingAnalyzer
from behaviors.signal_violation import SignalViolationAnalyzer
from behaviors.walking_smoking import WalkingSmokingAnalyzer
from behaviors.wrong_way import WrongWayAnalyzer
from detectors.pose_detector import PoseDetector
from detectors.signal_detector import SignalDetection, classify_signal_color
from detectors.yolo_detector import Detection, YOLODetector
from trackers.sort_tracker import SORTTracker

# Violation type -> color (BGR)
VIOLATION_COLORS = {
    "walking_smoking": (0, 0, 255),       # Red
    "bicycle_phone": (0, 165, 255),       # Orange
    "bicycle_umbrella": (0, 255, 255),    # Yellow
    "bicycle_wrong_way": (255, 0, 255),   # Magenta
    "signal_violation": (0, 0, 200),      # Dark Red
    "sidewalk_riding": (255, 128, 0),     # Cyan-ish
}


class VideoPipeline:
    """End-to-end video analysis pipeline."""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        det_cfg = self.config.get("detection", {})
        trk_cfg = self.config.get("tracking", {})

        self.detector = YOLODetector(
            model_path=det_cfg.get("model", "yolov8n.pt"),
            confidence=det_cfg.get("confidence", 0.4),
            iou_threshold=det_cfg.get("iou_threshold", 0.5),
            classes=det_cfg.get("classes"),
        )
        self.tracker = SORTTracker(
            max_age=trk_cfg.get("max_age", 30),
            min_hits=trk_cfg.get("min_hits", 3),
            iou_threshold=trk_cfg.get("iou_threshold", 0.3),
        )

        # Pose detector (optional, for improved walking smoking detection)
        self.use_pose = det_cfg.get("use_pose", False)
        self.pose_detector = None
        if self.use_pose:
            pose_model = det_cfg.get("pose_model", "yolov8n-pose.pt")
            self.pose_detector = PoseDetector(
                model_path=pose_model,
                confidence=det_cfg.get("confidence", 0.4),
            )

        self.analyzers = []
        if self.config.get("walking_smoking", {}).get("enabled", False):
            self.analyzers.append(WalkingSmokingAnalyzer(self.config["walking_smoking"]))
        phone_cfg = self.config.get("bicycle_phone", {})
        umbrella_cfg = self.config.get("bicycle_umbrella", {})
        if phone_cfg.get("enabled", False) or umbrella_cfg.get("enabled", False):
            self.analyzers.append(BicycleViolationAnalyzer(phone_cfg, umbrella_cfg))
        if self.config.get("bicycle_wrong_way", {}).get("enabled", False):
            self.analyzers.append(WrongWayAnalyzer(self.config["bicycle_wrong_way"]))
        if self.config.get("signal_violation", {}).get("enabled", False):
            self.analyzers.append(SignalViolationAnalyzer(self.config["signal_violation"]))
        if self.config.get("sidewalk_riding", {}).get("enabled", False):
            self.analyzers.append(SidewalkRidingAnalyzer(self.config["sidewalk_riding"]))

    def process_video(self, video_path: str, output_video: str | None = None,
                      output_json: str | None = None) -> list[dict]:
        """Process a video file and return violation events."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}", file=sys.stderr)
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_video:
            Path(output_video).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        all_events: list[ViolationEvent] = []
        frame_idx = 0

        print(f"Processing {video_path} ({total_frames} frames, {fps:.1f} fps)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect
            detections = self.detector.detect(frame)

            # Pose detection (optional)
            pose_detections = None
            if self.pose_detector:
                pose_detections = self.pose_detector.detect(frame)

            # Signal color classification
            signal_detections = None
            traffic_lights = [d for d in detections if d.class_name == "traffic light"]
            if traffic_lights:
                signal_detections = [
                    SignalDetection(
                        bbox=d.bbox,
                        color=classify_signal_color(frame, d.bbox),
                        confidence=d.confidence,
                    )
                    for d in traffic_lights
                ]

            # Track (only persons and bicycles)
            trackable = [d for d in detections if d.class_name in ("person", "bicycle")]
            tracks = self.tracker.update(trackable)

            # Analyze behaviors
            frame_events = []
            for analyzer in self.analyzers:
                if isinstance(analyzer, WalkingSmokingAnalyzer):
                    events = analyzer.update(frame_idx, tracks, detections, pose_detections)
                elif isinstance(analyzer, SignalViolationAnalyzer):
                    events = analyzer.update(frame_idx, tracks, detections, signal_detections)
                else:
                    events = analyzer.update(frame_idx, tracks, detections)
                frame_events.extend(events)
                all_events.extend(events)

            # Draw annotations
            if writer:
                annotated = self._draw_frame(frame, detections, tracks, frame_events, frame_idx)
                writer.write(annotated)

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  Frame {frame_idx}/{total_frames}")

        cap.release()
        if writer:
            writer.release()

        # Collect all finalized events
        final_events = []
        for analyzer in self.analyzers:
            final_events.extend(analyzer.finalize())

        event_dicts = [e.to_dict() for e in final_events]
        for e in event_dicts:
            e["start_time"] = round(e["start_frame"] / fps, 2)
            e["end_time"] = round(e["end_frame"] / fps, 2)

        # Save JSON
        if output_json:
            Path(output_json).parent.mkdir(parents=True, exist_ok=True)
            loc_cfg = self.config.get("location", {})
            result = {
                "video_id": Path(video_path).stem,
                "video_file": Path(video_path).name,
                "fps": fps,
                "total_frames": total_frames,
                "resolution": f"{width}x{height}",
                "events": event_dicts,
            }
            if loc_cfg.get("lat") and loc_cfg.get("lon"):
                result["location"] = {
                    "lat": loc_cfg["lat"],
                    "lon": loc_cfg["lon"],
                    "name": loc_cfg.get("name", ""),
                }
            with open(output_json, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(event_dicts)} events to {output_json}")

        if output_video:
            print(f"Saved annotated video to {output_video}")

        return event_dicts

    def process_stream(self, source, output_video: str | None = None,
                       output_json: str | None = None,
                       display: bool = False,
                       max_frames: int = 0) -> list[dict]:
        """Process a live stream (RTSP, webcam, HTTP) and return violation events."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Cannot open stream {source}", file=sys.stderr)
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

        writer = None
        if output_video:
            Path(output_video).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        all_events: list[ViolationEvent] = []
        frame_idx = 0

        source_name = str(source) if isinstance(source, str) else f"webcam:{source}"
        print(f"Streaming from {source_name} ({width}x{height}, {fps:.1f} fps)")
        print("Press 'q' to stop." if display else "Press Ctrl+C to stop.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    # Retry once for stream reconnection
                    cap.release()
                    cap = cv2.VideoCapture(source)
                    ret, frame = cap.read()
                    if not ret:
                        print("Stream ended or connection lost.")
                        break

                detections = self.detector.detect(frame)

                pose_detections = None
                if self.pose_detector:
                    pose_detections = self.pose_detector.detect(frame)

                signal_detections = None
                traffic_lights = [d for d in detections if d.class_name == "traffic light"]
                if traffic_lights:
                    signal_detections = [
                        SignalDetection(
                            bbox=d.bbox,
                            color=classify_signal_color(frame, d.bbox),
                            confidence=d.confidence,
                        )
                        for d in traffic_lights
                    ]

                trackable = [d for d in detections if d.class_name in ("person", "bicycle")]
                tracks = self.tracker.update(trackable)

                frame_events = []
                for analyzer in self.analyzers:
                    if isinstance(analyzer, WalkingSmokingAnalyzer):
                        events = analyzer.update(frame_idx, tracks, detections, pose_detections)
                    elif isinstance(analyzer, SignalViolationAnalyzer):
                        events = analyzer.update(frame_idx, tracks, detections, signal_detections)
                    else:
                        events = analyzer.update(frame_idx, tracks, detections)
                    frame_events.extend(events)
                    all_events.extend(events)

                if frame_events:
                    for e in frame_events:
                        print(f"  [ALERT] {e.violation_type} Track #{e.track_id} "
                              f"frame {frame_idx} (conf: {e.confidence:.2f})")

                annotated = self._draw_frame(frame, detections, tracks, frame_events, frame_idx)

                if writer:
                    writer.write(annotated)

                if display:
                    cv2.imshow("Urban Behavior Detector", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_idx += 1
                if max_frames > 0 and frame_idx >= max_frames:
                    break
                if frame_idx % 300 == 0:
                    print(f"  Frame {frame_idx} processed, {len(all_events)} events so far")

        except KeyboardInterrupt:
            print("\nStopped by user.")

        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        final_events = []
        for analyzer in self.analyzers:
            final_events.extend(analyzer.finalize())

        event_dicts = [e.to_dict() for e in final_events]
        for e in event_dicts:
            e["start_time"] = round(e["start_frame"] / fps, 2)
            e["end_time"] = round(e["end_frame"] / fps, 2)

        if output_json:
            Path(output_json).parent.mkdir(parents=True, exist_ok=True)
            result = {
                "video_id": source_name,
                "source": str(source),
                "fps": fps,
                "total_frames": frame_idx,
                "resolution": f"{width}x{height}",
                "events": event_dicts,
            }
            with open(output_json, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(event_dicts)} events to {output_json}")

        return event_dicts

    def _draw_frame(self, frame: np.ndarray, detections: list[Detection],
                    tracks: list, events: list[ViolationEvent],
                    frame_idx: int) -> np.ndarray:
        """Draw bounding boxes, track IDs, and violation alerts."""
        out = frame.copy()

        # Draw detections (thin boxes)
        for det in detections:
            if det.class_name not in ("person", "bicycle"):
                x1, y1, x2, y2 = det.bbox.astype(int)
                cv2.rectangle(out, (x1, y1), (x2, y2), (128, 128, 128), 1)
                cv2.putText(out, det.class_name, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

        # Draw tracked objects
        for track in tracks:
            x1, y1, x2, y2 = track.bbox.astype(int)
            color = (0, 255, 0) if track.class_name == "person" else (255, 200, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{track.class_name} #{track.track_id}"
            cv2.putText(out, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw violation alerts
        for event in events:
            color = VIOLATION_COLORS.get(event.violation_type, (0, 0, 255))
            label = f"VIOLATION: {event.violation_type} (#{event.track_id})"
            cv2.putText(out, label, (10, 30 + events.index(event) * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Frame counter
        cv2.putText(out, f"Frame: {frame_idx}", (10, out.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return out
