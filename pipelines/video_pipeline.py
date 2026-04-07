"""Main video processing pipeline."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

from behaviors.base import ViolationEvent
from behaviors.bicycle_violation import BicycleViolationAnalyzer
from behaviors.sidewalk_riding import SidewalkRidingAnalyzer
from behaviors.signal_violation import SignalViolationAnalyzer
from behaviors.walking_phone import WalkingPhoneAnalyzer
from behaviors.walking_smoking import WalkingSmokingAnalyzer
from behaviors.wrong_way import WrongWayAnalyzer
from detectors.cigarette_detector import CigaretteDetector
from detectors.pose_detector import PoseDetector
from detectors.signal_detector import (
    SignalDetection, classify_signal_color, detect_signals_from_rois,
)
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
    "walking_phone": (255, 200, 50),      # Light blue
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

        # Cigarette detector (optional, most accurate smoking detection)
        self.cigarette_detector = None
        cig_model = det_cfg.get("cigarette_model")
        if cig_model:
            self.cigarette_detector = CigaretteDetector(
                model_path=cig_model,
                confidence=det_cfg.get("cigarette_confidence", 0.25),
            )

        # ROI crop mode for far-view accuracy
        self.roi_crop = det_cfg.get("roi_crop", False)
        self.roi_crop_padding = det_cfg.get("roi_crop_padding", 0.2)
        self.roi_crop_min_size = det_cfg.get("roi_crop_min_size", 128)

        self.analyzers = []
        if self.config.get("walking_smoking", {}).get("enabled", False):
            self.analyzers.append(WalkingSmokingAnalyzer(self.config["walking_smoking"]))
        phone_cfg = self.config.get("bicycle_phone", {})
        umbrella_cfg = self.config.get("bicycle_umbrella", {})
        if phone_cfg.get("enabled", False) or umbrella_cfg.get("enabled", False):
            self.analyzers.append(BicycleViolationAnalyzer(phone_cfg, umbrella_cfg))
        if self.config.get("bicycle_wrong_way", {}).get("enabled", False):
            self.analyzers.append(WrongWayAnalyzer(self.config["bicycle_wrong_way"]))
        sig_cfg = self.config.get("signal_violation", {})
        if sig_cfg.get("enabled", False):
            self.analyzers.append(SignalViolationAnalyzer(sig_cfg))
        self.signal_rois = sig_cfg.get("signal_rois", [])
        if self.config.get("sidewalk_riding", {}).get("enabled", False):
            self.analyzers.append(SidewalkRidingAnalyzer(self.config["sidewalk_riding"]))
        if self.config.get("walking_phone", {}).get("enabled", False):
            self.analyzers.append(WalkingPhoneAnalyzer(self.config["walking_phone"]))

    @staticmethod
    def _write_jsonl(log_file, record: dict) -> None:
        """Write a single JSON line to the log file."""
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_file.flush()

    def process_video(self, video_path: str, output_video: str | None = None,
                      output_json: str | None = None,
                      log_jsonl: str | None = None) -> list[dict]:
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

        # JSONL structured log
        jsonl_file = None
        if log_jsonl:
            Path(log_jsonl).parent.mkdir(parents=True, exist_ok=True)
            jsonl_file = open(log_jsonl, "a")  # noqa: SIM115
            self._write_jsonl(jsonl_file, {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "pipeline_start",
                "source": video_path,
                "config": str(self.config),
            })

        all_events: list[ViolationEvent] = []
        event_snapshots: list[str] = []
        frame_idx = 0
        total_event_count = 0

        print(f"Processing {video_path} ({total_frames} frames, {fps:.1f} fps)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect
            detections = self.detector.detect(frame)

            # Signal color classification (YOLO auto-detect or fixed ROI)
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
            elif self.signal_rois:
                signal_detections = detect_signals_from_rois(frame, self.signal_rois)

            # Track (only persons and bicycles)
            trackable = [d for d in detections if d.class_name in ("person", "bicycle", "car", "motorcycle", "bus", "truck")]
            tracks = self.tracker.update(trackable)

            # Pose detection (optional)
            pose_detections = None
            if self.pose_detector:
                if self.roi_crop:
                    pose_detections = self._detect_on_crops(
                        frame, tracks, self.pose_detector)
                else:
                    pose_detections = self.pose_detector.detect(frame)

            # Cigarette detection (optional)
            cigarette_detections = None
            if self.cigarette_detector:
                if self.roi_crop:
                    cigarette_detections = self._detect_on_crops(
                        frame, tracks, self.cigarette_detector)
                else:
                    cigarette_detections = self.cigarette_detector.detect(frame)

            # Analyze behaviors
            frame_events = []
            for analyzer in self.analyzers:
                if isinstance(analyzer, WalkingSmokingAnalyzer):
                    events = analyzer.update(frame_idx, tracks, detections,
                                             pose_detections, cigarette_detections)
                elif isinstance(analyzer, WalkingPhoneAnalyzer):
                    events = analyzer.update(frame_idx, tracks, detections, pose_detections)
                elif isinstance(analyzer, SignalViolationAnalyzer):
                    events = analyzer.update(frame_idx, tracks, detections, signal_detections)
                else:
                    events = analyzer.update(frame_idx, tracks, detections)
                frame_events.extend(events)
                all_events.extend(events)

            # Write JSONL for real-time events
            if jsonl_file and frame_events:
                track_bboxes = {t.track_id: t.bbox.tolist() for t in tracks}
                for event in frame_events:
                    self._write_jsonl(jsonl_file, {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "frame_idx": frame_idx,
                        "event_type": "violation_detected",
                        "violation_type": event.violation_type,
                        "track_id": event.track_id,
                        "confidence": round(event.confidence, 2),
                        "bbox": track_bboxes.get(event.track_id, []),
                    })
                    total_event_count += 1

            # Draw annotations (include cigarette boxes)
            annotated = self._draw_frame(frame, detections, tracks, frame_events, frame_idx,
                                         cigarette_detections)
            if writer:
                writer.write(annotated)

            # Save snapshot for new violation events
            if frame_events and output_json:
                snap_dir = Path(output_json).parent / "snapshots"
                snap_dir.mkdir(parents=True, exist_ok=True)
                for event in frame_events:
                    snap_name = f"{Path(video_path).stem}_event_{len(event_snapshots)}.jpg"
                    cv2.imwrite(str(snap_dir / snap_name), annotated,
                                [cv2.IMWRITE_JPEG_QUALITY, 85])
                    event_snapshots.append(snap_name)

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  Frame {frame_idx}/{total_frames}")

        cap.release()
        if writer:
            writer.release()

        # Write pipeline_stop to JSONL and close
        if jsonl_file:
            self._write_jsonl(jsonl_file, {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "pipeline_stop",
                "total_frames": frame_idx,
                "total_events": total_event_count,
            })
            jsonl_file.close()

        # Collect all finalized events
        final_events = []
        for analyzer in self.analyzers:
            final_events.extend(analyzer.finalize())

        event_dicts = [e.to_dict() for e in final_events]
        for i, e in enumerate(event_dicts):
            e["start_time"] = round(e["start_frame"] / fps, 2)
            e["end_time"] = round(e["end_frame"] / fps, 2)
            if i < len(event_snapshots):
                e["snapshot"] = f"snapshots/{event_snapshots[i]}"

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
                       max_frames: int = 0,
                       log_jsonl: str | None = None) -> list[dict]:
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

        # JSONL structured log
        jsonl_file = None
        total_event_count = 0
        if log_jsonl:
            Path(log_jsonl).parent.mkdir(parents=True, exist_ok=True)
            jsonl_file = open(log_jsonl, "a")  # noqa: SIM115
            self._write_jsonl(jsonl_file, {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "pipeline_start",
                "source": str(source),
                "config": str(self.config),
            })

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
                elif self.signal_rois:
                    signal_detections = detect_signals_from_rois(frame, self.signal_rois)

                trackable = [d for d in detections if d.class_name in ("person", "bicycle", "car", "motorcycle", "bus", "truck")]
                tracks = self.tracker.update(trackable)

                pose_detections = None
                if self.pose_detector:
                    if self.roi_crop:
                        pose_detections = self._detect_on_crops(
                            frame, tracks, self.pose_detector)
                    else:
                        pose_detections = self.pose_detector.detect(frame)

                cigarette_detections = None
                if self.cigarette_detector:
                    if self.roi_crop:
                        cigarette_detections = self._detect_on_crops(
                            frame, tracks, self.cigarette_detector)
                    else:
                        cigarette_detections = self.cigarette_detector.detect(frame)

                frame_events = []
                for analyzer in self.analyzers:
                    if isinstance(analyzer, WalkingSmokingAnalyzer):
                        events = analyzer.update(frame_idx, tracks, detections,
                                                 pose_detections, cigarette_detections)
                    elif isinstance(analyzer, WalkingPhoneAnalyzer):
                        events = analyzer.update(frame_idx, tracks, detections, pose_detections)
                    elif isinstance(analyzer, SignalViolationAnalyzer):
                        events = analyzer.update(frame_idx, tracks, detections, signal_detections)
                    else:
                        events = analyzer.update(frame_idx, tracks, detections)
                    frame_events.extend(events)

                if frame_events:
                    for e in frame_events:
                        print(f"  [ALERT] {e.violation_type} Track #{e.track_id} "
                              f"frame {frame_idx} (conf: {e.confidence:.2f})")

                # Write JSONL for real-time events
                if jsonl_file and frame_events:
                    track_bboxes = {t.track_id: t.bbox.tolist() for t in tracks}
                    for event in frame_events:
                        self._write_jsonl(jsonl_file, {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "frame_idx": frame_idx,
                            "event_type": "violation_detected",
                            "violation_type": event.violation_type,
                            "track_id": event.track_id,
                            "confidence": round(event.confidence, 2),
                            "bbox": track_bboxes.get(event.track_id, []),
                        })
                        total_event_count += 1

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

                # Periodically prune analyzer state for long-running streams
                if frame_idx % 300 == 0:
                    active_track_ids = {t.track_id for t in self.tracker.tracks}
                    for analyzer in self.analyzers:
                        analyzer.prune_stale_tracks(active_track_ids)
                    print(f"  Frame {frame_idx} processed, {total_event_count} events so far")

        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

            # Write pipeline_stop to JSONL and close
            if jsonl_file:
                self._write_jsonl(jsonl_file, {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": "pipeline_stop",
                    "total_frames": frame_idx,
                    "total_events": total_event_count,
                })
                jsonl_file.close()

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

    def _crop_person_roi(self, frame: np.ndarray, bbox: np.ndarray):
        """Crop a person ROI from the frame with padding and optional upscale.

        Returns (crop, crop_x1, crop_y1, scale_factor) where scale_factor
        is the ratio by which the crop was upscaled (1.0 if no upscale).
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)
        bw, bh = x2 - x1, y2 - y1

        # Apply padding
        pad_x = int(bw * self.roi_crop_padding)
        pad_y = int(bh * self.roi_crop_padding)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)

        crop = frame[cy1:cy2, cx1:cx2]

        # Upscale if smaller than min size
        scale = 1.0
        crop_h, crop_w = crop.shape[:2]
        if crop_h > 0 and crop_w > 0:
            min_dim = min(crop_h, crop_w)
            if min_dim < self.roi_crop_min_size:
                scale = self.roi_crop_min_size / min_dim
                new_w = int(crop_w * scale)
                new_h = int(crop_h * scale)
                crop = cv2.resize(crop, (new_w, new_h),
                                  interpolation=cv2.INTER_LINEAR)

        return crop, cx1, cy1, scale

    def _detect_on_crops(self, frame, person_tracks, detector):
        """Run a detector on per-person crops and remap to full-frame coords.

        Args:
            frame: Full video frame.
            person_tracks: List of tracked persons.
            detector: A detector with a .detect(frame) method returning objects
                      that have a .bbox (np.ndarray [x1,y1,x2,y2]) attribute.

        Returns:
            List of detections with bboxes in full-frame coordinates.
        """
        all_detections = []
        for track in person_tracks:
            if track.class_name != "person":
                continue

            crop, cx1, cy1, scale = self._crop_person_roi(frame, track.bbox)
            if crop.size == 0:
                continue

            crop_dets = detector.detect(crop)
            for det in crop_dets:
                # Inverse scale then offset to full-frame coordinates
                local_bbox = det.bbox.copy()
                local_bbox = local_bbox / scale
                local_bbox[0] += cx1
                local_bbox[1] += cy1
                local_bbox[2] += cx1
                local_bbox[3] += cy1
                det.bbox = local_bbox
                all_detections.append(det)

        return all_detections

    def _draw_frame(self, frame: np.ndarray, detections: list[Detection],
                    tracks: list, events: list[ViolationEvent],
                    frame_idx: int,
                    cigarette_detections: list | None = None) -> np.ndarray:
        """Draw bounding boxes, track IDs, and violation alerts."""
        out = frame.copy()

        # Draw detections (thin boxes)
        for det in detections:
            if det.class_name not in ("person", "bicycle", "car", "motorcycle", "bus", "truck"):
                x1, y1, x2, y2 = det.bbox.astype(int)
                cv2.rectangle(out, (x1, y1), (x2, y2), (128, 128, 128), 1)
                cv2.putText(out, det.class_name, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

        # Draw cigarette detections (red boxes)
        if cigarette_detections:
            for cig in cigarette_detections:
                x1, y1, x2, y2 = cig.bbox.astype(int)
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(out, f"cigarette {cig.confidence:.2f}",
                            (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 0, 255), 1)

        # Draw tracked objects
        for track in tracks:
            x1, y1, x2, y2 = track.bbox.astype(int)
            _track_colors = {"person": (0, 255, 0), "bicycle": (255, 200, 0),
                            "car": (255, 150, 50), "motorcycle": (200, 100, 255),
                            "bus": (100, 200, 255), "truck": (100, 200, 255)}
            color = _track_colors.get(track.class_name, (200, 200, 200))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{track.class_name} #{track.track_id}"
            cv2.putText(out, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw signal ROIs
        if cigarette_detections is None and hasattr(self, 'signal_rois') and self.signal_rois:
            for roi in self.signal_rois:
                rx1, ry1, rx2, ry2 = [int(v) for v in roi]
                color_state = classify_signal_color(out, np.array(roi, dtype=float))
                roi_color = {"red": (0, 0, 255), "green": (0, 255, 0),
                             "yellow": (0, 255, 255)}.get(color_state.value, (128, 128, 128))
                cv2.rectangle(out, (rx1, ry1), (rx2, ry2), roi_color, 2)
                cv2.putText(out, f"signal:{color_state.value}", (rx1, ry1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, roi_color, 1)

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
