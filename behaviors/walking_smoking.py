"""Walking while smoking detection.

Detection strategy:
- Uses YOLOv8-pose to track wrist-to-nose distance over time
- Smoking gesture: hand repeatedly moves to mouth and back (oscillation)
- Phone usage: hand stays near face continuously (no oscillation)
- Requires the person to be walking (speed > threshold)
"""

from collections import defaultdict

import numpy as np

from behaviors.base import BehaviorAnalyzer, ViolationEvent, compute_confidence
from detectors.yolo_detector import Detection
from trackers.sort_tracker import Track


class WalkingSmokingAnalyzer(BehaviorAnalyzer):
    """Detect walking while smoking behavior via pose oscillation."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.speed_threshold = config.get("speed_threshold", 1.5)
        self.min_duration = config.get("min_duration_frames", 8)
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.pose_wrist_nose_ratio = config.get("pose_wrist_nose_ratio", 0.15)
        self.min_oscillations = config.get("min_oscillations", 2)

        # track_id -> list of (frame_idx, wrist_nose_distance_ratio)
        self._distance_history: dict[int, list[tuple[int, float]]] = defaultdict(list)
        # track_id -> list of frame indices where smoking detected
        self._candidates: dict[int, list[int]] = defaultdict(list)
        self._reported: set[int] = set()
        self._events: list[ViolationEvent] = []

    def _get_wrist_nose_distance(self, track: Track,
                                 pose_detections: list) -> float | None:
        """Get minimum wrist-to-nose distance ratio for a tracked person."""
        if not pose_detections:
            return None

        best_iou = 0.0
        best_pose = None
        for pose in pose_detections:
            iou = self._bbox_iou(track.bbox, pose.bbox)
            if iou > best_iou:
                best_iou = iou
                best_pose = pose

        if best_pose is None or best_iou < 0.5:
            return None

        nose = best_pose.keypoint(0)  # NOSE
        if nose is None:
            return None

        height = best_pose.height
        if height <= 0:
            return None

        min_dist = float("inf")
        for wrist_idx in (9, 10):  # LEFT_WRIST, RIGHT_WRIST
            wrist = best_pose.keypoint(wrist_idx)
            if wrist is not None:
                dist = float(np.linalg.norm(wrist - nose))
                min_dist = min(min_dist, dist)

        if min_dist == float("inf"):
            return None

        return min_dist / height

    def _detect_oscillation(self, track_id: int) -> bool:
        """Detect hand-to-mouth oscillation pattern in distance history.

        Smoking pattern: distance alternates between near (<threshold)
        and far (>threshold * 2), at least min_oscillations times.
        Phone pattern: distance stays consistently near.
        """
        history = self._distance_history.get(track_id, [])
        if len(history) < 6:
            return False

        threshold = self.pose_wrist_nose_ratio
        recent = [d for _, d in history[-30:]]  # last 30 data points

        # Count transitions: near->far and far->near
        near = [d < threshold for d in recent]
        transitions = 0
        for i in range(1, len(near)):
            if near[i] != near[i - 1]:
                transitions += 1

        # Need at least min_oscillations full cycles (near->far->near = 2 transitions)
        has_oscillation = transitions >= self.min_oscillations * 2

        # Also require some "near" frames (hand actually reaches mouth)
        near_count = sum(near)
        near_ratio = near_count / len(recent)

        # Smoking: 20-70% near (oscillating). Phone: >80% near (constant).
        is_smoking_pattern = 0.15 <= near_ratio <= 0.75

        return has_oscillation and is_smoking_pattern

    @staticmethod
    def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _check_cigarette_near_person(self, track: Track,
                                     cigarette_detections: list) -> bool:
        """Check if a cigarette detection overlaps with a person's upper body."""
        if not cigarette_detections:
            return False

        person_h = track.bbox[3] - track.bbox[1]
        upper_body = np.array([
            track.bbox[0], track.bbox[1],
            track.bbox[2], track.bbox[1] + person_h * 0.5,
        ])

        for cig in cigarette_detections:
            iou = self._bbox_iou(upper_body, cig.bbox)
            if iou > 0.0:
                return True
            cx, cy = cig.center
            if (track.bbox[0] <= cx <= track.bbox[2]
                    and track.bbox[1] <= cy <= track.bbox[1] + person_h * 0.6):
                return True
        return False

    def update(self, frame_idx: int, tracks: list[Track],
               all_detections: list[Detection],
               pose_detections: list | None = None,
               cigarette_detections: list | None = None) -> list[ViolationEvent]:
        if not self.enabled:
            return []

        new_events = []
        person_tracks = [t for t in tracks if t.class_name == "person"]

        for track in person_tracks:
            if track.speed < self.speed_threshold:
                continue

            smoking_detected = False

            has_oscillation = False
            has_cigarette = False

            # Check pose oscillation (hand-to-mouth pattern)
            if pose_detections:
                dist = self._get_wrist_nose_distance(track, pose_detections)
                if dist is not None:
                    self._distance_history[track.track_id].append((frame_idx, dist))
                    cutoff = frame_idx - 90  # ~3 seconds at 30fps
                    self._distance_history[track.track_id] = [
                        (f, d) for f, d in self._distance_history[track.track_id]
                        if f >= cutoff
                    ]
                    has_oscillation = self._detect_oscillation(track.track_id)

            # Check cigarette object detection
            if cigarette_detections:
                has_cigarette = self._check_cigarette_near_person(
                    track, cigarette_detections)

            # Decision logic:
            # - Both cigarette + oscillation = high confidence
            # - Cigarette only = medium (object visible)
            # - Oscillation only = not enough (too many false positives)
            if has_cigarette and has_oscillation:
                smoking_detected = True
            elif has_cigarette:
                smoking_detected = True

            if smoking_detected:
                self._candidates[track.track_id].append(frame_idx)
            else:
                frames = self._candidates.get(track.track_id, [])
                if frames and frame_idx - frames[-1] > 5:
                    self._candidates[track.track_id] = []

            frames = self._candidates.get(track.track_id, [])
            if (len(frames) >= self.min_duration
                    and track.track_id not in self._reported):
                conf = compute_confidence(frames)
                if has_cigarette and has_oscillation:
                    conf = min(0.98, conf + 0.15)
                elif has_cigarette:
                    conf = min(0.95, conf + 0.05)
                if conf >= self.confidence_threshold:
                    event = ViolationEvent(
                        violation_type="walking_smoking",
                        track_id=track.track_id,
                        start_frame=frames[0],
                        end_frame=frame_idx,
                        confidence=conf,
                    )
                    new_events.append(event)
                    self._events.append(event)
                    self._reported.add(track.track_id)

        return new_events

    def finalize(self) -> list[ViolationEvent]:
        return self._events
