"""Walking while using smartphone detection.

Logic:
- Person is walking (speed > threshold)
- Hand is consistently near face (wrist stays near nose) — NOT oscillating
- This is the opposite of smoking: smoking oscillates, phone usage is constant
- Also checks for "cell phone" object detection near face as supplementary signal
"""

from collections import defaultdict

import numpy as np

from behaviors.base import BehaviorAnalyzer, ViolationEvent, compute_confidence
from detectors.yolo_detector import Detection
from trackers.sort_tracker import Track


class WalkingPhoneAnalyzer(BehaviorAnalyzer):
    """Detect pedestrians using smartphones while walking."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.speed_threshold = config.get("speed_threshold", 1.0)
        self.min_duration = config.get("min_duration_frames", 10)
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.pose_wrist_nose_ratio = config.get("pose_wrist_nose_ratio", 0.20)
        self.min_near_ratio = config.get("min_near_ratio", 0.7)

        # track_id -> list of (frame_idx, wrist_nose_distance_ratio)
        self._distance_history: dict[int, list[tuple[int, float]]] = defaultdict(list)
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

    def _detect_constant_near(self, track_id: int) -> bool:
        """Detect hand constantly near face (phone usage pattern).

        Phone: hand stays near face (>70% of recent frames near).
        This is the opposite of smoking oscillation.
        """
        history = self._distance_history.get(track_id, [])
        if len(history) < 8:
            return False

        threshold = self.pose_wrist_nose_ratio
        recent = [d for _, d in history[-30:]]

        near_count = sum(1 for d in recent if d < threshold)
        near_ratio = near_count / len(recent)

        # Phone usage: hand consistently near face
        return near_ratio >= self.min_near_ratio

    def _check_phone_object_near_face(self, track: Track,
                                      all_detections: list[Detection]) -> bool:
        """Check if a cell phone detection is near person's face region."""
        phones = [d for d in all_detections if d.class_name == "cell phone"]
        if not phones:
            return False

        person_h = track.bbox[3] - track.bbox[1]
        face_region = np.array([
            track.bbox[0], track.bbox[1],
            track.bbox[2], track.bbox[1] + person_h * 0.4,
        ])

        for phone in phones:
            phone_center = phone.center
            if (face_region[0] <= phone_center[0] <= face_region[2]
                    and face_region[1] <= phone_center[1] <= face_region[3]):
                return True
        return False

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

    def update(self, frame_idx: int, tracks: list[Track],
               all_detections: list[Detection],
               pose_detections: list | None = None) -> list[ViolationEvent]:
        if not self.enabled:
            return []

        new_events = []
        person_tracks = [t for t in tracks if t.class_name == "person"]

        for track in person_tracks:
            if track.speed < self.speed_threshold:
                continue

            phone_detected = False

            has_constant_near = False
            has_phone_object = False

            # Check pose: hand constantly near face
            if pose_detections:
                dist = self._get_wrist_nose_distance(track, pose_detections)
                if dist is not None:
                    self._distance_history[track.track_id].append((frame_idx, dist))
                    cutoff = frame_idx - 90
                    self._distance_history[track.track_id] = [
                        (f, d) for f, d in self._distance_history[track.track_id]
                        if f >= cutoff
                    ]
                    has_constant_near = self._detect_constant_near(track.track_id)

            # Check YOLO: cell phone object near face
            has_phone_object = self._check_phone_object_near_face(
                track, all_detections)

            # Decision: require phone object detection (pose alone is not enough)
            # - Phone object + constant near = high confidence
            # - Phone object only = medium (object visible near face)
            # - Constant near only = not enough (could be eating, scratching etc.)
            if has_phone_object:
                phone_detected = True

            if phone_detected:
                self._candidates[track.track_id].append(frame_idx)
            else:
                frames = self._candidates.get(track.track_id, [])
                if frames and frame_idx - frames[-1] > 5:
                    self._candidates[track.track_id] = []

            frames = self._candidates.get(track.track_id, [])
            if (len(frames) >= self.min_duration
                    and track.track_id not in self._reported):
                conf = compute_confidence(frames)
                if has_phone_object and has_constant_near:
                    conf = min(0.98, conf + 0.10)
                if conf >= self.confidence_threshold:
                    event = ViolationEvent(
                        violation_type="walking_phone",
                        track_id=track.track_id,
                        start_frame=frames[0],
                        end_frame=frame_idx,
                        confidence=conf,
                    )
                    new_events.append(event)
                    self._events.append(event)
                    self._reported.add(track.track_id)

        return new_events

    def prune_stale_tracks(self, active_track_ids: set[int]) -> None:
        """Remove state for tracks no longer active in the tracker."""
        for tid in list(self._distance_history):
            if tid not in active_track_ids:
                del self._distance_history[tid]
        for tid in list(self._candidates):
            if tid not in active_track_ids:
                del self._candidates[tid]
        _MAX_REPORTED = 10000
        if len(self._reported) > _MAX_REPORTED:
            self._reported = set(sorted(self._reported)[-_MAX_REPORTED:])

    def finalize(self) -> list[ViolationEvent]:
        return self._events
