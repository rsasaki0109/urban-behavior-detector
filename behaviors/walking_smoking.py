"""Walking while smoking detection.

Supports two detection modes:
1. Pose-based (preferred): Uses YOLOv8-pose keypoints to check wrist-to-nose distance
2. Proxy-based (fallback): Uses small object (bottle/handbag) near mouth area

The mode is selected automatically based on whether pose detections are provided.
"""

from collections import defaultdict

import numpy as np

from behaviors.base import BehaviorAnalyzer, ViolationEvent, compute_confidence
from detectors.yolo_detector import Detection
from trackers.sort_tracker import Track


class WalkingSmokingAnalyzer(BehaviorAnalyzer):
    """Detect walking while smoking behavior."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.hand_mouth_distance = config.get("hand_mouth_distance", 0.12)
        self.speed_threshold = config.get("speed_threshold", 1.5)
        self.min_duration = config.get("min_duration_frames", 8)
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.pose_wrist_nose_ratio = config.get("pose_wrist_nose_ratio", 0.15)

        # track_id -> list of frame indices where smoking pose detected
        self._candidates: dict[int, list[int]] = defaultdict(list)
        self._reported: set[int] = set()
        self._events: list[ViolationEvent] = []

    def _is_near_mouth(self, person_bbox: np.ndarray, obj_bbox: np.ndarray) -> bool:
        """Check if a small object is near the mouth/hand area of a person."""
        person_h = person_bbox[3] - person_bbox[1]
        person_w = person_bbox[2] - person_bbox[0]

        # Mouth region: upper 20-35% of body, horizontally centered
        mouth_y = person_bbox[1] + person_h * 0.15
        mouth_y_end = person_bbox[1] + person_h * 0.30
        mouth_x = person_bbox[0] + person_w * 0.2
        mouth_x_end = person_bbox[2] - person_w * 0.2

        obj_center = np.array([(obj_bbox[0] + obj_bbox[2]) / 2, (obj_bbox[1] + obj_bbox[3]) / 2])

        # Check if object center is near mouth region
        dist_x = max(0, max(mouth_x - obj_center[0], obj_center[0] - mouth_x_end))
        dist_y = max(0, max(mouth_y - obj_center[1], obj_center[1] - mouth_y_end))
        dist = np.sqrt(dist_x**2 + dist_y**2)

        return dist < person_h * self.hand_mouth_distance

    def _check_smoking_proxy(self, track: Track,
                             all_detections: list[Detection]) -> bool:
        """Fallback: check if small proxy objects are near mouth area."""
        small_objects = [d for d in all_detections
                         if d.class_name in ("bottle", "handbag", "cell phone")
                         and d.width < 80 and d.height < 80]
        return any(self._is_near_mouth(track.bbox, obj.bbox) for obj in small_objects)

    def _check_smoking_pose(self, track: Track,
                            pose_detections: list) -> bool:
        """Preferred: check wrist-to-nose distance via pose keypoints."""
        # Find the pose detection that best matches this track
        best_iou = 0.0
        best_pose = None
        for pose in pose_detections:
            iou = self._bbox_iou(track.bbox, pose.bbox)
            if iou > best_iou:
                best_iou = iou
                best_pose = pose

        if best_pose is None or best_iou < 0.5:
            return False

        return best_pose.wrist_near_nose(self.pose_wrist_nose_ratio)

    @staticmethod
    def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        """Compute IoU between two boxes."""
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
        # Upper body region (top 50%)
        upper_body = np.array([
            track.bbox[0], track.bbox[1],
            track.bbox[2], track.bbox[1] + person_h * 0.5,
        ])

        for cig in cigarette_detections:
            # Check if cigarette bbox overlaps with upper body
            iou = self._bbox_iou(upper_body, cig.bbox)
            if iou > 0.0:
                return True
            # Also check proximity (cigarette center within person bbox)
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

            # Priority: cigarette model > pose > proxy
            if cigarette_detections:
                smoking_pose = self._check_cigarette_near_person(
                    track, cigarette_detections)
            elif pose_detections:
                smoking_pose = self._check_smoking_pose(track, pose_detections)
            else:
                smoking_pose = self._check_smoking_proxy(track, all_detections)

            if smoking_pose:
                self._candidates[track.track_id].append(frame_idx)
            else:
                # Allow small gaps (2 frames)
                frames = self._candidates.get(track.track_id, [])
                if frames and frame_idx - frames[-1] > 2:
                    self._candidates[track.track_id] = []

            # Check if violation threshold met
            frames = self._candidates.get(track.track_id, [])
            if (len(frames) >= self.min_duration
                    and track.track_id not in self._reported):
                conf = compute_confidence(frames)
                # Boost confidence for more reliable detection methods
                if cigarette_detections:
                    conf = min(0.98, conf + 0.10)
                elif pose_detections:
                    conf = min(0.98, conf + 0.05)
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
