"""Bicycle violation detection.

Supported violations:
- Smartphone while cycling: cell phone detected near cyclist's face
- Umbrella while cycling: umbrella overlapping with cyclist
"""

from collections import defaultdict

import numpy as np

from behaviors.base import BehaviorAnalyzer, ViolationEvent, compute_confidence
from detectors.yolo_detector import Detection
from trackers.sort_tracker import Track


def _bbox_overlap_ratio(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute overlap ratio of box_b within box_a."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / area_b if area_b > 0 else 0.0


class BicycleViolationAnalyzer(BehaviorAnalyzer):
    """Detect bicycle-related violations."""

    def __init__(self, phone_config: dict, umbrella_config: dict):
        self.phone_enabled = phone_config.get("enabled", True)
        self.umbrella_enabled = umbrella_config.get("enabled", True)
        self.phone_threshold = phone_config.get("phone_near_face_threshold", 0.15)
        self.phone_min_duration = phone_config.get("min_duration_frames", 6)
        self.phone_conf_threshold = phone_config.get("confidence_threshold", 0.5)
        self.umbrella_overlap = umbrella_config.get("umbrella_overlap_threshold", 0.3)
        self.umbrella_min_duration = umbrella_config.get("min_duration_frames", 5)
        self.umbrella_conf_threshold = umbrella_config.get("confidence_threshold", 0.5)

        self._phone_candidates: dict[int, list[int]] = defaultdict(list)
        self._umbrella_candidates: dict[int, list[int]] = defaultdict(list)
        self._reported: set[tuple[str, int]] = set()
        self._events: list[ViolationEvent] = []
        self.enabled = self.phone_enabled or self.umbrella_enabled
        self.config = {}

    def _find_nearby_cyclists(self, tracks: list[Track]) -> list[tuple[Track, Track]]:
        """Find person tracks that are near bicycle tracks (likely the rider).

        Uses a scoring system combining:
        - Bounding box overlap ratio
        - Center distance (normalized by person height)
        - Speed similarity (persons on bikes move at similar speed)
        """
        persons = [t for t in tracks if t.class_name == "person"]
        bicycles = [t for t in tracks if t.class_name == "bicycle"]

        pairs = []
        matched_bikes: set[int] = set()

        for person in persons:
            person_h = person.bbox[3] - person.bbox[1]
            if person_h <= 0:
                continue

            best_score = 0.0
            best_bike = None

            for bike in bicycles:
                if bike.track_id in matched_bikes:
                    continue

                # Score 1: Overlap ratio (0-1)
                overlap = _bbox_overlap_ratio(person.bbox, bike.bbox)

                # Score 2: Center distance, normalized by person height (inverted)
                center_dist = np.linalg.norm(person.center - bike.center)
                dist_score = max(0, 1.0 - center_dist / (person_h * 2))

                # Score 3: Speed similarity (both should be moving similarly)
                speed_diff = abs(person.speed - bike.speed)
                speed_score = max(0, 1.0 - speed_diff / 10.0) if person.speed > 0.5 else 0.0

                score = overlap * 0.5 + dist_score * 0.3 + speed_score * 0.2

                if score > best_score:
                    best_score = score
                    best_bike = bike

            if best_bike is not None and best_score > 0.15:
                pairs.append((person, best_bike))
                matched_bikes.add(best_bike.track_id)

        return pairs

    def update(self, frame_idx: int, tracks: list[Track],
               all_detections: list[Detection]) -> list[ViolationEvent]:
        if not self.enabled:
            return []

        cyclist_pairs = self._find_nearby_cyclists(tracks)
        phones = [d for d in all_detections if d.class_name == "cell phone"]
        umbrellas = [d for d in all_detections if d.class_name == "umbrella"]

        new_events = []

        for person, bike in cyclist_pairs:
            person_h = person.bbox[3] - person.bbox[1]

            # Check phone usage
            if self.phone_enabled:
                face_region = np.array([
                    person.bbox[0], person.bbox[1],
                    person.bbox[2], person.bbox[1] + person_h * 0.35
                ])
                phone_near = any(
                    _bbox_overlap_ratio(face_region, p.bbox) > 0.05
                    or np.linalg.norm(
                        np.array([(p.bbox[0]+p.bbox[2])/2, (p.bbox[1]+p.bbox[3])/2])
                        - np.array([(face_region[0]+face_region[2])/2, (face_region[1]+face_region[3])/2])
                    ) < person_h * self.phone_threshold
                    for p in phones
                )
                if phone_near:
                    self._phone_candidates[person.track_id].append(frame_idx)
                elif (self._phone_candidates.get(person.track_id)
                      and frame_idx - self._phone_candidates[person.track_id][-1] > 2):
                    self._phone_candidates[person.track_id] = []

                frames = self._phone_candidates.get(person.track_id, [])
                key = ("bicycle_phone", person.track_id)
                if len(frames) >= self.phone_min_duration and key not in self._reported:
                    conf = compute_confidence(frames, max_conf=0.9)
                    if conf >= self.phone_conf_threshold:
                        event = ViolationEvent(
                            violation_type="bicycle_phone",
                            track_id=person.track_id,
                            start_frame=frames[0],
                            end_frame=frame_idx,
                            confidence=conf,
                        )
                        new_events.append(event)
                        self._events.append(event)
                        self._reported.add(key)

            # Check umbrella
            if self.umbrella_enabled:
                umbrella_over = any(
                    _bbox_overlap_ratio(person.bbox, u.bbox) > self.umbrella_overlap
                    for u in umbrellas
                )
                if umbrella_over:
                    self._umbrella_candidates[person.track_id].append(frame_idx)
                elif (self._umbrella_candidates.get(person.track_id)
                      and frame_idx - self._umbrella_candidates[person.track_id][-1] > 2):
                    self._umbrella_candidates[person.track_id] = []

                frames = self._umbrella_candidates.get(person.track_id, [])
                key = ("bicycle_umbrella", person.track_id)
                if len(frames) >= self.umbrella_min_duration and key not in self._reported:
                    conf = compute_confidence(frames, max_conf=0.9)
                    if conf >= self.umbrella_conf_threshold:
                        event = ViolationEvent(
                            violation_type="bicycle_umbrella",
                            track_id=person.track_id,
                            start_frame=frames[0],
                            end_frame=frame_idx,
                            confidence=conf,
                        )
                        new_events.append(event)
                        self._events.append(event)
                        self._reported.add(key)

        return new_events

    def prune_stale_tracks(self, active_track_ids: set[int]) -> None:
        """Remove state for tracks no longer active in the tracker."""
        for tid in list(self._phone_candidates):
            if tid not in active_track_ids:
                del self._phone_candidates[tid]
        for tid in list(self._umbrella_candidates):
            if tid not in active_track_ids:
                del self._umbrella_candidates[tid]
        # Cap _reported to prevent unbounded growth
        _MAX_REPORTED = 10000
        if len(self._reported) > _MAX_REPORTED:
            self._reported = set(sorted(self._reported)[-_MAX_REPORTED:])

    def finalize(self) -> list[ViolationEvent]:
        return self._events
