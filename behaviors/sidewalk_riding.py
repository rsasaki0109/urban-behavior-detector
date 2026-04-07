"""Sidewalk riding detection for bicycles.

Logic:
- Sidewalk zones are defined as polygons in config (per-scene)
- A bicycle track's center point is checked against the polygon
- If the bicycle stays in the sidewalk zone for min_duration_frames, it's a violation
"""

from collections import defaultdict

import cv2
import numpy as np

from behaviors.base import BehaviorAnalyzer, ViolationEvent, compute_confidence
from trackers.sort_tracker import Track


class SidewalkRidingAnalyzer(BehaviorAnalyzer):
    """Detect bicycles riding on sidewalks."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.min_duration = config.get("min_duration_frames", 8)
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.min_speed = config.get("min_speed", 1.5)

        # Parse sidewalk zones from config
        raw_zones = config.get("sidewalk_zones", [])
        self.sidewalk_zones = [
            np.array(zone, dtype=np.int32) for zone in raw_zones
        ]

        self._candidates: dict[int, list[int]] = defaultdict(list)
        self._reported: set[int] = set()
        self._events: list[ViolationEvent] = []

    def _is_in_sidewalk(self, point: np.ndarray) -> bool:
        """Check if a point is inside any sidewalk zone polygon."""
        pt = (int(point[0]), int(point[1]))
        for zone in self.sidewalk_zones:
            if cv2.pointPolygonTest(zone, pt, False) >= 0:
                return True
        return False

    def update(self, frame_idx: int, tracks: list[Track],
               all_detections: list) -> list[ViolationEvent]:
        if not self.enabled or not self.sidewalk_zones:
            return []

        new_events = []
        bicycle_tracks = [t for t in tracks if t.class_name == "bicycle"]

        for track in bicycle_tracks:
            if track.speed < self.min_speed:
                continue

            if self._is_in_sidewalk(track.center):
                self._candidates[track.track_id].append(frame_idx)
            else:
                frames = self._candidates.get(track.track_id, [])
                if frames and frame_idx - frames[-1] > 2:
                    self._candidates[track.track_id] = []

            frames = self._candidates.get(track.track_id, [])
            if (len(frames) >= self.min_duration
                    and track.track_id not in self._reported):
                conf = compute_confidence(frames, max_conf=0.9)
                if conf >= self.confidence_threshold:
                    event = ViolationEvent(
                        violation_type="sidewalk_riding",
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
        for tid in list(self._candidates):
            if tid not in active_track_ids:
                del self._candidates[tid]
        _MAX_REPORTED = 10000
        if len(self._reported) > _MAX_REPORTED:
            self._reported = set(sorted(self._reported)[-_MAX_REPORTED:])

    def finalize(self) -> list[ViolationEvent]:
        return self._events
