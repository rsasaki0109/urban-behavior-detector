"""Wrong-way cycling detection.

Logic:
- Bicycle (cyclist) is moving in a direction opposite to the expected lane direction
- Direction deviation exceeds angle_tolerance for min_duration_frames
"""

from collections import defaultdict

from behaviors.base import BehaviorAnalyzer, ViolationEvent, compute_confidence
from trackers.sort_tracker import Track

# Named direction -> angle in degrees (0=right, 90=down, -90=up, 180=left)
DIRECTION_ANGLES = {
    "right": 0,
    "left": 180,
    "down": 90,
    "up": -90,
}


class WrongWayAnalyzer(BehaviorAnalyzer):
    """Detect bicycles traveling in the wrong direction."""

    def __init__(self, config: dict):
        super().__init__(config)
        direction_str = config.get("expected_direction", "right")
        self.expected_angle = DIRECTION_ANGLES.get(direction_str, 0)
        self.angle_tolerance = config.get("angle_tolerance", 45)
        self.min_duration = config.get("min_duration_frames", 10)
        self.confidence_threshold = config.get("confidence_threshold", 0.6)
        self.min_speed = config.get("speed_threshold", 2.0)

        self._candidates: dict[int, list[int]] = defaultdict(list)
        self._reported: set[int] = set()
        self._events: list[ViolationEvent] = []

    def _is_wrong_way(self, track: Track) -> bool:
        """Check if a track is moving opposite to the expected direction."""
        if track.speed < self.min_speed:
            return False

        direction = track.direction
        # Compute angular difference, normalized to [-180, 180]
        diff = direction - self.expected_angle
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360

        return abs(diff) > (180 - self.angle_tolerance)

    def update(self, frame_idx: int, tracks: list[Track],
               all_detections: list) -> list[ViolationEvent]:
        if not self.enabled:
            return []

        new_events = []
        bicycle_tracks = [t for t in tracks if t.class_name == "bicycle"]

        for track in bicycle_tracks:
            if self._is_wrong_way(track):
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
                        violation_type="bicycle_wrong_way",
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
