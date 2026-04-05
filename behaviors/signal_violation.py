"""Red light running detection for bicycles, pedestrians, and vehicles.

Logic:
- A traffic signal is detected as RED (via YOLO or fixed ROI)
- A target (bicycle/person/car) moves through a defined crossing zone while red
- The crossing zone is a polygon defined in config (required for accurate detection)
- Without a crossing zone, falls back to signal proximity check
"""

from collections import defaultdict

import cv2
import numpy as np

from behaviors.base import BehaviorAnalyzer, ViolationEvent, compute_confidence
from detectors.signal_detector import SignalColor, SignalDetection
from trackers.sort_tracker import Track


class SignalViolationAnalyzer(BehaviorAnalyzer):
    """Detect red light violations using crossing zone + signal state."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.min_duration = config.get("min_duration_frames", 5)
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.min_crossing_speed = config.get("min_crossing_speed", 2.0)
        self.detect_pedestrians = config.get("detect_pedestrians", True)
        self.detect_vehicles = config.get("detect_vehicles", True)
        self.target_classes = ["bicycle"]
        if self.detect_pedestrians:
            self.target_classes.append("person")
        if self.detect_vehicles:
            self.target_classes.extend(["car", "motorcycle", "bus", "truck"])

        # Crossing zone: polygon where people/vehicles cross
        raw_zones = config.get("crossing_zones", [])
        self.crossing_zones = [
            np.array(zone, dtype=np.int32) for zone in raw_zones
        ]

        # Fallback: proximity to signal (less accurate)
        self.proximity_threshold = config.get("proximity_threshold", 200)

        self._candidates: dict[int, list[int]] = defaultdict(list)
        self._reported: set[int] = set()
        self._events: list[ViolationEvent] = []

    def _is_in_crossing_zone(self, track: Track) -> bool:
        """Check if track center is inside any crossing zone polygon."""
        pt = (int(track.center[0]), int(track.center[1]))
        for zone in self.crossing_zones:
            if cv2.pointPolygonTest(zone, pt, False) >= 0:
                return True
        return False

    def _is_near_signal(self, track: Track, signal: SignalDetection) -> bool:
        """Fallback: check proximity to signal."""
        dist = np.linalg.norm(track.center - signal.center)
        return dist < self.proximity_threshold

    def update(self, frame_idx: int, tracks: list[Track],
               all_detections: list,
               signal_detections: list[SignalDetection] | None = None,
               ) -> list[ViolationEvent]:
        if not self.enabled or not signal_detections:
            return []

        red_signals = [s for s in signal_detections if s.color == SignalColor.RED]
        if not red_signals:
            return []

        new_events = []
        target_tracks = [t for t in tracks if t.class_name in self.target_classes]

        for track in target_tracks:
            if track.speed < self.min_crossing_speed:
                continue

            # Use crossing zone if defined (accurate), else proximity (fallback)
            if self.crossing_zones:
                in_violation = self._is_in_crossing_zone(track)
            else:
                in_violation = any(
                    self._is_near_signal(track, sig) for sig in red_signals
                )

            if in_violation:
                self._candidates[track.track_id].append(frame_idx)
            else:
                frames = self._candidates.get(track.track_id, [])
                if frames and frame_idx - frames[-1] > 3:
                    self._candidates[track.track_id] = []

            frames = self._candidates.get(track.track_id, [])
            if (len(frames) >= self.min_duration
                    and track.track_id not in self._reported):
                conf = compute_confidence(frames, max_conf=0.9)
                if conf >= self.confidence_threshold:
                    event = ViolationEvent(
                        violation_type="signal_violation",
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
