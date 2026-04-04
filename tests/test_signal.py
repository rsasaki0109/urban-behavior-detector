"""Tests for signal violation detection."""

import numpy as np

from behaviors.signal_violation import SignalViolationAnalyzer
from detectors.signal_detector import (
    SignalColor, SignalDetection, classify_signal_color,
)
from trackers.sort_tracker import Track


def _make_bicycle_track(track_id, bbox, speed_val=5.0):
    center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
    centers = [np.array([center[0] - speed_val * (5 - i), center[1]]) for i in range(6)]
    return Track(
        track_id=track_id,
        class_name="bicycle",
        bbox=np.array(bbox, dtype=float),
        center=center,
        hits=5,
        history=centers,
    )


def _make_signal(bbox, color):
    return SignalDetection(
        bbox=np.array(bbox, dtype=float),
        color=color,
        confidence=0.9,
    )


class TestSignalColorClassification:
    def test_red_signal(self):
        # Create a small image with a red region
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[:, :] = (0, 0, 255)  # BGR red
        color = classify_signal_color(img, np.array([0, 0, 50, 50]))
        assert color == SignalColor.RED

    def test_green_signal(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[:, :] = (0, 200, 0)  # BGR green
        color = classify_signal_color(img, np.array([0, 0, 50, 50]))
        assert color == SignalColor.GREEN

    def test_unknown_for_dark(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)  # all black
        color = classify_signal_color(img, np.array([0, 0, 50, 50]))
        assert color == SignalColor.UNKNOWN

    def test_empty_bbox(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        color = classify_signal_color(img, np.array([10, 10, 10, 10]))
        assert color == SignalColor.UNKNOWN


class TestSignalViolationAnalyzer:
    CONFIG = {
        "enabled": True,
        "proximity_threshold": 200,
        "min_duration_frames": 5,
        "confidence_threshold": 0.5,
        "min_crossing_speed": 2.0,
    }

    def test_disabled(self):
        analyzer = SignalViolationAnalyzer({"enabled": False})
        track = _make_bicycle_track(1, [100, 100, 200, 200])
        signal = _make_signal([150, 50, 180, 90], SignalColor.RED)
        events = analyzer.update(0, [track], [], [signal])
        assert events == []

    def test_red_light_violation(self):
        analyzer = SignalViolationAnalyzer(self.CONFIG)
        # Bicycle near a red signal
        track = _make_bicycle_track(1, [100, 100, 200, 200])
        signal = _make_signal([130, 50, 170, 90], SignalColor.RED)

        all_events = []
        for i in range(10):
            events = analyzer.update(i, [track], [], [signal])
            all_events.extend(events)

        assert len(all_events) == 1
        assert all_events[0].violation_type == "signal_violation"

    def test_green_light_no_violation(self):
        analyzer = SignalViolationAnalyzer(self.CONFIG)
        track = _make_bicycle_track(1, [100, 100, 200, 200])
        signal = _make_signal([130, 50, 170, 90], SignalColor.GREEN)

        all_events = []
        for i in range(10):
            events = analyzer.update(i, [track], [], [signal])
            all_events.extend(events)

        assert len(all_events) == 0

    def test_far_from_signal_no_violation(self):
        analyzer = SignalViolationAnalyzer(self.CONFIG)
        track = _make_bicycle_track(1, [100, 100, 200, 200])
        # Signal far away (>200px)
        signal = _make_signal([500, 50, 530, 90], SignalColor.RED)

        all_events = []
        for i in range(10):
            events = analyzer.update(i, [track], [], [signal])
            all_events.extend(events)

        assert len(all_events) == 0

    def test_stationary_no_violation(self):
        """Stopped bicycle at red light should not trigger."""
        analyzer = SignalViolationAnalyzer(self.CONFIG)
        track = _make_bicycle_track(1, [100, 100, 200, 200], speed_val=0.0)
        signal = _make_signal([130, 50, 170, 90], SignalColor.RED)

        all_events = []
        for i in range(10):
            events = analyzer.update(i, [track], [], [signal])
            all_events.extend(events)

        assert len(all_events) == 0

    def test_no_signals_no_crash(self):
        analyzer = SignalViolationAnalyzer(self.CONFIG)
        track = _make_bicycle_track(1, [100, 100, 200, 200])
        events = analyzer.update(0, [track], [], None)
        assert events == []
