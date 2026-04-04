"""Tests for walking phone detection."""

import numpy as np

from behaviors.walking_phone import WalkingPhoneAnalyzer
from detectors.pose_detector import PoseDetection, NOSE, RIGHT_WRIST
from detectors.yolo_detector import Detection
from trackers.sort_tracker import Track


def _make_keypoints(overrides: dict):
    kps = np.zeros((17, 3))
    for idx, (x, y, conf) in overrides.items():
        kps[idx] = [x, y, conf]
    return kps


def _make_walking_track(track_id, bbox, speed_val=3.0):
    center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
    centers = [np.array([center[0] - speed_val * (5 - i), center[1]]) for i in range(6)]
    return Track(
        track_id=track_id, class_name="person",
        bbox=np.array(bbox, dtype=float), center=center,
        hits=5, history=centers,
    )


class TestWalkingPhoneAnalyzer:
    CONFIG = {
        "enabled": True,
        "speed_threshold": 1.0,
        "min_duration_frames": 5,
        "confidence_threshold": 0.4,
        "pose_wrist_nose_ratio": 0.20,
        "min_near_ratio": 0.7,
    }

    def test_disabled(self):
        analyzer = WalkingPhoneAnalyzer({"enabled": False})
        track = _make_walking_track(1, [50, 0, 150, 200])
        events = analyzer.update(0, [track], [])
        assert events == []

    def test_constant_near_face_triggers(self):
        """Hand constantly near face = phone usage, should trigger."""
        analyzer = WalkingPhoneAnalyzer(self.CONFIG)
        track = _make_walking_track(1, [50, 0, 150, 200])
        bbox = [50, 0, 150, 200]

        # Hand always near nose
        kps = _make_keypoints({
            NOSE: (100, 30, 0.9),
            RIGHT_WRIST: (103, 33, 0.8),
        })
        pose = PoseDetection(
            bbox=np.array(bbox, dtype=float),
            confidence=0.9, keypoints=kps,
        )

        all_events = []
        for i in range(20):
            events = analyzer.update(i, [track], [], pose_detections=[pose])
            all_events.extend(events)

        assert len(all_events) == 1
        assert all_events[0].violation_type == "walking_phone"

    def test_oscillation_no_trigger(self):
        """Hand oscillating (smoking pattern) should NOT trigger phone."""
        analyzer = WalkingPhoneAnalyzer(self.CONFIG)
        track = _make_walking_track(1, [50, 0, 150, 200])
        bbox = [50, 0, 150, 200]

        all_events = []
        for i in range(30):
            if i % 4 < 2:
                kps = _make_keypoints({NOSE: (100, 30, 0.9), RIGHT_WRIST: (103, 33, 0.8)})
            else:
                kps = _make_keypoints({NOSE: (100, 30, 0.9), RIGHT_WRIST: (100, 150, 0.8)})
            pose = PoseDetection(bbox=np.array(bbox, dtype=float), confidence=0.9, keypoints=kps)
            events = analyzer.update(i, [track], [], pose_detections=[pose])
            all_events.extend(events)

        assert len(all_events) == 0

    def test_phone_object_near_face_triggers(self):
        """Cell phone detection near face + walking triggers."""
        analyzer = WalkingPhoneAnalyzer(self.CONFIG)
        track = _make_walking_track(1, [100, 100, 200, 300])
        phone = Detection(
            bbox=np.array([140, 120, 160, 145], dtype=float),
            class_id=67, class_name="cell phone", confidence=0.7,
        )

        all_events = []
        for i in range(15):
            events = analyzer.update(i, [track], [phone])
            all_events.extend(events)

        assert len(all_events) == 1
        assert all_events[0].violation_type == "walking_phone"

    def test_stationary_no_trigger(self):
        """Stationary person with phone should not trigger."""
        analyzer = WalkingPhoneAnalyzer(self.CONFIG)
        track = _make_walking_track(1, [100, 100, 200, 300], speed_val=0.0)
        phone = Detection(
            bbox=np.array([140, 120, 160, 145], dtype=float),
            class_id=67, class_name="cell phone", confidence=0.7,
        )

        all_events = []
        for i in range(15):
            events = analyzer.update(i, [track], [phone])
            all_events.extend(events)

        assert len(all_events) == 0
