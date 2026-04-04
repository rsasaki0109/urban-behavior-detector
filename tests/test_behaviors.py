"""Tests for behavior analyzers."""

import numpy as np

from behaviors.base import ViolationEvent
from behaviors.walking_smoking import WalkingSmokingAnalyzer
from behaviors.bicycle_violation import BicycleViolationAnalyzer, _bbox_overlap_ratio
from behaviors.wrong_way import WrongWayAnalyzer
from detectors.yolo_detector import Detection
from trackers.sort_tracker import Track


def _make_track(track_id, class_name, bbox, speed_val=0.0):
    """Create a Track with controllable speed via history."""
    centers = []
    center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
    if speed_val > 0:
        # Create history that produces the desired speed
        for i in range(6):
            centers.append(np.array([center[0] - speed_val * (5 - i), center[1]]))
    else:
        centers = [center.copy()]

    return Track(
        track_id=track_id,
        class_name=class_name,
        bbox=np.array(bbox, dtype=float),
        center=center,
        hits=5,
        history=centers,
    )


def _make_detection(class_name, bbox, confidence=0.8):
    class_map = {"person": 0, "bicycle": 1, "umbrella": 25, "bottle": 39, "cell phone": 67, "handbag": 26}
    return Detection(
        bbox=np.array(bbox, dtype=float),
        class_id=class_map.get(class_name, 0),
        class_name=class_name,
        confidence=confidence,
    )


# --- ViolationEvent ---

class TestViolationEvent:
    def test_to_dict(self):
        e = ViolationEvent("walking_smoking", 1, 10, 20, 0.7777)
        d = e.to_dict()
        assert d["type"] == "walking_smoking"
        assert d["confidence"] == 0.78

    def test_to_dict_fields(self):
        e = ViolationEvent("bicycle_phone", 5, 100, 200, 0.9)
        d = e.to_dict()
        assert d["track_id"] == 5
        assert d["start_frame"] == 100
        assert d["end_frame"] == 200


# --- bbox_overlap_ratio ---

class TestBboxOverlapRatio:
    def test_full_overlap(self):
        outer = np.array([0, 0, 100, 100])
        inner = np.array([10, 10, 50, 50])
        ratio = _bbox_overlap_ratio(outer, inner)
        assert ratio == 1.0

    def test_no_overlap(self):
        a = np.array([0, 0, 10, 10])
        b = np.array([20, 20, 30, 30])
        assert _bbox_overlap_ratio(a, b) == 0.0

    def test_partial_overlap(self):
        a = np.array([0, 0, 10, 10])
        b = np.array([5, 0, 15, 10])
        # intersection = 5*10=50, area_b = 10*10=100
        assert abs(_bbox_overlap_ratio(a, b) - 0.5) < 1e-6

    def test_zero_area_b(self):
        a = np.array([0, 0, 10, 10])
        b = np.array([5, 5, 5, 5])
        assert _bbox_overlap_ratio(a, b) == 0.0


# --- WalkingSmokingAnalyzer ---

class TestWalkingSmokingAnalyzer:
    DEFAULT_CONFIG = {
        "enabled": True,
        "hand_mouth_distance": 0.12,
        "speed_threshold": 1.5,
        "min_duration_frames": 8,
        "confidence_threshold": 0.5,
    }

    def test_disabled(self):
        analyzer = WalkingSmokingAnalyzer({"enabled": False})
        track = _make_track(1, "person", [100, 100, 200, 300], speed_val=3.0)
        det = _make_detection("bottle", [140, 130, 150, 140])
        events = analyzer.update(0, [track], [det])
        assert events == []

    def test_no_violation_stationary_person(self):
        analyzer = WalkingSmokingAnalyzer(self.DEFAULT_CONFIG)
        track = _make_track(1, "person", [100, 100, 200, 300], speed_val=0.0)
        det = _make_detection("bottle", [140, 130, 150, 140])
        for i in range(20):
            events = analyzer.update(i, [track], [det])
        assert events == []

    def test_violation_detected(self):
        """Walking person with small object near mouth triggers violation."""
        analyzer = WalkingSmokingAnalyzer(self.DEFAULT_CONFIG)
        # Person bbox: [100, 100, 200, 300] -> height=200
        # Mouth region: y=[100+200*0.15, 100+200*0.30] = [130, 160]
        #               x=[100+100*0.2, 200-100*0.2] = [120, 180]
        track = _make_track(1, "person", [100, 100, 200, 300], speed_val=3.0)
        # Small bottle near mouth area
        det = _make_detection("bottle", [145, 140, 155, 150])

        all_events = []
        for i in range(15):
            events = analyzer.update(i, [track], [det])
            all_events.extend(events)

        assert len(all_events) == 1
        assert all_events[0].violation_type == "walking_smoking"
        assert all_events[0].track_id == 1

    def test_no_duplicate_reports(self):
        """Same track should only be reported once."""
        analyzer = WalkingSmokingAnalyzer(self.DEFAULT_CONFIG)
        track = _make_track(1, "person", [100, 100, 200, 300], speed_val=3.0)
        det = _make_detection("bottle", [145, 140, 155, 150])

        event_count = 0
        for i in range(30):
            events = analyzer.update(i, [track], [det])
            event_count += len(events)

        assert event_count == 1

    def test_finalize_returns_all_events(self):
        analyzer = WalkingSmokingAnalyzer(self.DEFAULT_CONFIG)
        track = _make_track(1, "person", [100, 100, 200, 300], speed_val=3.0)
        det = _make_detection("bottle", [145, 140, 155, 150])

        for i in range(15):
            analyzer.update(i, [track], [det])

        events = analyzer.finalize()
        assert len(events) == 1


# --- BicycleViolationAnalyzer ---

class TestBicycleViolationAnalyzer:
    PHONE_CONFIG = {
        "enabled": True,
        "phone_near_face_threshold": 0.15,
        "min_duration_frames": 6,
        "confidence_threshold": 0.5,
    }
    UMBRELLA_CONFIG = {
        "enabled": True,
        "umbrella_overlap_threshold": 0.3,
        "min_duration_frames": 5,
        "confidence_threshold": 0.5,
    }

    def test_disabled(self):
        analyzer = BicycleViolationAnalyzer(
            {"enabled": False}, {"enabled": False}
        )
        assert analyzer.enabled is False

    def test_phone_violation(self):
        """Cyclist with phone near face triggers bicycle_phone."""
        analyzer = BicycleViolationAnalyzer(self.PHONE_CONFIG, {"enabled": False})
        # Person on bicycle
        person = _make_track(1, "person", [100, 100, 200, 300], speed_val=5.0)
        bike = _make_track(2, "bicycle", [90, 200, 210, 320], speed_val=5.0)
        # Phone near face (upper 35% of person: y=100 to 170)
        phone_det = _make_detection("cell phone", [140, 120, 160, 145])

        all_events = []
        for i in range(10):
            events = analyzer.update(i, [person, bike], [phone_det])
            all_events.extend(events)

        assert len(all_events) == 1
        assert all_events[0].violation_type == "bicycle_phone"

    def test_umbrella_violation(self):
        """Cyclist with umbrella triggers bicycle_umbrella."""
        analyzer = BicycleViolationAnalyzer({"enabled": False}, self.UMBRELLA_CONFIG)
        person = _make_track(1, "person", [100, 100, 200, 300], speed_val=5.0)
        bike = _make_track(2, "bicycle", [90, 200, 210, 320], speed_val=5.0)
        # Large umbrella overlapping person
        umbrella_det = _make_detection("umbrella", [80, 80, 220, 200])

        all_events = []
        for i in range(10):
            events = analyzer.update(i, [person, bike], [umbrella_det])
            all_events.extend(events)

        assert len(all_events) == 1
        assert all_events[0].violation_type == "bicycle_umbrella"

    def test_no_violation_without_bicycle(self):
        """Person without bicycle should not trigger."""
        analyzer = BicycleViolationAnalyzer(self.PHONE_CONFIG, self.UMBRELLA_CONFIG)
        person = _make_track(1, "person", [100, 100, 200, 300], speed_val=5.0)
        phone_det = _make_detection("cell phone", [140, 120, 160, 145])

        all_events = []
        for i in range(10):
            events = analyzer.update(i, [person], [phone_det])
            all_events.extend(events)

        assert len(all_events) == 0

    def test_finalize(self):
        analyzer = BicycleViolationAnalyzer(self.PHONE_CONFIG, {"enabled": False})
        person = _make_track(1, "person", [100, 100, 200, 300], speed_val=5.0)
        bike = _make_track(2, "bicycle", [90, 200, 210, 320], speed_val=5.0)
        phone_det = _make_detection("cell phone", [140, 120, 160, 145])

        for i in range(10):
            analyzer.update(i, [person, bike], [phone_det])

        events = analyzer.finalize()
        assert len(events) == 1


# --- WrongWayAnalyzer ---

def _make_bicycle_track(track_id, bbox, direction_points):
    """Create a bicycle track with specific movement direction."""
    centers = [np.array(p, dtype=float) for p in direction_points]
    return Track(
        track_id=track_id,
        class_name="bicycle",
        bbox=np.array(bbox, dtype=float),
        center=centers[-1],
        hits=5,
        history=centers,
    )


class TestWrongWayAnalyzer:
    DEFAULT_CONFIG = {
        "enabled": True,
        "expected_direction": "right",
        "angle_tolerance": 45,
        "min_duration_frames": 10,
        "confidence_threshold": 0.6,
        "speed_threshold": 2.0,
    }

    def test_disabled(self):
        analyzer = WrongWayAnalyzer({"enabled": False})
        track = _make_bicycle_track(1, [100, 100, 200, 200],
                                     [[200, 150], [190, 150], [180, 150]])
        events = analyzer.update(0, [track], [])
        assert events == []

    def test_correct_direction_no_violation(self):
        """Bicycle moving right (expected direction) should not trigger."""
        analyzer = WrongWayAnalyzer(self.DEFAULT_CONFIG)
        # Moving right: x increasing
        points = [[100 + i * 5, 150] for i in range(6)]
        track = _make_bicycle_track(1, [100, 100, 200, 200], points)

        all_events = []
        for i in range(15):
            events = analyzer.update(i, [track], [])
            all_events.extend(events)

        assert len(all_events) == 0

    def test_wrong_way_detected(self):
        """Bicycle moving left when expected right should trigger."""
        analyzer = WrongWayAnalyzer(self.DEFAULT_CONFIG)
        # Moving left: x decreasing (direction ~180 degrees)
        points = [[200 - i * 5, 150] for i in range(6)]
        track = _make_bicycle_track(1, [100, 100, 200, 200], points)

        all_events = []
        for i in range(15):
            events = analyzer.update(i, [track], [])
            all_events.extend(events)

        assert len(all_events) == 1
        assert all_events[0].violation_type == "bicycle_wrong_way"

    def test_stationary_no_violation(self):
        """Stationary bicycle should not trigger."""
        analyzer = WrongWayAnalyzer(self.DEFAULT_CONFIG)
        track = _make_bicycle_track(1, [100, 100, 200, 200], [[150, 150]])

        all_events = []
        for i in range(15):
            events = analyzer.update(i, [track], [])
            all_events.extend(events)

        assert len(all_events) == 0

    def test_person_ignored(self):
        """Non-bicycle tracks should be ignored."""
        analyzer = WrongWayAnalyzer(self.DEFAULT_CONFIG)
        track = _make_track(1, "person", [100, 100, 200, 200], speed_val=5.0)

        all_events = []
        for i in range(15):
            events = analyzer.update(i, [track], [])
            all_events.extend(events)

        assert len(all_events) == 0
