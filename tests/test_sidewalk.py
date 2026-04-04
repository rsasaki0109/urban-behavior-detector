"""Tests for sidewalk riding detection."""

import numpy as np

from behaviors.sidewalk_riding import SidewalkRidingAnalyzer
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


class TestSidewalkRidingAnalyzer:
    # Sidewalk zone: rectangle from (0,0) to (300,100)
    CONFIG = {
        "enabled": True,
        "sidewalk_zones": [[[0, 0], [300, 0], [300, 100], [0, 100]]],
        "min_duration_frames": 8,
        "confidence_threshold": 0.5,
        "min_speed": 1.5,
    }

    def test_disabled(self):
        analyzer = SidewalkRidingAnalyzer({"enabled": False})
        track = _make_bicycle_track(1, [50, 20, 100, 80])
        events = analyzer.update(0, [track], [])
        assert events == []

    def test_no_zones(self):
        analyzer = SidewalkRidingAnalyzer({"enabled": True, "sidewalk_zones": []})
        track = _make_bicycle_track(1, [50, 20, 100, 80])
        events = analyzer.update(0, [track], [])
        assert events == []

    def test_bicycle_on_sidewalk_violation(self):
        """Bicycle riding on sidewalk triggers violation."""
        analyzer = SidewalkRidingAnalyzer(self.CONFIG)
        # Center at (75, 50) which is inside the sidewalk zone
        track = _make_bicycle_track(1, [50, 20, 100, 80])

        all_events = []
        for i in range(15):
            events = analyzer.update(i, [track], [])
            all_events.extend(events)

        assert len(all_events) == 1
        assert all_events[0].violation_type == "sidewalk_riding"

    def test_bicycle_on_road_no_violation(self):
        """Bicycle on road (outside sidewalk) should not trigger."""
        analyzer = SidewalkRidingAnalyzer(self.CONFIG)
        # Center at (75, 250) which is outside the sidewalk zone
        track = _make_bicycle_track(1, [50, 200, 100, 300])

        all_events = []
        for i in range(15):
            events = analyzer.update(i, [track], [])
            all_events.extend(events)

        assert len(all_events) == 0

    def test_stationary_no_violation(self):
        """Stationary bicycle on sidewalk should not trigger."""
        analyzer = SidewalkRidingAnalyzer(self.CONFIG)
        track = _make_bicycle_track(1, [50, 20, 100, 80], speed_val=0.0)

        all_events = []
        for i in range(15):
            events = analyzer.update(i, [track], [])
            all_events.extend(events)

        assert len(all_events) == 0

    def test_person_ignored(self):
        """Person on sidewalk should not trigger."""
        analyzer = SidewalkRidingAnalyzer(self.CONFIG)
        center = np.array([75.0, 50.0])
        track = Track(
            track_id=1, class_name="person",
            bbox=np.array([50, 20, 100, 80], dtype=float),
            center=center, hits=5,
            history=[np.array([75.0 - 5 * (5 - i), 50.0]) for i in range(6)],
        )

        all_events = []
        for i in range(15):
            events = analyzer.update(i, [track], [])
            all_events.extend(events)

        assert len(all_events) == 0
