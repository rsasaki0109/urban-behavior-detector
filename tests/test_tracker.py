"""Tests for SORT tracker."""

import numpy as np

from detectors.yolo_detector import Detection
from trackers.sort_tracker import Track, SORTTracker, _iou


class TestIoU:
    def test_identical_boxes(self):
        box = np.array([0, 0, 10, 10])
        assert _iou(box, box) == 1.0

    def test_no_overlap(self):
        a = np.array([0, 0, 10, 10])
        b = np.array([20, 20, 30, 30])
        assert _iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = np.array([0, 0, 10, 10])
        b = np.array([5, 5, 15, 15])
        # intersection = 5*5=25, union = 100+100-25=175
        assert abs(_iou(a, b) - 25 / 175) < 1e-6

    def test_zero_area(self):
        a = np.array([0, 0, 0, 0])
        b = np.array([0, 0, 10, 10])
        assert _iou(a, b) == 0.0


class TestTrack:
    def _make_track(self, history_points):
        centers = [np.array(p, dtype=float) for p in history_points]
        return Track(
            track_id=1,
            class_name="person",
            bbox=np.array([0, 0, 50, 100]),
            center=centers[-1] if centers else np.array([25, 50]),
            history=centers,
        )

    def test_speed_no_history(self):
        t = self._make_track([])
        assert t.speed == 0.0

    def test_speed_single_point(self):
        t = self._make_track([[0, 0]])
        assert t.speed == 0.0

    def test_speed_uniform_motion(self):
        t = self._make_track([[0, 0], [3, 4], [6, 8]])
        # diffs: 5.0, 5.0 -> mean 5.0
        assert abs(t.speed - 5.0) < 1e-6

    def test_direction_right(self):
        t = self._make_track([[0, 0], [10, 0]])
        assert abs(t.direction - 0.0) < 1e-6

    def test_direction_down(self):
        t = self._make_track([[0, 0], [0, 10]])
        assert abs(t.direction - 90.0) < 1e-6


class TestSORTTracker:
    def _det(self, bbox, class_name="person"):
        return Detection(
            bbox=np.array(bbox, dtype=float),
            class_id=0,
            class_name=class_name,
            confidence=0.9,
        )

    def test_new_detections_create_tracks(self):
        tracker = SORTTracker(min_hits=1)
        dets = [self._det([0, 0, 10, 10])]
        # First frame: track created, not yet confirmed (needs match)
        tracker.update(dets)
        assert len(tracker.tracks) == 1
        # Second frame: track matched and confirmed
        tracks = tracker.update(dets)
        assert len(tracks) == 1
        assert tracks[0].track_id == 1

    def test_consistent_tracking(self):
        tracker = SORTTracker(min_hits=1)
        tracks = []
        for _ in range(5):
            tracks = tracker.update([self._det([0, 0, 10, 10])])
        assert len(tracks) == 1
        assert tracks[0].hits >= 3

    def test_no_detections_ages_tracks(self):
        tracker = SORTTracker(min_hits=1, max_age=3)
        tracker.update([self._det([0, 0, 10, 10])])
        for _ in range(5):
            tracker.update([])
        # Track should be removed after max_age
        assert len(tracker.tracks) == 0
