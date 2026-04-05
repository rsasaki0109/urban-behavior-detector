"""Tests for pose detection and pose-based smoking detection."""

import numpy as np

from behaviors.walking_smoking import WalkingSmokingAnalyzer
from detectors.pose_detector import (
    PoseDetection, NOSE, LEFT_WRIST, RIGHT_WRIST,
)
from trackers.sort_tracker import Track


def _make_keypoints(overrides: dict):
    """Create keypoints array (17, 3) with overrides {index: (x, y, conf)}."""
    kps = np.zeros((17, 3))
    for idx, (x, y, conf) in overrides.items():
        kps[idx] = [x, y, conf]
    return kps


def _make_pose(bbox, keypoints):
    return PoseDetection(
        bbox=np.array(bbox, dtype=float),
        confidence=0.9,
        keypoints=keypoints,
    )


def _make_walking_track(track_id, bbox, speed_val=3.0):
    center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
    centers = [np.array([center[0] - speed_val * (5 - i), center[1]]) for i in range(6)]
    return Track(
        track_id=track_id,
        class_name="person",
        bbox=np.array(bbox, dtype=float),
        center=center,
        hits=5,
        history=centers,
    )


class TestPoseDetection:
    def test_keypoint_valid(self):
        kps = _make_keypoints({NOSE: (100, 50, 0.9)})
        pose = _make_pose([50, 0, 150, 200], kps)
        nose = pose.keypoint(NOSE)
        assert nose is not None
        np.testing.assert_array_almost_equal(nose, [100, 50])

    def test_keypoint_low_confidence(self):
        kps = _make_keypoints({NOSE: (100, 50, 0.1)})
        pose = _make_pose([50, 0, 150, 200], kps)
        assert pose.keypoint(NOSE) is None

    def test_wrist_near_nose_true(self):
        kps = _make_keypoints({
            NOSE: (100, 50, 0.9),
            RIGHT_WRIST: (105, 55, 0.8),
        })
        pose = _make_pose([50, 0, 150, 200], kps)
        assert pose.wrist_near_nose(0.15) is True

    def test_wrist_near_nose_false(self):
        kps = _make_keypoints({
            NOSE: (100, 50, 0.9),
            RIGHT_WRIST: (100, 180, 0.8),
        })
        pose = _make_pose([50, 0, 150, 200], kps)
        assert pose.wrist_near_nose(0.15) is False

    def test_wrist_near_nose_no_nose(self):
        kps = _make_keypoints({RIGHT_WRIST: (100, 50, 0.8)})
        pose = _make_pose([50, 0, 150, 200], kps)
        assert pose.wrist_near_nose(0.15) is False

    def test_left_wrist_also_works(self):
        kps = _make_keypoints({
            NOSE: (100, 50, 0.9),
            LEFT_WRIST: (103, 52, 0.8),
        })
        pose = _make_pose([50, 0, 150, 200], kps)
        assert pose.wrist_near_nose(0.15) is True


class TestSmokingOscillation:
    """Test the oscillation-based smoking detection."""

    CONFIG = {
        "enabled": True,
        "speed_threshold": 1.5,
        "min_duration_frames": 5,
        "confidence_threshold": 0.4,
        "pose_wrist_nose_ratio": 0.15,
        "min_oscillations": 2,
    }

    def test_oscillation_alone_not_enough(self):
        """Oscillation without cigarette detection should NOT trigger."""
        analyzer = WalkingSmokingAnalyzer(self.CONFIG)
        track = _make_walking_track(1, [50, 0, 150, 200])
        bbox = [50, 0, 150, 200]

        all_events = []
        for i in range(40):
            if i % 4 < 2:
                kps = _make_keypoints({NOSE: (100, 30, 0.9), RIGHT_WRIST: (103, 33, 0.8)})
            else:
                kps = _make_keypoints({NOSE: (100, 30, 0.9), RIGHT_WRIST: (100, 150, 0.8)})
            pose = _make_pose(bbox, kps)
            events = analyzer.update(i, [track], [], pose_detections=[pose])
            all_events.extend(events)

        # Oscillation alone is not enough - requires cigarette detection
        assert len(all_events) == 0

    def test_cigarette_plus_oscillation_triggers(self):
        """Cigarette + oscillation = high confidence detection."""
        from detectors.cigarette_detector import CigaretteDetection
        analyzer = WalkingSmokingAnalyzer(self.CONFIG)
        track = _make_walking_track(1, [50, 0, 150, 200])
        bbox = [50, 0, 150, 200]
        cig = CigaretteDetection(bbox=np.array([90, 25, 110, 40], dtype=float), confidence=0.4)

        all_events = []
        for i in range(40):
            if i % 4 < 2:
                kps = _make_keypoints({NOSE: (100, 30, 0.9), RIGHT_WRIST: (103, 33, 0.8)})
            else:
                kps = _make_keypoints({NOSE: (100, 30, 0.9), RIGHT_WRIST: (100, 150, 0.8)})
            pose = _make_pose(bbox, kps)
            events = analyzer.update(i, [track], [], pose_detections=[pose],
                                     cigarette_detections=[cig])
            all_events.extend(events)

        assert len(all_events) == 1
        assert all_events[0].confidence >= 0.8  # high conf from both signals

    def test_no_pose_no_crash(self):
        """Without pose data, no events and no crash."""
        analyzer = WalkingSmokingAnalyzer(self.CONFIG)
        track = _make_walking_track(1, [50, 0, 150, 200])

        all_events = []
        for i in range(20):
            events = analyzer.update(i, [track], [], pose_detections=None)
            all_events.extend(events)

        assert len(all_events) == 0
