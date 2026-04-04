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
        # Person height = 200, threshold = 0.15 * 200 = 30
        # Wrist at (105, 55), nose at (100, 50), distance ~7
        kps = _make_keypoints({
            NOSE: (100, 50, 0.9),
            RIGHT_WRIST: (105, 55, 0.8),
        })
        pose = _make_pose([50, 0, 150, 200], kps)
        assert pose.wrist_near_nose(0.15) is True

    def test_wrist_near_nose_false(self):
        # Wrist far from nose
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


class TestWalkingSmokingWithPose:
    CONFIG = {
        "enabled": True,
        "hand_mouth_distance": 0.12,
        "speed_threshold": 1.5,
        "min_duration_frames": 8,
        "confidence_threshold": 0.5,
        "pose_wrist_nose_ratio": 0.15,
    }

    def test_pose_based_detection(self):
        """Smoking detected via pose keypoints."""
        analyzer = WalkingSmokingAnalyzer(self.CONFIG)
        track = _make_walking_track(1, [50, 0, 150, 200])

        # Pose with wrist near nose
        kps = _make_keypoints({
            NOSE: (100, 30, 0.9),
            RIGHT_WRIST: (105, 35, 0.8),
        })
        pose = _make_pose([50, 0, 150, 200], kps)

        all_events = []
        for i in range(15):
            events = analyzer.update(i, [track], [], pose_detections=[pose])
            all_events.extend(events)

        assert len(all_events) == 1
        assert all_events[0].violation_type == "walking_smoking"
        # Pose-based gets confidence boost
        assert all_events[0].confidence > 0.7

    def test_no_detection_when_wrist_far(self):
        """No violation when wrist is far from nose."""
        analyzer = WalkingSmokingAnalyzer(self.CONFIG)
        track = _make_walking_track(1, [50, 0, 150, 200])

        kps = _make_keypoints({
            NOSE: (100, 30, 0.9),
            RIGHT_WRIST: (100, 180, 0.8),
        })
        pose = _make_pose([50, 0, 150, 200], kps)

        all_events = []
        for i in range(15):
            events = analyzer.update(i, [track], [], pose_detections=[pose])
            all_events.extend(events)

        assert len(all_events) == 0

    def test_fallback_to_proxy_without_pose(self):
        """Falls back to proxy detection when no pose data."""
        from detectors.yolo_detector import Detection
        analyzer = WalkingSmokingAnalyzer(self.CONFIG)
        track = _make_walking_track(1, [100, 100, 200, 300])
        det = Detection(
            bbox=np.array([145, 140, 155, 150], dtype=float),
            class_id=39, class_name="bottle", confidence=0.8,
        )

        all_events = []
        for i in range(15):
            events = analyzer.update(i, [track], [det], pose_detections=None)
            all_events.extend(events)

        assert len(all_events) == 1
