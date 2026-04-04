"""Tests for Detection dataclass."""

import numpy as np

from detectors.yolo_detector import Detection


def test_detection_center():
    det = Detection(bbox=np.array([10, 20, 30, 40]), class_id=0, class_name="person", confidence=0.9)
    center = det.center
    np.testing.assert_array_almost_equal(center, [20.0, 30.0])


def test_detection_width_height():
    det = Detection(bbox=np.array([10, 20, 50, 80]), class_id=0, class_name="person", confidence=0.9)
    assert det.width == 40.0
    assert det.height == 60.0
