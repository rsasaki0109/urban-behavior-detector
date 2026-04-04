"""Traffic signal color detection using HSV analysis.

Uses YOLO to detect traffic lights (COCO class 9), then analyzes the
cropped region with HSV color thresholds to determine the signal state.
"""

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class SignalColor(Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    UNKNOWN = "unknown"


@dataclass
class SignalDetection:
    """A detected traffic signal with color state."""

    bbox: np.ndarray  # [x1, y1, x2, y2]
    color: SignalColor
    confidence: float

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.bbox[0] + self.bbox[2]) / 2,
                         (self.bbox[1] + self.bbox[3]) / 2])


# HSV ranges for signal colors
_HSV_RANGES = {
    SignalColor.RED: [
        (np.array([0, 100, 100]), np.array([10, 255, 255])),
        (np.array([160, 100, 100]), np.array([180, 255, 255])),
    ],
    SignalColor.YELLOW: [
        (np.array([15, 100, 100]), np.array([35, 255, 255])),
    ],
    SignalColor.GREEN: [
        (np.array([40, 80, 80]), np.array([90, 255, 255])),
    ],
}


def classify_signal_color(frame: np.ndarray, bbox: np.ndarray,
                          min_pixel_ratio: float = 0.02) -> SignalColor:
    """Classify the color of a traffic signal from its bounding box region.

    Args:
        frame: Full frame (BGR).
        bbox: [x1, y1, x2, y2] of the traffic light.
        min_pixel_ratio: Minimum ratio of colored pixels to trigger a color.

    Returns:
        Detected SignalColor.
    """
    x1, y1, x2, y2 = bbox.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return SignalColor.UNKNOWN

    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    total_pixels = crop.shape[0] * crop.shape[1]

    if total_pixels == 0:
        return SignalColor.UNKNOWN

    best_color = SignalColor.UNKNOWN
    best_ratio = min_pixel_ratio

    for color, ranges in _HSV_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv, lower, upper)
        ratio = np.count_nonzero(mask) / total_pixels
        if ratio > best_ratio:
            best_ratio = ratio
            best_color = color

    return best_color
