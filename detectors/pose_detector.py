"""YOLOv8-pose based person detector with keypoint extraction."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from ultralytics import YOLO


# COCO keypoint indices
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


@dataclass
class PoseDetection:
    """A person detection with keypoints."""

    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    keypoints: np.ndarray  # shape (17, 3) - x, y, confidence per keypoint

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2])

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    def keypoint(self, idx: int) -> Optional[np.ndarray]:
        """Get a keypoint [x, y] if confidence > 0.3, else None."""
        if self.keypoints[idx, 2] > 0.3:
            return self.keypoints[idx, :2]
        return None

    def wrist_near_nose(self, threshold_ratio: float = 0.15) -> bool:
        """Check if either wrist is near the nose (smoking gesture proxy)."""
        nose = self.keypoint(NOSE)
        if nose is None:
            return False

        threshold = self.height * threshold_ratio

        for wrist_idx in (LEFT_WRIST, RIGHT_WRIST):
            wrist = self.keypoint(wrist_idx)
            if wrist is not None:
                dist = np.linalg.norm(wrist - nose)
                if dist < threshold:
                    return True
        return False


class PoseDetector:
    """YOLOv8-pose wrapper for person pose estimation."""

    def __init__(self, model_path: str = "yolov8n-pose.pt",
                 confidence: float = 0.4):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> list[PoseDetection]:
        """Run pose estimation on a single frame."""
        results = self.model(frame, conf=self.confidence, verbose=False)

        detections = []
        for result in results:
            if result.boxes is None or result.keypoints is None:
                continue
            boxes = result.boxes
            kps = result.keypoints

            for i in range(len(boxes)):
                detections.append(PoseDetection(
                    bbox=boxes.xyxy[i].cpu().numpy(),
                    confidence=float(boxes.conf[i].item()),
                    keypoints=kps.data[i].cpu().numpy(),
                ))
        return detections
