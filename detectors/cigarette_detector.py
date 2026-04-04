"""Cigarette detection using a fine-tuned YOLO model.

Uses a specialized YOLO model trained on cigarette images to detect
cigarettes in video frames. Intended to be used alongside the main
person detector for walking-smoking analysis.
"""

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass
class CigaretteDetection:
    """A detected cigarette."""

    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.bbox[0] + self.bbox[2]) / 2,
                         (self.bbox[1] + self.bbox[3]) / 2])


class CigaretteDetector:
    """YOLO-based cigarette detector."""

    def __init__(self, model_path: str = "models/cigarette_yolov11m.pt",
                 confidence: float = 0.25):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> list[CigaretteDetection]:
        """Run cigarette detection on a frame."""
        results = self.model(frame, conf=self.confidence, verbose=False)

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for i in range(len(result.boxes)):
                detections.append(CigaretteDetection(
                    bbox=result.boxes.xyxy[i].cpu().numpy(),
                    confidence=float(result.boxes.conf[i].item()),
                ))
        return detections
