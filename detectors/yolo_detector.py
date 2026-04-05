"""YOLO-based object detector for urban scene analysis."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """Single detection result."""

    bbox: np.ndarray  # [x1, y1, x2, y2]
    class_id: int
    class_name: str
    confidence: float

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2])

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


class YOLODetector:
    """Wrapper around YOLOv8 for multi-class detection."""

    COCO_NAMES = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
        9: "traffic light",
        25: "umbrella",
        26: "handbag",
        27: "tie",
        39: "bottle",
        67: "cell phone",
    }

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.4,
                 iou_threshold: float = 0.5, classes: Optional[list[int]] = None):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.classes = classes

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a single frame."""
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=self.classes,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                detections.append(Detection(
                    bbox=boxes.xyxy[i].cpu().numpy(),
                    class_id=cls_id,
                    class_name=self.COCO_NAMES.get(cls_id, f"class_{cls_id}"),
                    confidence=float(boxes.conf[i].item()),
                ))
        return detections
