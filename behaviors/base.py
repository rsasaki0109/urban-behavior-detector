"""Base class for behavior analyzers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from trackers.sort_tracker import Track


@dataclass
class ViolationEvent:
    """A detected violation event."""

    violation_type: str
    track_id: int
    start_frame: int
    end_frame: int
    confidence: float

    def to_dict(self) -> dict:
        return {
            "type": self.violation_type,
            "track_id": self.track_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "confidence": round(self.confidence, 2),
        }


def compute_confidence(frames: list[int], base: float = 0.5,
                       max_conf: float = 0.95) -> float:
    """Compute violation confidence from frame evidence.

    Combines:
    - Duration: more frames = higher confidence
    - Consistency: ratio of detection frames to total span
    """
    n = len(frames)
    if n < 2:
        return base

    span = frames[-1] - frames[0] + 1
    consistency = n / span if span > 0 else 1.0

    # Duration component: logarithmic growth (diminishing returns)
    duration_score = min(0.3, 0.03 * n)

    # Consistency bonus: continuous detection is more reliable
    consistency_bonus = 0.1 * consistency

    return min(max_conf, base + duration_score + consistency_bonus)


class BehaviorAnalyzer(ABC):
    """Base class for all behavior analyzers."""

    def __init__(self, config: dict):
        self.config = config
        self.enabled = config.get("enabled", True)

    @abstractmethod
    def update(self, frame_idx: int, tracks: list[Track],
               all_detections: list) -> list[ViolationEvent]:
        """Process a frame and return any new violation events."""
        ...

    @abstractmethod
    def finalize(self) -> list[ViolationEvent]:
        """Finalize and return any pending violation events."""
        ...
