"""Simple IoU-based tracker (SORT-like) for object tracking."""

from dataclasses import dataclass, field

import numpy as np

from detectors.yolo_detector import Detection


@dataclass
class Track:
    """A tracked object across frames."""

    track_id: int
    class_name: str
    bbox: np.ndarray
    center: np.ndarray
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    history: list[np.ndarray] = field(default_factory=list)

    @property
    def speed(self) -> float:
        """Compute speed in pixels/frame from recent history."""
        if len(self.history) < 2:
            return 0.0
        recent = self.history[-5:]
        if len(recent) < 2:
            return 0.0
        diffs = [np.linalg.norm(recent[i + 1] - recent[i]) for i in range(len(recent) - 1)]
        return float(np.mean(diffs))

    @property
    def direction(self) -> float:
        """Compute direction angle in degrees (-180 to 180)."""
        if len(self.history) < 2:
            return 0.0
        delta = self.history[-1] - self.history[-min(10, len(self.history))]
        return float(np.degrees(np.arctan2(delta[1], delta[0])))


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class SORTTracker:
    """Simple Online and Realtime Tracker using IoU matching."""

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Track]:
        """Update tracks with new detections. Returns active tracks."""
        # Build cost matrix
        if self.tracks and detections:
            cost = np.zeros((len(self.tracks), len(detections)))
            for i, trk in enumerate(self.tracks):
                for j, det in enumerate(detections):
                    cost[i, j] = _iou(trk.bbox, det.bbox)

            # Greedy matching
            matched_trk = set()
            matched_det = set()
            # Sort by IoU descending
            indices = np.argwhere(cost > self.iou_threshold)
            if len(indices) > 0:
                scores = [cost[i, j] for i, j in indices]
                for idx in np.argsort(scores)[::-1]:
                    i, j = indices[idx]
                    if i not in matched_trk and j not in matched_det:
                        self.tracks[i].bbox = detections[j].bbox
                        self.tracks[i].center = detections[j].center
                        self.tracks[i].history.append(detections[j].center.copy())
                        self.tracks[i].hits += 1
                        self.tracks[i].time_since_update = 0
                        matched_trk.add(i)
                        matched_det.add(j)

            # Unmatched detections -> new tracks
            for j, det in enumerate(detections):
                if j not in matched_det:
                    track = Track(
                        track_id=self._next_id,
                        class_name=det.class_name,
                        bbox=det.bbox,
                        center=det.center,
                        history=[det.center.copy()],
                    )
                    self._next_id += 1
                    self.tracks.append(track)
        elif detections:
            for det in detections:
                track = Track(
                    track_id=self._next_id,
                    class_name=det.class_name,
                    bbox=det.bbox,
                    center=det.center,
                    history=[det.center.copy()],
                )
                self._next_id += 1
                self.tracks.append(track)

        # Age all tracks
        for trk in self.tracks:
            trk.age += 1
            trk.time_since_update += 1

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Return confirmed tracks (recently matched: time_since_update == 1)
        return [t for t in self.tracks if t.hits >= self.min_hits and t.time_since_update <= 1]
