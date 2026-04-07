#!/usr/bin/env python3
"""ROI polygon/rectangle calibration GUI tool.

Draw crossing_zones, signal_rois, and sidewalk_zones on a video frame,
then export as YAML for use in config files.

Controls:
  Left click  : Add point (polygon) or set corner (rectangle)
  Right click : Close current shape
  'n'         : Start a new shape
  't'         : Toggle polygon / rectangle mode
  'r'         : Reset current shape
  'f'         : Advance to next frame
  's'         : Save and export to YAML
  'q'         : Quit
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

# Colors (BGR)
COLORS = {
    "crossing_zone": (255, 150, 0),   # blue-ish
    "signal_roi": (0, 0, 255),        # red
    "sidewalk_zone": (0, 200, 0),     # green
}
DEFAULT_COLOR = (200, 200, 0)

LABEL_CHOICES = ["crossing_zone", "signal_roi", "sidewalk_zone"]


class Shape:
    """A single drawn shape (polygon or rectangle)."""

    def __init__(self, label: str, is_rect: bool = False):
        self.label = label
        self.points: list[list[int]] = []
        self.is_rect = is_rect
        self.closed = False

    @property
    def color(self):
        return COLORS.get(self.label, DEFAULT_COLOR)

    def as_bbox(self) -> list[int]:
        """Return [x1, y1, x2, y2] for rectangle shapes."""
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return [min(xs), min(ys), max(xs), max(ys)]


class Calibrator:
    """Interactive ROI calibration on a video frame."""

    WINDOW = "ROI Calibrator"

    def __init__(self, source: str, start_frame: int = 0):
        self.source = source
        self.start_frame = start_frame

        # Try to interpret source as camera index
        try:
            src = int(source)
        except ValueError:
            src = source
            if not Path(src).exists():
                print(f"Error: file not found: {src}", file=sys.stderr)
                sys.exit(1)

        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            print(f"Error: cannot open video source: {source}", file=sys.stderr)
            sys.exit(1)

        # Seek to start frame
        if start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        ret, self.frame = self.cap.read()
        if not ret:
            print("Error: cannot read frame from video", file=sys.stderr)
            sys.exit(1)

        self.frame_idx = start_frame
        self.shapes: list[Shape] = []
        self.current: Shape | None = None
        self.rect_mode = False
        self.mouse_pos = (0, 0)

    def _prompt_label(self) -> str:
        """Prompt user in terminal to pick a label for the new shape."""
        print("\nSelect label for new shape:")
        for i, name in enumerate(LABEL_CHOICES):
            print(f"  {i + 1}) {name}")
        while True:
            raw = input("Choice [1-3]: ").strip()
            if raw in ("1", "2", "3"):
                return LABEL_CHOICES[int(raw) - 1]
            print("Invalid choice, try again.")

    def _start_new_shape(self):
        """Finish current shape if any, then start a new one."""
        if self.current and len(self.current.points) >= 2:
            self.current.closed = True
            self.shapes.append(self.current)
        elif self.current and self.current.points:
            print("Shape discarded (too few points).")
        label = self._prompt_label()
        is_rect = self.rect_mode or (label == "signal_roi")
        if label == "signal_roi" and not is_rect:
            print("  (signal_roi forced to rectangle mode)")
            is_rect = True
        self.current = Shape(label, is_rect=is_rect)
        mode_str = "RECTANGLE" if is_rect else "POLYGON"
        print(f"Drawing {label} [{mode_str}] — click to add points, right-click to close.")

    def _close_current(self):
        """Close/finish the current shape."""
        if self.current is None:
            return
        min_pts = 2 if self.current.is_rect else 3
        if len(self.current.points) < min_pts:
            print(f"Need at least {min_pts} points to close shape.")
            return
        self.current.closed = True
        self.shapes.append(self.current)
        print(f"Shape closed: {self.current.label} with {len(self.current.points)} points.")
        self.current = None

    def _reset_current(self):
        """Discard current in-progress shape."""
        if self.current:
            self.current = Shape(self.current.label, is_rect=self.current.is_rect)
            print("Current shape reset.")

    def _mouse_cb(self, event, x, y, _flags, _param):
        self.mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current is None:
                self._start_new_shape()
            if self.current.is_rect and len(self.current.points) >= 2:
                # Rectangle already has 2 corners — replace second
                self.current.points[1] = [x, y]
            else:
                self.current.points.append([x, y])
            # Auto-close rectangle when 2 points placed
            if self.current.is_rect and len(self.current.points) == 2:
                self._close_current()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._close_current()

    def _draw_overlay(self) -> np.ndarray:
        """Return frame with all shapes drawn."""
        vis = self.frame.copy()
        overlay = vis.copy()

        all_shapes = list(self.shapes)
        if self.current and self.current.points:
            all_shapes.append(self.current)

        for shape in all_shapes:
            color = shape.color
            pts = shape.points
            if not pts:
                continue

            if shape.is_rect and len(pts) == 2:
                cv2.rectangle(overlay, tuple(pts[0]), tuple(pts[1]), color, -1)
                cv2.rectangle(vis, tuple(pts[0]), tuple(pts[1]), color, 2)
            elif len(pts) >= 2:
                arr = np.array(pts, dtype=np.int32)
                if shape.closed and len(pts) >= 3:
                    cv2.fillPoly(overlay, [arr], color)
                cv2.polylines(vis, [arr], isClosed=shape.closed, color=color, thickness=2)

            # Draw vertex circles
            for p in pts:
                cv2.circle(vis, tuple(p), 5, color, -1)

            # Label
            if pts:
                tx, ty = pts[0][0], pts[0][1] - 10
                cv2.putText(vis, shape.label, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Blend overlay for semi-transparent fill
        cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)

        # HUD
        mode_str = "RECT" if self.rect_mode else "POLY"
        status = (
            f"Frame {self.frame_idx} | Mode: {mode_str} | "
            f"Shapes: {len(self.shapes)} | "
            f"[n]ew [t]oggle [r]eset [f]rame [s]ave [q]uit"
        )
        cv2.putText(vis, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(vis, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        return vis

    def _advance_frame(self):
        """Read next frame from video."""
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame
            self.frame_idx += 1
            print(f"Advanced to frame {self.frame_idx}")
        else:
            print("No more frames.")

    def _export_yaml(self, output_path: str):
        """Export all shapes to YAML file."""
        signal_rois: list[list[int]] = []
        crossing_zones: list[list[list[int]]] = []
        sidewalk_zones: list[list[list[int]]] = []

        for shape in self.shapes:
            if shape.label == "signal_roi":
                signal_rois.append(shape.as_bbox())
            elif shape.label == "crossing_zone":
                crossing_zones.append(shape.points)
            elif shape.label == "sidewalk_zone":
                sidewalk_zones.append(shape.points)

        lines = [
            "# ROI Calibration Output",
            "# Copy the relevant sections into your config YAML",
            "",
        ]

        if signal_rois or crossing_zones:
            lines.append("signal_violation:")
            if signal_rois:
                lines.append("  signal_rois:")
                for roi in signal_rois:
                    lines.append(f"    - {roi}")
            if crossing_zones:
                lines.append("  crossing_zones:")
                for zone in crossing_zones:
                    lines.append(f"    - {zone}")

        if sidewalk_zones:
            if signal_rois or crossing_zones:
                lines.append("")
            lines.append("sidewalk_riding:")
            lines.append("  sidewalk_zones:")
            for zone in sidewalk_zones:
                lines.append(f"    - {zone}")

        lines.append("")
        content = "\n".join(lines)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(content)

        print(f"\nExported {len(self.shapes)} shapes to {output_path}")
        print("---")
        print(content)
        print("---")

    def run(self, output_path: str):
        """Main GUI loop."""
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.WINDOW, self._mouse_cb)

        print("ROI Calibrator ready.")
        print("Press 'n' to start drawing a new shape (or just click).")
        print()

        while True:
            vis = self._draw_overlay()
            cv2.imshow(self.WINDOW, vis)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("n"):
                self._start_new_shape()
            elif key == ord("t"):
                self.rect_mode = not self.rect_mode
                mode_str = "RECTANGLE" if self.rect_mode else "POLYGON"
                print(f"Switched to {mode_str} mode.")
            elif key == ord("r"):
                self._reset_current()
            elif key == ord("f"):
                self._advance_frame()
            elif key == ord("s"):
                # Close current shape first if valid
                if self.current and self.current.points:
                    min_pts = 2 if self.current.is_rect else 3
                    if len(self.current.points) >= min_pts:
                        self.current.closed = True
                        self.shapes.append(self.current)
                        self.current = None
                if not self.shapes:
                    print("No shapes to save.")
                else:
                    self._export_yaml(output_path)

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="ROI polygon/rectangle calibration tool. "
        "Draw zones on a video frame and export as YAML.",
    )
    parser.add_argument(
        "video",
        help="Input video file path or camera index (e.g. 0)",
    )
    parser.add_argument(
        "--frame", type=int, default=0,
        help="Start at this frame number (default: 0)",
    )
    parser.add_argument(
        "--output", "-o", default="outputs/roi_calibration.yaml",
        help="Output YAML path (default: outputs/roi_calibration.yaml)",
    )
    args = parser.parse_args()

    cal = Calibrator(args.video, start_frame=args.frame)
    cal.run(args.output)


if __name__ == "__main__":
    main()
