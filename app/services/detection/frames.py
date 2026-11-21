"""Panel frame lines: long straight edges, used where there is no background-coloured gutter.

Hand-drawn frames are broken by balloons and art, so a line is accepted from a Hough segment
that covers only part of a panel side. Two questions are answered from them:

* where can a region be cut (:meth:`FrameLines.cut_row`): a horizontal line that crosses a good
  part of the region is a frame edge, so the region splits there;
* is a region framed at all (:meth:`FrameLines.sides`): title lettering and caption strips have
  at most one frame line on their boundary, panels have several.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

_Region = tuple[int, int, int, int]  # y0, y1, x0, x1 (end-exclusive)
_THIN_GUTTER = 0.04  # of the page width: narrower gaps between pieces are gutters, not dark art


@dataclass(frozen=True, slots=True)
class FrameLines:
    """Axis-aligned line segments: horizontal (y, x0, x1) and vertical (x, y0, y1)."""

    horizontal: list[tuple[int, int, int]] = field(default_factory=list)
    vertical: list[tuple[int, int, int]] = field(default_factory=list)
    height: int = 0
    width: int = 0

    @classmethod
    def find(
        cls,
        page: NDArray[np.uint8],
        *,
        min_length: float = 0.12,
        max_tilt: float = 3.0,
    ) -> FrameLines:
        gray = cv2.cvtColor(page, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape
        found = cv2.HoughLinesP(
            cv2.Canny(gray, 60, 150),
            1,
            np.pi / 720,
            120,
            None,
            int(min(width, height) * min_length),
            12,
        )
        horizontal: list[tuple[int, int, int]] = []
        vertical: list[tuple[int, int, int]] = []
        if found is not None:
            for row in np.asarray(found).reshape(-1, 4).tolist():
                x0, y0, x1, y1 = (int(v) for v in row)
                angle = abs(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
                angle = min(angle, 180 - angle)  # 0 = horizontal, 90 = vertical
                if angle <= max_tilt:
                    horizontal.append(((y0 + y1) // 2, min(x0, x1), max(x0, x1)))
                elif 90 - angle <= max_tilt:
                    vertical.append(((x0 + x1) // 2, min(y0, y1), max(y0, y1)))
        gap = 0.1
        return cls(
            _merge(horizontal, gap * width, 0.008 * height),
            _merge(vertical, gap * height, 0.008 * width),
            height,
            width,
        )

    def cut_rows(self, region: _Region, min_cover: float = 0.3, margin: float = 0.1) -> list[int]:
        """Rows of lines that cover much of the region's width, longest first."""
        y0, y1, x0, x1 = region
        found: list[tuple[int, int]] = []
        for y, a, b in self.horizontal:
            cover = min(b, x1) - max(a, x0)
            inside = y0 + margin * (y1 - y0) < y < y1 - margin * (y1 - y0)
            if inside and cover >= min_cover * (x1 - x0):
                found.append((cover, y))
        return [y for _, y in sorted(found, reverse=True)]

    def touches_frame_row(
        self, region: _Region, min_cover: float = 0.25, tolerance: float = 0.02
    ) -> bool:
        """True if a horizontal frame line runs along the region's top or bottom boundary."""
        y0, y1, x0, x1 = region
        tol = tolerance * self.height
        return any(
            (abs(y - y0) <= tol or abs(y - y1) <= tol)
            and min(b, x1) - max(a, x0) >= min_cover * (x1 - x0)
            for y, a, b in self.horizontal
        )

    def encloses_as_one(
        self, left: _Region, right: _Region, *, joined: bool = False, tolerance: float = 0.02
    ) -> bool:
        """True if two side-by-side pieces sit inside one frame with no frame line between them.

        The union has long frame lines along its top and bottom and a frame line on at least one
        end, and no vertical frame line crosses the gap between the pieces. The gap is wide
        (``joined`` also allows a piece that overlaps one already joined). Dark art inside a
        panel on a dark page reads as paper and cuts the panel into such pieces.
        """
        y0, y1 = min(left[0], right[0]), max(left[1], right[1])
        x0, x1 = min(left[2], right[2]), max(left[3], right[3])
        tol_y, tol_x = tolerance * self.height, tolerance * self.width
        if not (y1 - y0 and x1 - x0):
            return False
        overlap = min(left[1], right[1]) - max(left[0], right[0])
        if overlap < 0.9 * min(left[1] - left[0], right[1] - right[0]):
            return False

        def spans(at: int) -> bool:
            return any(
                abs(y - at) <= tol_y and min(b, x1) - max(a, x0) >= 0.9 * (x1 - x0)
                for y, a, b in self.horizontal
            )

        def vertical(at_lo: float, at_hi: float, cover: float) -> bool:
            return any(
                at_lo <= x <= at_hi and min(b, y1) - max(a, y0) >= cover * (y1 - y0)
                for x, a, b in self.vertical
            )

        gap_lo, gap_hi = min(left[3], right[3]) - tol_x, max(left[2], right[2]) + tol_x
        gap = max(left[2], right[2]) - min(left[3], right[3])
        if gap <= _THIN_GUTTER * self.width and not (joined and gap <= 0):
            # A thin gap is a real gutter between two framed panels, and so is a slanted one whose
            # boxes overlap. Only a piece touching one already joined across a wide gap is part
            # of the same panel.
            return False
        end = vertical(x0 - tol_x, x0 + tol_x, 0.4) or vertical(x1 - tol_x, x1 + tol_x, 0.4)
        return spans(y0) and spans(y1) and end and not vertical(gap_lo, gap_hi, 0.9)

    def sides(self, region: _Region, min_cover: float = 0.4, tolerance: float = 0.02) -> int:
        """How many of the region's four sides lie on a frame line (0 to 4)."""
        y0, y1, x0, x1 = region
        tol_y, tol_x = tolerance * self.height, tolerance * self.width

        def covered(
            lines: list[tuple[int, int, int]], at: int, tol: float, lo: int, hi: int
        ) -> bool:
            return any(
                abs(pos - at) <= tol and min(b, hi) - max(a, lo) >= min_cover * (hi - lo)
                for pos, a, b in lines
            )

        return sum(
            (
                covered(self.horizontal, y0, tol_y, x0, x1),
                covered(self.horizontal, y1, tol_y, x0, x1),
                covered(self.vertical, x0, tol_x, y0, y1),
                covered(self.vertical, x1, tol_x, y0, y1),
            )
        )


def _merge(
    lines: list[tuple[int, int, int]], max_gap: float, tolerance: float
) -> list[tuple[int, int, int]]:
    """Join collinear fragments (same position within ``tolerance``, gap up to ``max_gap``).

    A frame line broken by a balloon or by art is found as several short segments; as one
    line it covers the panel side the way it really does.
    """
    merged = sorted(lines)
    changed = True
    while changed:  # groups formed early may only touch each other once both have grown
        changed = False
        out: list[tuple[int, int, int]] = []
        for pos, lo, hi in merged:
            for i, (mpos, mlo, mhi) in enumerate(out):
                if abs(pos - mpos) <= tolerance and lo <= mhi + max_gap and hi >= mlo - max_gap:
                    out[i] = ((pos + mpos) // 2, min(lo, mlo), max(hi, mhi))
                    changed = True
                    break
            else:
                out.append((pos, lo, hi))
        merged = out
    return merged
