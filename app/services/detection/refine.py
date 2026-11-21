"""Edge-snapping refiner: grow a detected box outward where its edge cuts through content.

Detectors sometimes place an edge in the middle of a balloon or caption, so the panel crops it.
For each edge the refiner looks at the line of pixels just outside the box:

* if that line holds ink, the edge is cutting content, so the edge moves outward;
* it keeps going until it meets a clean gutter (``gap`` consecutive ink-free lines), and the
  edge is placed right after the last inked line.

An expansion is discarded (the edge stays where the detector put it) when it never reaches a
gutter within ``max_expand_ratio`` of the page or meets a neighbouring panel first (content
shared with a neighbour is left for the neighbour to claim from its own side). Content that runs
right up to the page border grows to the border. That keeps noisy scans
and gutter-less layouts safe: the refiner can only ever make a box bigger, and only up to a
clean gutter.

Panels with slanted frames are the other case: their bounding box should reach the highest (or
lowest, leftmost, rightmost) corner, but a detector often places the edge part-way along the
slant. A long slanted frame line that crosses a box edge is therefore followed to its far end
(see ``slant_reach_ratio``), which puts the missing corner back inside the box.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import cv2
import numpy as np

from app.domain.models import BoundingBox, DetectedPanel
from app.services.detection.content import contrast_mask
from app.services.detection.frames import FrameLines
from app.services.detection.margin import normalize_unframed

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

_Rect = tuple[int, int, int, int]  # x0, y0, x1, y1 (end-exclusive)
_EDGES = ("top", "bottom", "left", "right")


@dataclass(frozen=True, slots=True)
class EdgeSnapRefiner:
    name: str = "edge-snap"
    version: str = "5"  # bump when a change could alter the output
    max_expand_ratio: float = 0.15  # farthest an edge may move, as a fraction of the page side
    gap: int = 3  # ink-free lines that count as a gutter
    ink_ratio: float = 0.003  # fraction of a line that must be ink to count as content
    min_ink_pixels: int = 2  # ...and at least this many pixels
    min_blob_pixels: int = 30  # smaller specks touching an edge are not content
    passes: int = 2  # a grown edge changes the span the perpendicular edges look at
    window: int = 31  # size of the neighbourhood an outline must stand out from
    contrast: float = 45.0  # gray-level difference that makes a pixel count as ink
    slant_reach_ratio: float = 0.04  # farthest a slanted frame line may extend an edge (page side)
    slant_min_cover: float = 0.5  # share of the box side the line must span
    slant_min_length: float = 0.15  # shortest slanted line to follow, as a fraction of page width
    slant_angles: tuple[float, float] = (3.0, 20.0)  # degrees off the axis; flatter is straight

    def refine(
        self, page: NDArray[np.uint8], panels: Sequence[DetectedPanel]
    ) -> tuple[DetectedPanel, ...]:
        if not panels:
            return ()
        mask = contrast_mask(page, self.window, self.contrast)
        rects = [(p.bbox.x, p.bbox.y, p.bbox.right, p.bbox.bottom) for p in panels]
        slants = self._slanted_lines(page)
        rects = [self._follow_slants(rect, slants, mask.shape) for rect in rects]
        for _ in range(self.passes):
            changed = False
            for index in range(len(rects)):
                grown = self._grow(mask, rects, index)
                changed |= grown != rects[index]
                rects[index] = grown
            if not changed:
                break
        rects = normalize_unframed(rects, FrameLines.find(page))
        return tuple(
            replace(p, bbox=BoundingBox(x=x0, y=y0, width=x1 - x0, height=y1 - y0))
            for p, (x0, y0, x1, y1) in zip(panels, rects, strict=True)
        )

    def _slanted_lines(self, page: NDArray[np.uint8]) -> list[tuple[int, int, int, int, bool]]:
        """Long slanted straight lines as (x0, y0, x1, y1, near_horizontal)."""
        gray = cv2.cvtColor(page, cv2.COLOR_RGB2GRAY)
        width = gray.shape[1]
        found = cv2.HoughLinesP(
            cv2.Canny(gray, 60, 150),
            1,
            np.pi / 720,
            150,
            None,
            int(width * self.slant_min_length),
            15,
        )
        if found is None:
            return []
        low, high = self.slant_angles
        lines: list[tuple[int, int, int, int, bool]] = []
        for row in np.asarray(found).reshape(-1, 4).tolist():
            x0, y0, x1, y1 = (int(v) for v in row)
            angle = abs(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
            angle = min(angle, 180 - angle)  # 0 = horizontal, 90 = vertical
            if low <= angle <= high:
                lines.append((x0, y0, x1, y1, True))
            elif low <= 90 - angle <= high:
                lines.append((x0, y0, x1, y1, False))
        return lines

    def _follow_slants(
        self,
        rect: _Rect,
        lines: Sequence[tuple[int, int, int, int, bool]],
        shape: tuple[int, ...],
    ) -> _Rect:
        """Extend the top/bottom edge to the far end of a slanted frame line that crosses it.

        Only near-horizontal lines are followed: slanted lines near the vertical are far more
        often artwork (trees, buildings) than frames, and extending to them hurt on the corpus.
        """
        x0, y0, x1, y1 = rect
        reach = shape[0] * self.slant_reach_ratio
        slack = shape[1] * 0.01  # a Hough end may land a few pixels past the box side
        top_edge, bottom_edge = y0, y1
        for ax, ay, bx, by, flat in lines:
            # A frame edge runs along most of the panel; shorter slants are artwork.
            if not flat or abs(bx - ax) < self.slant_min_cover * (x1 - x0):
                continue
            (hx, hy), (lx, ly) = ((ax, ay), (bx, by)) if ay < by else ((bx, by), (ax, ay))
            if hy < y0 <= ly and y0 - hy <= reach and x0 - slack <= hx <= x1 + slack:
                top_edge = min(top_edge, hy)  # the high end of the slant pokes above the box
            if hy <= y1 < ly and ly - y1 <= reach and x0 - slack <= lx <= x1 + slack:
                bottom_edge = max(bottom_edge, ly)  # the low end pokes below it
        return x0, top_edge, x1, bottom_edge

    def _grow(self, mask: NDArray[np.bool_], rects: list[_Rect], index: int) -> _Rect:
        x0, y0, x1, y1 = rects[index]
        height, width = mask.shape
        reach_y = round(height * self.max_expand_ratio)
        reach_x = round(width * self.max_expand_ratio)
        others = [r for i, r in enumerate(rects) if i != index]

        y0 = self._snap(mask, "top", (x0, y0, x1, y1), reach_y, others)
        y1 = self._snap(mask, "bottom", (x0, y0, x1, y1), reach_y, others)
        x0 = self._snap(mask, "left", (x0, y0, x1, y1), reach_x, others)
        x1 = self._snap(mask, "right", (x0, y0, x1, y1), reach_x, others)
        return x0, y0, x1, y1

    def _snap(
        self,
        mask: NDArray[np.bool_],
        edge: str,
        rect: _Rect,
        reach: int,
        others: Sequence[_Rect],
    ) -> int:
        """New coordinate of one edge (unchanged when it should not move)."""
        x0, y0, x1, y1 = rect
        vertical = edge in {"top", "bottom"}
        current = {"top": y0, "bottom": y1, "left": x0, "right": x1}[edge]
        outward = -1 if edge in {"top", "left"} else 1

        limit = self._limit(mask.shape, edge, rect, reach, others)
        if (limit - current) * outward <= 0:
            return current

        # Ink per line, walking outward from the current edge.
        if edge == "top":
            ink = mask[limit:current, x0:x1].sum(axis=1)[::-1]
        elif edge == "bottom":
            ink = mask[current:limit, x0:x1].sum(axis=1)
        elif edge == "left":
            ink = mask[y0:y1, limit:current].sum(axis=0)[::-1]
        else:
            ink = mask[y0:y1, current:limit].sum(axis=0)
        span = (x1 - x0) if vertical else (y1 - y0)
        inked = ink >= max(self.min_ink_pixels, self.ink_ratio * span)

        if not inked[0]:
            return current  # the edge already sits in a gutter: nothing is cut
        # Walk until `gap` clean lines in a row; the edge goes right after the last inked line.
        last, clean = 0, 0
        for step, has_ink in enumerate(inked):
            if has_ink:
                last, clean = step, 0
            else:
                clean += 1
                if clean >= self.gap:
                    return current + outward * (last + 1)
        page_border = 0 if outward < 0 else mask.shape[0 if vertical else 1]
        if limit == page_border and inked[-1]:
            return limit  # content runs off the page: the page border is the edge
        if limit != current + outward * reach:
            # No clean gutter before a neighbour: claim only the blobs that hang off this edge.
            return self._claim(mask, edge, rect, limit)
        return current  # never reached a gutter within reach: noise, bleed or touching art

    @staticmethod
    def _limit(
        shape: tuple[int, ...], edge: str, rect: _Rect, reach: int, others: Sequence[_Rect]
    ) -> int:
        """Farthest the edge may move: the allowance, the page border, the nearest neighbour."""
        x0, y0, x1, y1 = rect
        vertical = edge in {"top", "bottom"}
        current = {"top": y0, "bottom": y1, "left": x0, "right": x1}[edge]
        outward = -1 if edge in {"top", "left"} else 1
        limit = current + outward * reach
        limit = max(0, limit) if outward < 0 else min(shape[0] if vertical else shape[1], limit)
        for ox0, oy0, ox1, oy1 in others:
            # Only neighbours that share the edge's span and lie beyond the edge are walls.
            near, far = (oy0, oy1) if vertical else (ox0, ox1)
            shares_span = (ox0 < x1 and ox1 > x0) if vertical else (oy0 < y1 and oy1 > y0)
            if not shares_span:
                continue
            if near < current < far:
                return current  # the edge lies inside a neighbour's box: shared ground, no growth
            if outward < 0 and far <= current:
                limit = max(limit, far)
            elif outward > 0 and near >= current:
                limit = min(limit, near)
        return limit

    def _claim(self, mask: NDArray[np.bool_], edge: str, rect: _Rect, limit: int) -> int:
        """Grow an edge over the ink blobs in the gap that touch it but not the neighbour.

        The gap between two panels can hold balloons that belong to one of them while their
        ink runs into the other's frame, so no clean gutter exists. A blob touching the edge
        and not the neighbour is this panel's; one touching both is ambiguous and left alone.
        """
        x0, y0, x1, y1 = rect
        if edge == "top":
            band, current, outward = mask[limit:y0, x0:x1][::-1], y0, -1
        elif edge == "bottom":
            band, current, outward = mask[y1:limit, x0:x1], y1, 1
        elif edge == "left":
            band, current, outward = mask[y0:y1, limit:x0].T[::-1], x0, -1
        else:
            band, current, outward = mask[y0:y1, x1:limit].T, x1, 1
        # Row 0 touches this edge; the last row touches the neighbour.
        _, labels = cv2.connectedComponents(band.astype(np.uint8), connectivity=8)
        mine = set(np.unique(labels[0]).tolist()) - set(np.unique(labels[-1]).tolist()) - {0}
        reach = 0
        for label in mine:
            rows = np.flatnonzero((labels == label).any(axis=1))
            if np.count_nonzero(labels == label) >= self.min_blob_pixels:
                reach = max(reach, int(rows[-1]) + 1)
        return current + outward * reach
