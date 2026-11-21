"""Gutter-based panel detector using projection profiles (recursive XY-cut on pixels).

1. Estimate the page background colour from a thin strip around the page edge.
2. Mark every pixel whose colour differs from it as "content".
3. Recursively split a region wherever a run of rows (or columns) contains almost no
   content: that run is a gutter. Rows are tried first, then columns, then each piece
   is trimmed to its content and split again. Regions that cannot be split are panels.

Unlike contour detection this does not need panels to be disconnected blobs, so panels
whose borders or art nearly touch are still separated as long as a gutter runs between
them. When no straight gutter exists, a slanted one is followed with a minimum-content path
(seam). Known limitations: panels with no gutter at all stay merged; title banners are
not told apart from panels; borderless panels
whose artwork contains background-coloured bands can be split by mistake; a full-bleed
splash page (no margin to sample) yields no panels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

from app.domain.models import BoundingBox, DetectedPanel, Detection
from app.services.detection.content import (
    content_mask,
    estimate_background,
    local_content_mask,
)
from app.services.detection.frames import FrameLines

if TYPE_CHECKING:
    from numpy.typing import NDArray

    Mask = NDArray[np.bool_]

_Region = tuple[int, int, int, int]  # y0, y1, x0, x1 (end-exclusive)
_ALIGNED = 0.2  # share of a panel's side another panel must overlap to be in the way
_MATCH = 0.85  # share of the panel's extent the free strip must keep clear to count as matching it
_DARK_PAPER = 60  # mean page-edge colour below this: a dark page, where black art reads as margin
_FRAMED_SIDES = 3  # frame lines on this many sides: the panel has its own edges, keep them
_MIN_ABSORB = 0.04  # of the page: thinner unclaimed strips are margins or gutters
_ABSORB_INK = 0.05  # ink share an unclaimed strip needs to be art that was trimmed away
_MAX_FRAME_TRIES = 6  # frame lines tried per region, longest first


@dataclass(slots=True)
class _Frames:
    """Frame lines of the page and the rows already cut along them (per ``detect`` call)."""

    lines: FrameLines
    cuts: list[int] = field(default_factory=list)


def _bleeds(frames: _Frames, region: _Region) -> bool:
    """True if the region runs off the page on at least three sides: full-bleed artwork.

    Lettering sits on paper with margins; art that bleeds does not, so it is a panel even
    though it has no frame lines of its own.
    """
    y0, y1, x0, x1 = region
    height, width = frames.lines.height, frames.lines.width
    tol_y, tol_x = 0.01 * height, 0.01 * width
    edges = (y0 <= tol_y, y1 >= height - tol_y, x0 <= tol_x, x1 >= width - tol_x)
    return sum(edges) >= 3


def _free_strip(
    side: str, box: tuple[int, int, int, int], others: list[BoundingBox], page: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    """The space between a panel's side and the nearest panel (or page edge) beyond it.

    Returned as (x0, y0, x1, y1); it spans the panel's extent along that side, cut short where
    another panel intrudes, and is None when no space is left.
    """
    x0, y0, x1, y1 = box
    width, height = page
    horizontal = side in ("left", "right")
    lo, hi = (y0, y1) if horizontal else (x0, x1)
    reach = (0 if side == "left" else width) if horizontal else (0 if side == "top" else height)
    if side in ("left", "top"):
        edge = x0 if horizontal else y0
        beyond = [
            (o.x + o.width) if horizontal else (o.y + o.height)
            for o in others
            if _spans(o, lo, hi, horizontal)
            and ((o.x + o.width) if horizontal else (o.y + o.height)) <= edge
        ]
        start, end = max(beyond, default=reach), edge
        if start >= end:
            return None
    else:
        edge = x1 if horizontal else y1
        beyond = [
            o.x if horizontal else o.y
            for o in others
            if _spans(o, lo, hi, horizontal) and (o.x if horizontal else o.y) >= edge
        ]
        start, end = edge, min(beyond, default=reach)
        if start >= end:
            return None
    return (start, y0, end, y1) if horizontal else (x0, start, x1, end)


def _reaches_page_edge(side: str, free: tuple[int, int, int, int], page: tuple[int, int]) -> bool:
    """True if the free strip runs out to the page edge: space between panels is a gutter."""
    fx0, fy0, fx1, fy1 = free
    return {"left": fx0 <= 0, "right": fx1 >= page[0], "top": fy0 <= 0, "bottom": fy1 >= page[1]}[
        side
    ]


def _spans(other: BoundingBox, lo: int, hi: int, horizontal: bool) -> bool:
    """True if ``other`` overlaps the interval lo..hi along the axis the strip runs on."""
    o_lo, o_hi = (
        (other.y, other.y + other.height) if horizontal else (other.x, other.x + other.width)
    )
    return min(hi, o_hi) - max(lo, o_lo) > _ALIGNED * (hi - lo)


def _worth_absorbing(
    side: str,
    free: tuple[int, int, int, int],
    others: list[BoundingBox],
    mask: Mask,
) -> bool:
    fx0, fy0, fx1, fy1 = free
    height, width = mask.shape
    thickness = (fx1 - fx0) if side in ("left", "right") else (fy1 - fy0)
    page_extent = width if side in ("left", "right") else height
    if thickness < _MIN_ABSORB * page_extent:
        return False  # a thin strip is a margin or a gutter
    horizontal = side in ("left", "right")
    lo, hi = (fy0, fy1) if horizontal else (fx0, fx1)
    blocked = sorted(
        (
            max(lo, (o.y if horizontal else o.x)),
            min(hi, (o.y + o.height) if horizontal else (o.x + o.width)),
        )
        for o in others
        if o.x < fx1 and fx0 < o.x + o.width and o.y < fy1 and fy0 < o.y + o.height
    )
    longest, cursor = 0, lo
    for b0, b1 in blocked:
        longest, cursor = max(longest, b0 - cursor), max(cursor, b1)
    if max(longest, hi - cursor) < _MATCH * (hi - lo):
        return False  # other panels take too much of it: it is not the space of this panel
    return float(mask[fy0:fy1, fx0:fx1].mean()) >= _ABSORB_INK


def _is_wide(region: _Region, page_width: int) -> bool:
    """A piece wide enough to be a line of lettering or a caption, not a speck."""
    return region[3] - region[2] >= 0.05 * page_width


class ProjectionPanelDetector:
    name = "projection"
    version = "6"

    def __init__(  # noqa: PLR0913
        self,
        *,
        color_tolerance: float = 40.0,
        gutter_content_ratio: float = 0.005,
        gutter_relative_threshold: float = 0.2,
        min_gutter_ratio: float = 0.0015,
        min_area_ratio: float = 0.02,
        border_fraction: float = 0.01,
        max_depth: int = 12,
        seam_content_ratio: float = 0.02,
        min_seam_side_ratio: float = 0.03,
        seam_straightness: float = 0.01,
        clean_seam_ratio: float = 0.004,
        banner_area_ratio: float = 0.06,
    ) -> None:
        self._tolerance = color_tolerance
        self._gutter_ratio = gutter_content_ratio
        self._gutter_relative = gutter_relative_threshold
        self._min_gutter_ratio = min_gutter_ratio
        self._min_area_ratio = min_area_ratio
        self._border_fraction = border_fraction
        self._max_depth = max_depth
        self._seam_content_ratio = seam_content_ratio
        self._min_seam_side_ratio = min_seam_side_ratio
        self._seam_straightness = seam_straightness
        self._clean_seam_ratio = clean_seam_ratio
        self._banner_area_ratio = banner_area_ratio

    def detect(self, page: NDArray[np.uint8]) -> Detection:
        frames = _Frames(FrameLines.find(page))
        mask = self._content_mask(page)
        panels = self._detect_with(mask, frames, loose=False)
        if len(panels) <= 1:
            # Nothing was cut. On stained or unevenly toned paper the single page colour matches
            # only part of the page, so try the paper colour measured around each pixel.
            local = local_content_mask(page, self._tolerance, self._border_fraction)
            retry = self._detect_with(local, _Frames(frames.lines), loose=True)
            if len(retry) > len(panels):
                panels = retry
        if estimate_background(page, self._border_fraction).mean() < _DARK_PAPER:
            panels = self._absorb_unclaimed(panels, mask, frames.lines)
        return Detection(panels=tuple(panels), method=self.name)

    @staticmethod
    def _absorb_unclaimed(
        panels: list[DetectedPanel], mask: Mask, lines: FrameLines
    ) -> list[DetectedPanel]:
        """Hand ink that no panel claims to the panel it is aligned with.

        Only used on a dark page, where art that is dark reads as margin and is trimmed off a
        panel's side (on light paper the trim is reliable, and unclaimed ink is lettering that
        belongs to no panel). Unclaimed space between a panel and the page edge is taken back by
        that panel when it holds ink, is thick enough not to be a margin, and its extent along
        the shared side matches the panel's (same height beside it, same width above or below).
        A panel with a frame of its own is left alone.
        """
        height, width = mask.shape
        result = list(panels)
        for i, panel in enumerate(result):
            box = panel.bbox
            others = [p.bbox for j, p in enumerate(result) if j != i]
            x0, y0, x1, y1 = box.x, box.y, box.x + box.width, box.y + box.height
            if lines.sides((y0, y1, x0, x1)) >= _FRAMED_SIDES:
                continue
            for side in ("left", "right", "top", "bottom"):
                free = _free_strip(side, (x0, y0, x1, y1), others, (width, height))
                if (
                    free is None
                    or not _reaches_page_edge(side, free, (width, height))
                    or not _worth_absorbing(side, free, others, mask)
                ):
                    continue
                fx0, fy0, fx1, fy1 = free
                x0, y0, x1, y1 = min(x0, fx0), min(y0, fy0), max(x1, fx1), max(y1, fy1)
            result[i] = DetectedPanel(
                bbox=BoundingBox(x0, y0, x1 - x0, y1 - y0), confidence=panel.confidence
            )
        return result

    def _detect_with(self, mask: Mask, frames: _Frames, *, loose: bool) -> list[DetectedPanel]:
        """Panels of one pass. ``loose`` also gathers small lettering pieces into strips."""
        height, width = mask.shape
        regions: list[_Region] = []
        self._split(mask, (0, height, 0, width), 0, regions, frames)

        min_area = self._min_area_ratio * width * height
        big = [r for r in regions if (r[1] - r[0]) * (r[3] - r[2]) >= min_area]
        text = [r for r in regions if self._is_text_only(frames, r)]
        kept = [r for r in big if r not in text]
        strips = list(text)
        banner_area = self._banner_area_ratio * width * height
        if loose:
            # Second attempt on a page the first pass could not cut: caption lines are small wide
            # pieces, so gather them as strips.
            strips += [r for r in regions if r not in big and r not in text and _is_wide(r, width)]
            if kept:
                kept, strips = self._gather_title(
                    kept, strips, regions, frames, mask.shape, banner_area
                )
        kept = self._join_framed_pieces(kept, frames.lines)
        kept = self._attach_strips(kept, strips, 0.02 * height, banner_area, loose=loose)
        if loose:
            kept = self._snap_to_frame_lines(kept, frames.lines, 0.06 * height)
            kept = self._grow_into_gap_above(kept, 0.08 * height)
        return [
            DetectedPanel(
                bbox=BoundingBox(x=x0, y=y0, width=x1 - x0, height=y1 - y0),
                confidence=min(1.0, max(0.0, float(mask[y0:y1, x0:x1].mean()))),
            )
            for y0, y1, x0, x1 in kept
        ]

    @staticmethod
    def _join_framed_pieces(kept: list[_Region], lines: FrameLines) -> list[_Region]:
        """Rejoin pieces of one framed panel that dark art cut apart (see ``encloses_as_one``)."""
        result = sorted(kept, key=lambda r: (r[2], r[0]))
        joined: set[_Region] = set()
        merged = True
        while merged:
            merged = False
            for i, a in enumerate(result):
                for b in result[i + 1 :]:
                    if lines.encloses_as_one(a, b, joined=a in joined or b in joined):
                        result.remove(a)
                        result.remove(b)
                        union = (min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3]))
                        result.append(union)
                        joined.add(union)
                        merged = True
                        break
                if merged:
                    break
        return result

    @staticmethod
    def _attach_strips(
        kept: list[_Region],
        strips: list[_Region],
        tol: float,
        banner_area: float,
        *,
        loose: bool = False,
    ) -> list[_Region]:
        """Hand each text-only strip to its neighbours; a large one with nothing above is a panel.

        A strip between two panels (balloons, captions, a caption box crossing a frame) belongs
        to one of them, and which is not knowable from the pixels alone, so it is shared at 40%
        of its height: the panel above grows down to that line, the panel below grows up
        to it (balloons hang from the panel below more often than captions rise). A strip with
        no panel above it is a caption box at the top of the panel below or, if large, a title
        banner. Banners are returned as a first panel rather than discarded: the detector cannot
        tell lettering from a panel without a frame, and losing a panel costs more than a title.

        With ``loose`` (the second attempt on a page that could not be cut) strips are attached
        repeatedly, since one may touch a panel only after another has joined it, and what is
        left over with no panel above it is gathered into a single banner.
        """
        result = list(kept)
        if not loose:
            for strip in strips:
                has_panel_above = any(
                    y1 <= strip[0] + tol and min(x1, strip[3]) > max(x0, strip[2])
                    for _, y1, x0, x1 in result
                )
                if (
                    not has_panel_above
                    and (strip[1] - strip[0]) * (strip[3] - strip[2]) >= banner_area
                ):
                    result.append(strip)
                else:
                    ProjectionPanelDetector._attach_one(result, strip, tol)
            return result
        pending = sorted(strips)
        progress = True
        while progress:  # a strip may touch a panel only after another strip has joined it
            progress = False
            for strip in list(pending):
                if ProjectionPanelDetector._attach_one(result, strip, tol):
                    pending.remove(strip)
                    progress = True
        leftovers = [r for r in pending if not any(y1 <= r[0] + tol for _, y1, _, _ in result)]
        if leftovers:
            union = (
                min(r[0] for r in leftovers),
                max(r[1] for r in leftovers),
                min(r[2] for r in leftovers),
                max(r[3] for r in leftovers),
            )
            if (union[1] - union[0]) * (union[3] - union[2]) >= banner_area:
                result.append(union)
        return result

    @staticmethod
    def _gather_title(  # noqa: PLR0913, PLR0917
        kept: list[_Region],
        strips: list[_Region],
        regions: list[_Region],
        frames: _Frames,
        shape: tuple[int, ...],
        banner_area: float,
    ) -> tuple[list[_Region], list[_Region]]:
        """Make one title banner of everything unframed that lies above the main panel.

        Lettering that gutters split into words and lines (with ascenders and descenders as
        separate specks) would otherwise be lost, or its lower line attached to the main panel.
        The banner lines up with the main panel's left edge, as a title does on the page.
        """
        height, width = shape[:2]
        body = max(kept, key=lambda r: (r[1] - r[0]) * (r[3] - r[2]))
        framed = [r for r in kept if r[1] <= body[0] and frames.lines.sides(r) > 0]
        loose_kept = [r for r in kept if r[1] <= body[0] and r not in framed]
        pieces = [
            r
            for r in regions
            if r[1] <= body[0]
            and r not in framed
            and r[0] > 0.015 * height  # not the scan edge
            and 0.01 * width < r[2]
            and r[3] < 0.99 * width
        ]
        if not pieces:
            return kept, strips
        union = (
            min(r[0] for r in pieces),
            max(r[1] for r in pieces),
            body[2]
            if min(r[2] for r in pieces) >= body[2] - 0.03 * width
            else min(r[2] for r in pieces),
            max(r[3] for r in pieces),
        )
        if (union[1] - union[0]) * (union[3] - union[2]) < banner_area:
            return kept, strips
        remaining = [r for r in kept if r not in loose_kept] + [union]
        return remaining, [r for r in strips if r not in pieces]

    @staticmethod
    def _snap_to_frame_lines(
        panels: list[_Region], lines: FrameLines, reach: float, min_cover: float = 0.7
    ) -> list[_Region]:
        """Move a panel's top or bottom edge out to the frame line just beyond it.

        Pale panel edges (sky, white balloons near the paper colour) leave the box short of its
        own frame, cutting the first lines of a balloon. A frame line that spans most of the
        panel and lies within ``reach`` beyond the edge is the panel's frame. A line at or inside
        a neighbouring panel's box is that panel's frame, not this one's.
        """
        result = list(panels)
        for i, (y0, y1, x0, x1) in enumerate(panels):
            others = [p for j, p in enumerate(panels) if j != i]
            usable = [
                y
                for y, a, b in lines.horizontal
                if min(b, x1) - max(a, x0) >= min_cover * (x1 - x0)
                and not any(
                    oy0 - 4 <= y <= oy1 + 4 and min(ox1, x1) > max(ox0, x0)
                    for oy0, oy1, ox0, ox1 in others
                )
            ]
            above = [y for y in usable if y0 - reach <= y < y0]
            below = [y for y in usable if y1 < y <= y1 + reach]
            result[i] = (min(above, default=y0), max(below, default=y1), x0, x1)
        return result

    @staticmethod
    def _grow_into_gap_above(panels: list[_Region], max_gap: float) -> list[_Region]:
        """Let a panel grow up to the bottom of the panel above it when the gap is small.

        On a page that needed the second attempt, what lies between two stacked panels (the
        caption lines of a row) hangs from the panel below, so that panel takes the whole gap.
        """
        result = list(panels)
        for i, (y0, y1, x0, x1) in enumerate(panels):
            gaps = [
                y0 - oy1
                for j, (_, oy1, ox0, ox1) in enumerate(panels)
                if j != i
                and 0 < y0 - oy1 <= max_gap
                and min(x1, ox1) - max(x0, ox0) >= 0.3 * (x1 - x0)
            ]
            if gaps:
                result[i] = (y0 - min(gaps), y1, x0, x1)
        return result

    @staticmethod
    def _attach_one(result: list[_Region], strip: _Region, tol: float) -> bool:
        sy0, sy1, sx0, sx1 = strip
        above = [
            i
            for i, (_, y1, x0, x1) in enumerate(result)
            if y1 <= sy0 + tol and min(x1, sx1) > max(x0, sx0)
        ]
        middle = sy0 + (sy1 - sy0) * 2 // 5 if above else sy0
        attached = False
        for i, (y0, y1, x0, x1) in enumerate(result):
            if min(x1, sx1) - max(x0, sx0) < 0.3 * (x1 - x0):
                continue
            if i in above and abs(y1 - sy0) <= tol:
                result[i] = (y0, max(y1, middle), x0, x1)
                attached = True
            elif abs(y0 - sy1) <= tol:
                result[i] = (min(y0, middle), y1, x0, x1)
                attached = True
        return attached

    def _content_mask(self, page: NDArray[np.uint8]) -> Mask:
        return content_mask(page, self._tolerance, self._border_fraction)

    def _split(
        self, mask: Mask, region: _Region, depth: int, out: list[_Region], frames: _Frames
    ) -> None:
        y0, y1, x0, x1 = region
        sub = mask[y0:y1, x0:x1]

        rows = np.flatnonzero(sub.mean(axis=1) > self._gutter_ratio)
        if rows.size == 0:
            return
        top, bottom = int(rows[0]), int(rows[-1]) + 1
        cols = np.flatnonzero(sub[top:bottom].mean(axis=0) > self._gutter_ratio)
        if cols.size == 0:
            return
        left, right = int(cols[0]), int(cols[-1]) + 1

        y0, y1, x0, x1 = y0 + top, y0 + bottom, x0 + left, x0 + right
        sub = mask[y0:y1, x0:x1]

        if depth < self._max_depth:
            for axis in (1, 0):  # rows first (horizontal gutters), then columns
                profile = sub.mean(axis=axis)
                segments = self._segments(
                    profile, self._threshold(profile), self._min_gutter(len(profile))
                )
                if len(segments) > 1:
                    for s0, s1 in segments:
                        child = (
                            (y0 + s0, y0 + s1, x0, x1) if axis == 1 else (y0, y1, x0 + s0, x0 + s1)
                        )
                        self._split(mask, child, depth + 1, out, frames)
                    return
        if depth < self._max_depth:
            halves = self._seam_split(sub)
            if halves is not None:
                for half in halves:
                    padded = np.zeros_like(mask)
                    padded[y0:y1, x0:x1] = half
                    self._split(padded, (y0, y1, x0, x1), depth + 1, out, frames)
                return
            big_enough = (y1 - y0) * (x1 - x0) >= self._min_area_ratio * mask.shape[0] * mask.shape[
                1
            ]
            candidates = frames.lines.cut_rows((y0, y1, x0, x1)) if big_enough else []
            for cut in candidates[:_MAX_FRAME_TRIES]:
                if self._cut_pays_off(mask, (y0, y1, x0, x1), cut, depth, frames):
                    frames.cuts.append(cut)
                    self._split(mask, (y0, cut, x0, x1), depth + 1, out, frames)
                    self._split(mask, (cut, y1, x0, x1), depth + 1, out, frames)
                    return
        out.append((y0, y1, x0, x1))

    def _cut_pays_off(
        self, mask: Mask, region: _Region, cut: int, depth: int, frames: _Frames
    ) -> bool:
        """A frame line is only trusted if cutting there is useful.

        Lines inside artwork look like frame edges too. A real frame cut either sets off
        something that is not a panel (title lettering, a caption strip) or lets a part split
        further (stacked panels with no gutter); a cut through one panel does neither.
        """
        y0, y1, x0, x1 = region
        trial = _Frames(frames.lines, [*frames.cuts, cut])
        pieces: list[_Region] = []
        self._split(mask, (y0, cut, x0, x1), depth + 1, pieces, trial)
        self._split(mask, (cut, y1, x0, x1), depth + 1, pieces, trial)
        min_area = self._min_area_ratio * mask.shape[0] * mask.shape[1]
        big = [r for r in pieces if (r[1] - r[0]) * (r[3] - r[2]) >= min_area]
        text_only = [r for r in big if self._is_text_only(trial, r)]
        return bool(text_only) or len(big) > 2

    @staticmethod
    def _is_text_only(frames: _Frames, region: _Region) -> bool:
        """A region cut off along a frame line that has at most one frame side is not a panel.

        Title lettering and caption strips sit next to a panel's frame line but have no frame
        of their own; a real panel has frame lines on several of its sides.
        """
        y0, y1, *_ = region
        touches_cut = any(abs(y - cut) <= 4 for cut in frames.cuts for y in (y0, y1))
        return touches_cut and frames.lines.sides(region) <= 1 and not _bleeds(frames, region)

    def _seam_split(self, sub: Mask) -> tuple[Mask, Mask] | None:
        """Cut a region along a slanted gutter: the cheapest path through the content mask.

        Used only when no straight row/column gutter exists. The path may drift one pixel per
        step, so it follows a slanted gutter; it is accepted only if it runs through (almost)
        no content for the whole region, i.e. it really is a gutter and not a cut through art.
        """
        best: tuple[float, Mask, Mask] | None = None
        for transpose in (False, True):  # False: seam runs left to right, cutting rows apart
            oriented = sub.T if transpose else sub
            found = self._cheapest_seam(oriented)
            if found is None:
                continue
            cost, upper, lower = found
            if best is None or cost < best[0]:
                best = (cost, upper.T if transpose else upper, lower.T if transpose else lower)
        return None if best is None else (best[1], best[2])

    def _cheapest_seam(self, sub: Mask) -> tuple[float, Mask, Mask] | None:
        height, width = sub.shape
        clearance = self._min_gutter(height) | 1
        blocked = cv2.dilate(sub.astype(np.uint8), np.ones((clearance, 1), np.uint8)) > 0
        cost = blocked.astype(np.float32)
        total = np.zeros((height, width), dtype=np.float32)
        step = np.zeros((height, width), dtype=np.int8)
        total[:, 0] = cost[:, 0]
        inf = np.float32(np.inf)
        for x in range(1, width):
            prev = total[:, x - 1]
            up = np.concatenate(([inf], prev[:-1]))
            down = np.concatenate((prev[1:], [inf]))
            # Prefer going straight on ties, so a straight gutter stays straight.
            stacked = np.stack((prev, up + 1e-3, down + 1e-3))
            choice = stacked.argmin(axis=0)
            total[:, x] = stacked.min(axis=0) + cost[:, x]
            step[:, x] = choice
        end = int(total[:, -1].argmin())
        if total[end, -1] > self._seam_content_ratio * width * 10:  # blocked pixels allowed
            return None
        path = np.empty(width, dtype=np.int64)
        y = end
        for x in range(width - 1, -1, -1):
            path[x] = y
            c = step[y, x]
            y += 0 if c == 0 else (-1 if c == 1 else 1)
        blocked_ratio = float(blocked[path, np.arange(width)].mean())
        if blocked_ratio > self._seam_content_ratio:
            return None
        if blocked_ratio > self._clean_seam_ratio:
            # Bridged by a little art (jagged borders): only believable if it is a straight line.
            xs = np.arange(width)
            slope, intercept = np.polyfit(xs, path, 1)
            deviation = float(np.abs(path - (slope * xs + intercept)).max())
            if deviation > self._seam_straightness * width:
                return None
        rows = np.arange(height)[:, None]
        upper = sub & (rows < path[None, :])
        lower = sub & (rows > path[None, :])
        area = sub.sum()
        floor = self._min_seam_side_ratio * area
        if upper.sum() < floor or lower.sum() < floor:
            return None
        return float(blocked[path, np.arange(width)].mean()), upper, lower

    def _min_gutter(self, length: int) -> int:
        return max(2, round(length * self._min_gutter_ratio))

    def _threshold(self, profile: NDArray[np.float64]) -> float:
        """Content fraction at or below which a row/column counts as gutter.

        Relative to how full the region's rows/columns normally are, because hand-drawn
        or jagged borders leave a little content in even a perfectly good gutter.
        """
        reference = float(np.percentile(profile, 75))
        return min(max(self._gutter_ratio, self._gutter_relative * reference), 0.2)

    @staticmethod
    def _segments(
        profile: NDArray[np.float64], threshold: float, min_gutter: int
    ) -> list[tuple[int, int]]:
        """Content segments of a profile, separated by gutter runs of at least min_gutter."""
        is_gutter = profile <= threshold
        segments: list[tuple[int, int]] = []
        start = 0
        i = 0
        n = len(profile)
        while i < n:
            if is_gutter[i]:
                j = i
                while j < n and is_gutter[j]:
                    j += 1
                if j - i >= min_gutter:
                    if i > start:
                        segments.append((start, i))
                    start = j
                i = j
            else:
                i += 1
        if start < n:
            segments.append((start, n))
        return segments
