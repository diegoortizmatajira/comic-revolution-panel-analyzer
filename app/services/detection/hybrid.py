"""Hybrid detector: a primary detector, with an ML detector as fallback and as a second opinion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from app.domain.models import DetectedPanel, Detection
    from app.domain.ports import PanelDetector

# Bump when the policy or selection rule changes (4: extra panels; 3: agreement; 2: title exempt).
VERSION = "4"


@dataclass(frozen=True, slots=True)
class ReliabilityPolicy:
    """When is the primary result not trusted, so the ML detector is asked as well?

    Two cheap signals, both visible in the primary's own output (see docs/adr/0002):

    * very few panels: a page with fewer than ``min_panels`` panels is usually a page where
      panels were merged into one (comic pages with one or two panels are rare);
    * a very thin panel: a box whose long side is at least ``max_aspect`` times its short side
      is nearly always a caption strip counted as a panel. The one exception is a thin panel
      that lies wholly above every other panel: that is the title banner the primary detector
      returns on purpose as a first panel, and the ML model would only drop it.

    They are deliberately conservative. A third rule is about boxes, not about trust: when the
    primary result is plausible and the model finds the same number of panels (at least
    ``min_panels``) and each of the primary's panels overlaps its own model box by at least
    ``agree_iou``, the two agree on the layout and the model's boxes are used. They sit on the
    frames more tightly, which projection loses where white balloons or captions read as gutter.
    """

    min_panels: int = 3
    max_aspect: float = 5.0
    agree_iou: float = 0.6
    # Prototype knobs (off by default, so version 3 behaviour is unchanged):
    # * prefer_model: use the model's answer whenever it is not degenerate (finds at least
    #   ``min_model_panels`` panels), whatever the primary looks like;
    # * extra_panels: when the primary looks plausible but the model finds at least this many
    #   more panels, the primary probably merged panels, so use the model (0 = off).
    prefer_model: bool = False
    min_model_panels: int = 2
    extra_panels: int = 1

    def prefers_model(self, primary: Detection, other: Detection) -> bool:
        n, m = len(primary.panels), len(other.panels)
        if self.prefer_model:
            return m >= self.min_model_panels or (m > 0 and n == 0)
        return self.extra_panels > 0 and m >= n + self.extra_panels

    def distrusts(self, primary: Detection) -> bool:
        panels = primary.panels
        if len(panels) < self.min_panels:
            return True
        return any(self._is_thin(p) and not _is_title(p, panels) for p in panels)

    def agrees(self, primary: Detection, other: Detection) -> bool:
        """Same layout: equal count (enough panels) and every panel has its own overlapping box."""
        a, b = primary.panels, other.panels
        if len(a) != len(b) or len(a) < self.min_panels:
            return False
        best = [max(range(len(b)), key=lambda j: _iou(p, b[j])) for p in a]
        return len(set(best)) == len(a) and all(
            _iou(p, b[j]) >= self.agree_iou for p, j in zip(a, best, strict=True)
        )

    def _is_thin(self, panel: DetectedPanel) -> bool:
        box = panel.bbox
        return max(box.width, box.height) >= self.max_aspect * min(box.width, box.height)


def _iou(a: DetectedPanel, b: DetectedPanel) -> float:
    left, top = max(a.bbox.x, b.bbox.x), max(a.bbox.y, b.bbox.y)
    right, bottom = min(a.bbox.right, b.bbox.right), min(a.bbox.bottom, b.bbox.bottom)
    inter = max(0, right - left) * max(0, bottom - top)
    union = a.bbox.width * a.bbox.height + b.bbox.width * b.bbox.height - inter
    return inter / union if union else 0.0


def _is_title(panel: DetectedPanel, panels: tuple[DetectedPanel, ...]) -> bool:
    """A panel wholly above all the others: the title banner, not a caption strip between them."""
    others = [p for p in panels if p is not panel]
    return bool(others) and all(panel.bbox.bottom <= p.bbox.y for p in others)


class HybridPanelDetector:
    name = "hybrid"

    def __init__(
        self,
        primary: PanelDetector,
        fallback: PanelDetector,
        policy: ReliabilityPolicy | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._policy = policy or ReliabilityPolicy()
        # Results depend on both detectors (and their weights) and on the policy.
        self.version = (
            f"{VERSION}[{primary.name}@{primary.version}|{fallback.name}@{fallback.version}]"
        )

    def detect(self, page: NDArray[np.uint8]) -> Detection:
        first = self._primary.detect(page)
        second = self._fallback.detect(page)
        if self._policy.prefers_model(first, second):
            return second
        if self._policy.distrusts(first):
            # Keep the primary answer unless the fallback actually found something.
            return second if second.panels else first
        return second if self._policy.agrees(first, second) else first
