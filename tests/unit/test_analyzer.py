from collections.abc import Sequence

import numpy as np
from app.domain.models import AnalysisResult, DetectedPanel, Detection, ImageSize, ReadingDirection
from app.services.analyzer import FALLBACK_CONFIDENCE, PanelAnalyzer
from app.services.image.decoder import DecodedImage
from app.services.ordering.xy_cut import XYCutOrdering
from numpy.typing import NDArray

from tests.conftest import panel


class FakeDetector:
    name = "fake"
    version = "7"

    def __init__(self, panels: Sequence[DetectedPanel]) -> None:
        self._panels = panels

    def detect(self, page: NDArray[np.uint8]) -> Detection:
        return Detection(panels=tuple(self._panels), method=self.name)


def run(
    panels: Sequence[DetectedPanel], direction: ReadingDirection = ReadingDirection.LTR
) -> AnalysisResult:
    image = DecodedImage(np.zeros((1000, 500, 3), dtype=np.uint8), ImageSize(500, 1000))
    return PanelAnalyzer(FakeDetector(panels), XYCutOrdering()).analyze(image, direction)


def test_numbers_panels_in_reading_order_with_normalized_boxes() -> None:
    result = run([panel(250, 0, 250, 500, 0.8), panel(0, 0, 250, 500, 0.9)])
    assert [(p.id, p.order, p.bbox.x) for p in result.panels] == [(1, 1, 0), (2, 2, 250)]
    assert result.panels[0].normalized_bbox.width == 0.5
    assert result.method == "fake"
    assert result.version == "fake@7+xycut@6"
    assert result.whole_page_fallback is False
    assert result.warnings == ()


def test_no_panels_returns_whole_page_fallback() -> None:
    result = run([])
    assert result.whole_page_fallback is True
    (only,) = result.panels
    assert (only.id, only.order) == (1, 1)
    assert (only.bbox.x, only.bbox.y, only.bbox.width, only.bbox.height) == (0, 0, 500, 1000)
    assert (only.normalized_bbox.width, only.normalized_bbox.height) == (1.0, 1.0)
    assert only.confidence == FALLBACK_CONFIDENCE
    assert result.sequence_confidence == FALLBACK_CONFIDENCE
    assert "whole page" in result.warnings[0]
    assert len(result.warnings) == 1


def test_detected_single_panel_is_not_a_fallback() -> None:
    result = run([panel(0, 0, 500, 1000)])
    assert result.whole_page_fallback is False


def test_single_panel_warns() -> None:
    assert "splash" in run([panel(0, 0, 500, 1000)]).warnings[0]


def test_ambiguous_layout_warns() -> None:
    result = run([panel(0, 0, 60, 60), panel(40, 40, 60, 60)])
    assert any("ambiguous" in w for w in result.warnings)
