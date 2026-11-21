"""End-to-end detection on committed pages (public-domain scans plus synthetic edge cases).

Where the detector is correct the tests assert it. Where it is known to fall short the test is a
strict xfail, so the suite stays green today and fails loudly (XPASS) when detection improves.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
from app.domain.models import AnalysisResult, ImageSize, ReadingDirection
from app.evaluation.labels import LabeledPage, discover
from app.evaluation.metrics import Prediction, score_page
from app.services.analyzer import PanelAnalyzer
from app.services.detection.projection import ProjectionPanelDetector
from app.services.detection.refine import EdgeSnapRefiner
from app.services.image.decoder import DecodedImage, ImageLimits, decode_image
from app.services.ordering.xy_cut import XYCutOrdering

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
LIMITS = ImageLimits(max_bytes=10_000_000, max_pixels=40_000_000)
analyzer = PanelAnalyzer(ProjectionPanelDetector(), XYCutOrdering())


def labelled(name: str) -> LabeledPage:
    """A committed fixture page with its hand-made ground truth (``*.labels.json``)."""
    return next(p for p in discover([FIXTURES]) if p.id == f"fixtures/{name}")


def matches_ground_truth(name: str) -> bool:
    """Every labelled panel found (IoU >= 0.5), nothing extra, same reading order."""
    page = labelled(name)
    result = analyzer.analyze(page.image, page.direction)
    prediction = Prediction(
        boxes=tuple(p.normalized_bbox for p in result.panels),
        sequence_confidence=result.sequence_confidence,
        whole_page_fallback=result.whole_page_fallback,
    )
    return score_page(page.id, page.tags, prediction, page.panels).correct


PUBLIC_DOMAIN_PAGES = sorted(p.name for p in FIXTURES.glob("*.jpg"))


def analyze_file(name: str, direction: ReadingDirection = ReadingDirection.LTR) -> AnalysisResult:
    image = decode_image((FIXTURES / name).read_bytes(), LIMITS)
    return analyzer.analyze(image, direction)


def analyze_array(
    page: np.ndarray, direction: ReadingDirection = ReadingDirection.LTR
) -> AnalysisResult:
    height, width = page.shape[:2]
    return analyzer.analyze(DecodedImage(page, ImageSize(width, height)), direction)


def assert_well_formed(result: AnalysisResult) -> None:
    assert result.panels, "at least one panel (or the whole-page fallback) is always returned"
    assert [p.order for p in result.panels] == list(range(1, len(result.panels) + 1))
    assert 0.0 <= result.sequence_confidence <= 1.0
    for p in result.panels:
        assert 0.0 <= p.confidence <= 1.0
        n = p.normalized_bbox
        assert 0.0 <= n.x <= 1.0
        assert 0.0 <= n.y <= 1.0
        assert n.x + n.width <= 1.0 + 1e-9
        assert n.y + n.height <= 1.0 + 1e-9


@pytest.mark.parametrize("name", PUBLIC_DOMAIN_PAGES)
@pytest.mark.parametrize("direction", list(ReadingDirection))
def test_public_domain_pages_yield_well_formed_results(
    name: str, direction: ReadingDirection
) -> None:
    assert_well_formed(analyze_file(name, direction))


# Pages the projection detector gets exactly right, and pages it is known to get wrong (#11).
CORRECT_PAGES = [
    "western_fighters_6",
    "nature_boy_4",
    "public_001",
    "moon-mullings-003-01",
    "western_fighters_8",
    "public_004",
    "public_005",
    "midnight-mistery-004-03",
    "midnight-mistery-004-04",
    "midnight-mistery-004-10",
    "moon-mullings-003-05",
    "moon-mullings-003-10",
    "moon-mullings-003-14",
    "moon-mullings-003-15",
    "moon-mullings-003-17",
    "blurred-002",
]
KNOWN_FAILURES = {
    "public_002": "L-shaped panel and bleeding art leave one page-sized panel",
    "public_003": "gutters are crossed by captions and art (12-17% of any cut path is ink); "
    "hybrid is fine",
    "midnight-mistery-004-07": "narrow gutter holds 22% ink, above the cut threshold; a valley cut "
    "finds it but then white balloons read as gutters and split other panels",
    "midnight-mistery-004-08": "last panel split in two by large lettering and its caption strip; "
    "white balloons and caption areas read as gutters",
    "midnight-mistery-004-14": "caption boxes at the top of panels are split off as extra panels",
    "moon-mullings-003-03": "two open-framed bottom panels with no gutter stay merged; a frame "
    "line between panels cannot be told from a line inside one panel",
    "moon-mullings-003-09": "the two top panels touch through art (27% of the path between them "
    "is ink)",
    "moon-mullings-003-16": "open frames and overlapping balloons leave no clean gutter",
    "blurred-001": "top art bleeds behind the framed panels and the middle-row frames are "
    "staggered: no gutter, and a full-width frame line cannot be told from one through art",
    "blurred-003": "full-bleed art with two white-framed insets over its bottom edge: the art is "
    "cut into fragments and the insets are split off as separate pieces",
}


@pytest.mark.parametrize("name", CORRECT_PAGES)
def test_page_matches_its_labels(name: str) -> None:
    assert matches_ground_truth(name)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(name, marks=pytest.mark.xfail(strict=True, reason=f"{reason} (#11)"))
        for name, reason in KNOWN_FAILURES.items()
    ],
)
def test_known_failure_page_matches_its_labels(name: str) -> None:
    assert matches_ground_truth(name)


def test_western_fighters_6_reads_the_title_first_with_and_without_edge_snapping() -> None:
    page = labelled("western_fighters_6")
    for refiner in (None, EdgeSnapRefiner()):
        result = PanelAnalyzer(ProjectionPanelDetector(), XYCutOrdering(), refiner).analyze(
            page.image, page.direction
        )
        title, body, *bottom = result.panels
        assert title.bbox.y < body.bbox.y + 100  # the title lettering, read before the big panel
        assert title.bbox.bottom < body.bbox.bottom
        assert body.bbox.bottom <= min(p.bbox.bottom for p in bottom)


def test_western_fighters_6_bottom_left_panel_takes_the_caption_above_it() -> None:
    page = labelled("western_fighters_6")
    for refiner in (None, EdgeSnapRefiner()):
        result = PanelAnalyzer(ProjectionPanelDetector(), XYCutOrdering(), refiner).analyze(
            page.image, page.direction
        )
        body, bottom_left = result.panels[1], result.panels[2]
        # The caption lines under the big panel hang from the bottom row: its box starts where
        # the big panel ends, so the caption (and the top of the first balloon) is inside it.
        assert abs(bottom_left.bbox.y - body.bbox.bottom) <= 5


def test_western_fighters_6_title_lines_up_with_the_big_panel() -> None:
    page = labelled("western_fighters_6")
    for refiner in (None, EdgeSnapRefiner()):
        result = PanelAnalyzer(ProjectionPanelDetector(), XYCutOrdering(), refiner).analyze(
            page.image, page.direction
        )
        title, body = result.panels[0], result.panels[1]
        assert title.bbox.x == body.bbox.x  # the title's left edge is the first row's
        assert title.bbox.y < 100  # the whole lettering, ascenders included, is inside


def test_every_fixture_page_is_accounted_for() -> None:
    stems = {p.stem for p in (*FIXTURES.glob("*.jpg"), *FIXTURES.glob("*.webp"))}
    assert stems == set(CORRECT_PAGES) | set(KNOWN_FAILURES)


def test_western_fighters_8_rtl_reads_bottom_right_before_bottom_left() -> None:
    result = analyze_file("western_fighters_8.jpg", ReadingDirection.RTL)
    title, top, first_bottom, second_bottom = result.panels
    assert title.bbox.y < 100  # the title lettering is the first panel
    assert abs(top.bbox.y - 268) < 10
    assert first_bottom.bbox.x > second_bottom.bbox.x


# --- synthetic edge cases ---------------------------------------------------------------------


def splash_page() -> np.ndarray:
    """Full-bleed art with no margin at all."""
    y, x = np.mgrid[0:1200, 0:800]
    return np.stack([(x / 800 * 255), (y / 1200 * 255), np.full_like(x, 120)], axis=2).astype(
        np.uint8
    )


def borderless_page() -> np.ndarray:
    """Coloured panels with no frame lines, separated only by white gutters."""
    page = np.full((1200, 800, 3), 255, dtype=np.uint8)
    for (x, y, w, h), colour in [
        ((40, 40, 720, 500), (200, 90, 60)),
        ((40, 580, 340, 580), (60, 140, 200)),
        ((420, 580, 340, 580), (90, 180, 90)),
    ]:
        page[y : y + h, x : x + w] = colour
    return page


def diagonal_page() -> np.ndarray:
    """Two panels split by a diagonal gutter: no clean row or column exists."""
    page = np.full((1200, 800, 3), 255, dtype=np.uint8)
    cv2.fillPoly(page, [np.array([[40, 40], [760, 40], [760, 420], [40, 700]])], (60, 90, 200))
    cv2.fillPoly(page, [np.array([[40, 760], [760, 480], [760, 1160], [40, 1160]])], (200, 90, 60))
    return page


def test_splash_page_yields_a_single_full_page_panel() -> None:
    result = analyze_array(splash_page())
    assert_well_formed(result)
    assert len(result.panels) == 1
    assert (result.panels[0].normalized_bbox.width, result.panels[0].normalized_bbox.height) == (
        1.0,
        1.0,
    )


def test_borderless_panels_are_separated_by_white_gutters() -> None:
    result = analyze_array(borderless_page())
    assert_well_formed(result)
    assert [(p.bbox.x, p.bbox.y) for p in result.panels] == [(40, 40), (40, 580), (420, 580)]


def test_diagonal_gutter_panels_are_separated() -> None:
    assert len(analyze_array(diagonal_page()).panels) == 2


def test_diagonal_gutter_degrades_gracefully() -> None:
    result = analyze_array(diagonal_page())
    assert_well_formed(result)  # never an error: worst case is one larger panel
