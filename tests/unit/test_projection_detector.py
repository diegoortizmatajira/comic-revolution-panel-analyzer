from pathlib import Path

import cv2
import numpy as np
import pytest
from app.domain.models import BoundingBox, DetectedPanel, ReadingDirection
from app.services.detection.frames import FrameLines
from app.services.detection.projection import ProjectionPanelDetector, _Frames, _has_band
from app.services.image.decoder import ImageLimits, decode_image
from app.services.ordering.xy_cut import XYCutOrdering

from tests.conftest import make_comic_page
from tests.unit.test_content import stained_page

detector = ProjectionPanelDetector()
MAROON = (79, 18, 18)

# Layout modelled on a real page: tall left panel, two stacked right, two across the bottom.
REAL_LAYOUT = [
    (40, 40, 220, 480),
    (290, 40, 470, 220),
    (290, 290, 470, 230),
    (40, 560, 340, 420),
    (410, 560, 350, 420),
]


def boxes(page: np.ndarray) -> list[tuple[int, int, int, int]]:
    return sorted(
        (p.bbox.x, p.bbox.y, p.bbox.width, p.bbox.height) for p in detector.detect(page).panels
    )


def assert_close(actual: tuple[int, ...], expected: tuple[int, ...], tol: int = 3) -> None:
    assert all(abs(a - e) <= tol for a, e in zip(actual, expected, strict=True)), (actual, expected)


def test_grid_on_white_page() -> None:
    rects = [(40, 40, 340, 540), (420, 40, 340, 540), (40, 620, 340, 540), (420, 620, 340, 540)]
    found = boxes(make_comic_page(rects))
    assert len(found) == 4
    for actual, expected in zip(found, sorted(rects), strict=True):
        assert_close(actual, expected)


def test_dark_background_with_black_frames() -> None:
    # Black frames are close to a dark page in grayscale; colour distance still separates them.
    assert len(boxes(make_comic_page(REAL_LAYOUT, background=MAROON))) == 5


def test_noisy_background_and_art() -> None:
    assert len(boxes(make_comic_page(REAL_LAYOUT, background=MAROON, noise=8.0))) == 5


def test_real_layout_reading_order_ltr_and_rtl() -> None:
    panels = detector.detect(make_comic_page(REAL_LAYOUT, background=MAROON)).panels
    ltr = XYCutOrdering().order(panels, ReadingDirection.LTR)
    got = [(p.bbox.x, p.bbox.y) for p in ltr.panels]
    expected = [(40, 40), (290, 40), (290, 290), (40, 560), (410, 560)]
    for actual, exp in zip(got, expected, strict=True):
        assert_close(actual, exp)
    assert ltr.confidence == 1.0
    rtl = XYCutOrdering().order(panels, ReadingDirection.RTL)
    assert rtl.panels[0].bbox.x > 200  # top-right block first in manga order


def test_art_bleeding_off_the_page_edge_does_not_break_background_detection() -> None:
    rects = [(0, 0, 800, 500), (40, 540, 340, 300), (420, 540, 340, 300)]
    found = boxes(make_comic_page(rects, size=(800, 900)))
    assert len(found) == 3


def test_blank_page_has_no_panels() -> None:
    assert detector.detect(np.full((300, 200, 3), 255, dtype=np.uint8)).panels == ()


def test_ignores_specks_below_min_area() -> None:
    assert boxes(make_comic_page([(20, 20, 30, 30)])) == []


def test_panels_touching_along_a_flat_frame_divider_are_split() -> None:
    # No paper between them, but their two frames make a band of one flat colour down the full
    # height, and a frame line runs along it: that is a divider, not an edge inside a panel.
    found = boxes(make_comic_page([(40, 40, 340, 500), (380, 40, 340, 500)]))
    assert len(found) == 2


def test_confidence_is_within_range() -> None:
    for p in detector.detect(make_comic_page(REAL_LAYOUT, background=MAROON)).panels:
        assert 0.0 <= p.confidence <= 1.0


SAMPLES = Path(__file__).resolve().parents[2] / "samples"


@pytest.mark.skipif(not (SAMPLES / "001.webp").exists(), reason="local sample not present")
def test_real_sample_001_is_split_into_separate_panels() -> None:
    # Local, git-ignored sample (dark page, jagged hand-drawn frames). The contour
    # detector found 2 panels here; the projection detector separates at least 4.
    image = decode_image((SAMPLES / "001.webp").read_bytes(), ImageLimits(10**8, 10**8))
    assert len(detector.detect(image.pixels).panels) >= 4


def test_slanted_gutter_is_cut_along_its_slope() -> None:
    page = np.full((1200, 800, 3), 255, dtype=np.uint8)
    cv2.fillPoly(page, [np.array([[40, 40], [760, 40], [760, 420], [40, 700]])], (60, 90, 200))
    cv2.fillPoly(page, [np.array([[40, 760], [760, 480], [760, 1160], [40, 1160]])], (200, 90, 60))
    assert len(detector.detect(page).panels) == 2


def wobbly_gutter_page(*, bridged: bool) -> np.ndarray:
    """Two dark-framed panels split by a thin, slightly slanted, jagged gutter."""
    height, width = 700, 1100
    page = np.full((height, width, 3), MAROON, dtype=np.uint8)
    rng = np.random.default_rng(1)
    rows = np.arange(height)
    centre = 520 + 0.015 * rows + np.cumsum(rng.integers(-1, 2, height)) * 0.3
    for y in rows:
        left, right = int(centre[y]) - 5, int(centre[y]) + 5
        page[y, 20:left] = (200, 120, 60)
        page[y, right : width - 20] = (60, 140, 200)
    page[:20] = page[-20:] = MAROON
    if bridged:  # art from both panels touches across the gutter for a few rows
        for bridge in range(300, 310):
            page[bridge, int(centre[bridge]) - 6 : int(centre[bridge]) + 6] = (20, 20, 20)
    return page


@pytest.mark.parametrize("bridged", [False, True])
def test_wobbly_gutter_splits_even_when_art_bridges_it(bridged: bool) -> None:
    assert len(detector.detect(wobbly_gutter_page(bridged=bridged)).panels) == 2


def test_wobbly_gutter_does_not_split_a_single_panel() -> None:
    page = np.full((700, 1100, 3), MAROON, dtype=np.uint8)
    page[20:680, 20:1080] = (200, 120, 60)
    assert len(detector.detect(page).panels) == 1


def banner_page() -> np.ndarray:
    """Title lettering above two framed panels that touch the lettering's baseline."""
    page = np.full((1200, 800, 3), 240, dtype=np.uint8)
    rng = np.random.default_rng(3)
    for x in range(70, 650, 40):  # overlapping letters of uneven height
        page[int(rng.integers(80, 150)) : 230, x : x + 60] = (30, 30, 200)
    page[230:236, 40:760] = 0  # the top frame line sits right under the lettering
    page[236:1160, 40:760] = (200, 170, 120)
    page[700:706, 40:760] = 0  # a frame line between the two stacked panels
    page[236:1160, 40:46] = 0
    page[236:1160, 754:760] = 0
    page[1154:1160, 40:760] = 0
    return page


def test_title_lettering_above_a_frame_line_is_returned_as_a_first_panel() -> None:
    title, *rest = sorted(boxes(banner_page()), key=lambda box: box[1])
    assert rest, "the framed panels are still found"
    _, title_top, _, title_height = title
    assert title_top < 100  # the lettering band
    assert title_top + title_height <= 240  # ...ends at the frame line
    assert all(y >= 225 for _, y, _, _ in rest)


def test_a_strip_between_two_panels_is_shared_between_them() -> None:
    kept = [(0, 100, 0, 100), (200, 300, 0, 100)]  # y0, y1, x0, x1
    strip = (100, 200, 0, 100)
    above, below = ProjectionPanelDetector._attach_strips(kept, [strip], 10, 5_000)
    assert above == (0, 140, 0, 100)
    assert below == (140, 300, 0, 100)


def test_a_small_strip_at_the_top_goes_to_the_panel_below_but_a_banner_is_its_own_panel() -> None:
    panel = [(200, 300, 0, 100)]
    small, banner = (150, 200, 0, 100), (0, 200, 0, 100)
    assert ProjectionPanelDetector._attach_strips(panel, [small], 10, 20_000) == [
        (150, 300, 0, 100)
    ]
    assert ProjectionPanelDetector._attach_strips(panel, [banner], 10, 20_000) == [*panel, banner]


def test_full_bleed_art_touching_the_next_row_is_a_panel_not_a_banner() -> None:
    page = np.full((1200, 800, 3), 255, dtype=np.uint8)
    ramp = np.linspace(60, 200, 420, dtype=np.uint8)[:, None]
    page[0:420] = np.stack([ramp, ramp // 2, np.full_like(ramp, 90)], axis=2)  # art off the page
    page[420:426, 60:740] = 0  # the next row's top frame, with no gutter above it
    page[426:1140, 60:740] = (200, 170, 120)
    page[426:1140, 60:66] = page[426:1140, 734:740] = 0
    page[1134:1140, 60:740] = 0
    found = boxes(page)
    assert any(y == 0 and w >= 780 for _, y, w, _ in found), found


def test_stained_paper_page_is_cut_through_the_local_paper_colour() -> None:
    assert len(boxes(stained_page())) == 2


def lines_at(*rows: tuple[int, int, int]) -> FrameLines:
    return FrameLines(horizontal=list(rows), vertical=[], height=1000, width=800)


def test_a_panel_edge_snaps_out_to_the_frame_line_just_beyond_it() -> None:
    panel = [(300, 600, 50, 450)]  # y0, y1, x0, x1: pale top and bottom were trimmed
    lines = lines_at((270, 40, 460), (625, 40, 460))
    assert ProjectionPanelDetector._snap_to_frame_lines(panel, lines, 60) == [(270, 625, 50, 450)]


def test_snapping_ignores_short_distant_and_neighbouring_lines() -> None:
    panel, above = (300, 600, 50, 450), (100, 262, 50, 450)  # a panel whose bottom frame is y=262
    lines = lines_at((270, 40, 120), (150, 40, 460), (262, 40, 460))
    snapped = ProjectionPanelDetector._snap_to_frame_lines([above, panel], lines, 60)
    assert snapped[1] == panel  # too short, too far, and the panel above's own frame


def test_a_panel_grows_up_to_the_bottom_of_the_panel_above_across_a_small_gap() -> None:
    above, below = (100, 400, 50, 450), (460, 800, 50, 450)  # a 60px gap holding caption lines
    grown = ProjectionPanelDetector._grow_into_gap_above([above, below], 80)
    assert grown == [above, (400, 800, 50, 450)]


def test_a_large_gap_or_a_panel_beside_is_left_alone() -> None:
    above, far, beside = (100, 400, 50, 450), (700, 900, 50, 450), (460, 800, 460, 860)
    panels = [above, far, beside]
    assert ProjectionPanelDetector._grow_into_gap_above(panels, 80) == panels


def dark_page() -> np.ndarray:
    """A black page: art bleeding off the top and right (its left part as dark as the page), and
    a framed strip of art blocks separated by black bands wider than a gutter."""
    page = np.zeros((1200, 800, 3), dtype=np.uint8)
    page[0:500, 300:800] = (200, 120, 60)
    for x in range(0, 300, 20):  # streaks of detail in the dark left part
        page[0:500, x : x + 4] = (110, 110, 120)
    cv2.rectangle(page, (40, 800), (760, 1150), (200, 40, 80), 6)
    # slanted edges, like art: a straight full-height edge would read as a panel divider
    cv2.fillPoly(page, [np.array([[52, 812], [240, 812], [200, 1138], [52, 1138]])], (60, 140, 200))
    cv2.fillPoly(
        page, [np.array([[470, 812], [748, 812], [748, 1138], [520, 1138]])], (90, 180, 90)
    )
    return page


def test_dark_art_inside_a_frame_does_not_split_the_panel() -> None:
    strip = [b for b in boxes(dark_page()) if b[1] >= 800]
    assert len(strip) == 1  # black bands between the art read as gutters, the frame says one panel
    x, _, w, _ = strip[0]
    assert x <= 52
    assert x + w >= 748


def test_dark_page_art_runs_to_the_page_edge_it_bleeds_towards() -> None:
    top = min(boxes(dark_page()), key=lambda b: b[1])
    assert top[0] == 0  # the black left part of the art is not margin
    assert top[0] + top[2] == 800


def test_light_page_margin_beside_bleeding_art_is_kept() -> None:
    page = np.full((1200, 800, 3), 255, dtype=np.uint8)
    page[0:500, 300:800] = (200, 120, 60)
    for x in range(0, 300, 20):  # lettering-like marks on paper that belong to no panel
        page[100:400, x : x + 4] = (110, 110, 120)
    page[700:1150, 40:760] = (60, 140, 200)
    top = min(boxes(page), key=lambda b: b[1])
    assert top[0] >= 100  # on paper the margin is real: unclaimed ink is not folded into a panel


def test_a_thin_gap_between_framed_pieces_is_still_a_gutter() -> None:
    page = np.zeros((1200, 800, 3), dtype=np.uint8)
    cv2.rectangle(page, (40, 100), (760, 600), (200, 40, 80), 6)
    page[112:588, 52:380] = (60, 140, 200)
    page[112:588, 400:748] = (90, 180, 90)  # a 20px gap, 2.5% of the page width
    pieces = [b for b in boxes(page) if b[1] < 700]
    assert len(pieces) == 2


def test_inset_frames_finds_framed_rectangles_that_run_off_the_page() -> None:
    # Frame lines measured on the blurred-003 fixture: two white-framed insets over the bottom edge
    lines = FrameLines(
        horizontal=[(1945, 111, 515), (2197, 790, 1574), (2916, 932, 1987)],
        vertical=[
            (81, 1976, 2687),
            (106, 1960, 3056),
            (686, 2564, 3056),
            (709, 2683, 3056),
            (761, 2178, 2929),
            (785, 2313, 2903),
        ],
        height=3057,
        width=1988,
    )
    left, right = sorted(lines.inset_frames())
    assert left == (1945, 3057, 106, 686)  # top, left and right lines; the page edge closes it
    assert right == (2197, 2916, 785, 1988)  # top, left and bottom lines; the page edge closes it


def test_inset_frames_ignores_a_rectangle_with_only_two_frame_sides() -> None:
    lines = FrameLines(
        horizontal=[(1000, 100, 700)], vertical=[(100, 1000, 1900)], height=2000, width=1000
    )
    assert lines.inset_frames() == []


def lines_011() -> FrameLines:
    """Frame lines measured on a full-bleed page with two framed insets over its bottom edge."""
    return FrameLines(
        horizontal=[
            (1933, 75, 715),
            (1961, 109, 678),
            (2200, 787, 1987),
            (2872, 254, 519),
            (2903, 787, 1987),
            (2928, 761, 1987),
        ],
        vertical=[
            (80, 2055, 3056),
            (111, 1967, 3056),
            (683, 1962, 3056),
            (709, 1927, 3056),
            (760, 2171, 2927),
            (789, 2248, 2905),
        ],
        height=3057,
        width=1988,
    )


def panel_at(x: int, y: int, w: int, h: int) -> DetectedPanel:
    return DetectedPanel(bbox=BoundingBox(x=x, y=y, width=w, height=h), confidence=1.0)


def test_inset_cut_into_pieces_is_one_panel_and_the_art_slice_beside_it_rejoins() -> None:
    # What the first pass returns: the art, the left inset cut in two at its top frame line, a
    # slice of the art cut off at that line, and the right inset.
    first = [
        panel_at(0, 0, 1988, 1933),
        panel_at(114, 1962, 531, 238),
        panel_at(113, 2200, 566, 664),
        panel_at(715, 1933, 1273, 239),
        panel_at(790, 2203, 1198, 728),
    ]
    mask = np.ones((3057, 1988), dtype=bool)
    layout = ProjectionPanelDetector._inset_layout(first, mask, _Frames(lines_011()))
    found = sorted((p.bbox.x, p.bbox.y, p.bbox.width, p.bbox.height) for p in layout)
    assert len(found) == 3
    assert found[0] == (0, 0, 1988, 2172)  # the art took its slice back
    assert found[1][0] == 111  # the left inset is whole again, down to the page's lower part
    assert found[1][1] <= 1962
    assert found[1][1] + found[1][3] >= 2860


def test_inset_layout_leaves_alone_an_inset_that_was_not_cut_apart() -> None:
    first = [
        panel_at(0, 0, 1988, 1933),
        panel_at(113, 1962, 566, 902),
        panel_at(790, 2203, 1198, 728),
    ]
    mask = np.ones((3057, 1988), dtype=bool)
    assert ProjectionPanelDetector._inset_layout(first, mask, _Frames(lines_011())) == []


def test_cut_cols_wants_a_line_down_nearly_the_whole_height() -> None:
    lines = FrameLines(
        vertical=[(500, 0, 900), (300, 100, 400), (50, 0, 900)], height=1000, width=1000
    )
    assert lines.cut_cols((0, 1000, 0, 1000)) == [500]  # 300 is too short, 50 too near the side


def test_framed_sides_counts_page_edges_and_frame_lines() -> None:
    lines = FrameLines(horizontal=[(900, 0, 1000)], height=1000, width=1000)
    assert lines.framed_sides((0, 900, 0, 1000)) == 4  # top, left, right are page edges
    assert lines.framed_sides((100, 800, 100, 800)) == 0


def test_has_band_finds_a_flat_colour_column_run_and_not_a_lone_line() -> None:
    page = np.random.default_rng(2).integers(0, 255, (600, 1000, 3)).astype(np.uint8)
    region = (0, 600, 0, 1000)
    assert not _has_band(page, region, 500)
    page[:, 490:515] = 255  # a white border between two panels
    assert _has_band(page, region, 500)
    noisy = np.random.default_rng(2).integers(0, 255, (600, 1000, 3)).astype(np.uint8)
    noisy[:, 500] = 0  # one dark line inside art is not a band
    assert not _has_band(noisy, region, 500)
    assert not _has_band(None, region, 500)


def test_caption_box_resting_on_a_panel_is_joined_to_it() -> None:
    # A framed caption box cut off from the top of its panel, with the panel's lettering between
    lines = FrameLines(
        horizontal=[(57, 82, 533), (121, 82, 533), (450, 82, 535)],
        vertical=[(82, 57, 450), (533, 57, 450)],
        height=1443,
        width=975,
    )
    caption, body = (57, 121, 82, 533), (160, 450, 82, 535)
    assert ProjectionPanelDetector._join_caption_boxes([caption, body], lines) == [
        (57, 450, 82, 535)
    ]


def test_unframed_lettering_and_a_thin_panel_are_not_joined_to_the_panel_below() -> None:
    lines = FrameLines(horizontal=[(450, 82, 535)], height=1443, width=975)
    banner, body = (57, 121, 82, 533), (160, 450, 82, 535)  # no frame lines: a title banner
    assert ProjectionPanelDetector._join_caption_boxes([banner, body], lines) == [banner, body]
    tall = (57, 300, 82, 533)  # not short, so it is a panel
    assert ProjectionPanelDetector._join_caption_boxes([tall, (310, 900, 82, 535)], lines) == [
        tall,
        (310, 900, 82, 535),
    ]
