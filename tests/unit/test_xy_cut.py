from app.domain.models import DetectedPanel, ReadingDirection
from app.services.ordering.xy_cut import XYCutOrdering
from hypothesis import given
from hypothesis import strategies as st

from tests.conftest import panel

ordering = XYCutOrdering()


def positions(panels: tuple[DetectedPanel, ...]) -> list[tuple[int, int]]:
    return [(p.bbox.x, p.bbox.y) for p in panels]


def test_empty_page() -> None:
    result = ordering.order([], ReadingDirection.LTR)
    assert result.panels == ()
    assert result.confidence == 1.0


def test_grid_ltr() -> None:
    grid = [
        panel(420, 620, 340, 540),
        panel(40, 40, 340, 540),
        panel(420, 40, 340, 540),
        panel(40, 620, 340, 540),
    ]
    result = ordering.order(grid, ReadingDirection.LTR)
    assert positions(result.panels) == [(40, 40), (420, 40), (40, 620), (420, 620)]
    assert result.confidence == 1.0


def test_grid_rtl_reverses_columns_not_rows() -> None:
    grid = [
        panel(40, 40, 340, 540),
        panel(420, 40, 340, 540),
        panel(40, 620, 340, 540),
        panel(420, 620, 340, 540),
    ]
    result = ordering.order(grid, ReadingDirection.RTL)
    assert positions(result.panels) == [(420, 40), (40, 40), (420, 620), (40, 620)]


def test_tall_panel_left_then_stacked_right() -> None:
    layout = [panel(420, 340, 340, 300), panel(40, 40, 340, 600), panel(420, 40, 340, 260)]
    result = ordering.order(layout, ReadingDirection.LTR)
    assert positions(result.panels) == [(40, 40), (420, 40), (420, 340)]
    assert result.confidence == 1.0


def test_interlocking_layout_is_ambiguous() -> None:
    # Overlapping extents on both axes cannot be cut cleanly.
    layout = [panel(0, 0, 60, 60), panel(40, 40, 60, 60)]
    result = ordering.order(layout, ReadingDirection.LTR)
    assert positions(result.panels) == [(0, 0), (40, 40)]
    assert result.confidence == 0.5


def test_ambiguous_fallback_rtl() -> None:
    layout = [panel(0, 0, 60, 60), panel(40, 0, 60, 60)]
    rtl = ordering.order(layout, ReadingDirection.RTL)
    assert positions(rtl.panels) == [(40, 0), (0, 0)]


def test_uncuttable_group_reads_panels_with_near_equal_tops_left_to_right() -> None:
    # Diagonal gutters make the boxes overlap on both axes, so no clean cut exists. The
    # bottom-right panel's top is 75px above the bottom-left's: still the same row.
    layout = [
        panel(0, 0, 1988, 1188),
        panel(0, 1190, 1641, 939),
        panel(1642, 1297, 346, 724),
        panel(1284, 1823, 704, 1233),
        panel(0, 1898, 1283, 1158),
    ]
    result = ordering.order(layout, ReadingDirection.LTR)
    assert positions(result.panels) == [(0, 0), (0, 1190), (1642, 1297), (0, 1898), (1284, 1823)]
    rtl = ordering.order(layout[3:], ReadingDirection.RTL)
    assert positions(rtl.panels) == [(1284, 1823), (0, 1898)]


rect = st.builds(
    panel,
    st.integers(0, 500),
    st.integers(0, 500),
    st.integers(1, 200),
    st.integers(1, 200),
)


@given(st.lists(rect, max_size=12), st.sampled_from(ReadingDirection))
def test_ordering_is_a_permutation_with_valid_confidence(
    panels: list[DetectedPanel], direction: ReadingDirection
) -> None:
    result = ordering.order(panels, direction)
    assert sorted(map(id, result.panels)) == sorted(map(id, panels))
    assert 0.0 <= result.confidence <= 1.0


def test_small_overlaps_still_separate_panels_with_the_default_tolerance() -> None:
    # Ragged ML boxes overlap by a few pixels; the order must still be found without guessing.
    row = [
        panel(0, 0, 105, 100),
        panel(100, 0, 105, 100),
        panel(0, 102, 105, 100),
        panel(100, 98, 105, 100),
    ]
    result = ordering.order(row, ReadingDirection.LTR)
    assert positions(result.panels) == [(0, 0), (100, 0), (0, 102), (100, 98)]
    assert result.confidence == 1.0


def test_tolerance_zero_keeps_the_strict_behaviour() -> None:
    row = [panel(0, 0, 105, 100), panel(100, 0, 105, 100)]
    strict = XYCutOrdering(gap_tolerance=0.0).order(row, ReadingDirection.LTR)
    assert strict.confidence == 0.5  # overlapping extents: no clean cut, order is a guess


def test_large_overlaps_are_still_ambiguous() -> None:
    result = ordering.order([panel(0, 0, 60, 60), panel(40, 40, 60, 60)], ReadingDirection.LTR)
    assert result.confidence == 0.5


def test_a_title_over_the_first_panel_is_read_first_even_when_the_tops_are_close() -> None:
    # The title box reaches down into the first panel and starts level with it: the two sit over
    # each other, so they are not one row and the title (which ends first) comes first.
    layout = [
        panel(10, 50, 850, 800),  # first panel, top grown up to the lettering
        panel(40, 50, 800, 210),  # title lettering
        panel(30, 900, 400, 330),
        panel(450, 860, 470, 440),
    ]
    result = ordering.order(layout, ReadingDirection.LTR)
    assert positions(result.panels)[:2] == [(40, 50), (10, 50)]


STAGGERED = [  # a tall panel, then two rows beside it whose columns do not line up
    panel(10, 620, 270, 360),  # 1: tall, full height
    panel(290, 620, 400, 160),  # 2
    panel(690, 620, 290, 160),  # 3: right of 2, above 6 and 7
    panel(330, 780, 170, 210),  # 4
    panel(510, 780, 140, 200),  # 5
    panel(660, 790, 160, 190),  # 6
    panel(830, 780, 150, 200),  # 7
]


def test_a_tall_panel_beside_staggered_rows_does_not_make_the_block_read_by_columns() -> None:
    # The gutter at x=690 runs between 2 and 3 and between 5 and 6, so cutting columns first
    # would read 2, 4, 5, 3, 6, 7. The rows beside the tall panel are read row by row.
    result = ordering.order(list(reversed(STAGGERED)), ReadingDirection.LTR)
    assert result.panels == tuple(STAGGERED)


def test_the_same_staggered_layout_mirrored_reads_right_to_left_by_rows() -> None:
    mirrored = [
        panel(1000 - p.bbox.x - p.bbox.width, p.bbox.y, p.bbox.width, p.bbox.height)
        for p in STAGGERED
    ]
    result = ordering.order(list(reversed(mirrored)), ReadingDirection.RTL)
    assert result.panels == tuple(mirrored)


def test_columns_stay_columns_when_the_rest_has_no_horizontal_gutter() -> None:
    # tall panel, then two columns whose panels interlock: nothing to cut into rows
    layout = [
        panel(0, 0, 300, 1000),
        panel(310, 0, 300, 600),
        panel(310, 610, 300, 390),
        panel(620, 0, 300, 400),
        panel(620, 410, 300, 590),
    ]
    result = ordering.order(layout, ReadingDirection.LTR)
    assert positions(result.panels) == [(0, 0), (310, 0), (310, 610), (620, 0), (620, 410)]


# The page from feedback sample abcec3d8: a full-bleed scene (4th) sits behind the second and third
# panels' column, and its top edge is higher than the third panel's. Pixels of a 1952x3000 page.
SCENE_BEHIND_INSETS = [
    panel(80, 0, 1798, 438),  # 1: title strip
    panel(80, 459, 755, 525),  # 2: inset, starts above the scene
    panel(76, 921, 775, 507),  # 3: inset, inside the scene
    panel(0, 534, 1952, 2112),  # 4: full-bleed scene
    panel(45, 2574, 1827, 342),  # 5: bottom strip
]


def test_a_scene_behind_insets_reads_after_the_inset_that_starts_above_it() -> None:
    result = ordering.order(list(reversed(SCENE_BEHIND_INSETS)), ReadingDirection.LTR)
    assert positions(result.panels) == positions(tuple(SCENE_BEHIND_INSETS))


def test_a_scene_that_starts_above_its_insets_is_still_read_first() -> None:
    # Feedback sample 9ac0200d: the background starts at the top and the insets lie on its right.
    layout = [
        panel(0, 0, 1000, 1000),
        panel(510, 370, 470, 145),
        panel(515, 520, 460, 135),
        panel(440, 655, 535, 110),
    ]
    result = ordering.order(list(reversed(layout)), ReadingDirection.LTR)
    assert positions(result.panels) == positions(tuple(layout))


def test_a_scene_whose_top_was_claimed_above_its_insets_still_reads_after_them() -> None:
    # What the hybrid detector returns for abcec3d8 once captions are claimed: the scene's top
    # edge now lies above the first inset's.
    layout = [
        panel(81, 0, 1798, 438),
        panel(80, 459, 756, 526),
        panel(76, 921, 774, 507),
        panel(0, 399, 1952, 2246),
        panel(667, 2570, 1204, 343),
    ]
    result = ordering.order(list(reversed(layout)), ReadingDirection.LTR)
    assert [p.bbox.y for p in result.panels] == [0, 459, 921, 399, 2570]
