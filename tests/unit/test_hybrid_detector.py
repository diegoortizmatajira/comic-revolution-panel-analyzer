import numpy as np
import pytest
from app.domain.models import DetectedPanel, Detection
from app.services.detection.hybrid import HybridPanelDetector, ReliabilityPolicy

from tests.conftest import panel

NO_EXTRA = ReliabilityPolicy(extra_panels=0)  # isolate the older rules from the extra-panels rule
PAGE = np.zeros((1000, 1000, 3), dtype=np.uint8)


class StubDetector:
    def __init__(self, name: str, version: str, panels: list[DetectedPanel]) -> None:
        self.name = name
        self.version = version
        self._panels = panels
        self.calls = 0

    def detect(self, page: np.ndarray) -> Detection:
        self.calls += 1
        return Detection(tuple(self._panels), self.name)


def grid(n: int) -> list[DetectedPanel]:
    return [panel(i * 100, 0, 90, 90) for i in range(n)]


def test_a_plausible_primary_result_stands_when_the_model_finds_another_layout() -> None:
    primary, fallback = StubDetector("projection", "1", grid(4)), StubDetector("ml", "9", grid(6))
    result = HybridPanelDetector(primary, fallback, NO_EXTRA).detect(PAGE)
    assert result.method == "projection"
    assert len(result.panels) == 4


@pytest.mark.parametrize("count", [0, 1, 2])
def test_too_few_panels_asks_the_fallback(count: int) -> None:
    primary, fallback = (
        StubDetector("projection", "1", grid(count)),
        StubDetector("ml", "9", grid(5)),
    )
    result = HybridPanelDetector(primary, fallback).detect(PAGE)
    assert result.method == "ml"  # the method reflects who actually answered this page
    assert len(result.panels) == 5


def test_a_very_thin_panel_asks_the_fallback() -> None:
    strip = [panel(0, 0, 90, 90), panel(0, 150, 500, 100), panel(0, 300, 90, 90)]  # 5:1 caption
    primary, fallback = StubDetector("projection", "1", strip), StubDetector("ml", "9", grid(4))
    assert HybridPanelDetector(primary, fallback).detect(PAGE).method == "ml"


def test_just_below_the_thin_threshold_is_trusted() -> None:
    almost = [panel(0, 0, 90, 90), panel(0, 150, 499, 100), panel(0, 300, 90, 90)]
    primary, fallback = StubDetector("projection", "1", almost), StubDetector("ml", "9", grid(4))
    assert HybridPanelDetector(primary, fallback, NO_EXTRA).detect(PAGE).method == "projection"


def test_a_thin_panel_above_all_the_others_is_a_title_and_is_trusted() -> None:
    title = [panel(0, 0, 500, 60), panel(0, 100, 90, 90), panel(100, 100, 90, 90)]  # 8:1 banner
    primary, fallback = StubDetector("projection", "1", title), StubDetector("ml", "9", grid(4))
    result = HybridPanelDetector(primary, fallback, NO_EXTRA).detect(PAGE)
    assert result.method == "projection"  # the model would only drop the title


def test_a_thin_panel_that_is_not_above_the_rest_is_still_distrusted() -> None:
    beside = [panel(0, 0, 500, 60), panel(0, 30, 90, 90), panel(100, 100, 90, 90)]
    primary, fallback = StubDetector("projection", "1", beside), StubDetector("ml", "9", grid(4))
    assert HybridPanelDetector(primary, fallback).detect(PAGE).method == "ml"


def test_thin_in_either_direction_counts() -> None:
    tall = [panel(0, 0, 100, 500), panel(200, 0, 90, 90), panel(300, 0, 90, 90)]
    primary, fallback = StubDetector("projection", "1", tall), StubDetector("ml", "9", grid(4))
    assert HybridPanelDetector(primary, fallback).detect(PAGE).method == "ml"


def test_an_empty_fallback_answer_keeps_the_primary_one() -> None:
    primary, fallback = StubDetector("projection", "1", grid(1)), StubDetector("ml", "9", [])
    result = HybridPanelDetector(primary, fallback).detect(PAGE)
    assert result.method == "projection"
    assert len(result.panels) == 1
    assert fallback.calls == 1


def test_version_covers_both_detectors_so_cached_results_invalidate() -> None:
    a = HybridPanelDetector(StubDetector("projection", "1", []), StubDetector("ml", "1:abc", []))
    b = HybridPanelDetector(StubDetector("projection", "1", []), StubDetector("ml", "1:def", []))
    assert a.name == "hybrid"
    assert a.version == "5[projection@1|ml@1:abc]"
    assert a.version != b.version  # new weights -> new version


def test_the_policy_is_configurable() -> None:
    lenient = ReliabilityPolicy(min_panels=1, max_aspect=100.0, extra_panels=0)
    primary, fallback = StubDetector("projection", "1", grid(1)), StubDetector("ml", "9", grid(5))
    assert HybridPanelDetector(primary, fallback, lenient).detect(PAGE).method == "projection"


def shifted(panels: list[DetectedPanel], dy: int) -> list[DetectedPanel]:
    return [panel(p.bbox.x, p.bbox.y + dy, p.bbox.width, p.bbox.height) for p in panels]


def test_the_model_boxes_are_used_when_both_find_the_same_layout() -> None:
    # The model's boxes sit 5 px lower (IoU 0.89): the same four panels, set on the frames.
    primary = StubDetector("projection", "1", grid(4))
    fallback = StubDetector("ml", "9", shifted(grid(4), 5))
    result = HybridPanelDetector(primary, fallback).detect(PAGE)
    assert result.method == "ml"
    assert [p.bbox.y for p in result.panels] == [5, 5, 5, 5]


def test_boxes_that_only_barely_overlap_are_not_the_same_layout() -> None:
    primary = StubDetector("projection", "1", grid(4))
    fallback = StubDetector("ml", "9", shifted(grid(4), 60))  # IoU 0.2
    assert HybridPanelDetector(primary, fallback).detect(PAGE).method == "projection"


def test_two_primary_panels_cannot_share_one_model_box() -> None:
    twice = [panel(0, 0, 90, 90), panel(0, 0, 90, 90), panel(100, 0, 90, 90)]
    primary = StubDetector("projection", "1", twice)
    model = [panel(0, 0, 90, 90), panel(100, 0, 90, 90), panel(300, 0, 9, 9)]
    fallback = StubDetector("ml", "9", model)
    assert HybridPanelDetector(primary, fallback).detect(PAGE).method == "projection"


def test_a_different_panel_count_is_not_the_same_layout() -> None:
    primary = StubDetector("projection", "1", grid(4))
    fallback = StubDetector("ml", "9", shifted(grid(5), 5))
    assert HybridPanelDetector(primary, fallback, NO_EXTRA).detect(PAGE).method == "projection"


def test_agreement_can_be_turned_off() -> None:
    primary = StubDetector("projection", "1", grid(4))
    fallback = StubDetector("ml", "9", shifted(grid(4), 5))
    off = ReliabilityPolicy(agree_iou=1.0)
    assert HybridPanelDetector(primary, fallback, off).detect(PAGE).method == "projection"


def test_a_model_that_finds_more_panels_than_a_plausible_primary_wins() -> None:
    primary, fallback = StubDetector("projection", "1", grid(4)), StubDetector("ml", "9", grid(6))
    result = HybridPanelDetector(primary, fallback).detect(PAGE)
    assert result.method == "ml"
    assert len(result.panels) == 6


def test_the_extra_panels_rule_needs_the_threshold_and_can_be_turned_off() -> None:
    primary, fallback = StubDetector("projection", "1", grid(4)), StubDetector("ml", "9", grid(6))
    needs_three = ReliabilityPolicy(extra_panels=3)
    assert HybridPanelDetector(primary, fallback, needs_three).detect(PAGE).method == "projection"
    assert HybridPanelDetector(primary, fallback, NO_EXTRA).detect(PAGE).method == "projection"


def test_a_model_with_fewer_panels_does_not_replace_the_primary() -> None:
    primary, fallback = StubDetector("projection", "1", grid(5)), StubDetector("ml", "9", grid(3))
    assert HybridPanelDetector(primary, fallback).detect(PAGE).method == "projection"


def test_model_first_uses_the_model_unless_it_is_degenerate() -> None:
    model_first = ReliabilityPolicy(prefer_model=True, extra_panels=0)
    primary = StubDetector("projection", "1", grid(5))
    three = HybridPanelDetector(primary, StubDetector("ml", "9", grid(3)), model_first)
    assert three.detect(PAGE).method == "ml"
    one = HybridPanelDetector(primary, StubDetector("ml", "9", grid(1)), model_first)
    assert one.detect(PAGE).method == "projection"


def test_a_large_primary_panel_the_model_left_uncovered_is_added() -> None:
    # the model found only two small insets; projection also has the bleeding art under them
    bleed = panel(0, 0, 1000, 800)
    primary = StubDetector("projection", "1", [bleed])
    model = [panel(100, 100, 200, 200), panel(600, 500, 200, 200)]
    result = HybridPanelDetector(primary, StubDetector("ml", "9", model)).detect(PAGE)
    assert result.method == "ml"
    assert result.panels[-1] == bleed
    assert len(result.panels) == 3


def test_a_primary_panel_the_model_already_covers_is_not_added() -> None:
    merged = panel(0, 0, 1000, 400)  # projection merged two panels that the model split
    primary = StubDetector("projection", "1", [merged])
    model = [panel(0, 0, 500, 400), panel(500, 0, 500, 400)]
    result = HybridPanelDetector(primary, StubDetector("ml", "9", model)).detect(PAGE)
    assert result.panels == tuple(model)


def test_a_small_uncovered_primary_panel_is_not_added() -> None:
    primary = StubDetector("projection", "1", [panel(0, 0, 100, 100)])  # 1% of the page
    model = [panel(500, 500, 400, 400)]
    result = HybridPanelDetector(primary, StubDetector("ml", "9", model)).detect(PAGE)
    assert result.panels == tuple(model)


def test_the_bleed_fill_can_be_turned_off() -> None:
    primary = StubDetector("projection", "1", [panel(0, 0, 1000, 800)])
    model = [panel(100, 100, 200, 200), panel(600, 500, 200, 200)]
    off = ReliabilityPolicy(fill_max_covered=0.0)
    result = HybridPanelDetector(primary, StubDetector("ml", "9", model), off).detect(PAGE)
    assert result.panels == tuple(model)
