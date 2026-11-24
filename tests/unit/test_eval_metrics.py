import pytest
from app.domain.models import NormalizedBox
from app.evaluation.metrics import (
    Prediction,
    auc,
    iou,
    match,
    order_agreement,
    overlap_order_agreement,
    score_page,
    summarize,
)


def box(x: float, y: float, w: float, h: float) -> NormalizedBox:
    return NormalizedBox(x, y, w, h)


A = box(0.0, 0.0, 0.5, 0.5)
B = box(0.5, 0.0, 0.5, 0.5)
C = box(0.0, 0.5, 1.0, 0.5)


def pred(*boxes: NormalizedBox, conf: float = 1.0, fallback: bool = False) -> Prediction:
    return Prediction(boxes=tuple(boxes), sequence_confidence=conf, whole_page_fallback=fallback)


def test_iou_identical_disjoint_and_partial() -> None:
    assert iou(A, A) == 1.0
    assert iou(A, B) == 0.0
    assert iou(A, box(0.25, 0.0, 0.5, 0.5)) == pytest.approx(1 / 3)


def test_iou_is_scale_invariant() -> None:
    # Independent x/y scaling (image width vs height) must not change the score.
    assert iou(box(0, 0, 0.5, 1.0), box(0.25, 0, 0.5, 1.0)) == iou(
        box(0, 0, 0.25, 1.0), box(0.125, 0, 0.25, 1.0)
    )


def test_match_is_one_to_one_and_thresholded() -> None:
    pairs = match([A, A], [A, B], 0.5)
    assert pairs == [(0, 0)]  # the duplicate prediction finds nothing left
    assert match([box(0.25, 0, 0.5, 0.5)], [A], 0.5) == []  # IoU 1/3 < 0.5


def test_match_prefers_the_best_overlap() -> None:
    good = box(0.0, 0.0, 0.5, 0.5)
    rough = box(0.0, 0.0, 0.6, 0.5)
    assert match([rough, good], [A], 0.5) == [(1, 0)]


def test_order_agreement_counts_concordant_pairs() -> None:
    assert order_agreement([(0, 0), (1, 1), (2, 2)]) == (3, 3)
    assert order_agreement([(0, 1), (1, 0)]) == (0, 1)
    assert order_agreement([(0, 0)]) == (0, 0)
    assert order_agreement([(0, 0), (2, 1), (1, 2)]) == (2, 3)


def test_score_page_perfect() -> None:
    r = score_page("p", frozenset(), pred(A, B, C), [A, B, C])
    assert r.correct
    assert (r.matched_loose, r.matched_strict, r.order_concordant, r.order_total) == (3, 3, 3, 3)


def test_score_page_wrong_order_is_incorrect() -> None:
    r = score_page("p", frozenset(), pred(B, A, C), [A, B, C])
    assert not r.correct
    assert r.matched_loose == 3
    assert (r.order_concordant, r.order_total) == (2, 3)


def test_score_page_extra_and_missing_panels_are_incorrect() -> None:
    assert not score_page(
        "p", frozenset(), pred(A, B, C, box(0.1, 0.1, 0.1, 0.1)), [A, B, C]
    ).correct
    assert not score_page("p", frozenset(), pred(A, B), [A, B, C]).correct


def test_score_page_strict_iou_is_stricter() -> None:
    sloppy = box(0.0, 0.0, 0.5, 0.65)  # IoU 0.77 -> still strict; shrink further below
    assert score_page("p", frozenset(), pred(sloppy), [A]).matched_strict == 1
    loose_only = box(0.0, 0.0, 0.5, 0.8)  # IoU 0.625
    r = score_page("p", frozenset(), pred(loose_only), [A])
    assert (r.matched_loose, r.matched_strict) == (1, 0)


def test_auc() -> None:
    assert auc([1.0, 0.9], [0.5, 0.4]) == 1.0
    assert auc([0.4], [0.9]) == 0.0
    assert auc([0.5], [0.5]) == 0.5  # ties count half
    assert auc([], [0.5]) is None


def test_summarize_aggregates_micro_averaged() -> None:
    good = score_page(
        "g",
        frozenset(),
        pred(A, B),
        [A, B],
    )
    bad = score_page("b", frozenset(), pred(A), [A, B, C])
    s = summarize([good, bad])
    assert s.pages == 2
    assert (s.panels_truth, s.panels_predicted) == (5, 3)
    assert s.precision_50 == 1.0
    assert s.recall_50 == pytest.approx(3 / 5)
    assert s.f1_50 == pytest.approx(2 * 1.0 * 0.6 / 1.6)
    assert s.page_accuracy == 0.5
    assert s.order_accuracy == 1.0


def test_summarize_confidence_signal() -> None:
    right = score_page("r", frozenset(), pred(A, conf=0.9), [A])
    wrong = score_page("w", frozenset(), pred(B, conf=0.2), [A])
    s = summarize([right, wrong])
    assert s.confidence_auc == 1.0
    assert (s.confidence_correct, s.confidence_incorrect) == (0.9, 0.2)


def test_summarize_empty_has_no_numbers() -> None:
    s = summarize([])
    assert s.pages == 0
    assert s.f1_50 is None
    assert s.page_accuracy is None
    assert s.metrics() == {
        "f1_50": None,
        "f1_75": None,
        "order_accuracy": None,
        "page_accuracy": None,
    }


def test_overlap_order_only_counts_pairs_that_overlap_in_the_truth() -> None:
    inset = NormalizedBox(0.1, 0.3, 0.3, 0.2)
    scene = NormalizedBox(0.0, 0.2, 1.0, 0.7)  # covers the inset
    apart = NormalizedBox(0.0, 0.95, 1.0, 0.05)
    truth = [inset, scene, apart]
    right = overlap_order_agreement([(0, 0), (1, 1), (2, 2)], truth)
    swapped = overlap_order_agreement([(0, 1), (1, 0), (2, 2)], truth)
    assert right == (1, 1)  # only the inset/scene pair overlaps
    assert swapped == (0, 1)
    assert overlap_order_agreement([(0, 0), (1, 2)], truth) == (0, 0)
