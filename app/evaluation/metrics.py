"""Scoring: match predicted panels to ground truth and aggregate."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from app.domain.models import NormalizedBox

IOU_LOOSE = 0.5
IOU_STRICT = 0.75
OVERLAP_SHARE = 0.5  # of the smaller box, for two ground-truth panels to count as overlapping


def iou(a: NormalizedBox, b: NormalizedBox) -> float:
    """Intersection over union; invariant to independent x/y scaling, so normalized is fine."""
    ix = max(0.0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x))
    iy = max(0.0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y))
    inter = ix * iy
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def match(
    predicted: Sequence[NormalizedBox], truth: Sequence[NormalizedBox], threshold: float
) -> list[tuple[int, int]]:
    """One-to-one greedy matching by descending IoU. Returns (predicted_idx, truth_idx) pairs."""
    # Best overlap first; on a tie the earlier prediction / earlier ground-truth panel wins.
    candidates = sorted(
        ((iou(p, t), pi, ti) for pi, p in enumerate(predicted) for ti, t in enumerate(truth)),
        key=lambda c: (-c[0], c[1], c[2]),
    )
    used_p: set[int] = set()
    used_t: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for score, pi, ti in candidates:
        if score < threshold:
            break
        if pi in used_p or ti in used_t:
            continue
        used_p.add(pi)
        used_t.add(ti)
        pairs.append((pi, ti))
    return pairs


def order_agreement(pairs: Sequence[tuple[int, int]]) -> tuple[int, int]:
    """(concordant, total) pairs-of-panels whose relative order matches the ground truth."""
    concordant = total = 0
    for (p1, t1), (p2, t2) in combinations(sorted(pairs), 2):
        total += 1
        concordant += (p1 < p2) == (t1 < t2)
    return concordant, total


def _overlaps(a: NormalizedBox, b: NormalizedBox) -> bool:
    """True if one box covers more than half of the other (an inset, or a scene behind an inset)."""
    ix = max(0.0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x))
    iy = max(0.0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y))
    return ix * iy > OVERLAP_SHARE * min(a.width * a.height, b.width * b.height)


def overlap_order_agreement(
    pairs: Sequence[tuple[int, int]], truth: Sequence[NormalizedBox]
) -> tuple[int, int]:
    """(concordant, total) for the matched pairs of panels that overlap in the ground truth.

    Panels that sit over one another have no clean top-to-bottom order: a full-bleed scene starts
    above the inset that reads before it. Plain `order_agreement` counts these pairs among all
    the others; this isolates them, so a change to how overlaps are read shows up as a number.
    """
    concordant = total = 0
    for (p1, t1), (p2, t2) in combinations(sorted(pairs), 2):
        if _overlaps(truth[t1], truth[t2]):
            total += 1
            concordant += (p1 < p2) == (t1 < t2)
    return concordant, total


@dataclass(frozen=True, slots=True)
class PageResult:
    id: str
    tags: frozenset[str]
    n_truth: int
    n_predicted: int
    matched_loose: int  # IoU >= 0.5
    matched_strict: int  # IoU >= 0.75
    order_concordant: int
    order_total: int
    correct: bool  # every panel found at IoU >= 0.5, nothing extra, same order
    sequence_confidence: float
    whole_page_fallback: bool
    overlap_concordant: int = 0  # order_* restricted to panels that overlap in the ground truth
    overlap_total: int = 0


@dataclass(frozen=True, slots=True)
class Prediction:
    boxes: tuple[NormalizedBox, ...]  # in the order the service reports them
    sequence_confidence: float
    whole_page_fallback: bool


def score_page(
    page_id: str,
    tags: frozenset[str],
    prediction: Prediction,
    truth: Sequence[NormalizedBox],
) -> PageResult:
    predicted = prediction.boxes
    loose = match(predicted, truth, IOU_LOOSE)
    strict = match(predicted, truth, IOU_STRICT)
    concordant, total = order_agreement(loose)
    overlap_concordant, overlap_total = overlap_order_agreement(loose, truth)
    correct = len(predicted) == len(truth) == len(loose) and concordant == total
    return PageResult(
        id=page_id,
        tags=tags,
        n_truth=len(truth),
        n_predicted=len(predicted),
        matched_loose=len(loose),
        matched_strict=len(strict),
        order_concordant=concordant,
        order_total=total,
        correct=correct,
        sequence_confidence=prediction.sequence_confidence,
        whole_page_fallback=prediction.whole_page_fallback,
        overlap_concordant=overlap_concordant,
        overlap_total=overlap_total,
    )


def auc(positives: Sequence[float], negatives: Sequence[float]) -> float | None:
    """P(random positive scores higher than random negative); ties count half."""
    if not positives or not negatives:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


@dataclass(frozen=True, slots=True)
class Summary:
    pages: int
    panels_truth: int
    panels_predicted: int
    precision_50: float | None
    recall_50: float | None
    f1_50: float | None
    precision_75: float | None
    recall_75: float | None
    f1_75: float | None
    order_accuracy: float | None
    overlap_order_accuracy: float | None  # order accuracy on panels that overlap in the truth
    page_accuracy: float | None
    confidence_auc: float | None  # does sequence_confidence rank correct pages above wrong ones?
    confidence_correct: float | None  # mean sequence_confidence on correct pages
    confidence_incorrect: float | None  # mean sequence_confidence on wrong pages

    def metrics(self) -> dict[str, float | None]:
        """The headline numbers compared against a baseline."""
        return {
            "f1_50": self.f1_50,
            "f1_75": self.f1_75,
            "order_accuracy": self.order_accuracy,
            "page_accuracy": self.page_accuracy,
        }


def summarize(results: Iterable[PageResult]) -> Summary:
    rs = list(results)
    truth = sum(r.n_truth for r in rs)
    predicted = sum(r.n_predicted for r in rs)
    loose = sum(r.matched_loose for r in rs)
    strict = sum(r.matched_strict for r in rs)
    p50, r50 = _ratio(loose, predicted), _ratio(loose, truth)
    p75, r75 = _ratio(strict, predicted), _ratio(strict, truth)
    right = [r.sequence_confidence for r in rs if r.correct]
    wrong = [r.sequence_confidence for r in rs if not r.correct]
    return Summary(
        pages=len(rs),
        panels_truth=truth,
        panels_predicted=predicted,
        precision_50=p50,
        recall_50=r50,
        f1_50=_f1(p50, r50),
        precision_75=p75,
        recall_75=r75,
        f1_75=_f1(p75, r75),
        order_accuracy=_ratio(sum(r.order_concordant for r in rs), sum(r.order_total for r in rs)),
        overlap_order_accuracy=_ratio(
            sum(r.overlap_concordant for r in rs), sum(r.overlap_total for r in rs)
        ),
        page_accuracy=_ratio(len(right), len(rs)),
        confidence_auc=auc(right, wrong),
        confidence_correct=sum(right) / len(right) if right else None,
        confidence_incorrect=sum(wrong) / len(wrong) if wrong else None,
    )
