"""Text and JSON reports, plus baseline comparison."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from app.evaluation.labels import LabeledPage
    from app.evaluation.metrics import Summary
    from app.evaluation.runner import DetectorReport

SCHEMA_VERSION = 1


def _num(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def to_json(pages: Sequence[LabeledPage], reports: Sequence[DetectorReport]) -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "corpus": {"pages": sorted(p.id for p in pages)},
        "detectors": {
            r.detector: {
                "version": r.version,
                "overall": asdict(r.overall),
                "by_tag": {tag: asdict(s) for tag, s in r.by_tag.items()},
                "pages": [{**asdict(p), "tags": sorted(p.tags)} for p in r.pages],
            }
            for r in reports
        },
    }


def _summary_row(name: str, extra: str, s: Summary) -> str:
    return (
        f"{name:<18}{extra:<9}{s.pages:>5}  {_num(s.precision_50):>7} {_num(s.recall_50):>6} "
        f"{_num(s.f1_50):>6} {_num(s.f1_75):>6}  {_num(s.order_accuracy):>6} "
        f"{_num(s.overlap_order_accuracy):>7} {_num(s.page_accuracy):>8}  "
        f"{_num(s.confidence_auc):>8}"
    )


_HEADER = (
    f"{'':<18}{'version':<9}{'pages':>5}  {'prec@50':>7} {'rec@50':>6} {'f1@50':>6} "
    f"{'f1@75':>6}  {'order':>6} {'overlap':>7} {'page_acc':>8}  {'conf_auc':>8}"
)


def to_text(
    pages: Sequence[LabeledPage], reports: Sequence[DetectorReport], *, verbose: bool = False
) -> str:
    synthetic = sum(p.source == "synthetic" for p in pages)
    truth = sum(len(p.panels) for p in pages)
    lines = [
        f"Corpus: {len(pages)} pages ({len(pages) - synthetic} labelled files, "
        f"{synthetic} synthetic), {truth} ground-truth panels",
        "",
        "Detector comparison",
        _HEADER,
        *(_summary_row(r.detector, r.version, r.overall) for r in reports),
        "",
        "page_acc: every panel found (IoU>=0.5), nothing extra, same order.",
        "order: share of matched panel pairs read in the ground-truth order.",
        "overlap: the same, only for pairs that overlap in the ground truth (insets, scenes).",
        "conf_auc: chance a correct page gets a higher sequence_confidence than a wrong one "
        "(0.5 = no signal, 1.0 = perfect).",
    ]
    for report in reports:
        lines += ["", f"{report.detector}: by tag", _HEADER]
        lines += [_summary_row(tag, "", s) for tag, s in report.by_tag.items()]
        if verbose:
            lines += ["", f"{report.detector}: pages"]
            lines += [
                f"  {'ok ' if p.correct else 'BAD'} {p.id:<34} truth={p.n_truth} "
                f"pred={p.n_predicted} matched@50={p.matched_loose} "
                f"seq_conf={p.sequence_confidence:.2f}"
                f"{' fallback' if p.whole_page_fallback else ''}"
                for p in report.pages
            ]
    return "\n".join(lines)


def compare(current: dict[str, Any], baseline: dict[str, Any], max_drop: float) -> list[str]:
    """Regressions of the current run against a stored baseline (empty list = fine)."""
    problems: list[str] = []
    if current["corpus"]["pages"] != baseline["corpus"]["pages"]:
        return [
            "The corpus differs from the baseline's; refresh it with --write-baseline "
            "(comparing different pages is meaningless)."
        ]
    for name, now in current["detectors"].items():
        before = baseline["detectors"].get(name)
        if before is None:
            continue
        for metric, old in before["overall"].items():
            if metric not in {"f1_50", "f1_75", "order_accuracy", "page_accuracy"}:
                continue
            new = now["overall"][metric]
            if old is not None and (new is None or new < old - max_drop):
                problems.append(
                    f"{name}: {metric} dropped from {_num(old)} to {_num(new)} "
                    f"(allowed drop {max_drop:.2f})"
                )
    return problems
