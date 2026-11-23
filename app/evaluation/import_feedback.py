"""CLI: ``uv run python -m app.evaluation.import_feedback`` (see docs/evaluation.md).

Copies the samples stored by ``POST /v1/panels/feedback`` into an evaluation corpus folder, as
``feedback-<id>`` pages tagged ``feedback``, so the harness scores them (and reports them as their
own tag row) next to the pages labelled by hand. The feedback folder is already in the corpus
format, so ``--corpus <feedback dir>`` also works without importing.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from app.evaluation.labels import image_files, labels_path

PREFIX = "feedback-"
DEFAULT_SOURCE = Path("samples/feedback")
DEFAULT_DEST = Path("samples")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m app.evaluation.import_feedback",
        description="Copy stored feedback samples into an evaluation corpus folder.",
    )
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="PANELS_FEEDBACK_DIR")
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="corpus folder to copy into")
    p.add_argument("--overwrite", action="store_true", help="replace pages that differ")
    p.add_argument("--dry-run", action="store_true", help="only report what would be copied")
    return p


def import_samples(
    source: Path, dest: Path, *, overwrite: bool = False, dry_run: bool = False
) -> dict[str, list[str]]:
    """Copy every complete sample. Returns the sample ids per outcome."""
    outcome: dict[str, list[str]] = {"added": [], "updated": [], "unchanged": [], "skipped": []}
    for image in image_files(source) if source.is_dir() else []:
        labels = labels_path(image)
        if not labels.is_file():
            continue  # incomplete sample
        target = dest / f"{PREFIX}{image.stem}{image.suffix.lower()}"
        target_labels = labels_path(target)
        if target.exists() and target_labels.exists():
            if (
                target.read_bytes() == image.read_bytes()
                and target_labels.read_bytes() == labels.read_bytes()
            ):
                outcome["unchanged"].append(image.stem)
                continue
            if not overwrite:
                outcome["skipped"].append(image.stem)
                continue
            kind = "updated"
        else:
            kind = "added"
        outcome[kind].append(image.stem)
        if not dry_run:
            dest.mkdir(parents=True, exist_ok=True)
            for old in image_files(dest):  # the image format may have changed
                if old.stem == target.stem and old != target:
                    old.unlink()
            shutil.copyfile(image, target)
            shutil.copyfile(labels, target_labels)
    return outcome


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.source.is_dir():
        print(f"No feedback folder at {args.source}.", file=sys.stderr)
        return 2
    outcome = import_samples(args.source, args.dest, overwrite=args.overwrite, dry_run=args.dry_run)
    verb = "would be " if args.dry_run else ""
    for kind, ids in outcome.items():
        if ids:
            print(f"{len(ids)} {verb}{kind}: {', '.join(ids)}")
    if outcome["skipped"]:
        print("Skipped pages differ from the feedback sample; use --overwrite to replace them.")
    if not any(outcome.values()):
        print("No feedback samples found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
