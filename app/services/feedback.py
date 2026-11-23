"""Filesystem store for feedback samples: a page image plus its labels, in the corpus format.

Each sample is three sibling files named after the reference's hash::

    <sample_id>.<jpg|png|webp>   the page, exactly as the client sent it
    <sample_id>.labels.json      the corrected panels (docs/evaluation.md), readable by the harness
    <sample_id>.feedback.json    the reference, image hash, time received and what was detected

The reference is opaque and client-chosen, so it never becomes a file name itself.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from PIL import Image

from app.core.errors import FeedbackNotFoundError, FeedbackStorageFullError
from app.evaluation.labels import IMAGE_SUFFIXES, LABELS_SUFFIX, PageLabels

if TYPE_CHECKING:
    from pathlib import Path

META_SUFFIX = ".feedback.json"
FEEDBACK_TAG = "feedback"
_EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


def sample_id_for(reference: str) -> str:
    return hashlib.sha256(reference.encode()).hexdigest()[:32]


@dataclass(frozen=True, slots=True)
class SaveOutcome:
    sample_id: str
    replaced: bool
    unchanged: bool
    image_sha256: str


class FeedbackStore:
    def __init__(self, root: Path, *, max_samples: int, max_total_bytes: int) -> None:
        self.root = root
        self.max_samples = max_samples
        self.max_total_bytes = max_total_bytes
        self._lock = threading.Lock()

    def save(
        self,
        reference: str,
        image: bytes,
        labels: PageLabels,
        detected: dict[str, Any] | None,
    ) -> SaveOutcome:
        """Store a sample, replacing the one with the same reference. Raises when over a limit."""
        sample_id = sample_id_for(reference)
        digest = hashlib.sha256(image).hexdigest()
        labels_text = labels.model_dump_json(indent=2) + "\n"
        extension = _extension(image)

        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            existing = self._read_meta(sample_id)
            replaced = existing is not None
            if (
                existing is not None
                and existing.get("image_sha256") == digest
                and existing.get("detected") == detected
                and self._labels_path(sample_id).read_text() == labels_text
            ):
                return SaveOutcome(sample_id, replaced=True, unchanged=True, image_sha256=digest)

            meta = {
                "reference": reference,
                "image_sha256": digest,
                "received_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "detected": detected,
            }
            meta_text = json.dumps(meta, indent=2) + "\n"
            self._check_limits(
                sample_id, len(image) + len(labels_text.encode()) + len(meta_text.encode())
            )

            # Labels go last: the harness only sees a sample once it is complete.
            self._labels_path(sample_id).unlink(missing_ok=True)
            for old in self._images(sample_id):
                old.unlink()
            _write_atomic(self.root / f"{sample_id}{extension}", image)
            _write_atomic(self._meta_path(sample_id), meta_text.encode())
            _write_atomic(self._labels_path(sample_id), labels_text.encode())
        return SaveOutcome(sample_id, replaced=replaced, unchanged=False, image_sha256=digest)

    def delete(self, reference: str) -> None:
        sample_id = sample_id_for(reference)
        with self._lock:
            if self._read_meta(sample_id) is None:
                msg = "No feedback sample has this reference."
                raise FeedbackNotFoundError(msg)
            self._labels_path(sample_id).unlink(missing_ok=True)
            self._meta_path(sample_id).unlink(missing_ok=True)
            for image in self._images(sample_id):
                image.unlink()

    def _check_limits(self, sample_id: str, incoming: int) -> None:
        count = 0
        total = 0
        for meta in self.root.glob(f"*{META_SUFFIX}"):
            other = meta.name.removesuffix(META_SUFFIX)
            if other == sample_id:
                continue  # about to be replaced
            count += 1
            total += sum(f.stat().st_size for f in self.root.glob(f"{other}.*"))
        if count + 1 > self.max_samples:
            msg = "The feedback folder holds the maximum number of samples."
            raise FeedbackStorageFullError(msg, {"max_samples": self.max_samples})
        if total + incoming > self.max_total_bytes:
            msg = "The feedback folder has no room for this sample."
            raise FeedbackStorageFullError(msg, {"max_total_bytes": self.max_total_bytes})

    def _read_meta(self, sample_id: str) -> dict[str, Any] | None:
        path = self._meta_path(sample_id)
        if not path.is_file():
            return None
        data: dict[str, Any] = json.loads(path.read_text())
        return data

    def _images(self, sample_id: str) -> list[Path]:
        return [p for p in self.root.glob(f"{sample_id}.*") if p.suffix.lower() in IMAGE_SUFFIXES]

    def _labels_path(self, sample_id: str) -> Path:
        return self.root / f"{sample_id}{LABELS_SUFFIX}"

    def _meta_path(self, sample_id: str) -> Path:
        return self.root / f"{sample_id}{META_SUFFIX}"


def _extension(image: bytes) -> str:
    with Image.open(io.BytesIO(image)) as img:  # already validated by the decoder
        return _EXTENSIONS[str(img.format)]


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
