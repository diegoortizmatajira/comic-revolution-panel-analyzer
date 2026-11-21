"""ML panel detector that runs a YOLO-style ONNX model with onnxruntime.

No weights ship with this project: the model file is supplied by whoever deploys it
(``PANELS_ML_MODEL``) and its licence is that person's responsibility, see docs/adr/0002.

The model must take a square NCHW float32 RGB input in 0-1 and produce either

* ``yolov8``  - ``(1, 4 + classes, N)`` or ``(1, N, 4 + classes)``: centre x, centre y, width,
  height, then one score per class, in input pixels (what Ultralytics ``format=onnx`` writes), or
* ``end2end`` - ``(1, N, 6)``: x1, y1, x2, y2, score, class, already de-duplicated.

or, as ``dfine`` (D-FINE / RT-DETR exports), take ``images`` plus ``orig_target_sizes`` (the
page's ``[width, height]``, int64) and return ``labels``, ``boxes`` (x1, y1, x2, y2) and
``scores`` by name. The page is stretched to the square, not letterboxed, and the boxes come
back in page pixels.
"""

from __future__ import annotations

import copy
import hashlib
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np

from app.domain.models import BoundingBox, DetectedPanel, Detection

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from numpy.typing import NDArray

OutputFormat = Literal["yolov8", "end2end", "dfine"]
PAD_VALUE = 114 / 255  # Ultralytics' letterbox grey
MIN_SIDE_PX = 2
VERSION = "1"


class InferenceSession(Protocol):
    """The slice of ``onnxruntime.InferenceSession`` this detector uses."""

    def get_inputs(self) -> Sequence[Any]: ...

    def run(self, output_names: object, input_feed: dict[str, object]) -> Sequence[object]: ...


class ModelError(RuntimeError):
    """The ML model is missing, cannot be loaded, or returned something unexpected."""


def _default_session_factory(path: Path) -> InferenceSession:
    try:
        import onnxruntime as ort  # noqa: PLC0415  (optional dependency)
    except ImportError as exc:
        msg = "onnxruntime is not installed; install the 'ml' extra (uv sync --extra ml)."
        raise ModelError(msg) from exc
    session: InferenceSession = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session


def file_digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


@dataclass(frozen=True, slots=True)
class Letterbox:
    """How a page was fitted into the square model input, to map boxes back to the page."""

    scale: float
    pad_x: int
    pad_y: int
    page_width: int
    page_height: int

    def to_box(self, x1: float, y1: float, x2: float, y2: float) -> BoundingBox | None:
        left = float(np.clip((x1 - self.pad_x) / self.scale, 0, self.page_width))
        top = float(np.clip((y1 - self.pad_y) / self.scale, 0, self.page_height))
        right = float(np.clip((x2 - self.pad_x) / self.scale, 0, self.page_width))
        bottom = float(np.clip((y2 - self.pad_y) / self.scale, 0, self.page_height))
        if right - left < MIN_SIDE_PX or bottom - top < MIN_SIDE_PX:
            return None
        x, y = round(left), round(top)
        return BoundingBox(
            x=x, y=y, width=max(1, round(right) - x), height=max(1, round(bottom) - y)
        )


def letterbox(page: NDArray[np.uint8], size: int) -> tuple[NDArray[np.float32], Letterbox]:
    """Fit the page into a size x size square, keeping its aspect ratio, padded with grey."""
    import cv2  # noqa: PLC0415

    height, width = page.shape[:2]
    scale = min(size / height, size / width)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(page, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.float32)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized.astype(np.float32) / 255.0
    blob = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None])
    return blob, Letterbox(scale, pad_x, pad_y, width, height)


def stretch(page: NDArray[np.uint8], size: int) -> NDArray[np.float32]:
    """Resize the page to size x size ignoring its aspect ratio (how D-FINE is trained)."""
    import cv2  # noqa: PLC0415

    resized = cv2.resize(page, (size, size), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray((resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])


def nms(boxes: NDArray[np.float32], scores: NDArray[np.float32], iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression on x1, y1, x2, y2 boxes. Returns kept indices."""
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size:
        best = int(order[0])
        keep.append(best)
        rest = order[1:]
        if not rest.size:
            break
        x1 = np.maximum(boxes[best, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[best, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[best, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[best, 3], boxes[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_best = (boxes[best, 2] - boxes[best, 0]) * (boxes[best, 3] - boxes[best, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        union = area_best + area_rest - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= iou_threshold]
    return keep


class _SessionHolder:
    """The inference session, loaded on first use and shared by detectors made from one another."""

    def __init__(self) -> None:
        self._session: InferenceSession | None = None
        self._lock = threading.Lock()

    def get(self, load: Callable[[], InferenceSession]) -> InferenceSession:
        if self._session is None:
            with self._lock:
                if self._session is None:  # weights are loaded on first use, once
                    self._session = load()
        return self._session


class OnnxPanelDetector:
    name = "ml"

    def __init__(  # noqa: PLR0913
        self,
        model_path: Path,
        *,
        input_size: int = 640,
        confidence: float = 0.25,
        nms_iou: float = 0.45,
        output_format: OutputFormat = "yolov8",
        panel_class: int = 0,
        session_factory: Callable[[Path], InferenceSession] = _default_session_factory,
    ) -> None:
        if not model_path.is_file():
            msg = f"ML model not found: {model_path}"
            raise ModelError(msg)
        self._path = model_path
        self._size = input_size
        self._confidence = confidence
        self._nms_iou = nms_iou
        self._format = output_format
        self._panel_class = panel_class
        self._factory = session_factory
        self._holder = _SessionHolder()
        # The weights are part of what produced a result, so they are part of the version.
        self.version = f"{VERSION}:{file_digest(model_path)[:12]}"

    def detect(self, page: NDArray[np.uint8]) -> Detection:
        if self._format == "dfine":
            return self._detect_dfine(page)
        blob, fit = letterbox(page, self._size)
        session = self._load()
        input_name = session.get_inputs()[0].name
        raw = np.asarray(session.run(None, {input_name: blob})[0])
        boxes, scores = self._decode(raw)
        panels: list[DetectedPanel] = []
        for (x1, y1, x2, y2), score in zip(boxes, scores, strict=True):
            box = fit.to_box(float(x1), float(y1), float(x2), float(y2))
            if box is not None:
                panels.append(DetectedPanel(bbox=box, confidence=float(np.clip(score, 0.0, 1.0))))
        return Detection(panels=tuple(panels), method=self.name)

    def _detect_dfine(self, page: NDArray[np.uint8]) -> Detection:
        height, width = page.shape[:2]
        session = self._load()
        inputs = session.get_inputs()
        if len(inputs) < 2:
            msg = "A dfine ML model needs two inputs: images and orig_target_sizes"
            raise ModelError(msg)
        feed = {
            inputs[0].name: stretch(page, self._size),
            inputs[1].name: np.array([[width, height]], dtype=np.int64),
        }
        labels, boxes, scores = (
            np.asarray(out) for out in session.run(["labels", "boxes", "scores"], feed)
        )
        if boxes.ndim != 3 or boxes.shape[2] != 4:
            msg = f"Unexpected dfine ML model boxes shape {boxes.shape}"
            raise ModelError(msg)
        keep = (scores[0] >= self._confidence) & (labels[0] == self._panel_class)
        found, found_scores = boxes[0][keep].astype(np.float32), scores[0][keep].astype(np.float32)
        fit = Letterbox(1.0, 0, 0, width, height)  # boxes are already in page pixels
        panels: list[DetectedPanel] = []
        for index in nms(found, found_scores, self._nms_iou):
            box = fit.to_box(*(float(v) for v in found[index]))
            if box is not None:
                panels.append(
                    DetectedPanel(bbox=box, confidence=float(np.clip(found_scores[index], 0, 1)))
                )
        return Detection(panels=tuple(panels), method=self.name)

    def with_thresholds(self, confidence: float, nms_iou: float) -> OnnxPanelDetector:
        """A detector with other score thresholds that shares this one's loaded weights."""
        clone = copy.copy(self)
        clone._confidence = confidence
        clone._nms_iou = nms_iou
        return clone

    def _load(self) -> InferenceSession:
        return self._holder.get(lambda: self._factory(self._path))

    def _decode(self, raw: NDArray[Any]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        out = raw[0] if raw.ndim == 3 else raw
        if out.ndim != 2:
            msg = f"Unexpected ML model output shape {raw.shape}"
            raise ModelError(msg)
        out = out.astype(np.float32)
        if self._format == "end2end":
            return self._decode_end2end(out)
        if out.shape[0] < out.shape[1]:  # (4 + classes, N) -> (N, 4 + classes)
            out = out.T
        column = 4 + self._panel_class
        if out.shape[1] <= column:
            msg = f"ML model output has no class {self._panel_class}: shape {raw.shape}"
            raise ModelError(msg)
        keep = out[:, column] >= self._confidence
        out = out[keep]
        if not out.shape[0]:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
        cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        scores = out[:, column]
        kept = nms(boxes, scores, self._nms_iou)
        return boxes[kept], scores[kept]

    def _decode_end2end(
        self, out: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if out.shape[1] != 6:
            msg = f"Unexpected end2end ML model output shape {out.shape}"
            raise ModelError(msg)
        keep = (out[:, 4] >= self._confidence) & (out[:, 5] == self._panel_class)
        return out[keep, :4], out[keep, 4]
