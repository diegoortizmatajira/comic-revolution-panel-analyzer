"""The ONNX detector, exercised with a fake session: CI needs no weights and no onnxruntime."""

import sys
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from app.services.detection.onnx import (
    Letterbox,
    ModelError,
    OnnxPanelDetector,
    _default_session_factory,
    letterbox,
    nms,
    stretch,
)

SIZE = 64


class FakeInput:
    name = "images"


class FakeSession:
    """Returns a canned output and records what it was fed."""

    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.fed: list[np.ndarray] = []

    def get_inputs(self) -> Sequence[Any]:
        return [FakeInput()]

    def run(self, output_names: object, input_feed: dict[str, object]) -> Sequence[object]:
        blob = input_feed["images"]
        assert isinstance(blob, np.ndarray)
        self.fed.append(blob)
        return [self.output]


def model_file(tmp_path: Path, content: bytes = b"weights-v1") -> Path:
    path = tmp_path / "panels.onnx"
    path.write_bytes(content)
    return path


def detector(
    tmp_path: Path, output: np.ndarray, **kwargs: Any
) -> tuple[OnnxPanelDetector, FakeSession]:
    session = FakeSession(output)
    det = OnnxPanelDetector(
        model_file(tmp_path), input_size=SIZE, session_factory=lambda _: session, **kwargs
    )
    return det, session


def yolo(*rows: tuple[float, float, float, float, float], anchors: int = 16) -> np.ndarray:
    """(1, 5, N) like Ultralytics: cx, cy, w, h, score per column, padded with empty anchors.

    Real exports have thousands of anchors (8400 at 640 px), always more than channels, which is
    how the decoder tells the layouts apart.
    """
    padded = [*rows, *[(0.0, 0.0, 0.0, 0.0, 0.0)] * (anchors - len(rows))]
    return np.array(padded, dtype=np.float32).T[None]


PAGE = np.full((50, 100, 3), 255, dtype=np.uint8)  # 100 wide, 50 tall -> scale 0.64, pad_y 16


def test_letterbox_keeps_aspect_ratio_and_pads_with_grey() -> None:
    blob, fit = letterbox(PAGE, SIZE)
    assert blob.shape == (1, 3, SIZE, SIZE)
    assert blob.dtype == np.float32
    assert (fit.scale, fit.pad_x, fit.pad_y) == (0.64, 0, 16)
    assert blob[0, :, 0, 0] == pytest.approx(114 / 255)  # padded top rows
    assert blob[0, :, SIZE // 2, SIZE // 2] == pytest.approx(1.0)  # the white page


def test_letterbox_maps_boxes_back_to_the_page() -> None:
    fit = Letterbox(scale=0.64, pad_x=0, pad_y=16, page_width=100, page_height=50)
    box = fit.to_box(16, 32, 48, 48)
    assert box is not None
    assert (box.x, box.y, box.width, box.height) == (25, 25, 50, 25)


def test_letterbox_clips_to_the_page_and_drops_slivers() -> None:
    fit = Letterbox(scale=1.0, pad_x=0, pad_y=0, page_width=100, page_height=50)
    clipped = fit.to_box(-10, -10, 200, 200)
    assert clipped is not None
    assert (clipped.x, clipped.y, clipped.width, clipped.height) == (0, 0, 100, 50)
    assert fit.to_box(10, 10, 11, 40) is None  # one pixel wide


def test_nms_keeps_the_best_of_overlapping_boxes() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], dtype=np.float32)
    scores = np.array([0.6, 0.9, 0.5], dtype=np.float32)
    assert sorted(nms(boxes, scores, 0.45)) == [1, 2]
    assert sorted(nms(boxes, scores, 0.99)) == [0, 1, 2]  # only identical boxes are suppressed


def test_nms_of_nothing_is_nothing() -> None:
    assert nms(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), 0.5) == []


def test_decodes_ultralytics_output_and_maps_to_page_pixels(tmp_path: Path) -> None:
    out = yolo(
        (32, 24, 32, 16, 0.9),  # the panel, in letterboxed coordinates
        (33, 24, 32, 16, 0.6),  # a duplicate that NMS must drop
        (10, 10, 8, 8, 0.1),  # below the confidence threshold
    )
    det, _ = detector(tmp_path, out)
    result = det.detect(PAGE)
    assert result.method == "ml"
    (found,) = result.panels
    # x1,y1 = (16,16) in the letterboxed square; the 16 px top padding is removed, then / 0.64.
    assert (found.bbox.x, found.bbox.y, found.bbox.width, found.bbox.height) == (25, 0, 50, 25)
    assert found.confidence == pytest.approx(0.9)


def test_accepts_the_transposed_layout(tmp_path: Path) -> None:
    det, _ = detector(tmp_path, yolo((32, 24, 32, 16, 0.9)).transpose(0, 2, 1))
    assert len(det.detect(PAGE).panels) == 1


def test_picks_the_configured_class(tmp_path: Path) -> None:
    col = np.array([[32], [24], [32], [16], [0.9], [0.2]], dtype=np.float32)
    out = np.concatenate([col, np.zeros((6, 15), np.float32)], axis=1)[None]  # 2 classes
    first, _ = detector(tmp_path, out, panel_class=0)
    second, _ = detector(tmp_path, out, panel_class=1, confidence=0.1)
    assert len(first.detect(PAGE).panels) == 1
    assert second.detect(PAGE).panels[0].confidence == pytest.approx(0.2)
    missing, _ = detector(tmp_path, out, panel_class=2)
    with pytest.raises(ModelError, match="no class 2"):
        missing.detect(PAGE)


def test_decodes_end2end_output(tmp_path: Path) -> None:
    out = np.array(
        [[16, 32, 48, 48, 0.8, 0], [0, 0, 5, 5, 0.9, 1], [0, 0, 5, 5, 0.1, 0]], np.float32
    )
    det, _ = detector(tmp_path, out[None], output_format="end2end")
    (found,) = det.detect(PAGE).panels  # wrong class and low score are dropped
    assert found.bbox.width == 50


def test_unexpected_output_shapes_are_reported(tmp_path: Path) -> None:
    bad, _ = detector(tmp_path, np.zeros((2, 3, 4, 5), np.float32))
    with pytest.raises(ModelError, match="Unexpected ML model output shape"):
        bad.detect(PAGE)
    bad_e2e, _ = detector(tmp_path, np.zeros((1, 7, 5), np.float32), output_format="end2end")
    with pytest.raises(ModelError, match="end2end"):
        bad_e2e.detect(PAGE)


def test_no_detections_is_an_empty_result(tmp_path: Path) -> None:
    det, _ = detector(tmp_path, yolo((32, 24, 32, 16, 0.01)))
    assert det.detect(PAGE).panels == ()


def test_feeds_the_model_a_letterboxed_blob(tmp_path: Path) -> None:
    det, session = detector(tmp_path, yolo((32, 24, 32, 16, 0.9)))
    det.detect(PAGE)
    assert session.fed[0].shape == (1, 3, SIZE, SIZE)


def test_missing_model_file_fails_at_construction(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="not found"):
        OnnxPanelDetector(tmp_path / "nope.onnx")


def test_version_identifies_the_weights(tmp_path: Path) -> None:
    a = OnnxPanelDetector(
        model_file(tmp_path, b"one"), session_factory=lambda _: FakeSession(yolo())
    )
    again = OnnxPanelDetector(
        model_file(tmp_path, b"one"), session_factory=lambda _: FakeSession(yolo())
    )
    b = OnnxPanelDetector(
        model_file(tmp_path, b"two"), session_factory=lambda _: FakeSession(yolo())
    )
    assert a.version == again.version
    assert a.version != b.version
    assert a.version.startswith("1:")


def test_weights_load_lazily_and_only_once(tmp_path: Path) -> None:
    calls: list[Path] = []

    def factory(path: Path) -> FakeSession:
        calls.append(path)
        return FakeSession(yolo((32, 24, 32, 16, 0.9)))

    det = OnnxPanelDetector(model_file(tmp_path), input_size=SIZE, session_factory=factory)
    assert calls == []  # nothing loaded at construction
    det.detect(PAGE)
    det.detect(PAGE)
    assert len(calls) == 1


def test_concurrent_first_use_loads_the_model_once(tmp_path: Path) -> None:
    calls: list[int] = []
    gate = threading.Barrier(4)

    def factory(path: Path) -> FakeSession:
        calls.append(1)
        return FakeSession(yolo((32, 24, 32, 16, 0.9)))

    det = OnnxPanelDetector(model_file(tmp_path), input_size=SIZE, session_factory=factory)

    def worker() -> None:
        gate.wait()
        det.detect(PAGE)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1


def test_missing_onnxruntime_has_an_actionable_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", None)  # makes `import onnxruntime` fail
    with pytest.raises(ModelError, match="uv sync --extra ml"):
        _default_session_factory(model_file(tmp_path))


def test_factory_type_is_a_plain_callable() -> None:
    factory: Callable[[Path], Any] = _default_session_factory
    assert callable(factory)


def test_other_thresholds_share_the_loaded_weights(tmp_path: Path) -> None:
    calls: list[Path] = []

    def factory(path: Path) -> FakeSession:
        calls.append(path)
        return FakeSession(yolo((32, 24, 32, 16, 0.4)))

    strict = OnnxPanelDetector(
        model_file(tmp_path), input_size=SIZE, confidence=0.5, session_factory=factory
    )
    loose = strict.with_thresholds(0.3, 0.45)
    assert strict.detect(PAGE).panels == ()  # score 0.4 is under 0.5
    assert len(loose.detect(PAGE).panels) == 1  # ...and over 0.3
    assert len(calls) == 1  # one session, loaded by whichever ran first
    assert loose.version == strict.version


class DfineSession:
    """A D-FINE style session: two inputs, three named outputs."""

    def __init__(self, labels: list[int], boxes: list[list[float]], scores: list[float]) -> None:
        self.outputs = [
            np.array([labels], np.int64),
            np.array([boxes], np.float32).reshape(1, len(labels), 4),
            np.array([scores], np.float32),
        ]
        self.fed: dict[str, np.ndarray] = {}
        self.output_names: object = None

    def get_inputs(self) -> Sequence[Any]:
        return [FakeInput(), type("Sizes", (), {"name": "orig_target_sizes"})()]

    def run(self, output_names: object, input_feed: dict[str, object]) -> Sequence[object]:
        self.output_names = output_names
        self.fed = {k: np.asarray(v) for k, v in input_feed.items()}
        return self.outputs


def dfine(tmp_path: Path, session: object, **kwargs: Any) -> OnnxPanelDetector:
    return OnnxPanelDetector(
        model_file(tmp_path),
        input_size=SIZE,
        output_format="dfine",
        session_factory=lambda _: session,  # type: ignore[arg-type,return-value]
        **kwargs,
    )


def test_stretch_ignores_the_aspect_ratio() -> None:
    blob = stretch(PAGE, SIZE)
    assert blob.shape == (1, 3, SIZE, SIZE)
    assert blob.dtype == np.float32
    assert np.allclose(blob, 1.0)  # no grey padding


def test_decodes_dfine_output_in_page_pixels(tmp_path: Path) -> None:
    session = DfineSession(
        labels=[0, 0, 1, 0, 0],
        boxes=[[10, 5, 60, 45], [11, 5, 60, 45], [0, 0, 20, 20], [70, 5, 95, 45], [1, 1, 2, 2]],
        scores=[0.9, 0.8, 0.9, 0.7, 0.1],
    )
    det = dfine(tmp_path, session)
    panels = det.detect(PAGE).panels
    # duplicate (NMS), other class and low score are dropped
    assert [(p.bbox.x, p.bbox.width) for p in panels] == [(10, 50), (70, 25)]
    assert session.output_names == ["labels", "boxes", "scores"]
    assert session.fed["images"].shape == (1, 3, SIZE, SIZE)
    assert session.fed["orig_target_sizes"].tolist() == [[100, 50]]  # width, height
    assert session.fed["orig_target_sizes"].dtype == np.int64


def test_dfine_with_no_detections_is_an_empty_result(tmp_path: Path) -> None:
    det = dfine(tmp_path, DfineSession([0], [[0, 0, 50, 50]], [0.01]))
    assert det.detect(PAGE).panels == ()


def test_dfine_model_shape_problems_are_reported(tmp_path: Path) -> None:
    class OneInput(DfineSession):
        def get_inputs(self) -> Sequence[Any]:
            return [FakeInput()]

    with pytest.raises(ModelError, match="orig_target_sizes"):
        dfine(tmp_path, OneInput([0], [[0, 0, 5, 5]], [0.9])).detect(PAGE)

    bad = DfineSession([0], [[0, 0, 5, 5]], [0.9])
    bad.outputs[1] = np.zeros((1, 1, 5), np.float32)
    with pytest.raises(ModelError, match="dfine ML model boxes"):
        dfine(tmp_path, bad).detect(PAGE)
