from pathlib import Path

import pytest
from app.container import build_container, ml_config
from app.core.config import Settings
from app.services.detection.registry import MlConfig, build_detector
from pydantic import ValidationError


def weights(tmp_path: Path) -> Path:
    path = tmp_path / "panels.onnx"
    path.write_bytes(b"not a real model: it is never loaded here")
    return path


def settings(**kwargs: object) -> Settings:
    return Settings(api_keys="k", log_json=False, _env_file=None, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["ml", "hybrid"])
def test_ml_detectors_need_a_model(name: str) -> None:
    with pytest.raises(ValueError, match="PANELS_ML_MODEL"):
        build_detector(name)


def test_builds_the_ml_detector(tmp_path: Path) -> None:
    detector = build_detector("ml", MlConfig(model=weights(tmp_path)))
    assert detector.name == "ml"
    assert detector.version.startswith("1:")


def test_builds_a_hybrid_over_projection_and_the_model(tmp_path: Path) -> None:
    detector = build_detector("hybrid", MlConfig(model=weights(tmp_path)))
    assert detector.name == "hybrid"
    assert "projection@9" in detector.version
    assert "|ml@1:" in detector.version


def test_a_missing_model_file_fails_when_the_detector_is_built(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not found"):
        build_detector("ml", MlConfig(model=tmp_path / "missing.onnx"))


@pytest.mark.parametrize("name", ["ml", "hybrid"])
def test_settings_reject_an_ml_detector_without_a_model(name: str) -> None:
    with pytest.raises(ValidationError, match="PANELS_ML_MODEL"):
        settings(detector=name)


def test_settings_accept_an_ml_detector_with_a_model(tmp_path: Path) -> None:
    assert settings(detector="hybrid", ml_model=weights(tmp_path)).detector == "hybrid"


def test_the_default_needs_no_model() -> None:
    s = settings()
    assert s.detector == "projection"
    assert s.ml_model is None
    assert ml_config(s) is None


def test_ml_config_carries_every_setting(tmp_path: Path) -> None:
    s = settings(
        detector="ml",
        ml_model=weights(tmp_path),
        ml_input_size=1024,
        ml_confidence=0.4,
        ml_nms_iou=0.5,
        ml_output_format="end2end",
        ml_panel_class=2,
    )
    config = ml_config(s)
    assert config == MlConfig(
        model=weights(tmp_path),
        input_size=1024,
        confidence=0.4,
        nms_iou=0.5,
        output_format="end2end",
        panel_class=2,
    )


def test_the_container_builds_a_hybrid_analyzer(tmp_path: Path) -> None:
    container = build_container(settings(detector="hybrid", ml_model=weights(tmp_path)))
    assert container.analyzer is not None  # built, but the weights are not loaded until used
