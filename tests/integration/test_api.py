import base64
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from app.container import build_container
from app.core.config import Settings
from app.main import create_app
from app.services.analyzer import PanelAnalyzer
from app.services.detection.hybrid import HybridPanelDetector
from app.services.detection.onnx import OnnxPanelDetector
from app.services.detection.projection import ProjectionPanelDetector
from app.services.ordering.xy_cut import XYCutOrdering
from fastapi.testclient import TestClient

from tests.conftest import make_page

KEY = "test-key"
HEADERS = {"X-API-Key": KEY}


class FakeFetcher:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.urls: list[str] = []

    async def fetch(self, url: str) -> bytes:
        self.urls.append(url)
        return self.data


@pytest.fixture
def settings() -> Settings:
    return Settings(api_keys=KEY, rate_limit_per_minute=1000, log_json=False, _env_file=None)


@pytest.fixture
def client(settings: Settings, page_2x2: bytes) -> Iterator[TestClient]:
    container = replace(build_container(settings), fetcher=FakeFetcher(page_2x2))
    with TestClient(create_app(settings, container)) as c:
        yield c


def test_health(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}


def assert_four_panels(body: dict[str, Any], direction: str = "ltr") -> None:
    assert body["image"] == {"width": 800, "height": 1200}
    assert body["reading_direction"] == direction
    assert [p["order"] for p in body["panels"]] == [1, 2, 3, 4]
    assert body["sequence_confidence"] == 1.0
    assert body["method"] == "projection"
    assert body["version"] == "projection@9+xycut@5"
    assert body["whole_page_fallback"] is False
    first = body["panels"][0]
    assert first["bbox"] == {"x": 40, "y": 40, "width": 340, "height": 540}
    assert first["bbox_normalized"]["x"] == pytest.approx(0.05)


def test_analyze_base64(client: TestClient, page_2x2: bytes) -> None:
    payload = {"image": {"base64": base64.b64encode(page_2x2).decode()}}
    response = client.post("/v1/panels/analyze", json=payload, headers=HEADERS)
    assert response.status_code == 200
    assert_four_panels(response.json())
    assert response.headers["x-request-id"]


def test_analyze_url_rtl(client: TestClient) -> None:
    payload = {
        "image": {"url": "https://example.com/p.png"},
        "options": {"reading_direction": "rtl"},
    }
    response = client.post("/v1/panels/analyze", json=payload, headers=HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["panels"][0]["bbox"]["x"] == 420  # right panel first in manga order


def test_analyze_upload(client: TestClient, page_2x2: bytes) -> None:
    response = client.post(
        "/v1/panels/analyze/upload",
        files={"file": ("page.png", page_2x2, "image/png")},
        data={"reading_direction": "ltr"},
        headers=HEADERS,
    )
    assert response.status_code == 200
    assert_four_panels(response.json())


def test_upload_too_large(client: TestClient, settings: Settings) -> None:
    big = b"x" * (settings.max_image_bytes + 1)
    response = client.post(
        "/v1/panels/analyze/upload", files={"file": ("a.png", big)}, headers=HEADERS
    )
    assert response.status_code == 413
    assert response.json()["code"] == "image_too_large"


def test_requires_api_key(client: TestClient) -> None:
    for headers in ({}, {"X-API-Key": "nope"}):
        response = client.post(
            "/v1/panels/analyze", json={"image": {"url": "https://a/b"}}, headers=headers
        )
        assert response.status_code == 401
        assert response.json()["code"] == "unauthorized"


def test_validation_error_is_structured(client: TestClient) -> None:
    response = client.post("/v1/panels/analyze", json={"image": {}}, headers=HEADERS)
    body = response.json()
    assert response.status_code == 422
    assert body["code"] == "validation_error"
    assert body["request_id"]


def test_both_sources_rejected(client: TestClient) -> None:
    payload = {"image": {"url": "https://a/b", "base64": "AAAA"}}
    assert client.post("/v1/panels/analyze", json=payload, headers=HEADERS).status_code == 422


def test_invalid_image_error_shape(client: TestClient) -> None:
    payload = {"image": {"base64": base64.b64encode(b"junk").decode()}}
    response = client.post("/v1/panels/analyze", json=payload, headers=HEADERS)
    assert response.status_code == 400
    assert set(response.json()) == {"code", "message", "details", "request_id"}


def test_unsupported_format(client: TestClient) -> None:
    gif = make_page([], size=(10, 10), fmt="GIF")
    response = client.post(
        "/v1/panels/analyze",
        json={"image": {"base64": base64.b64encode(gif).decode()}},
        headers=HEADERS,
    )
    assert response.status_code == 415


def test_rate_limit_sets_retry_after(page_2x2: bytes) -> None:
    settings = Settings(api_keys=KEY, rate_limit_per_minute=1, log_json=False, _env_file=None)
    payload = {"image": {"base64": base64.b64encode(page_2x2).decode()}}
    with TestClient(create_app(settings)) as c:
        assert c.post("/v1/panels/analyze", json=payload, headers=HEADERS).status_code == 200
        limited = c.post("/v1/panels/analyze", json=payload, headers=HEADERS)
    assert limited.status_code == 429
    assert int(limited.headers["retry-after"]) >= 1


def test_demo_page_disabled_by_default(client: TestClient) -> None:
    assert client.get("/demo").status_code == 404


def test_demo_page_served_when_enabled() -> None:
    settings = Settings(api_keys=KEY, enable_demo_page=True, log_json=False, _env_file=None)
    with TestClient(create_app(settings)) as c:
        response = c.get("/demo")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "/v1/panels/analyze/upload" in response.text


def demo_settings(samples: Path, fixtures: Path, feedback: Path | None = None) -> Settings:
    return Settings(
        feedback_dir=feedback,
        api_keys=KEY,
        enable_demo_page=True,
        samples_dir=samples,
        fixtures_dir=fixtures,
        log_json=False,
        _env_file=None,
    )


def test_demo_lists_and_serves_samples_and_fixtures(tmp_path: Path, page_2x2: bytes) -> None:
    samples, fixtures = tmp_path / "samples", tmp_path / "fixtures"
    samples.mkdir()
    fixtures.mkdir()
    (samples / "a.png").write_bytes(page_2x2)
    (fixtures / "a.png").write_bytes(b"fixture-bytes")  # same name, different source
    (fixtures / "b.jpg").write_bytes(b"jpg")
    (samples / "notes.txt").write_text("ignore me")
    (samples / ".gitignore").write_text("*")
    with TestClient(create_app(demo_settings(samples, fixtures))) as c:
        assert c.get("/demo/samples").json() == [
            {"source": "samples", "name": "a.png"},
            {"source": "fixtures", "name": "a.png"},
            {"source": "fixtures", "name": "b.jpg"},
        ]
        assert c.get("/demo/samples/samples/a.png").content == page_2x2
        assert c.get("/demo/samples/fixtures/a.png").content == b"fixture-bytes"
        assert c.get("/demo/samples/samples/notes.txt").status_code == 404
        assert c.get("/demo/samples/nope/a.png").status_code == 404
        assert c.get("/demo/samples/samples/..%2Fsecret.png").status_code == 404


def test_demo_lists_and_serves_feedback_samples(tmp_path: Path, page_2x2: bytes) -> None:
    feedback = tmp_path / "feedback"
    feedback.mkdir()
    (feedback / "abc.png").write_bytes(page_2x2)
    (feedback / "abc.labels.json").write_text("{}")
    settings = demo_settings(tmp_path / "nope", tmp_path / "nada", feedback)
    with TestClient(create_app(settings)) as c:
        assert c.get("/demo/samples").json() == [{"source": "feedback", "name": "abc.png"}]
        assert c.get("/demo/samples/feedback/abc.png").content == page_2x2
        assert c.get("/demo/samples/feedback/abc.labels.json").status_code == 404


def test_demo_samples_missing_folders_are_empty(tmp_path: Path) -> None:
    settings = demo_settings(tmp_path / "nope", tmp_path / "nada")
    with TestClient(create_app(settings)) as c:
        assert c.get("/demo/samples").json() == []


def test_detector_is_selectable_by_setting(page_2x2: bytes) -> None:
    settings = Settings(api_keys=KEY, detector="classical", log_json=False, _env_file=None)
    with TestClient(create_app(settings)) as c:
        response = c.post(
            "/v1/panels/analyze/upload", files={"file": ("p.png", page_2x2)}, headers=HEADERS
        )
    assert response.json()["method"] == "classical"


def test_blank_page_returns_whole_page_fallback(client: TestClient) -> None:
    blank = make_page([], size=(300, 400))
    response = client.post(
        "/v1/panels/analyze/upload", files={"file": ("blank.png", blank)}, headers=HEADERS
    )
    body = response.json()
    assert response.status_code == 200
    assert body["whole_page_fallback"] is True
    assert len(body["panels"]) == 1
    assert body["panels"][0]["bbox"] == {"x": 0, "y": 0, "width": 300, "height": 400}
    assert body["panels"][0]["bbox_normalized"]["width"] == 1.0
    assert body["sequence_confidence"] == 0.1


def _fake_ml_output(
    boxes: list[tuple[int, int, int, int]], page: tuple[int, int], size: int
) -> Any:
    """A YOLO-format output that decodes to the given page boxes (inverse of the letterbox)."""
    width, height = page
    scale = min(size / height, size / width)
    pad_x, pad_y = (size - round(width * scale)) // 2, (size - round(height * scale)) // 2
    cols = []
    for x, y, w, h in boxes:
        cols.append(
            ((x + w / 2) * scale + pad_x, (y + h / 2) * scale + pad_y, w * scale, h * scale, 0.9)
        )
    cols += [(0.0, 0.0, 0.0, 0.0, 0.0)] * (16 - len(cols))
    return np.array(cols, dtype=np.float32).T[None]


def test_hybrid_falls_back_to_the_ml_detector_and_reports_it(
    tmp_path: Path, page_2x2: bytes
) -> None:
    # One merged panel from the classical detector is suspicious, so the model is asked.
    merged = make_page([(40, 40, 720, 1120)])
    four = [(40, 40, 340, 540), (420, 40, 340, 540), (40, 620, 340, 540), (420, 620, 340, 540)]
    weights = tmp_path / "m.onnx"
    weights.write_bytes(b"w")

    class Session:
        def get_inputs(self) -> list[Any]:
            return [type("I", (), {"name": "images"})()]

        def run(self, names: object, feed: dict[str, object]) -> list[Any]:
            return [_fake_ml_output(four, (800, 1200), 64)]

    ml = OnnxPanelDetector(weights, input_size=64, session_factory=lambda _: Session())
    settings = Settings(api_keys=KEY, log_json=False, _env_file=None)
    container = replace(
        build_container(settings),
        analyzer=PanelAnalyzer(HybridPanelDetector(ProjectionPanelDetector(), ml), XYCutOrdering()),
    )
    with TestClient(create_app(settings, container)) as c:
        body = c.post(
            "/v1/panels/analyze/upload", files={"file": ("p.png", merged)}, headers=HEADERS
        ).json()
    assert body["method"] == "ml"
    assert body["version"].startswith("hybrid@6[projection@9|ml@1:")
    assert body["version"].endswith("+xycut@5")
    assert [p["order"] for p in body["panels"]] == [1, 2, 3, 4]
    assert abs(body["panels"][0]["bbox"]["x"] - 40) <= 20
