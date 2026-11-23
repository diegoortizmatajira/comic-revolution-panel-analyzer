import base64
import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from app.container import build_container
from app.core.config import Settings
from app.evaluation.labels import discover
from app.main import create_app
from fastapi.testclient import TestClient

from tests.conftest import make_page
from tests.integration.test_api import HEADERS, KEY, FakeFetcher

PANELS = [
    {"x": 0.05, "y": 0.03, "width": 0.43, "height": 0.45},
    {"x": 0.52, "y": 0.03, "width": 0.43, "height": 0.45},
]


def make_client(tmp_path: Path, page: bytes, **overrides: Any) -> TestClient:
    settings = Settings(
        api_keys=KEY,
        rate_limit_per_minute=1000,
        log_json=False,
        feedback_dir=tmp_path / "feedback",
        _env_file=None,
        **overrides,
    )
    container = replace(build_container(settings), fetcher=FakeFetcher(page))
    return TestClient(create_app(settings, container))


@pytest.fixture
def fb(tmp_path: Path, page_2x2: bytes) -> Iterator[TestClient]:
    with make_client(tmp_path, page_2x2) as c:
        yield c


def payload(page: bytes, reference: str = "book-1/page-3", **extra: Any) -> dict[str, Any]:
    return {
        "image": {"base64": base64.b64encode(page).decode()},
        "reference": reference,
        "reading_direction": "ltr",
        "panels": PANELS,
        **extra,
    }


def send(c: TestClient, page: bytes, reference: str) -> int:
    r = c.post("/v1/panels/feedback", json=payload(page, reference), headers=HEADERS)
    return int(r.status_code)


def folder(tmp_path: Path) -> Path:
    return tmp_path / "feedback"


def test_creates_a_sample_in_the_corpus_format(
    fb: TestClient, page_2x2: bytes, tmp_path: Path
) -> None:
    r = fb.post("/v1/panels/feedback", json=payload(page_2x2), headers=HEADERS)
    assert r.status_code == 201
    body = r.json()
    assert body["replaced"] is False
    assert body["panels"] == 2
    stem = body["sample_id"]
    assert (folder(tmp_path) / f"{stem}.png").read_bytes() == page_2x2
    labels = json.loads((folder(tmp_path) / f"{stem}.labels.json").read_text())
    assert labels["verified"] is True
    assert labels["tags"] == ["feedback"]
    assert labels["panels"][0]["x"] == 0.05

    [page] = discover([folder(tmp_path)])
    assert len(page.panels) == 2


def test_resending_is_idempotent_and_replacing_reports_it(
    fb: TestClient, page_2x2: bytes, tmp_path: Path
) -> None:
    assert send(fb, page_2x2, "book-1/page-3") == 201
    before = {p.name: p.read_bytes() for p in folder(tmp_path).iterdir()}

    again = fb.post("/v1/panels/feedback", json=payload(page_2x2), headers=HEADERS)
    assert again.status_code == 200
    assert again.json()["replaced"] is True
    assert again.json()["unchanged"] is True
    assert {p.name: p.read_bytes() for p in folder(tmp_path).iterdir()} == before

    changed = payload(page_2x2, panels=PANELS[:1])
    r = fb.post("/v1/panels/feedback", json=changed, headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["replaced"] is True
    assert r.json()["unchanged"] is False
    assert len(list(folder(tmp_path).glob("*.labels.json"))) == 1


def test_a_new_image_under_the_same_reference_replaces_the_old_one(
    fb: TestClient, page_2x2: bytes, tmp_path: Path
) -> None:
    fb.post("/v1/panels/feedback", json=payload(page_2x2), headers=HEADERS)
    other = make_page([(40, 40, 700, 500)], fmt="JPEG")
    r = fb.post("/v1/panels/feedback", json=payload(other), headers=HEADERS)
    assert r.status_code == 200
    images = [p for p in folder(tmp_path).iterdir() if p.suffix in {".png", ".jpg"}]
    assert [p.suffix for p in images] == [".jpg"]


def test_detected_is_recorded(fb: TestClient, page_2x2: bytes, tmp_path: Path) -> None:
    detected = {
        "version": "projection@9+xycut@4",
        "method": "projection",
        "panels": [{"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0, "order": 1}],
        "sequence_confidence": 0.5,
    }
    r = fb.post("/v1/panels/feedback", json=payload(page_2x2, detected=detected), headers=HEADERS)
    meta = json.loads((folder(tmp_path) / f"{r.json()['sample_id']}.feedback.json").read_text())
    assert meta["reference"] == "book-1/page-3"
    assert meta["detected"]["version"] == "projection@9+xycut@4"
    assert meta["image_sha256"] == r.json()["image_sha256"]


def test_upload_form(fb: TestClient, page_2x2: bytes) -> None:
    r = fb.post(
        "/v1/panels/feedback/upload",
        files={"file": ("p.png", page_2x2, "image/png")},
        data={"reference": "u-1", "reading_direction": "rtl", "panels": json.dumps(PANELS)},
        headers=HEADERS,
    )
    assert r.status_code == 201
    assert r.json()["panels"] == 2


def test_upload_form_with_bad_json_is_a_validation_error(fb: TestClient, page_2x2: bytes) -> None:
    r = fb.post(
        "/v1/panels/feedback/upload",
        files={"file": ("p.png", page_2x2, "image/png")},
        data={"reference": "u-1", "reading_direction": "ltr", "panels": "not json"},
        headers=HEADERS,
    )
    assert r.status_code == 422
    assert r.json()["code"] == "validation_error"


def test_url_source(fb: TestClient) -> None:
    body = payload(b"")
    body["image"] = {"url": "https://example.com/p.png"}
    assert fb.post("/v1/panels/feedback", json=body, headers=HEADERS).status_code == 201


@pytest.mark.parametrize(
    "panels",
    [
        [],
        [PANELS[0]] * 65,
        [{"x": 0.0, "y": 0.0, "width": 0.005, "height": 0.5}],  # under 1%
        [{"x": 0.0, "y": 0.0, "width": 0.5, "height": 0.005}],
        [{"x": 0.6, "y": 0.0, "width": 0.5, "height": 0.5}],  # outside the page
        [{"x": -0.1, "y": 0.0, "width": 0.5, "height": 0.5}],
        [{"x": 0.0, "y": 0.0, "width": 0.5}],  # missing height
        [{"x": 0.0, "y": 0.0, "width": 0.5, "height": 0.5, "extra": 1}],
        [{"x": "NaN", "y": 0.0, "width": 0.5, "height": 0.5}],
    ],
)
def test_bad_panels_are_a_422(fb: TestClient, page_2x2: bytes, panels: list[Any]) -> None:
    r = fb.post("/v1/panels/feedback", json=payload(page_2x2, panels=panels), headers=HEADERS)
    assert r.status_code == 422
    assert r.json()["code"] == "validation_error"


@pytest.mark.parametrize(
    ("field", "bad"),
    [("reference", ""), ("reference", "x" * 201), ("reading_direction", "diagonal")],
)
def test_bad_scalars_are_a_422(fb: TestClient, page_2x2: bytes, field: str, bad: str) -> None:
    r = fb.post("/v1/panels/feedback", json=payload(page_2x2, **{field: bad}), headers=HEADERS)
    assert r.status_code == 422


def test_invalid_image_is_rejected_and_nothing_is_stored(fb: TestClient, tmp_path: Path) -> None:
    r = fb.post("/v1/panels/feedback", json=payload(b"not an image"), headers=HEADERS)
    assert r.status_code == 400
    assert not folder(tmp_path).exists() or not list(folder(tmp_path).iterdir())


def test_image_size_limit(tmp_path: Path, page_2x2: bytes) -> None:
    with make_client(tmp_path, page_2x2, max_image_bytes=100) as c:
        r = c.post("/v1/panels/feedback", json=payload(page_2x2), headers=HEADERS)
    assert r.status_code == 413


def test_sample_count_limit(tmp_path: Path, page_2x2: bytes) -> None:
    with make_client(tmp_path, page_2x2, feedback_max_samples=2) as c:
        assert send(c, page_2x2, "a") in {200, 201}
        assert send(c, page_2x2, "b") in {200, 201}
        full = c.post("/v1/panels/feedback", json=payload(page_2x2, "c"), headers=HEADERS)
        assert full.status_code == 507
        assert full.json()["code"] == "feedback_storage_full"
        # replacing an existing sample needs no room
        assert send(c, page_2x2, "a") in {200, 201}


def test_total_size_limit(tmp_path: Path, page_2x2: bytes) -> None:
    with make_client(tmp_path, page_2x2, feedback_max_total_bytes=len(page_2x2) + 1500) as c:
        assert send(c, page_2x2, "a") in {200, 201}
        r = c.post("/v1/panels/feedback", json=payload(page_2x2, "b"), headers=HEADERS)
        assert r.status_code == 507


def test_default_folder_is_samples_feedback() -> None:
    assert Settings(_env_file=None).feedback_dir == Path("samples/feedback")


def test_empty_setting_turns_feedback_off(monkeypatch: pytest.MonkeyPatch, page_2x2: bytes) -> None:
    monkeypatch.setenv("PANELS_FEEDBACK_DIR", "")
    settings = Settings(api_keys=KEY, log_json=False, _env_file=None)
    assert settings.feedback_dir is None
    container = replace(build_container(settings), fetcher=FakeFetcher(page_2x2))
    client = TestClient(create_app(settings, container))
    r = client.post("/v1/panels/feedback", json=payload(page_2x2), headers=HEADERS)
    assert r.status_code == 501
    assert r.json()["code"] == "feedback_disabled"
    assert client.delete("/v1/panels/feedback/x", headers=HEADERS).status_code == 501


def test_requires_an_api_key(fb: TestClient, page_2x2: bytes) -> None:
    assert fb.post("/v1/panels/feedback", json=payload(page_2x2)).status_code == 401
    wrong = {"X-API-Key": "nope"}
    assert fb.post("/v1/panels/feedback", json=payload(page_2x2), headers=wrong).status_code == 401
    assert fb.delete("/v1/panels/feedback/x").status_code == 401


def test_delete(fb: TestClient, page_2x2: bytes, tmp_path: Path) -> None:
    fb.post("/v1/panels/feedback", json=payload(page_2x2, "book/1"), headers=HEADERS)
    assert fb.delete("/v1/panels/feedback/book/1", headers=HEADERS).status_code == 204
    assert list(folder(tmp_path).iterdir()) == []
    gone = fb.delete("/v1/panels/feedback/book/1", headers=HEADERS)
    assert gone.status_code == 404
    assert gone.json()["code"] == "feedback_not_found"
