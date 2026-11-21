import base64
from collections.abc import Iterator
from typing import Any

import pytest
from app.core.config import Settings
from app.main import create_app
from fastapi.testclient import TestClient

KEY = "test-key"
HEADERS = {"X-API-Key": KEY}
BASE_VERSION = "projection@8+xycut@4"


def make_client(**kwargs: Any) -> TestClient:
    settings = Settings(
        api_keys=KEY, rate_limit_per_minute=1000, log_json=False, _env_file=None, **kwargs
    )
    return TestClient(create_app(settings))


@pytest.fixture
def client() -> Iterator[TestClient]:
    with make_client() as c:
        yield c


def analyze(client: TestClient, page: bytes, **options: Any) -> dict[str, Any]:
    payload = {"image": {"base64": base64.b64encode(page).decode()}, "options": options}
    response = client.post("/v1/panels/analyze", json=payload, headers=HEADERS)
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def test_options_list_defaults_ranges_and_descriptions(client: TestClient) -> None:
    response = client.get("/v1/panels/options", headers=HEADERS)
    assert response.status_code == 200
    options = response.json()["options"]
    assert options["color_tolerance"]["default"] == 40.0
    assert options["color_tolerance"]["minimum"] == 5
    assert options["color_tolerance"]["maximum"] == 120
    assert options["detector"]["enum"] == ["projection", "classical", "ml", "hybrid"]
    assert options["refine_max_expand"]["exclusive_minimum"] == 0
    assert options["claim_captions"]["type"] == "boolean"
    assert options["hybrid_min_panels"]["type"] == "integer"
    assert all(o["description"] for o in options.values())


def test_options_need_an_api_key(client: TestClient) -> None:
    assert client.get("/v1/panels/options").status_code == 401


def test_the_listed_defaults_are_the_configured_ones() -> None:
    with make_client(color_tolerance=55.0, reading_direction="rtl") as c:
        options = c.get("/v1/panels/options", headers=HEADERS).json()["options"]
    assert options["color_tolerance"]["default"] == 55.0
    assert options["reading_direction"]["default"] == "rtl"


def test_an_override_changes_the_result_and_the_version(
    client: TestClient, page_2x2: bytes
) -> None:
    plain = analyze(client, page_2x2)
    assert len(plain["panels"]) == 4
    # Each panel is 19% of the page: with a 20% minimum no panel is big enough to be cut out.
    tuned = analyze(client, page_2x2, min_area_ratio=0.2)
    assert len(tuned["panels"]) == 1
    assert tuned["version"].startswith(BASE_VERSION + "+opts@")
    assert plain["version"] == BASE_VERSION


def test_the_same_overrides_give_the_same_version(client: TestClient, page_2x2: bytes) -> None:
    a = analyze(client, page_2x2, min_area_ratio=0.2, color_tolerance=50)
    b = analyze(client, page_2x2, color_tolerance=50, min_area_ratio=0.2)
    assert a["version"] == b["version"]


def test_an_override_equal_to_the_default_changes_nothing(
    client: TestClient, page_2x2: bytes
) -> None:
    assert analyze(client, page_2x2, min_area_ratio=0.02)["version"] == BASE_VERSION


def test_the_configured_default_applies_without_an_override(page_2x2: bytes) -> None:
    with make_client(min_area_ratio=0.2) as c:
        assert len(analyze(c, page_2x2)["panels"]) == 1
        assert len(analyze(c, page_2x2, min_area_ratio=0.02)["panels"]) == 4


def test_the_configured_reading_direction_is_the_default(page_2x2: bytes) -> None:
    with make_client(reading_direction="rtl") as c:
        assert analyze(c, page_2x2)["reading_direction"] == "rtl"
        assert analyze(c, page_2x2, reading_direction="ltr")["reading_direction"] == "ltr"


def test_reading_direction_does_not_change_the_version(client: TestClient, page_2x2: bytes) -> None:
    assert analyze(client, page_2x2, reading_direction="rtl")["version"] == BASE_VERSION


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("color_tolerance", 500),
        ("min_area_ratio", 0),
        ("hybrid_min_panels", 0),
        ("detector", "magic"),
        ("claim_captions", "maybe"),
        ("not_an_option", 1),
    ],
)
def test_invalid_options_are_rejected(
    client: TestClient, page_2x2: bytes, name: str, value: object
) -> None:
    payload = {"image": {"base64": base64.b64encode(page_2x2).decode()}, "options": {name: value}}
    response = client.post("/v1/panels/analyze", json=payload, headers=HEADERS)
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


@pytest.mark.parametrize("detector", ["ml", "hybrid"])
def test_an_ml_detector_without_a_model_is_refused(
    client: TestClient, page_2x2: bytes, detector: str
) -> None:
    payload = {
        "image": {"base64": base64.b64encode(page_2x2).decode()},
        "options": {"detector": detector},
    }
    response = client.post("/v1/panels/analyze", json=payload, headers=HEADERS)
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_option"


def test_the_classical_detector_can_be_chosen_per_request(
    client: TestClient, page_2x2: bytes
) -> None:
    assert analyze(client, page_2x2, detector="classical")["method"] == "classical"


def test_the_post_processing_steps_can_be_switched_on(client: TestClient, page_2x2: bytes) -> None:
    body = analyze(client, page_2x2, refine_edges=True, claim_captions=True)
    assert len(body["panels"]) == 4
    assert body["version"].startswith("projection@8+chain@edge-snap@")
    assert "+caption-claim@" in body["version"]


def test_upload_takes_the_options_as_form_fields(client: TestClient, page_2x2: bytes) -> None:
    response = client.post(
        "/v1/panels/analyze/upload",
        files={"file": ("page.png", page_2x2, "image/png")},
        data={"reading_direction": "rtl", "min_area_ratio": "0.2", "claim_captions": "true"},
        headers=HEADERS,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["reading_direction"] == "rtl"
    assert len(body["panels"]) == 1
    assert "caption-claim@" in body["version"]


def test_upload_rejects_an_out_of_range_form_field(client: TestClient, page_2x2: bytes) -> None:
    response = client.post(
        "/v1/panels/analyze/upload",
        files={"file": ("page.png", page_2x2, "image/png")},
        data={"color_tolerance": "500"},
        headers=HEADERS,
    )
    assert response.status_code == 422
