# Comic Panel Analyzer

API that analyzes a comic book page and returns its panels, reading order, coordinates and
a reliability score, so third-party apps can offer guided panel reading.

## Run

```sh
uv sync
cp .env.example .env            # set PANELS_API_KEYS
uv run uvicorn app.main:app_factory --factory --reload
```

Demo page (upload/paste a page and see numbered panel overlays): set
`PANELS_ENABLE_DEMO_PAGE=true` and open http://localhost:8000/demo. Images placed in
`samples/` (git-ignored, your own pages) and the committed `tests/fixtures/` pages show up in its
"Sample pages" dropdown, in separate groups.

Interactive docs: http://localhost:8000/docs · OpenAPI: `/openapi.json` · Health: `/healthz`

Or with Docker: `docker build -t panels . && docker run -p 8000:8000 -e PANELS_API_KEYS=change-me panels`

## API

All analyze endpoints require an `X-API-Key` header.

```sh
# URL
curl -X POST localhost:8000/v1/panels/analyze -H 'X-API-Key: change-me' \
  -H 'Content-Type: application/json' \
  -d '{"image": {"url": "https://example.com/page.png"}, "options": {"reading_direction": "rtl"}}'

# Base64 (or a data: URI)
curl -X POST localhost:8000/v1/panels/analyze -H 'X-API-Key: change-me' \
  -H 'Content-Type: application/json' \
  -d "{\"image\": {\"base64\": \"$(base64 -w0 page.png)\"}}"

# Upload
curl -X POST localhost:8000/v1/panels/analyze/upload -H 'X-API-Key: change-me' \
  -F file=@page.png -F reading_direction=ltr
```

Response:

```json
{
  "image": {"width": 800, "height": 1200},
  "reading_direction": "ltr",
  "panels": [
    {"id": 1, "order": 1,
     "bbox": {"x": 40, "y": 40, "width": 340, "height": 540},
     "bbox_normalized": {"x": 0.05, "y": 0.033, "width": 0.425, "height": 0.45},
     "confidence": 1.0}
  ],
  "sequence_confidence": 1.0,
  "method": "projection",
  "version": "projection@1+xycut@4",
  "whole_page_fallback": false,
  "warnings": []
}
```

`confidence` is per-panel detection quality; `sequence_confidence` is how unambiguous the
reading order is (1.0 = every panel separated by clean gutters).

### Processing options (per request)

Every setting that shapes the analysis has a server-wide default (an environment variable, see
`.env.example`) that a request can override, by `options` in the JSON body or by form fields of the
same name on the upload endpoint. A request that sets nothing gets the defaults.

```sh
curl -X POST localhost:8000/v1/panels/analyze -H 'X-API-Key: change-me' \
  -H 'Content-Type: application/json' \
  -d '{"image": {"url": "https://example.com/page.png"},
       "options": {"min_area_ratio": 0.005, "claim_captions": true}}'

curl -X POST localhost:8000/v1/panels/analyze/upload -H 'X-API-Key: change-me' \
  -F file=@page.png -F color_tolerance=60 -F refine_edges=true
```

`GET /v1/panels/options` lists every option with this server's default, range and a description:
`reading_direction`, `detector`, `refine_edges`, `refine_max_expand`, `claim_captions`,
`color_tolerance`, `gutter_content_ratio`, `min_gutter_ratio`, `min_area_ratio`,
`hybrid_min_panels`, `hybrid_max_aspect`, `hybrid_agree_iou`, `ml_confidence`, `ml_nms_iou`, `order_gap_tolerance`.
The demo page builds its "Processing options" controls from that list and sends only the values
you changed.

- An out-of-range value or an unknown option in the JSON body is a `422 validation_error`. On the
  upload endpoint, unknown form fields are ignored (the file itself is one).
- `detector` may be `ml` or `hybrid` only when the server has a model (`PANELS_ML_MODEL`); otherwise
  the request fails with `422 invalid_option`.
- Overrides that differ from the defaults add `+opts@<hash>` to the response `version`; the same
  overrides always give the same hash, and an override equal to the default adds nothing. So
  `version` still tells you when two results are comparable. The reading direction is not part of
  it, as before.
- The ML weights load once per server; a request that changes `ml_confidence` or `ml_nms_iou`
  reuses them.

### Client examples

Python (`pip install httpx`):

```python
import httpx

API = "http://localhost:8000"
HEADERS = {"X-API-Key": "change-me"}

# Upload a file
with open("page.png", "rb") as f:
    result = httpx.post(
        f"{API}/v1/panels/analyze/upload",
        headers=HEADERS,
        files={"file": ("page.png", f, "image/png")},
        data={"reading_direction": "ltr"},
        timeout=30,
    ).json()

# Or let the service download it
result = httpx.post(
    f"{API}/v1/panels/analyze",
    headers=HEADERS,
    json={"image": {"url": "https://example.com/page.png"}},
    timeout=30,
).json()

if result["whole_page_fallback"] or result["sequence_confidence"] < 0.6:
    print("not reliable: show the whole page")
for panel in result["panels"]:
    b = panel["bbox_normalized"]  # fractions of the page, independent of image size
    print(panel["order"], b["x"], b["y"], b["width"], b["height"])
```

JavaScript (browser or Node 18+):

```js
const API = "http://localhost:8000";

async function analyze(file, readingDirection = "ltr") {
  const form = new FormData();
  form.append("file", file);
  form.append("reading_direction", readingDirection);
  const response = await fetch(`${API}/v1/panels/analyze/upload`, {
    method: "POST",
    headers: { "X-API-Key": "change-me" },
    body: form,
  });
  const body = await response.json();
  if (!response.ok) throw new Error(`${body.code}: ${body.message}`);
  return body.panels; // ordered; use bbox_normalized to position a zoom/pan viewport
}
```

Do not ship the API key in a public web page; call the service from your own backend.

`version` identifies the detector and ordering algorithms that produced the result. Compare it
for equality: cache results by `(image, version)` and recompute when it changes.

When no panel is found the API still returns one panel covering the whole page, with
`confidence` and `sequence_confidence` of 0.1, `whole_page_fallback: true` and a warning, so
a reader can fall back to showing the page. A single detected panel is returned as is.

`method` names the detector that produced the result. Choose it with `PANELS_DETECTOR`:
`projection` (default; gutter-based, works on colour), `classical` (contour-based), or the optional
`ml` and `hybrid` detectors (see below). With `hybrid`, `method` is the one that answered that page.
`projection` (version 9) also follows a slightly slanted gutter when no straight row or column is
clean, and uses long straight frame lines to cut where there is no gutter at all and to set off
title banners, which are returned as a first panel. Caption strips and balloons that sit between panels are shared between the panel
above and the one below instead of being lost. On a dark page it rejoins a framed panel that its own
black art cut apart, and gives a panel that bleeds off the page the dark space between it and the page edge. Full-bleed art with framed insets over its edge
(frame lines on three sides of a rectangle that runs off the page) is returned as the art plus the insets. Panels with white borders and no paper between them are
cut along the frame line that runs down the white band separating them, and a framed caption box
cut off the top of its panel is joined back to it.
Known limitations: panels with no gutter between them (art that bleeds into the next
panel, open frames), L-shaped
panels, title banners and caption strips on pages whose frames are too broken to find (the ML detector
never returns titles),
borderless panels on unevenly toned paper (a page where nothing could be cut is retried against
the paper colour measured around each pixel, which handles stained paper but not every case), and dense pages where balloons cross
gutters (white balloons and caption areas read as gutters, which also splits some panels). Those pages return fewer, larger panels,
or extra ones, and a warning when only one is found. On light paper, ink between a panel and the page
edge is treated as lettering that belongs to no panel, so it is not added to the panel. `hybrid` handles `public_003`
(and the local `samples/003`). See `docs/evaluation.md` for per-page numbers and the committed pages that are
known failures (`tests/integration/test_fixture_pages.py`).

### Edge snapping (optional)

Detectors sometimes cut a panel edge through a balloon or caption. Set `PANELS_REFINE_EDGES=true`
to run a post-processing step after any detector: each edge that cuts through ink moves outward
until it reaches a clean gutter. Where balloons sit in the gap between two panels with no clean
gutter, each panel claims only the blobs that touch its own edge. It only ever grows a box, by at most 15% of the page side (`PANELS_REFINE_MAX_EXPAND`), and
leaves an edge alone when no gutter is found in reach, a neighbouring panel is in the way, or the
edge already lies inside another panel's box (a title that overlaps the first panel).
Panels with no frame of their own (a title above the first panel) are stretched to the margin box,
the bounding box of the framed panels, and set against their neighbours, so they begin and end
where the panels around them do.
It also follows a long slanted top or bottom frame line to its far end (within 4% of the page height),
so a panel with a slanted edge keeps its highest or lowest corner.
`version` then includes `edge-snap@<n>`. Score it with `python -m app.evaluation --refine-edges`.
Labels in the corpus are frame-tight, so it lowers strict (IoU 0.75) scores even when the grown
box is the more useful one (see docs/evaluation.md).

### Caption claiming (optional)

Detectors return the frame of a panel, but in western comics the narration box and the first
balloons often hang above it. Set `PANELS_CLAIM_CAPTIONS=true` and each panel's top edge moves up
over the lettering that touches it, so a reader zoomed to the box does not lose the text. Lettering
in the gap between two rows belongs to the row below it and is claimed once, and a top edge never
moves past the panel above, into rows that another panel's box covers (boxes of diagonal or
full-bleed layouts overlap), or more than 12% of the page height. Only top edges move. It combines
with `PANELS_REFINE_EDGES` (edge snapping runs first), and `version` then includes
`caption-claim@<n>`. Score it with `python -m app.evaluation --claim-captions`; see
docs/evaluation.md for the labels it is scored against.

### Optional ML detector

The `ml` and `hybrid` detectors run a YOLO-style ONNX panel model. **No weights are included**:
you supply a model you are allowed to use (its licence is yours to check, see
[ADR 0002](docs/adr/0002-optional-ml-detector.md)).

```sh
uv sync --extra ml
PANELS_DETECTOR=hybrid PANELS_ML_MODEL=/path/to/panels.onnx uv run uvicorn app.main:app_factory --factory

# Docker: onnxruntime is opt-in, weights are mounted, never baked in
docker build --build-arg WITH_ML=true -t panels-ml .
docker run -p 8000:8000 -e PANELS_API_KEYS=k -e PANELS_DETECTOR=hybrid \
  -e PANELS_ML_MODEL=/models/panels.onnx -v "$PWD/models:/models:ro" panels-ml
```

`hybrid` runs projection and the model on every page. The model's answer is used when projection
looks wrong (fewer than 3 panels, or a very thin panel), and also when the two agree on the layout
(same number of panels, each projection box overlapping its own model box by at least 60%, see
`hybrid_agree_iou`), because the model's boxes then sit on the frames more tightly. When they
disagree and projection looks plausible, projection's answer stands. The `version` field includes
the model's hash, so results from different weights are never mixed.

Errors share one shape: `{"code", "message", "details", "request_id"}` with statuses
400, 401, 413, 415, 422, 429 (with `Retry-After`), 502 and 504.

## Development

```sh
uv run pytest          # tests + coverage gate (90%)
uv run ruff check . && uv run ruff format --check .
uv run mypy            # strict
uv run python -m app.evaluation --verbose   # score the detectors, see docs/evaluation.md
uv run pre-commit install   # runs ruff (with --fix), ruff format and mypy before every commit
```

CI runs the same three checks, so a commit that skips the hook fails there. If `pre-commit install`
refuses with "Cowardly refusing to install hooks with `core.hooksPath` set" (a global hooks directory
is configured), install it with the override cleared for that one command; the global hooks hand off
to `.git/hooks`, so the repo hook still runs:

```sh
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.hooksPath GIT_CONFIG_VALUE_0= uv run pre-commit install
```

Architecture notes: [docs/adr](docs/adr). Configuration: see `.env.example`.
