# ADR 0002: Optional ML detector, bring your own weights

- Status: accepted
- Context: issue #4; depends on the evaluation harness (#2)

## Decision

- The service can run a YOLO-style **ONNX** panel model through `onnxruntime`, alone (`ml`) or as a
  fallback to the classical detector (`hybrid`). It is selected with `PANELS_DETECTOR`.
- **No weights ship with this project.** The deployer supplies a model file
  (`PANELS_ML_MODEL=/path/to/model.onnx`) and is responsible for its licence. The Docker image
  never contains weights: mount them. `onnxruntime` is an optional extra (`uv sync --extra ml`,
  `docker build --build-arg WITH_ML=true`), so the default install is unchanged.
- The model's SHA-256 (first 12 hex digits) is part of the response `version`, so results from
  different weights never share a cache entry.
- `PanelDetector.detect` now returns `Detection(panels, method)`, so a composite detector reports
  which component actually answered each page.

## Why not ship a model

No candidate has a clean licence chain (checked against the model cards on 2026-11-19; nothing
was verified beyond what the cards say):

| Candidate | Weights licence | Training data | Notes |
|---|---|---|---|
| [Inkwell panel detector](https://huggingface.co/cedarrapidsboy/inkwell-panel-models) | AGPL-3.0, including network use | ~3,800 annotated pages; origin and licence not stated | YOLOv8-nano, 640 px, ships `best.onnx`, reported mAP50 0.977 |
| [mosesb YOLOv12x](https://huggingface.co/mosesb/best-comic-panel-detection) | card says Apache-2.0 | custom Roboflow set; licence not stated | extra-large model; the Ultralytics code behind it is AGPL, so the claim is doubtful |
| [Oliverdsfdsf YOLO26n-seg](https://huggingface.co/Oliverdsfdsf/comic-panels-text-detect) | card says Apache-2.0 | not stated | targets comics, manga and manhwa; no metrics |
| [kernald/panel-detection](https://git.enoent.fr/kernald/panel-detection/releases) `model-v0.1` | repo MIT; weights' licence not stated | not stated; its benchmark names Pepper&Carrot, Mini Fantasy Theater, Golden Age, Krazy Kat, Little Nemo | D-FINE-L, 1024 px, `detector.onnx` (119 MiB) plus a reading-order `orderer.onnx` we do not need |
| [DeepPanel](https://github.com/smather/DeepPanel) | code Apache-2.0, weights unlicensed | private, 550 pages | TensorFlow U-Net |

This repository is MIT. Shipping or requiring an AGPL model would change what hosting the service
obliges you to publish. Training our own needs a licensable annotated dataset we do not have (the
corpus has 34 pages). So the project provides the mechanism and leaves the choice of weights, and
its licence, to the deployer.

## Measurements

All numbers are from the evaluation harness (`docs/evaluation.md`) on the 34-page corpus (23 real
pages, 11 generated), using the Inkwell `best.onnx` (`1:f240e1296efd`) **for local measurement
only: it was not committed or distributed**. "Pages exactly right" is the harness `page_acc`.

| Detector | Real pages exactly right | Real F1@50 | Real F1@75 | Generated pages exactly right |
|---|---|---|---|---|
| `projection` | 15 of 23 | 0.90 | 0.75 | 8 of 11 |
| `ml` | 16 of 23 | 0.96 | 0.79 | 7 of 11 |
| `hybrid` | 19 of 23 | 0.95 | 0.88 | 8 of 11 |

Re-measured on 2026-11-20 with `projection@5` and the same weights. The first measurement
(`projection` 11 of 23, `ml` 19 of 23, `hybrid` 18 of 23) predates the projection work on slanted
gutters, frame lines, title banners and frameless panels, which lifted `projection` by four pages.
`ml` went from 19 to 16 of 23 with the same weights; the cause was not investigated (the ordering
code, the labels and the corpus changed in the meantime), so the two tables are not directly
comparable.

- **`hybrid` is now the best option on real pages** (19 of 23, against 16 for `ml` and 15 for
  `projection`). It keeps the pages `projection` already gets right and lets the model rescue the
  rest; `ml` alone has no such safety net and loses pages `projection` gets right. Over the whole
  34-page corpus `hybrid` is exactly right on 0.79 of pages, `ml` and `projection` on 0.68 each.
  This reverses the earlier conclusion that `ml` alone beat `hybrid`.
- The earlier claims that hybrid never does worse than `projection` on any page, and that the model
  ran on 9 of 23 real pages, were not re-checked after the projection changes.
- Reading order matters as much as detection. With the old strict XY-cut the same model got only
  10 of 23 real pages exactly right even though it found nearly every panel, because its boxes
  overlap by a few pixels and no clean cut exists. The XY-cut now has a **gap tolerance** of 8% of
  the group's extent (ordering version 2). Against ML boxes the result rises to 19 of 23 and plateaus
  from 8% to 12%; `projection` is unaffected.

### A second model: D-FINE (`dfine` output format)

Measured on 2026-11-21 with `projection@9` on the 37-page corpus (26 labelled real pages, 11
generated, 182 panels), with the `kernald/panel-detection` `model-v0.1` `detector.onnx`
(`1:34b4e048931a`), **for local measurement only: it was not committed or distributed**. It runs with
`PANELS_ML_OUTPUT_FORMAT=dfine`, `PANELS_ML_INPUT_SIZE=1024`; the page is stretched to the square
(letterboxing gave duplicate, shifted boxes) and `orig_target_sizes` is `[width, height]`. Inkwell
was re-measured on the same corpus and code (`1:f240e1296efd`, threshold 0.25; D-FINE 0.5).

| Detector | Inkwell f1@50 | Inkwell f1@75 | D-FINE f1@50 | D-FINE f1@75 |
|---|---|---|---|---|
| `projection` | 0.94 | 0.85 | 0.94 | 0.85 |
| `ml` | 0.96 | 0.83 | 0.87 | 0.79 |
| `hybrid` | 0.96 | 0.90 | 0.96 | 0.93 |

Recall@50 of `ml` alone is 0.95 for Inkwell and 0.81 for D-FINE (worst on `borderless` pages and
`balloons-outside-frame`); `page_acc` of `hybrid` is 0.78 for both, the same as `projection`.

- **Alone, Inkwell is clearly better**; D-FINE misses more panels but its boxes are tighter when it
  finds one. Thresholds from 0.2 to 0.6 did not change that (`ml` f1@50 0.84 to 0.88).
- **As the `hybrid` partner, D-FINE is slightly ahead** on f1@75 (0.93 against 0.90), because
  `hybrid` only uses the model where `projection` looks wrong or the two agree. Three points on 37
  pages is not decisive.
- The same contamination and licence caveats as Inkwell apply, and the D-FINE weights' licence is
  not stated at all.

### Hybrid policy

`projection` is distrusted, and the model asked, when its result has **fewer than 3 panels** or a
panel whose long side is **at least 5 times its short side** (a caption strip). A thin panel that
lies wholly above every other panel is the title banner `projection` returns on purpose as a first
panel, so it does not count: the model never returns titles and would only drop it.
On the original measurement these rules flagged 9 of the 12 pages `projection` got wrong and none of the 11 it
gets right; the model's answer replaces `projection`'s when it finds at least one panel.

**Agreement (policy version 3).** The model now runs on every page, and when `projection` looks
plausible but the model finds the **same number of panels** (at least 3) and each `projection` box
overlaps its own, distinct model box by **IoU 0.6 or more**, the model's boxes are used. The
detectors agree on the layout, and the model's boxes sit on the frames, which `projection` loses
where white balloons, captions or halftone read as gutter (`public_001`: row 2 started 140 px below
its frame). On the real pages where they agree, both detectors were exactly right on all 7, so
agreement is a safe signal. Result on the 23 real pages: f1@75 0.85 to 0.88, exactly-right pages
unchanged at 19; four pages gain (`public_001` 4 to 8 of 8 panels at IoU 0.75,
`midnight-mistery-004-10` 6 to 7, `moon-mullings-003-05` 2 to 4) and `public_005` falls from 6 to 3
at IoU 0.75, because the model's boxes include the captions that its frame-tight labels exclude.
It can be switched off per request with `hybrid_agree_iou=1`.

When the two disagree and `projection` looks plausible, `projection` still wins, and that is a
coin flip: on the 16 real pages where they disagree each is exactly right on 8 and 9. A tie-break
(for example trusting the model when it finds more panels, which would fix the three pages
`projection` under-segments) was left out: it rests on about three pages and was found after
seeing the results.

**Extra panels (policy version 4).** Tried on 15 feedback pages (one modern colour issue, full-bleed,
inset and overlapping panels) added to the 32 real pages, with the Inkwell model. On the feedback
pages `projection` looks plausible and is wrong (`hybrid` v3: f1@50 0.67, 0 of 15 pages exactly right).
Rule: when the model finds **at least one more panel** than a plausible `projection`, `projection`
probably merged panels, so the model's answer is used (`extra_panels`, 0 turns it off).

| Policy | 32 earlier real pages f1@50 / f1@75 / page_acc | 15 feedback pages | all 47 |
|---|---|---|---|
| v3 | 0.96 / 0.88 / 0.81 | 0.67 / 0.49 / 0.00 | 0.88 / 0.77 / 0.55 |
| extra >= 1 (default) | 0.97 / 0.87 / 0.81 | 0.82 / 0.70 / 0.20 | 0.92 / 0.82 / 0.62 |
| extra >= 2 | 0.97 / 0.86 / 0.78 | 0.81 / 0.66 / 0.13 | 0.92 / 0.80 / 0.57 |
| extra >= 3 | 0.97 / 0.88 / 0.81 | 0.79 / 0.63 / 0.07 | 0.92 / 0.80 / 0.57 |
| model first (`prefer_model`, off) | 0.95 / 0.79 / 0.59 | 0.85 / 0.74 / 0.27 | 0.92 / 0.78 / 0.49 |

Model-first is better on the feedback pages but costs the earlier pages 0.09 f1@75 and 0.22
page_acc, so it stays an off-by-default knob. `extra >= 1` was chosen on these same 15 pages, which
come from one issue, so it is a prototype: confirm it on pages from other publishers. Not done: a
gutter-fraction signal to distrust `projection` on full-bleed pages.

**Bleed fill (policy version 5).** What the model misses on the feedback pages is mostly large
borderless or full-bleed panels (8 of the 11 missed panels: 26 to 98% of the page, many with insets
over them); a lower threshold does not find them (recall 0.85 to 0.88 for precision 0.86 to 0.71),
and `projection` already has a matching box for 6 of the 11. Rule: when the model's answer is used,
a `projection` panel of at least 10% of the page that the model's boxes cover at most 50% of is added
(`fill_max_covered`, `fill_min_area`; 0 turns it off). A 100x100 coverage grid is used.

| Policy | 32 earlier real pages f1@50 / f1@75 / page_acc | 15 feedback pages | all 47 |
|---|---|---|---|
| v4 | 0.97 / 0.87 / 0.81 | 0.82 / 0.70 / 0.20 | 0.92 / 0.82 / 0.62 |
| v5, covered <= 0.5 (default) | 0.97 / 0.87 / 0.81 | 0.84 / 0.70 / 0.27 | 0.93 / 0.82 / 0.64 |
| covered <= 0.8 | 0.97 / 0.87 / 0.81 | 0.85 / 0.71 / 0.27 | 0.93 / 0.82 / 0.64 |

The gain is small (one more page exactly right) and the earlier pages do not move. The result is flat
from 0.5 to 0.8, so the conservative end was taken. It was found on the same 15 pages from one issue.

The thresholds (3 panels, 5:1, IoU 0.6, 8% tolerance) were chosen on this small corpus and are
untuned beyond it. IoU 0.5 also takes the model's boxes on `moon-mullings-003-14` and loses panels
there, and 0.7 changes nothing, so the choice is fragile. Revisit them when the corpus grows.

### Cost (this machine, 24 cores; pinning to one core gave the same figures)

| | |
|---|---|
| Model file | 12.3 MB (mounted, not in the image) |
| Inference | median 15 ms per ~975x1400 page (`projection`: 37 ms) |
| Cold start | 0.27 s in process (import `onnxruntime`, hash the file, load, first inference); container ready 0.6 s |
| Docker image | 429 MB without the extra, 513 MB with `WITH_ML=true` (+84 MB) |

## Caveats

- **Possible test-set contamination.** The Inkwell training pages are undocumented. If they
  include golden-age scans like ours, its numbers here are optimistic.
- **AGPL.** Measuring the Inkwell weights locally is not distribution. Deploying them behind a
  service others use is a different matter; read its licence and decide before doing so.
- **Small corpus.** 23 real pages, all western comics, 15 of them from two issues.
- The model must be YOLO-style single-class (or multi-class with `PANELS_ML_PANEL_CLASS`) and
  take a square input; `PANELS_ML_OUTPUT_FORMAT=end2end` selects the `(N, 6)` NMS-free layout, and
  `dfine` the D-FINE layout (`images` + `orig_target_sizes` in, `labels`/`boxes`/`scores` out, page
  stretched to the square). Other architectures need another decoder.
