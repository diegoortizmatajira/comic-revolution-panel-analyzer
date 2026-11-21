# Detector evaluation

Panel detectors are scored against hand-labelled pages, so a change can be judged by numbers
instead of by eye: did it fix the page it targets, and did it break any page that worked?

```sh
uv run python -m app.evaluation --verbose                       # everything available
uv run python -m app.evaluation --corpus tests/fixtures         # committed pages + synthetic only
uv run python -m app.evaluation --detector projection --json out.json
```

By default the corpus is `tests/fixtures` (committed, public-domain), `samples/` (local,
git-ignored, your own pages) and a set of generated pages. Every registered detector is scored
through the same analyzer and reading-order code the API uses.

## What is measured

| Metric | Meaning |
|---|---|
| `prec@50`, `rec@50`, `f1@50` | Panels matched one-to-one at IoU >= 0.5, micro-averaged over all panels |
| `f1@75` | The same at IoU >= 0.75 (tighter boxes) |
| `order` | Share of matched panel pairs that are read in the ground-truth order |
| `page_acc` | Share of pages that are **exactly** right: every panel found, nothing extra, same order |
| `conf_auc` | Chance that a correct page gets a higher `sequence_confidence` than a wrong one. 0.5 means the score carries no information, 1.0 means it separates right from wrong perfectly |

IoU is computed on normalized boxes, which gives the same value as on pixels. Results are also
broken down by tag (`dark`, `borderless`, `spread`, `rtl`, ...) so a weakness shows up as a row
instead of being averaged away.

### Reading the numbers

- `page_acc` is the strictest and the most useful for readers: one missing or merged panel makes
  the page wrong.
- `conf_auc` is currently about 0.5: `sequence_confidence` only measures how cleanly the layout
  could be cut, not whether detection was right, so it does not predict a wrong page. Treat it
  as "layout ambiguity", not "probability of being correct", until that changes.
- The whole-page fallback counts as one predicted panel, so it is correct on a real splash page.

## Scoring an ML model

`ml` and `hybrid` are scored when you give a model (it is never part of the committed baseline):

```sh
uv run python -m app.evaluation --corpus tests/fixtures --ml-model /path/to/panels.onnx --verbose
```

Keep the weights outside the repository (`/models/` is git-ignored). Results for the model tried so
far are in [ADR 0002](adr/0002-optional-ml-detector.md).

## Scoring the edge refiner

`--refine-edges` applies the edge-snapping refiner after each detector. The labels in the corpus are
frame-tight: balloons and captions that hang outside a panel's frame are not part of the label.
The refiner's job is the opposite, to keep those balloons inside the box, so on open-frame pages it
moves boxes away from the labels and strict scores drop. With `projection@6` on the committed
corpus (37 pages: 26 labelled, 11 synthetic): f1@50 0.898 either way, f1@75 0.773 to 0.744;
`classical` is unchanged (f1@50 0.502, f1@75 0.411). Several pages (for example `moon-mullings-003-10`,
`-14` and `-17`, and `western_fighters_8`) grow by 140 to 180 px to include balloons above the frame;
`midnight-mistery-004-14`, whose caption boxes are already split off as extra panels, grows by 470 px.

IoU cannot reward "no content is cropped", so judge the refiner by looking at the boxes, not by
f1@75. A truth-coverage metric (share of each labelled box covered) would be the fair number.

## Scoring caption claiming

`--claim-captions` runs the caption claimer after every detector (it moves each panel's top edge up
over the lettering that hangs above the frame, see README). The labels are frame-tight, so pages
whose lettering sits above the frame carry a second list, `caption_panels`: the same panels in the
same order with that lettering included. With the flag a page is scored against `caption_panels`
when it has one and against `panels` otherwise; without the flag every page is scored against
`panels`, so the committed baseline is unchanged.

Nine pages carry `caption_panels` (the `captions` tag). They were made by running the claimer on the
hand-made frame labels and then checking every box against the page by eye. Real pages, `--claim-captions`
against without it (f1@75, with the model of ADR 0002):

| detector | `captions` pages | all 23 real pages |
|---|---|---|
| `projection` | 0.81 to 0.93 | 0.75 to 0.76 |
| `ml` | 0.88 to 0.97 | 0.79 to 0.80 |
| `hybrid` | 0.90 to 0.95 | 0.88 to 0.84 |

The `captions` rows now agree with what a reader sees. Across all real pages `hybrid` loses a little
(0.88 to 0.84): the pages without a `caption_panels` list are still scored frame-tight, and the
claimer also grows boxes on them wherever lettering touches a top edge. Exactly-right pages do not
move (`page_acc` is 0.65, 0.65 and 0.83 on real pages, as before, for `projection`, `ml` and `hybrid`)
except `ml`, which goes from 0.70 to 0.65 (the cause was not investigated).

## Label format

A labelled page is an image plus a sibling `<name>.labels.json`:

```json
{
  "version": 1,
  "reading_direction": "ltr",
  "verified": true,
  "tags": ["borderless"],
  "notes": "free text",
  "panels": [
    {"x": 0.05, "y": 0.03, "width": 0.90, "height": 0.30},
    {"x": 0.05, "y": 0.36, "width": 0.43, "height": 0.30}
  ]
}
```

- `panels` are boxes **normalized to the image** (0-1, origin top-left) **in reading order** for
  `reading_direction` (`ltr` or `rtl`).
- A panel is its **frame**. Page numbers, credit lines and captions outside the frame are not
  panels; a caption box inside the frame is part of it. A **title banner** (large lettering above
  the first frame) is labelled as the first panel and tagged `title-banner`: the detector cannot
  tell lettering from a panel that has no frame, and losing a panel costs more than showing a
  title. Where a caption straddles a frame line,
  use the frame line and say so in `notes`. Labels are good to a few pixels, which IoU 0.5
  absorbs.
- Lettering above a frame (narration boxes, the first balloons) is outside the panel here. If you
  also want it counted, add `caption_panels` (same panels, same order, with the lettering above
  each frame included; only the top edge differs).
- Spreads: label every panel of both pages, in reading order across the whole spread.
- Open frames: when a panel has no frame line on a side, the box is the extent of its artwork, and
  balloons or captions floating outside that artwork are left out. Boxes on such pages are only
  accurate to a few tens of pixels; say so in `notes`.
- Unknown keys and boxes outside the page are rejected when the file is loaded.
- `verified: false` marks a draft; the harness ignores drafts unless `--include-unverified`.

Suggested tags: `clean`, `dark`, `noisy`, `aged-paper`, `borderless`, `bleed`, `splash`,
`spread`, `rtl`, `diagonal`, `no-gutter`, `narrow-gutter`, `title-banner`, `captions`,
`circular-inset`, `dense-art`. Generated pages also carry `synthetic`.

## Adding a page

1. Put the image in `tests/fixtures/` (public domain only, with its provenance in
   `tests/fixtures/LICENSES.md`) or in `samples/` (local only; the folder's `.gitignore` keeps
   both the images and their labels out of git).
2. Create a draft: `uv run python -m app.evaluation --propose tests/fixtures/page.jpg`.
   The draft comes from a detector, so **it is biased towards what the detector already does**.
3. Correct every box by looking at the page, set the reading direction and tags, and change
   `"verified"` to `true`.
4. Re-run the evaluation. If the page is in `tests/fixtures`, refresh the baseline (below).

Drafts are only a starting point: never accept one without checking it.

## Baseline and CI

`docs/evaluation/baseline.json` holds the results on the committed corpus (`tests/fixtures`
plus the generated pages). CI runs

```sh
uv run python -m app.evaluation --corpus tests/fixtures --baseline docs/evaluation/baseline.json
```

and fails when `f1@50`, `f1@75`, `order` or `page_acc` of any detector drops by more than
`--max-drop` (default 0.02). Improvements pass. If the corpus itself changed, the comparison is
refused until the baseline is refreshed:

```sh
uv run python -m app.evaluation --corpus tests/fixtures --write-baseline docs/evaluation/baseline.json
```

Commit the refreshed baseline together with the change that caused it. Local `samples/` are not
part of the baseline because they differ between machines.

## Baseline numbers (committed corpus: 23 real pages + 11 generated)

Read the **real** rows. Generated pages are cleaner than scans and flatter the averages.

| detector | pages | f1@50 | f1@75 | order | page_acc | conf_auc |
|---|---|---|---|---|---|---|
| projection, real | 23 | 0.90 | 0.75 | 1.00 | 0.65 | 0.50 |
| projection, synthetic | 11 | 0.99 | 0.96 | 0.96 | 0.73 | 0.44 |
| classical, real | 23 | 0.34 | 0.26 | 1.00 | 0.00 | - |
| classical, synthetic | 11 | 0.94 | 0.86 | 0.96 | 0.55 | 0.52 |

On real pages `projection` gets 15 of 23 pages exactly right and `classical` none (`conf_auc` is
undefined for `classical` because no real page is correct). With an ML model
([ADR 0002](adr/0002-optional-ml-detector.md)), `hybrid` gets 19 of 23 (f1@75 0.88), `ml` alone 16. `projection`'s
`sequence_confidence` does not separate right from wrong pages (AUC 0.50).

Known weaknesses visible in the per-tag rows: pages with no gutter, double-page spreads (panels
of both pages are ordered together, see #5), diagonal gutters for `projection`, and on real
scans: caption text or title lettering in the gutters, thin frames on newsprint, L-shaped panels,
slanted gutters, caption boxes split off from their panel, narrow gutters between panels, and
open frames with balloons overlapping the gutters. Each real page that fails is a strict xfail in
`tests/integration/test_fixture_pages.py`.

## Current limits of the corpus

The corpus is small: 23 labelled real pages and 11 generated pages in the repository (plus
whatever you label locally in `samples/`). Treat the numbers as a regression guard, not as an
accuracy estimate.

- No real manga (RTL) page, double-page spread, full-page splash or dense modern page is
  committed yet; RTL, spreads and splashes are covered only by generated stand-ins.
- All real pages are golden-age or silver-age western comics, which is what public-domain archives
  mostly hold. Fifteen of them come from two issues, so they share a house style.
- Labels were made by one person looking at overlays. For two pages (`public_004`, `public_005`)
  the first draft came from the detector and was then checked by eye, so those labels may lean
  towards what the detector does; the `midnight-mistery-*` and `moon-mullings-*` labels were made
  fully by hand before the detector was run on them.
- Provenance of some pages is incomplete or unverified, see `tests/fixtures/LICENSES.md`.
