# ADR 0001: Architecture and detection approach

- Status: accepted
- Context: issue #1

## Decisions

- **FastAPI** for the HTTP layer: async, generated OpenAPI contract, Pydantic validation.
- **Hexagonal layout.** `app/domain` holds value objects and ports (Protocols) with no
  framework or I/O imports. `app/services` holds adapters and application services.
  `app/api` is a thin HTTP adapter. `app/container.py` is the only composition root.
- **Hybrid detection, behind `PanelDetector`.** v1 ships a classical OpenCV detector.
  An ML detector can be added as another implementation without touching callers
  (open/closed). Known limitation of the classical detector: panels that touch with no
  gutter merge into one.
- **Two classical detectors, selected with `PANELS_DETECTOR`.** `projection` (default)
  works on colour: it estimates the background from the dominant edge colour, marks
  non-background pixels as content, and recursively cuts along gutters found in row and
  column profiles. `classical` takes outer contours of content blobs on a grayscale
  image. Projection separated panels the contour detector merged on a dark, hand-framed
  page, but both fail where there is no gutter: touching panels, art that bleeds into
  the next panel and dense pages with balloons crossing gutters. Projection v2 adds a
  fallback for slanted gutters: when no clean row or column exists it cuts along the
  minimum-content path across the region, accepted only if it is nearly empty (up to 2%
  blocked pixels, and then only if nearly straight). That fixed the diagonal and
  slightly slanted pages. Projection v3 then falls back to frame lines (`frames.py`): Hough
  segments, merged when collinear, so a frame broken by balloons still reads as one line. A
  region is cut at a frame line only if the cut pays off, meaning it sets off a region with
  at most one frame side (title lettering, a caption strip) or lets a part split further. Such a
  strip is not thrown away: balloons and captions between panels are shared at 40% of the
  strip's height between the panel above and the one below. A large strip with no panel above
  it (a title banner) is returned as the first panel: lettering and a panel without a frame look
  the same to the detector, and dropping a real panel (samples/002's bleed art) costs more than
  showing a title. Labels for title pages include the banner for the same reason;
  otherwise lines inside artwork would cut real panels (a plain long-line cut broke
  `western_fighters_8`). This fixed `nature_boy_4`, and treating art that bleeds off three
  page borders as a panel fixed `samples/002`. Projection v4 adds a second attempt for a page
  where nothing could be cut: the paper colour is measured around each pixel instead of once at
  the page edge (paper reachable from the border without crossing an edge, smoothed over the
  page), which fixed `western_fighters_6`, where stained paper made almost the whole page read as
  content. It is only a second attempt because on pages that already work it fragments titles.
  The same retry also moves a panel's top or bottom edge out to a frame line that spans the
  panel just beyond it (pale edges leave a box short of its own frame and cut the first lines of
  a balloon); applied to every page that snap loosened boxes that were already right.
  A panel there also grows up to the bottom of the panel above it across a small gap, since
  what sits in it (a row's caption lines) hangs from the panel below.
  The edge refiner finally normalizes panels with no frame of their own: the margin box (the
  bounding box of the framed panels, `margin.py`) gives a title above the first panel the same
  left and right edges as the panels below and ends it where they begin. Only wide, short
  panels are touched, and only when at least two panels are framed; applied to every unframed
  panel it stretched ordinary panels on pages where frame detection fails and cost 15 points
  of page accuracy with refinement on.
  Pages whose frames are too broken for the lines to be found keep their banners and caption
  strips. Projection v6 targets pages with a dark background, where black art reads as paper.
  A framed strip whose art is split by black bands was cut into pieces; pieces side by side are
  now rejoined when the union has long frame lines along its top and bottom and on at least one
  end, and no full-height frame line crosses the gap between them (`FrameLines.encloses_as_one`).
  A gap under 4% of the page width stays a gutter, and slanted pieces whose boxes overlap are only
  joined to one already joined across a wide gap, otherwise two framed panels with a slanted
  gutter would merge. Also on dark pages only, unclaimed space between an unframed panel and the
  page edge is absorbed by that panel if it holds ink, is at least 4% of the page across, and the
  other panels leave 85% of its extent clear. Applied to every page, that step took lettering
  and caption zones that the labels deliberately keep outside the boxes and cost 6 points of page
  accuracy, so it is not a general rule: on light paper the trim is reliable. Both steps are
  limited to what the corpus shows: a perfectly straight, full-height edge on a piece inside a
  frame still counts as a divider and blocks the rejoin.
  Projection v7 handles full-bleed art with framed insets over its edge (`blurred-003`, and its
  original, which is not committed). Nothing can be cut on the blurred copy (the art is the page),
  and the paper-colour retry only fragmented it into five pieces; on the original the first pass cuts
  the art at the inset's top frame line and the inset in two. `FrameLines.inset_frames` finds
  rectangles with frame lines on at least three sides whose other sides run off the page (the
  best-framed, smallest first, so a double frame is one inset). A single page-sized panel
  becomes the insets plus the largest band of page they leave free; on a cut page only an inset
  that was cut into two or more pieces is put back together, and a slice of the art cut off beside
  it rejoins the art. Insets drawn on other frame lines added extra panels on `009` and
  `blurred-001` when every inset was used, hence that rule.
  What still fails, tracked in issue 11 (strict `xfail` in `tests/integration/test_fixture_pages.py`):
  staggered frames over art that bleeds behind them (`blurred-001`); an L-shaped panel (`public_002`); slanted gutters
  crossed by captions and art (`public_003`, `moon-mullings-003-09`); open frames with no gutter
  (`moon-mullings-003-03`, `-16`); a narrow gutter holding 22% ink (`midnight-mistery-004-07`);
  and captions or large lettering splitting a panel (`midnight-mistery-004-08`, `-14`). Several
  share one cause: white balloons and caption areas look like gutters, so a valley cut that would
  find the narrow gutters also splits other panels. Tried and not kept: cutting at any frame line
  spanning 90% of the region (broke pages that were correct, because captions and art also carry
  full-width lines), vertical frame-line cuts, a valley cut on the column profile, counting paper
  enclosed by outlines as content, and letting a box claim ink across a neighbour's wall. These
  cases are what the ML detector is for. A blurred copy of a page keeps its frame geometry but
  not its ink statistics, so the blurred fixtures do not cover the dark-page cases; the synthetic
  tests in `tests/unit/test_projection_detector.py` do.
- **Reading order:** recursive XY-cut honouring `ltr`/`rtl`. `sequence_confidence`
  measures layout ambiguity only (share of panels that could not be cut cleanly);
  per-panel `confidence` measures detection quality.
- **Synchronous processing.** CPU-bound work runs in a thread pool. Async jobs are a
  possible follow-up.
- **Security.** API keys (compared by SHA-256 digest in constant time), in-memory
  sliding-window rate limit (single process; replace with a shared store to scale out),
  SSRF validation on every redirect hop. Each hop resolves the hostname once, rejects the
  request if any answer is non-public, then connects to that validated IP (Host header and
  TLS server name preserved), so a second, different DNS answer cannot redirect the request
  (DNS rebinding). Embedded URL credentials and private IP literals are rejected.
