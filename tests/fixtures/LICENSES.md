# Test fixture provenance

Real comic pages used to test panel detection. All three are labelled **Public domain** on
Wikimedia Commons (the licence is as stated there; this is not an independent legal opinion)
and originate from [Comic Book Plus](https://comicbookplus.com/), which hosts public-domain
comics. Files are the unmodified originals.

| File | Commons page | Comic Book Plus id | Why it is useful |
|---|---|---|---|
| `nature_boy_4.jpg` | https://commons.wikimedia.org/wiki/File:Pecos_Bill_in_Nature_Boy_4_page1.jpg | 18201 | aged paper, scan border, title banner, caption box overlapping a frame, narrow gutters |
| `western_fighters_6.jpg` | https://commons.wikimedia.org/wiki/File:Pecos_Bill_in_Western_Fighters_6_page1.jpg | 39853 | caption text sitting in the gutters between panels, title lettering |
| `western_fighters_8.jpg` | https://commons.wikimedia.org/wiki/File:Pecos_Bill_in_Western_Fighters_8_page1.jpg | 39966 | title banner, captions, circular inset panel |

The git-ignored `samples/` folder is for local, possibly copyrighted pages and is never committed.

## Pages added by the repository owner

`public_001.jpg` to `public_005.jpg` were added by the repository owner, who obtained them from
[Comic Book Plus](https://comicbookplus.com/), a site that hosts comics it treats as public domain,
and stated they are in the public domain. The files carry no EXIF data; their JPEG comment
(`gd-jpeg`) and 975 px width are consistent with that source.

**The specific title, issue and page URL of each file was not recorded.** Add them here (and the
basis for its public-domain status) so the provenance can be checked per page.

| File | What it is | Why it is useful |
|---|---|---|
| `public_001.jpg` | black-and-white newsprint public-service page | no colour, thin frames, title banner, 2-3-2 grid |
| `public_002.jpg` | colour action page | L-shaped panel wrapping an inset, irregular layout |
| `public_003.jpg` | colour action page, dense art | slanted gutters, captions on frame lines |
| `public_004.jpg` | colour page on aged paper | 3x2 grid, floating captions, partly unframed art |
| `public_005.jpg` | colour page on aged paper with a scan edge | 3x2 grid, floating captions |

## Midnight Mystery #4 (six pages)

`midnight-mistery-004-<page>.jpg` are pages 3, 4, 7, 8, 10 and 14 (the number in the file name is
the page) of *Midnight Mystery* no. 4 (July 1961, American Comics Group), story "A Slight Change
of Luck!", downloaded from Comic Book Plus: <https://comicbookplus.com/?dlid=13178>. They were
added by the repository owner.

**Verify before relying on this.** Page 3 carries a printed notice "© 1961 by Best Syndicated
Features Inc.". Works published in the US in 1961 with a notice are in the public domain only if
the copyright was not renewed 28 years later; Comic Book Plus lists the issue as public domain,
but renewal records have not been checked here. Check the Copyright Office renewal records (or the
Stanford Copyright Renewal Database) and note the result below before keeping these files in a
public repository.

Renewal check: _not done_.

| Page | What it is | Why it is useful |
|---|---|---|
| `-03` | large framed top panel with two panels below, printed credits | one big panel plus small ones, copyright text |
| `-04` | rows of 3, 2 and 2 panels | irregular row structure |
| `-07` | rows of 2, 2 and 3 panels | narrow gutters in the bottom row |
| `-08` | rows of 2, 3 and 2 panels | huge lettering inside a panel |
| `-10` | rows of 3, 2 and 2 panels | open frame edges on the middle-right panel |
| `-14` | three rows of two | caption boxes at panel tops |

## Moon Mullins no. 3 (nine pages)

`moon-mullings-003-<page>.jpg` are pages 1, 3, 5, 9, 10, 14, 15, 16 and 17 (the number in the file
name is the page; the file name spells the title "mullings") of the issue listed on Comic Book Plus as
*Moon Mullins* no. 3 (April 1948, English), <https://comicbookplus.com/?dlid=35122>. The listing
describes the scan as "incomplete" (`Moon_Mullins_003__1948___incomplete___comicrazys_.cbr`, uploaded
2013-07-19 by darkmark). The pages shown are Milt Gross's *That's My Pop!* strip. They were added by the
repository owner.

**Verify before relying on this.** I could not see a copyright notice on these nine pages. A 1948
US comic is in the public domain if it was published without a notice or if the copyright was not
renewed, and the licence of the underlying syndicated feature may differ from the comic book's.
Comic Book Plus lists the issue as public domain, but that has not been checked here. Check the
Copyright Office renewal records for the issue and the feature, and note the result below.

Renewal check: _not done_.

Useful because the frames are hand-drawn and crooked, many panels are open on one or more sides, and
balloons float over the gutters. Boxes on these pages are approximate (about 30 px).

Each page has a hand-made ground-truth file next to it (`<name>.labels.json`, format in
`docs/evaluation.md`) used by the evaluation harness.

## Blurred layout fixtures

`blurred-001.webp` to `blurred-003.webp` are heavily blurred copies of copyrighted pages from the repository owner's
local `samples/` folder, with the lettering removed. Only the panel layout survives (frames, white
borders, bleeding art), so it is not a reproduction of the page and can be committed.

| File | What it is | Why it is useful |
|---|---|---|
| `blurred-001.webp` | six panels, top-left art bleeding behind framed panels | staggered frames on the middle row, no gutter (#11) |
| `blurred-002.webp` | four panels under a curved black outline | slanted gutters, red-framed strip, dark page (projection gets it right) |
| `blurred-003.webp` | full-bleed art with two white-framed insets | bleed with insets over its bottom edge, no gutter; the page-sized-panel case (projection gets it right) |
