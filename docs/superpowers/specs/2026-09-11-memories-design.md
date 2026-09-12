# M2 — the memory engine (design)

*Date: 2026-09-11. Supersedes the recipe sketch in
[writing-recipes.md](../../writing-recipes.md), which described a protocol that
never existed.*

A **memory** is a short slideshow built from photos that belong together. This
document specifies how rekindle chooses those photos, how it refuses to choose
the wrong ones, and how it turns a choice into a GIF or an MP4.

Everything here is **deterministic**: no model, no network, no randomness, no
wall-clock dependence except where the user asks for "today". The same index in
produces the same bytes out. The GPT caption layer (§12) is strictly additive —
delete it and v1 is still complete, and so is the optional embedding signal of
§5C: it reads vectors already on disk, adds no randomness, and its absence
changes the result but never breaks it.

---

## 1. What was measured first

Every number in this document was measured against the real index
(`data/rekindle.sqlite`, 19,480 rows from a 61 GB Takeout export) at design
time, not carried in from the brief. Where the brief disagreed, the measurement
won and the disagreement is recorded in §14.

```
19,480 photos          18,363 images · 1,117 videos
19,318 live            162 archived · 0 trashed
19,480 with a capture date (100%)
24 distinct years      2000-2026 (the brief's table began at 2008 and omitted
                       13 photos in 2000, 2003, 2005, 2006 and 2007)
10,887 with people (55.9%)      8,433 live photos have NO face tag (43.7%)
 2,330 with GPS (12.0%)         2,318 live, in 19 cells at 0.25 degrees
    40 distinct people          66 albums (43 named + 23 auto `Photos from YYYY`)
     3 favourites               142 descriptions
```

Two measurements changed the design:

**Videos carry no dimensions.** All 1,117 video rows have `width`/`height`
NULL, because `read_exif` never opened them. Anything that ranks on resolution
must tolerate NULL, and the fingerprint pass (§4) cannot decode them at all.

**A 30-second window is not a burst.** 59.2% of consecutive live images are
within 30s of the previous one. Chaining on time alone groups 14,425 of 18,201
images into multi-photo runs, the largest containing **80 photos spanning 40
minutes**. Collapsing those to one photo each would destroy the library. Time is
only the *gate*; perceptual near-identity is the *decision* (§5).

---

## 2. Architecture

```
        PhotoStore (SQLite, schema v4)
                  |
                  v
        +--------------------+
        |    MemoryIndex     |  <-- THE CHOKEPOINT. Guardrails applied once,
        | (already filtered) |      here, before any recipe sees a row.
        +--------------------+
                  |
     offers()     |     select()
                  v
        +--------------------+
        |      Recipe        |  8 of them. Pure functions over the index.
        +--------------------+
                  |
                  v
        +--------------------+
        |      Engine        |  dedup -> rank -> cap -> order -> caption
        +--------------------+
                  |
                  v
             MemorySpec (JSON, deterministic)
                  |
         +--------+--------+
         v                 v
       GIF               MP4  (only when ffmpeg is on PATH)
```

Module layout under `src/rekindle/memory/`:

| Module | Responsibility |
|---|---|
| `policy.py` | `ExclusionPolicy`, the public-safe rule, TOML config load |
| `index.py` | `MemoryIndex` — the chokepoint |
| `fingerprint.py` | dHash, sharpness, brightness and post-rotation size, one decode per photo |
| `dedup.py` | burst collapse |
| `strata.py` | stratified selection - the spread across a memory's own dimension (§6.2a) |
| `diversity.py` | content dissimilarity, and the seam a semantic signal plugs into (§6.2b) |
| `composition.py` | orientation cohesion, canvas, resolution/aspect/quality gates (§5A) |
| `history.py` | dismissal and resurfacing state (§5B) |
| `spec.py` | `MemorySpec`, `Shot`, `FactSheet` |
| `recipes/` | the eight recipes plus the registry |
| `engine.py` | the pipeline above |
| `captions.py` | deterministic titles and captions |
| `render/` | `frames.py`, `gif.py`, `mp4.py`, `music.py` |
| `watch.py` | the foreground watcher |
| `llm.py` | the additive GPT caption layer (§12) |

---

## 3. Guardrails: one chokepoint, structurally enforced

> *"If a recipe can bypass the guardrail, the guardrail does not exist."*

The guardrail is not a filter that each recipe remembers to apply. It is the
**only way to obtain a `Photo` at all**.

```python
class MemoryIndex:
    @classmethod
    def open(cls, store, policy): ...
    # every query method reads self._photos, which the constructor already filtered
```

`MemoryIndex.open()` loads every row, applies the policy once, and stores the
survivors in a private tuple. **No method on `MemoryIndex` can return a photo
that the policy rejected**, because no rejected photo is ever stored on the
object. A recipe is handed a `MemoryIndex` and nothing else — it has no
`PhotoStore`, no path, no connection. Bypassing the guardrail requires importing
`PhotoStore` yourself, which is exactly the reviewable act we want it to be.

> **Amended after this spec was written.** The private tuple is gone: the
> index holds SQLite rowids and loads `Photo` objects on demand, because at
> 300,000 rows the tuple is 681 MB. The chokepoint, the interface and the
> guarantee are unchanged — what makes the guarantee true is now that exactly
> one function turns a row into a `Photo` and it applies `deny_reason`, rather
> than that no rejected photo is stored. See
> [the scale decision log](../../decision-log-memory-scale.md).

Filtering is done in `policy.deny_reason(photo)`, a single predicate, in this
order:

1. **`archived` or `trashed` → denied. Not configurable.** 162 rows. There is no
   flag to turn this off; the brief called these must-not-surface and a
   user-settable escape hatch would be a foot-gun, not a feature.
2. **`media_type is UNKNOWN` → denied.** Nothing can render it.
3. **No capture date → denied.** Every recipe is temporal. (0 rows today, but
   the predicate must not assume that.)
4. **Person exclusions** — denied if *any* of the photo's people are excluded.
   Deliberately "any", not "all": a photo containing an excluded person is
   excluded even if five welcome people are also in it.
5. **Date-range exclusions** — denied if `taken_at_local` falls inside any
   excluded closed interval. Local, not UTC: a user excluding a date means the
   date they lived, not the UTC instant.
6. **Album exclusions** — denied if any album matches (casefolded).
7. **Path-prefix exclusions** — denied if any of the photo's paths sits under an
   excluded directory.
8. **Public-safe mode**, when enabled — §3.1.

Order matters only for the report: `MemoryIndex.open` returns an
`ExclusionReport` counting denials *by first-matching reason*, so
`rekindle memories` can say what it withheld and why without naming the photos.

### 3.1 The public-safe rule

The user publishes some memories to a public GitHub repo. The rule they
specified: **only photos with no other person in them may be public.**

```python
def is_public_safe(photo, allow):
    people = {p for p in photo.meta.people if p}
    return bool(people) and people <= allow
```

Both halves are load-bearing, and **default deny** is the point:

- **non-empty** — an untagged photo is *not* public-safe. Face tags come from
  Google and cover 55.9% of the library; **8,433 live photos (43.7%) carry no
  tag at all** and any of them may contain anyone. Treating "no tags" as "no
  people" would publish strangers' faces. This is the single most dangerous
  widening available in this codebase and it is refused.
- **subset of the allow-list** — not "intersects". A photo of Abhik *and*
  Paramita is not publishable under an allow-list of `{"Abhik Maiti"}`.

Measured on the real library with `allow = {"Abhik Maiti"}`: **959 of 19,318
live photos (4.96%)** are public-safe — 912 images and 47 videos, spread over
2009-2026. That is the honest ceiling. It is small on purpose.

Public-safe is evaluated **per photo** and lifted to the memory:
`MemorySpec.public_safe` is true only when *every* shot in it is public-safe.
`--public-safe` additionally runs the index in public-safe mode, so
non-qualifying photos are never even candidates.

### 3.2 Configuration

`data/exclusions.toml`, read with stdlib `tomllib`, absent by default:

```toml
people = ["Someone You Would Rather Not See"]
albums = ["Hospital"]
paths  = ["D:/photos/2019-difficult-year"]
public_safe_allow = ["Abhik Maiti"]

[[dates]]
from = "2019-03-01"
to   = "2019-09-30"
```

Written by hand or by `rekindle exclude --person NAME` / `--from D --to D`,
which appends and never rewrites what it did not add. A malformed file is a
**hard error, not a warning**: silently continuing with an unparsed exclusion
list is precisely the failure this whole section exists to prevent.

---

## 4. Fingerprints: schema v3 and v4

Dedup needs a perceptual hash per photo. Decoding 18,363 images is the
expensive part, so it is computed once and stored.

### 4.1 The hash

**64-bit dHash** (row-wise gradient over a 9x8 grayscale), chosen over aHash by
measurement. Distances between 414 real photos in real 30-second runs:

| Threshold T | within-burst pairs at ≤T | random unrelated pairs at ≤T |
|---|---|---|
| 4 | 15.99% | 0.000% |
| **6** | **18.37%** | **0.033%** |
| 8 | 22.79% | 0.067% |
| 12 | 29.25% | 0.133% |

aHash at comparable recall is 4-5x less specific (T=12: 42.18% within but 1.37%
random), so dHash it is. **Default threshold 6**: one false collapse per ~3,000
unrelated comparisons, and only temporally adjacent photos are ever compared.
The bias is deliberate — under-collapsing leaves a near-duplicate in a montage,
over-collapsing destroys a distinct memory, and only one of those is
recoverable.

Note the median within-burst distance is **22**. Most photos taken seconds apart
are genuinely different photos. This is the measurement that makes the two-stage
design mandatory.

### 4.2 Sharpness

> **Revised after M2 shipped.** The original design measured a mean absolute
> horizontal gradient of a **128x128** grayscale copy. That measure was
> adequate for its stated job — ranking frames of one burst from one camera —
> and useless for the job the user actually wanted, which is keeping mildly
> blurred photos out of memories. Measured: it scored a sharp photo above a
> mildly blurred copy of itself **51.9%** of the time. A coin flip. The rest
> of this section describes what replaced it.

The image is reduced to an **aspect-preserving** copy 1024 pixels on its long
edge (never enlarged), cut into tiles of roughly 128 pixels, and each tile is
**reblurred by one pixel**. The tile's score is the *fraction* of its absolute
gradient energy that the reblur destroys; the image's score is its **sharpest
tile**. The result is in [0, 1].

Three properties, each measured rather than argued:

| | old (128x128 mean gradient) | new (tiled reblur ratio) |
|---|---|---|
| P(sharp > mildly blurred), 123 real photos | 0.519 | **0.903** |
| P(sharp > grossly blurred) | 0.704 | **1.000** |
| P(sharp > mild), 44 photos hand-graded at 100% | 0.682 | **0.793** |
| P(sharp > gross), same 44 | 0.600 | **0.800** |
| per-year p5 spread, all 18,363 photos | 3.84x | **1.75x** |
| shallow-DoF retention (synthetic) | — | **0.800** |

**Resolution.** 128 was not a tuning choice that came out low; it was the
wrong *kind* of choice. Downscaling a 4000px photo to 128px is itself a
low-pass filter, and it removes precisely the fine detail that mild blur
removes. There was no signal left to measure. 1024 is where the measured
separation stops improving materially against what it costs.

**A ratio, not a magnitude.** This is the load-bearing decision, and it is
what makes a *single* threshold safe across a library spanning 2000-2026. A
gradient magnitude cannot tell a low-contrast photo from a blurred one, so the
old measure's per-year 5th percentile spanned 3.84x across the whole library
and the gate had to cower in the tail to avoid gutting the early years.
Dividing by the tile's own gradient energy cancels scene contrast, and the
spread falls to 1.75x. On a matched sample a plain gradient at the same 1024
resolution still spans 4.44x, so the resolution is not what fixed this — the
normalisation is.

**The sharpest tile, not the mean.** A portrait with a sharp face against a
deliberately blurred background is often the best photo in the set and scores
badly on any whole-image measure. Face boxes are M3's, so this cannot be
solved here; taking the maximum over tiles is the available mitigation. On a
synthetic shallow-DoF version of each of 123 real photos, the median retention
of the fully-sharp score is **0.800** taking the max tile, 0.330 at the 90th
percentile and 0.006 at the 75th. The max is doing nearly all the work; the
tiling alone would not have helped.

**What it is blind to, and why the checkerboard test failed.** The absolute
gradient summed across an isolated step edge is unchanged when the edge is
spread over three pixels instead of one. A picture made only of hard edges is
therefore nearly invisible to this measure — a 40px checkerboard scores 0.043
sharp and 0.009 after a radius-4 blur. What the measure reads is the loss of
fine *texture*, which is the right thing for a photograph and is exactly why
it is contrast-invariant, but a genuinely flat-and-hard-edged subject (a sign,
a document) scores low whether or not it is in focus. The same weakness makes
it unreliable on heavily compressed sub-megapixel files, whose JPEG blocking is
all hard edges — and those are **53% of 2011** against 7.5% of the library,
which is where the threshold below had to be settled by looking rather than by
arithmetic. `is_screenshot` removes the common case; the gate sitting *below*
the 1st percentile absorbs the rest.

It remains a **relative** measure. A grossly blurred photo of a high-contrast
scene can still outscore a sharp photo of a soft one — measured, it happens —
so it is not a quality score and nothing may use it as one.

### 4.3 Cost and storage

One decode yields all five measurements, using `Image.draft()` to let libjpeg
do DCT-scaled decoding. Measured on 123 real files:

| draft | ms/photo | median long edge |
|---|---|---|
| `("L", (64, 64))` — what M2 shipped | 13.6 | 486 |
| `("RGB", (64, 64))` | 14.5 | 486 |
| `("RGB", (1024, 1024))` — now | 24.6 | 1984 |

Both arguments changed. The **size**, because the sharpness measure cannot see
mild blur in a 486px copy of a 4000px photo. The **mode**, because `"L"` tells
libjpeg to decode the luma plane only — which meant `colour_signature` was
running on a grey image and every stored "colour histogram" was a luminance
histogram. Measured over 2,000 real rows: a median of **42 of the 64 bins were
exactly zero** and 55% of the mass sat on the four grey bins. Nothing failed;
the diversity signal simply carried a fraction of what it claimed to.

End to end the pass now measures **59 ms/photo** against 18.6 — about **18
minutes** for the reference library against 5.6. It is a one-time cost and it
stays resumable; without `draft()` at all it would be far worse.

Schema v3 and v4 add nullable columns to `photos`, via the existing additive
`ALTER TABLE ADD COLUMN` pattern:

```
phash        INTEGER   -- 64-bit dHash, NULL when never computed or undecodable
sharpness    REAL      -- relative focus measure
phash_error  TEXT      -- why it is NULL, so a retry does not redo known failures
brightness   REAL      -- mean luminance, for the near-black/blown-out gates
colour       TEXT      -- v4: hex 4x4x4 RGB histogram, for diversity (§6.2b)
```

`iter_unfingerprinted` re-offers a row that HAS a hash but lacks a later
measurement, which is what let the v4 histogram reach an index already
fingerprinted under v3 without a full re-index.

`_migrate` becomes a **ladder** (`{1: _to_v2, 2: _to_v3}`) applied in sequence
while the stored version is a known step. It keeps v2's property that a database
at an *unknown* version is left untouched for `__init__`'s explicit error, and
it fixes the latent bug v2's own comment predicted: the old `!= 1` guard would
have skipped a v1 database straight past v3.

`rekindle fingerprint` runs the pass over an existing index, resumable (skips
rows that already have a `phash` or a recorded `phash_error`) and interruptible.
`rekindle index` does **not** run it: indexing is IO-bound and fast,
fingerprinting is CPU-bound and slow, and welding them would make a re-index
cost six minutes for nothing.

Videos are never fingerprinted (`phash_error = "video"`) and therefore never
deduped. Extracting a frame needs ffmpeg, which must stay optional.

---

## 5. Dedup: burst-only, two-stage

```
sort live images by (taken_at_utc, file_hash)
for each photo:
    joins the current burst  <=>  gap to PREVIOUS photo <= 30s
                             AND  dHash distance to the burst's ANCHOR <= 6
    otherwise: starts a new burst, and becomes its anchor
```

- **Time gates, pixels decide.** Both conditions, always. §1 shows why.
- **Anchored, not chained, on pixels.** Comparing each photo to the burst's
  first member prevents a drift chain where A≈B≈C but A is nothing like C.
  Comparing to the previous photo would allow exactly that.
- **Chained on time.** A 30s gap to the previous photo, not to the anchor, so a
  genuine 12-frame burst is one burst. A long run of *visually different*
  photos still splits, because the pixel test splits it.
- A photo with no `phash` (video, undecodable, or not yet fingerprinted)
  **never joins a burst and never absorbs one**. Unknown is not similar.
- **When an embedding store exists**, the pixel test is *or*-ed with a cosine
  test against the same anchor — a dHash calls two framings of one moment
  unrelated, and it did, five seconds apart. See §5C for the threshold and the
  six pairs it was read off.

**Winner:** highest `sharpness`, then largest `width*height` (NULL sorts last),
then **lexicographically smallest `file_hash`**. The final tiebreak is the file
content itself, so it is stable across filesystems, orderings and re-runs — a
path- or mtime-based tiebreak is not.

Dedup runs in the engine *after* the recipe selects and *before* the cap, so the
cap is filled with 24 distinct photos rather than 24 slots of which 6 are
duplicates.

---

## 5A. Composition guardrails: what can be shown side by side

Selection answers *which photos belong together*. This answers *which of those
can actually be watched together* — and it is the same promise, because a
memory that mixes a portrait phone photo with a 9.41:1 panorama and a 101x24
barcode looks broken however well chosen its contents were.

**Every rule rejects, every rejection is counted with a reason, and the count is
surfaced.** A guardrail that drops photos silently is the defect, not the
feature: a user who expected a photo and did not get it must be able to find out
why. `CompositionReport` keeps one example filename per reason so the CLI can
say *which kind* of photo a rule is catching.

### 5A.1 The orientation trap, and a genuine M0 bug

A camera stores a rotated portrait as **landscape pixels plus an orientation
tag**. `Image.size` is the raw stored size — Pillow does not apply the tag — and
M0's `read_exif` recorded exactly that.

Measured: **13.6% of a 456-file sample carry a 90/270° tag, so roughly 2,475 of
18,201 live images were indexed with width and height swapped.** This is a real
M0 defect, fixed at source in `meta/exif.py` rather than worked around in the
memory layer. It stayed latent through two milestones because `w*h` is invariant
under the swap, so even the dedup tiebreak that reads those columns could not
notice; only a rule that asks "is this taller than it is wide?" exposes it.

Only orientations **5-8** exchange the axes. 2, 3 and 4 are mirrors and 180°
rotations, so the tempting `orientation != 1` test would corrupt photos that are
currently correct. Both readings are pinned by tests.

Existing indexes keep the wrong values until re-indexed, so **the fingerprint
pass repairs them in place** — it decodes every image anyway, and the
alternative was telling every existing user to re-index from scratch.
Orientation is applied exactly once, at measurement time; nothing downstream may
re-apply it.

### 5A.2 Orientation cohesion — majority wins

Classification: **square when the long edge is within 1.05x the short one**,
otherwise portrait or landscape. Measured justification: 178 live images are
*exactly* 1:1 and 210 are within 1.02, while only 21 more fall in the 1.02-1.05
band. There is a natural cliff at 1.02 and the band above it is nearly empty, so
the exact tolerance barely matters; 1.05 forgives a crop that is a pixel off
without swallowing a real 4:3 (1.33).

**A mixed set keeps the majority orientation and drops the minority**, rather
than splitting into one memory per orientation. Two reasons: several recipes are
not splittable (`then_and_now` is exactly two shots; `person_years` walks one
person through time and halving it destroys the narrative), and splitting
doubles the memory count while halving each one, working against the "return
fewer photos than you think" rule the engine follows elsewhere.

- **Ties break towards landscape.** 73% of this library is landscape and a
  montage is watched in a landscape frame. It is a choice, and it is
  deterministic.
- **Square is never the minority.** It letterboxes acceptably into either
  canvas, so it is a compatible minority rather than a competing majority.
- **Orientation is decided *after* the per-photo gates.** Otherwise a pile of
  rejected portrait thumbnails outvotes the real landscape photos and empties
  the memory.

### 5A.3 Canvas: the median of the set, and padding below it

**The first rule here was the minimum, and it was catastrophic.** Measured
across 45 rendered memories:

```
11 of 45 memories rendered at 640x480
 1 rendered at 1105x510
25 distinct canvases, most of them tiny

person_years-paramita   canvas 640x480
  source photos         min 640x480 (ONE photo, from 2014)
                        median 3984 wide
                        max 7008x4672 (two photos, from 2024)
```

One 640x480 photo from 2014 pinned the entire memory to 640x480, rendering two
7008x4672 photos at **a 120th of their pixel count**. `album_story-kashmir`
landed on 1280x960 and `person_years-paramita` on 640x480 purely because of
which single weakest photo happened to be selected.

The intent — never blow a small photo up into mush — was right. The lever was
wrong: on a library spanning 2000-2026, the oldest phone photo in the set
dictated the resolution of everything.

**The rule now:**

| Photo vs canvas | What happens |
|---|---|
| larger | downscale to fit, preserving aspect |
| within **1.25x** below | upscale to fit — imperceptible |
| further below | **rendered at NATIVE size and padded** |
| — | **nothing is ever excluded for being small** |

The canvas is the **median** width and the **median** height of the selected
set, taken independently so a set mixing 4:3 and 16:9 gets a box both can sit
in. The *lower* median, so it is always a real size at least half the set can
meet or exceed.

The 1.25x tolerance exists because without it a photo 3% below the canvas would
be padded, which reads as an inconsistency rather than as a deliberate signal.
A 25% linear stretch is about 1.6x the pixel count and is imperceptible at
viewing size.

#### Why sub-canvas photos are padded, not excluded

Excluding them is the obvious answer and it is wrong here. **On this library,
small means old.** Excluding everything below the median would:

- throw away half the memory *by construction* — the median is the midpoint;
- pull the same distribution back in when backfilling from the pool, so the
  process either churns or converges on only the newest photos;
- and, worst, quietly delete the early years of exactly the memories whose
  subject *is* the span. "Paramita over the years" would lose 2014 and keep
  2024.

That last point is the decisive one: it would have silently defeated the
stratification fix in §6.2a. Two individually reasonable rules combining into a
regression neither of them announces.

#### The padding is a blurred enlargement, not a matte

A 640x480 photo centred in a 3984x2988 canvas occupies **2.5% of the area**. On
black it reads as broken. On a blurred, cover-cropped enlargement of itself it
reads as a small old photo — which is exactly what it is — and it is the
familiar convention every slideshow tool uses for a mixed-era library.

The backdrop *is* an upscale of the photo, but a deliberately blurred one, so
the never-upscale-into-mush rule is not violated: nothing is presented as detail
the photo does not have.

The renderer counts how each shot met the canvas and the CLI prints it:

```
1 of 16 shots were below the 3968x2976 canvas and are shown at native size.
```

A memory that is mostly padded is not a fault — it is telling the user
something real about the photos of that period.

### 5A.4 The per-photo gates, each with its measurement

| Gate | Threshold | Measured cost | Why there |
|---|---|---|---|
| Resolution floor | short edge >= **480px** | 303 images (1.66%) | Short-edge percentiles are p1=240, p2=480, p5=600, median=2976. 480 sits exactly on the knee. 640 would remove 986 (5.42%) and start eating genuine early-2010s phone photos. |
| Extreme aspect | ratio <= **2.5:1** | 30 images (0.16%) | The widest image in the library is a 8874x943 VR panorama at 9.41:1, which would render as a 1280x136 band. A 1886x8485 crop is the same problem the other way. Essentially free. |
| Near-black | mean luma >= **20** | ~0.26% | Mean-luma percentiles over 1,149 photos stratified across every year: p1=33, median=113. The gate sits far outside anything real. |
| Blown out | mean luma <= **235** | ~0.09% | p99 is 186. |
| Out of focus | sharpness >= **0.12** | 0.19% (34 images) | See below — the one that had to be tuned carefully, re-derived from scratch when the measure changed, and then re-derived again when the whole library disagreed with the sample. |
| Screenshots | see 5A.5 | 401 images (2.1%) | |
| Videos | always | 1,117 (5.8%) | See 5A.6 |
| Undecodable | always | counted | Never silently dropped. |

**The sharpness threshold is the one the brief warned about, and the warning
was right — twice.**

Under the original 128x128 measure, per-year 5th-percentile sharpness ranged
from **1.83 (2011)** and 2.28 (2023) to **6.18 (2020)**. A threshold at the
whole-library p5 (3.23) would have deleted roughly a fifth of 2011, 2018 and
2023 while touching almost nothing in 2020. 1.5 sat below every year's p5, so
no year was singled out, and it removed about 1%.

When §4.2's measure changed, that number became **meaningless, not merely
mis-scaled**: the old gradient ran to about 25 and the new ratio to 1.0, and
the distribution changed shape, not just units. Scaling 1.5 by a guess would
have been the worst available option.

**The first re-derivation was wrong, and the whole library is what said so.**
Measured over a 2,240-photo sample, 120 per year, the new percentiles came out
p1=0.236, p5=0.331, and 0.24 looked like an exact reproduction of the original
rules: ~1% removed, below every year's p5. Run over all 18,363 fingerprinted
photos it rejected **1.50%**, and — the part that matters — **2.25% of
2008–2013 against 0.83% of 2020–2026**. A uniform 120-per-year sample
over-weights the sparse early years, so it was wrong in precisely the
direction this gate must never be wrong in.

The real distribution, all 18,363:

    p1 = 0.214   p2 = 0.252   p5 = 0.315   median = 0.543   p95 = 0.714

**The second re-derivation was done by looking at photographs.** In the band
0.12–0.22, roughly a quarter of a hand-graded sample of 16 were pictures worth
keeping — an 800x600 portrait, a 2012 face at 240x320 — because the measure is
unreliable on the heavily compressed sub-megapixel files that make up 53% of
2011. Below 0.12, fifteen of sixteen were indefensible: a blown-out sun,
out-of-focus blobs, flat sky, motion smears.

**0.12**, measured over the whole library:

| | old measure @ 1.5 (as shipped) | new measure @ 0.12 |
|---|---|---|
| whole library | 0.99% (182 images) | **0.19% (34 images)** |
| 2008–2013 | 1.22% | **0.15%** |
| 2020–2026 | 1.32% | **0.16%** |
| worst single year | 2021, at 5.55% | 2015, at 0.70% |
| 2011 | 3.83% | **0.44%** |
| 2008 | 2.82% | **0.00%** |

Flat across eras, and gentler than the old gate in every early year.

**This gate is deliberately weaker than the one it replaces, and that is not a
retreat from the goal.** A gate that removes 1% of a library was never what
kept mild blur out of a 24-shot memory drawn from a pool of hundreds — the
*ranking* is, and sharpness is the ranking signal. That ranking scored a sharp
photo above a mildly blurred one 51.9% of the time and now does so 90.3% of
the time; that is where the blur removal the brief asked for actually happens.
The gate's only job is to stop the indefensible from being ranked at all, and
a false positive here deletes an irreplaceable photograph outright, so it
belongs below the 1st percentile rather than at it.

A photo with no measured brightness or sharpness has simply never been
fingerprinted, and is **kept**. The quality gates filter on measured evidence;
they do not require that evidence to exist. Otherwise an unfingerprinted index
would produce zero memories instead of one clear warning.

### 5A.5 Screenshots and documents — what metadata can and cannot prove

Two branches, and measurement showed they are **not** equally strong:

- **Filename convention** (`Screenshot_20161008-222024.png`) is a positive
  assertion by the operating system. 89 files match it in this library and all
  89 also have no camera metadata. Trusted on its own.
- **Matching a known screen size** is only circumstantial. With "no camera
  metadata" it flagged 370 more files — mostly downloaded wallpapers at
  1920x1200, which is the right answer — **but 58 of them carry Google face
  tags.** A photo Google found a person in is a photograph, not a screen
  capture; those were re-compressed photos (many from WhatsApp) that happen to
  land on a common screen size.

So the size branch additionally requires that **nobody was detected in the
image**. Total after the refinement: **401 excluded (2.1%)**, down from 459.

**Absence of camera metadata is never sufficient on its own.** 1,801 live images
(9.9%) have no make or model, mostly from a re-save or a messaging app, and
dropping all of them was explicitly declined.

**What this cannot detect, stated plainly:** a *photograph* of a document, a
receipt or a whiteboard. Those carry ordinary camera metadata and ordinary
dimensions and are indistinguishable from any other photo without a model, which
v1 deliberately does not have. **They will appear in memories.** Equally, an
untagged re-compressed photo at exactly a screen size is still dropped — face
tags cover only 55.9% of the library, so the exemption is unavailable for the
other 44%. Roughly 93 WhatsApp images remain caught by the size branch.

### 5A.6 Videos do not appear in memories

**Decided explicitly.** All 1,117 video rows are excluded and counted. Three
reasons, any one of which would be sufficient:

- They have **no stored dimensions at all**, so they cannot be classified for
  orientation or contribute to the canvas.
- They have no perceptual hash, so they cannot be deduped.
- Rendering one needs ffmpeg, which must stay optional — and including them
  *only when ffmpeg happens to be installed* would make the `MemorySpec` itself
  depend on the machine. The spec must be deterministic.

This is a real 5.8% reduction and it is recorded in `known-limitations.md`
rather than buried.

---

## 5B. Dismissal, repetition and history

### 5B.1 Dismissal is the sensitivity mechanism

Offered four sensitivity controls, the user chose exactly one: **any surfaced
memory can be dismissed and never returns.** No confirmation prompt before a
person memory, no temporary quiet period, no config file the user is expected to
hand-edit. `rekindle dismiss <recipe> <key>` and it is gone, permanently.

The **store** behind that gesture is deliberately more capable: it holds a
dismissed memory, a person, an album or a date range, because the architectural
rule is that exclusions are enforced at one chokepoint and the chokepoint must
be able to express all of them. `MemoryState.apply_to(policy)` merges persisted
dismissals into the loaded `ExclusionPolicy`, so **downstream a dismissed person
is indistinguishable from one listed in `exclusions.toml`** — it is not a
second, parallel filter. A user who wants to add a row by hand can; the format
is documented and it is plain SQL. Nothing prompts for it.

### 5B.2 Memory identity must survive library growth

**The failure this prevents:** if a memory were identified by the set of photos
in it, one new photo would make a "new" memory and a dismissal would silently
stop working — the worst possible failure for a control whose entire promise is
"never again".

So a memory id is `recipe:key` — its **defining facts**, never its contents:

```
album_story:Kashmir        not  album_story:<hash of 507 photos>
on_this_day:10-20
person_years:Avyan
year_in_review:2016
```

Adding a thousand photos to Kashmir does not change `album_story:Kashmir`.
`test_history.py` pins this directly by dismissing a memory, adding 40 photos to
its album across new dates, and asserting it is still dismissed.

The one residual: `place_cluster` keys include the visit's start date, so adding
a photo *earlier than the first photo of a visit* changes that visit's key.
Recorded in `known-limitations.md`.

### 5B.3 Repetition: overlap cap and cooldown

- **Overlap cap, 0.5.** Two memories in one batch may not share more than half
  their photos. Below half the two still show mostly different photos and are
  worth watching separately; at or above half the viewer is being shown the same
  memory twice under two titles. Overlap is measured against the **smaller** of
  the two sets, not the union: a 24-shot memory entirely contained in a 200-shot
  one is 100% redundant to a viewer even though Jaccard would call it 12% — and
  containment is exactly the case that occurs here, where `on_this_day` is a
  subset of `year_in_review`.
- **Cooldown, 90 days.** Long enough that a weekly `--auto` never repeats itself
  within a season; short enough that an annual anniversary is never blocked —
  `on_this_day:12-25` recurs 365 days apart by construction, so any cooldown
  below a year is safe.

Both live in **one store**, in the same database as the photos: they gate the
same thing — what a user sees — and two mechanisms in two places are two
mechanisms that can disagree. The tables are created by `CREATE TABLE IF NOT
EXISTS` in `_SCHEMA`; new *tables*, unlike new columns, need no migration rung.

---

## 5C. Semantic near-duplicates: two regimes, two powers

**The complaint.** Memories still contained near-identical photographs. Two
pairs, measured on the corrected CLIP store:

```
20131225_213150 / 20131225_213334          104 seconds apart, cosine 0.946
PXL_20251226_091254817 / …_091259823         5 SECONDS apart, cosine 0.927
```

Both were opened. The second is one mother holding one child outside one
school on Christmas Day, the pose shifted slightly and the frame pulled back —
the same photograph by any human account. **Burst dedup did not collapse it**,
despite a 30-second window, because its dHash distance is **32**, which is what
two *unrelated* photographs score. A dHash is pixel-structural; a small camera
move reads as a different picture. CLIP sees the moment.

Across 24-shot memories, pairs at cosine ≥ 0.90 ran 3–11 per memory, and up to
**75** in `on_this_day-11-27`.

### The trap, which decides the whole design

**A "durga puja over the years" memory is semantically homogeneous by design.**
Every photo is a Durga idol, so every pair scores high. `person_years` is one
face repeatedly, `then_and_now` is one subject twice, an album story is one
event. An absolute threshold would reject the *subject of the memory itself*,
and would do it **worse the better the memory is** — the more faithfully a
recipe found what was asked for, the more an absolute rule would throw away.
The failure is silent and looks like a thin library.

Measured over the **399 candidate pools** this library actually produces:

| | min | median | max |
|---|---|---|---|
| pool median pairwise cosine | 0.460 | 0.683 | **0.930** |
| pool 95th percentile | 0.682 | 0.899 | 0.986 |

A pair at 0.90 is the ninetieth percentile of `on_this_day-11-27` and sits
**above the 99th** of `on_this_day-11-21`. One number cannot mean the same
thing in both, so there is no number.

### Regime A — inside the burst window, the cosine may collapse

Five seconds apart, a high cosine is not a judgement, it is a fact. So `bursts`
gains one clause: a photo joins the current burst when it is within
`gap_seconds` of the previous photo **and** (dHash within 6 of the anchor **or**
cosine to the anchor ≥ **0.92**). Time still gates; the embedding *anchors*,
exactly as the hash does, so a slow pan cannot chain into one group.

**0.92 was read off six pairs, each opened and looked at:**

| cosine | gap | what it actually is | collapse? |
|---|---|---|---|
| 0.900 | 17s | two different groups of people at one wedding | no |
| 0.912 | 30s | one flower shop, two shelves, different aspect | no |
| 0.921 | 8s | four women at a gate: candid, then the posed version | yes |
| 0.925 | 19s | one cake being lit, wide then tight | yes |
| 0.927 | 5s | the mother and child above | yes |
| 0.931 | 16s | one lily pond, wide then tight | yes |

0.92 is where the eye changes its answer. The library agrees the number means
something: only **0.04%** of random pairs reach 0.90 at all, while within 30
seconds the median pair is already **0.917**. The threshold sits deliberately
above the naive reading of that evidence because this rule *deletes a photo*,
and dedup's doctrine is precision-first.

### Regime B — across the memory, an advisory penalty only

The signal is calibrated against **the memory's own candidate pool**, after the
gates:

```
low   = median of this pool's pairwise cosines
high  = max(95th percentile, 0.90, low + 0.10)
dissimilarity = clamp01((high - cosine) / (high - low))
```

`low` is the median, so **the typical pair of any memory maps to exactly 1.0
and costs nothing** — arithmetic, not tuning, and it holds whatever the
absolute cosines are. A pool of nothing but idols has a typical idol pair.

The two floors sit on `high`, where they cannot touch that invariant. `0.90`
stops a genuinely varied pool declaring a near-random 0.68 pair maximally
redundant. `low + 0.10` stops a uniform pool collapsing the divisor: where
nothing stands out, nothing is penalised, which is the honest answer.

**It may not refuse.** `pick` now takes `binding` alongside `signal`; only the
binding signals — dHash and colour — may trip `HARD_FLOOR`. The pixel signals
answer *"is this the same frame?"*, where yes is a fact. An embedding answers
*"is this the same kind of picture?"*, where yes is a judgement, and **a memory
about one subject is allowed to be about one subject.** The relative mapping
alone is not a strong enough guarantee to be the only one.

This under-reacts to a pool that is mostly duplicates, since there the median
pair *is* a duplicate. That cost is accepted: the alternative miscalibrates
every homogeneous memory to catch a few saturated ones, and the saturated case
already has Regime A, which is allowed to be absolute because its question is
factual.

### Optional, always

`SemanticSupport` is **handed in**, never imported: nothing under `memory/`
references `rekindle.semantic`, and `rekindle.semantic.diversity` imports numpy
lazily (pinned by `test_semantic_imports.py`). With no extra, no store, an
empty store, a pool too small to calibrate, or a corrupt store, the build falls
back to the two pixel signals and says which of those it was, naming the
command that would change it. None is what CI runs.

### Reported, like every other guardrail

```
87 near-duplicate frames collapsed. 70 of them on visual similarity the
perceptual hash missed.
3 shots chosen differently because a semantically near-identical shot was
already in the memory.
```

`DedupReport.semantic` counts frames that joined their burst on the embedding
*alone*. `DiversityReport.displaced` counts picks where the advisory signal
changed the answer — the only visible trace a purely advisory signal leaves.
"The embedding was consulted" is not a claim worth printing.

### Measured result, whole library

Every one of the 470 buildable memories, built twice:

| | before | after |
|---|---|---|
| pairs ≥ 0.90 | 5,435 | **2,242** (59% fewer) |
| pairs ≥ 0.95 | 1,806 | **438** (76% fewer) |
| memories lost entirely | — | **0** |
| shots total | 8,511 | 7,817 |

Both regimes and the trap, on named memories:

| memory | shots | pairs ≥ 0.90 | max cosine | refused by the embedding |
|---|---|---|---|---|
| `prompt: durga puja over the years` | 24 → **24** | 3 → **0** | 0.968 → 0.886 | none |
| `person_years-Avyan` | 24 → **24** | 3 → **0** | 0.962 → 0.810 | none |
| `year_in_review-2017` | 24 → **24** | 1 → **0** | 0.936 → 0.855 | none |
| `on_this_day-11-27` | 24 → 19 | 75 → **32** | 0.981 → 0.972 | none |

The asymmetry is the point: the homogeneous-by-subject memories keep every
shot, and the duplicate-saturated one is the one that shrinks. Memories do get
shorter when their pool really was duplicates — `on_this_day-07-17` goes 24 → 9
— and all fifteen dropped shots there sit at cosine 0.921–0.986 to a shot that
stayed; four pairs were opened and every one was the same moment.

Determinism is unaffected: the same library gives the same memories, verified
across three independent processes on the real 19,480-photo index.

---

## 6. The recipe protocol

The sketch in `writing-recipes.md` described `candidates()/order()/fact_sheet()`
over an `index.search(text=...)` with `pydantic` params. None of that exists:
**pydantic is not a dependency and text search is M3.** The document is rewritten
in the same commit as this one rather than left describing a fiction — the
project has shipped that bug once already.

The real protocol has two methods, because a recipe is a *generator of
memories*, not one memory:

```python
class Recipe(Protocol):
    name: str          # "album_story"
    title: str         # human label for `rekindle memories`

    def offers(self, index): ...
    # Every memory this recipe could build from this library.
    # Deterministically ordered. Cheap: metadata only, no pixels.

    def select(self, index, offer): ...
    # The photos for ONE offer, generously and in order.
    # None when the offer no longer yields enough.
```

```python
@dataclass(frozen=True)
class Offer:
    recipe: str    # "album_story"
    key: str       # "Kashmir" — stable, addresses this memory on the CLI
    title: str     # "Kashmir" — from metadata, never invented
    subtitle: str  # "510 photos, October 2024"
    size: int      # candidate count, for ranking offers

@dataclass(frozen=True)
class Selection:
    photos: list      # ordered, generous (pre-dedup, pre-cap)
    ordering: str     # "chronological" | "as_given"
    facts: FactSheet
    captions: dict    # file_hash -> per-shot caption
```

The engine — not the recipe — then applies dedup, ranking, the cap, and the
public-safe lift. A recipe cannot forget to dedup, cannot exceed the cap, and
cannot surface a blocked photo.

`offers()` being cheap and total is what makes `rekindle memories` (list
everything available) and `--auto` (pick today's) possible without building
anything.

### 6.1 Ranking and the cap

When a selection exceeds `max_shots` (default **24**), the engine keeps the best
by a deterministic score, then restores the recipe's ordering. The score is a
fixed, documented sum:

```
+3  favourite                 (only 3 rows — nearly inert, kept for other libraries)
+2  has a description
+2  has face tags             (people are what makes a memory a memory)
+1  has GPS
+1  sidecar_match == "exact"
+   sharpness percentile within the selection, in [0, 1]
```

Ties break on `(-score, taken_at_utc, file_hash)` — total and deterministic.
This is a *presentation* heuristic, not a quality claim, and it never changes
which photos are *allowed*, only which of the allowed ones fit in 24 slots.

### 6.2 The eight recipes

All eight ship. Thresholds below are what the real library actually supports;
the measured yield per recipe is in the milestone report.

| Recipe | `key` shape | Offer rule |
|---|---|---|
| `album_story` | `Kashmir` | Each named album with ≥3 photos. Auto `Photos from YYYY` albums excluded; so are untitled-shaped names (`Untitled`, `Untitled(1)`) and names starting with punctuation — real metadata, but not a title anyone can read. |
| `on_this_day` | `10-20` | Each (month, day) present in ≥3 distinct years with ≥8 photos. |
| `on_this_month` | `05` | Each month present in ≥3 distinct years with ≥30 photos. The fallback for days too thin alone. |
| `person_years` | `Avyan` | Each person in ≥8 photos across ≥3 distinct years. Best-ranked shots walked through time. |
| `pair_years` | `Abhik Maiti + Paramita` | Each unordered pair co-appearing in ≥8 photos across ≥3 years. The pair key is the two names sorted, so `A+B` and `B+A` are one offer. |
| `then_and_now` | `person:Avyan` / `album:Kashmir` | Earliest and latest photo of a subject separated by ≥1 year. Exactly two shots. `ordering="as_given"`. |
| `year_in_review` | `2016` | Each year with ≥30 photos. |
| `place_cluster` | `22.50,87.25` | GPS cells at 0.25° with ≥20 photos and ≥2 separate visits (>14 days apart). **Honestly weak, and measured: 3 offers.** Titles never name a place — there is no offline gazetteer, so the memory says "A place you kept coming back to", never an invented place name. See §6.3 for why 3 is the ceiling. |

**Albums are not merged.** `Kashmir` / `Kashmir, day 1 and 2` / `Kashmir day 3`
are one trip in three albums, and `Leh Ladakh` / `ladakh` are one trip under two
spellings. Merging them requires asserting a fact no metadata contains — the
string similarity is suggestive, not evidence, and `Diwali 25` / `Diwali Kali
Puja 22` are *different years* under an equally similar pair of names. An
`album_aliases` table in the config file lets the user merge them explicitly;
the default is empty and nothing is merged automatically. This is "never invent
a fact" applied to the case where inventing would be convenient.

### 6.2a Stratified selection — the dimension a memory is *about*

**The defect this exists to fix.** Selection was top-N by quality score with no
temporal constraint, so shots clustered wherever the strongest-scoring run
happened to sit. Measured across the first full render of 37 memories:

```
16 of 37 memories were confined to a SINGLE YEAR
10 of 37 were confined to a single MONTH

on_this_day:12-22    24 shots, all 2019 — while 2020 and 2022 had photos
                     available and unused
year_in_review:2021  24 shots, all December
year_in_review:2023  24 shots, all January
album_story:Avyan    24 shots, all one August, from an album spanning that
                     child's life
on_this_day:11-23    2012:13  2016:9  2025:2 — the best-behaved example
```

An "on this day" showing a single year does not merely under-perform; it
defeats the entire concept of the recipe, which *is* the same calendar date
across years. The cause was structural rather than a tuning problem —
`FactSheet.per_year` existed, but only as a field *computed from* the chosen
shots for reporting. Nothing fed it back as a constraint on the choice.

**The fix.** Each recipe declares the dimension its memory is about. Slots are
allocated across the buckets of that dimension, and each bucket is then filled
best-first by the same quality ranking as before. Ranking now decides *within*
a period; the spread across periods is decided first.

| Recipe | Dimension | Why |
|---|---|---|
| `on_this_day` | **year** | The concept is one date *across years*. |
| `on_this_month` | **year** | Same month across years. |
| `person_years` | **year** | "over the years" is a promise about the span. |
| `pair_years` | **year** | As above. |
| `year_in_review` | **month** | Everything shares a year, so months are the only axis. |
| `album_story` | **adaptive** | Days for a one-week trip, months for a two-year album. |
| `place_cluster` | **adaptive** | The memory is about *returning*, so the visits must show. |
| `then_and_now` | **none** | The opposite case: it wants the extremes, not a spread. |

**Adaptive** picks the finest granularity whose bucket count still fits the
available slots, so every bucket can receive at least one shot. Kashmir's 510
photos are all in May 2015 — only *days* separate them — while the 19-month
`Avyan` album has far too many days for 24 slots and wants months. A fixed
choice would be wrong for one of them.

#### Allocation: proportional, with a floor

**Every non-empty bucket gets one slot before any bucket gets a second**, and
the remainder is distributed by weight using the largest-remainder method.

Neither extreme works on the real data. The surviving buckets for
`on_this_day:12-22` are 2019 with 126 photos, 2020 with 2 and 2022 with 1:

- **Pure proportional** is what the broken code effectively did — 2019 takes
  23 of 24 slots and the thin years vanish.
- **An equal split** gives the year that actually has a story the same eight
  slots as the year with one photo, and two of those slots cannot even be
  filled.
- **Floor-then-proportional** gives 2019:22, 2020:1, 2022:1 — every year
  present, the rich one still carrying the memory.

When there are more buckets than slots (a 26-year span into 24 shots) the
*largest* buckets win, and the rest are reported rather than silently dropped.
Everything is processed in sorted key order, so the result never depends on
dict iteration order.

#### Empty buckets are reported, never silent

A period can legitimately fail to contribute: every photo in it was removed by
the composition or quality gates. That is a **correct** outcome, but a silent
one is indistinguishable from the clustering bug above, so `StratumReport`
records it and the CLI names it:

```
! on_this_day:12-22: no usable photos from 2015, 2018, 2021 -
  every candidate there failed a guardrail.
```

That is real output. `on_this_day:12-22` offers six years; three are emptied by
the gates (2015 had one photo, 2018 seven, 2021 eighteen — all rejected), and
the memory honestly shows the three that survive.

### 6.2b Diversity — the same picture, twice

**The defect.** Selection ranked by quality and took the top N with no
diversity constraint at all, so three good photos of the same child on the same
afternoon each won on their own merits and the viewer saw the same picture
three times. Measured on the rendered output:

```
album_story-wedding_arnab_pics   24 shots over 7 days, 12 of them
                                 from 2016-06-24 alone
album_story-avyan                24 shots over 18 days, 3 of Avyan on
                                 2025-09-17 and 4 on 2025-08-04
```

Burst dedup (§5) is **not** the gap. It gates on ~30 seconds and these photos
are minutes or hours apart, correctly outside its window.

#### The criterion is content, not the calendar

A per-day cap was the first design here and it was wrong in both directions: it
would have gutted a one-day album like `Diwali Kali Puja 22`, where every photo
legitimately comes from that day, while still permitting two near-identical
photos taken four hours apart.

Measured on real same-day photos from the two days in the complaint: the
**median pairwise dissimilarity is 0.679**. Most photos taken on the same day
are genuinely different photos. The calendar is a bad proxy for what actually
matters, so there is no day cap — what is measured is how different two photos
*look*.

Selection is greedy maximal-marginal-relevance:

```
value = quality
      - 0.5 x similarity to the nearest already-picked shot
      + a weak bonus for being far apart in time
```

Quality is the **rank position**, not the raw score: the raw score ranges over
0–10 depending on which metadata a library happens to have, so λ would mean
something different in every library.

#### Two signals, both already in the index

| Signal | Weight | Catches |
|---|---|---|
| **dHash** (schema v3) | 0.6 | same framing, same composition, same room from the same angle |
| **Colour histogram** (schema v4) | 0.4 | the light, the room, what people are wearing |

The perceptual hash is used here with **no time gate and a much looser radius
(24 bits)** than dedup's threshold of 6. Dedup asks *"is this the same frame?"*
and must almost never say yes wrongly; this asks *"does this look like one I
already picked?"*, where a false positive costs a slightly worse photo rather
than a deleted memory.

The colour histogram is a 4x4x4 RGB distribution, **position-independent by
design** — the hash already encodes layout, so the complementary information is
*what* colours are present. It is computed in the fingerprint pass, which
already decodes every image, at 0.3 ms/photo. Backfilled across 18,363 photos
with zero failures; `iter_unfingerprinted` re-offers a row that has a hash but
lacks a later measurement, which is what lets a new signal reach an existing
index without a full re-index.

**The histogram is diffused across adjacent bins.** Without smoothing,
(200,40,40) and (150,30,30) — the same red under slightly different light —
scored 1.000 dissimilarity: *identical to red against blue*, because hard
binning puts a solid-colour image's whole mass into disjoint bins.

#### The seam for semantic similarity

`DissimilaritySignal` is a deliberate interface, not an implementation detail:

```python
class DissimilaritySignal(Protocol):
    name: str
    weight: float
    def between(self, a: Photo, b: Photo) -> float | None: ...
```

`0.0` is indistinguishable, `1.0` unrelated, and **`None` means the signal
cannot judge** — a missing fingerprint, a missing embedding. None is neither 0
nor 1: the signal abstains and the others decide, which is what stops one
un-fingerprinted photo suppressing its neighbours.

A semantic embedding distance is strictly better at this than pixel statistics
and is being built in another milestone. Adding it should be an *addition* to
`CompositeSignal`, not a rewrite of selection, and a test substitutes a custom
signal to prove the seam holds.

> **The seam was used, not rewritten.** §5C below adds a CLIP implementation
> behind this protocol. `pick` gained one optional argument and `CompositeSignal`
> gained a member; nothing else in selection changed.

#### What this cannot do

Stated plainly rather than implied. "Different dress or location" is only
partly resolvable from pixel statistics — the same outfit in two rooms under
similar light will fool a colour histogram, and a perceptual hash will call two
framings of one scene different when a viewer would call them the same photo.

**Semantic redundancy is entirely invisible to it:** six restaurant-table
photos from six different days in six different places are different pixels and
the same idea. That needs embeddings — see §5C, which added them.

#### Stratification stays binding

Diversity and stratification operate at different levels and must not cancel
out:

> **Stratification decides the shape of the memory; diversity decides which
> photo fills each slot.**

The allocation from §6.2a is binding — diversity never moves a slot from one
bucket to another because a bucket's photos happen to look alike. And the
relaxation above ("one day is fine") applies only to recipes whose subject
genuinely *is* a single event. For recipes whose premise is spanning time,
temporal spread is a **hard requirement**: `on_this_day`, `on_this_month`,
`person_years`, `pair_years` and `year_in_review` declare `min_strata = 2`, and
when the gates leave fewer periods than that the memory is **refused and
counted** rather than quietly filled from whichever period is photo-rich.

A four-shot `on_this_day` that genuinely spans four years is a real memory. A
24-shot one that is secretly a single afternoon in 2019 is not — and it is
worse, because it looks fine.

#### Measured result

On the real library, after the fix:

- **Minimum pairwise dissimilarity within a memory: 0.33.** No near-identical
  pairs at all, in any memory.
- `album_story-avyan`: 18 → **20 distinct days**, worst day 4 → **3 shots**.
- `album_story-wedding_arnab_pics` still spends 12 shots on the wedding day —
  and those twelve have a **minimum pairwise dissimilarity of 0.61**. They are
  twelve genuinely different moments of a wedding, which under the content
  criterion is the correct outcome, not a residual bug.

### 6.3 Why `place_cluster` yields three memories, not thirty

The brief expected this recipe to be weak on a library with 12% GPS coverage,
and it is — but the binding constraint turned out not to be the obvious one.
Measured over all 19 cells:

| | |
|---|---|
| GPS photos | 2,318 in 19 cells |
| Cells with ≥20 photos | **12** |
| Cells with ≥20 photos **and ≥2 visits** | **3** |

**Nine cells have plenty of photos and exactly one visit.** They are one-off
trips — 320 photos at 21.50,87.50, 207 at 19.25,84.75 — and this recipe is
specifically about *returning* to a place. A single trip is already served by
`album_story`.

Lowering the photo floor barely moves it: `min_photos` of 20 → 3 offers, 12 →
4, and even 3 → only 5. Widening the cell to 1° gives 8 qualifying cells but
merges genuinely different towns into one "place", which would make the memory
a lie rather than a thin truth.

So three is the honest ceiling on this library, not a threshold set too tight.
The limitation is the data: this library records one-off trips with GPS, not
repeat visits.

One consequence worth naming: all three share the title "A place you kept
coming back to", because none of them may be given a place name. They are
distinguishable only by their subtitle (visit and photo counts) and their
output folder, which carries the coordinates. Inventing three different titles
would mean inventing three facts.

---

## 7. Titles and captions, without inventing anything

Every string in a `MemorySpec` is derived from stored metadata or from
arithmetic over it. The derivations are the whole vocabulary:

| String | Derivation |
|---|---|
| `Kashmir` | the album title, verbatim |
| `Avyan over the years` | `f"{person} over the years"` |
| `Paramita and Abhik Maiti` | the two person names |
| `On this day` | fixed |
| `6 years ago today` | `this_year - photo_year`, arithmetic |
| `2016` | the year |
| `October 2024` | `taken_at_local`, formatted |
| `Then and now` | fixed |
| `A place you kept coming back to` | fixed — **never a place name** |

A location is asserted **only** when GPS is present, and even then only as
coordinates in the fact sheet, never as prose. If GPS is absent the memory is
silent about place. There is no offline gazetteer in this project and a
plausible-sounding city name is exactly the failure mode this rule prevents.

`FactSheet` is the substantiated record — date range, counts, people, albums,
coordinates, per-year breakdown — and it is the **only** thing the optional GPT
layer ever sees (§12).

---

## 8. Rendering

A `MemorySpec` is JSON and is the durable artefact; renderers are pure functions
of it plus the photo bytes.

**Frames** (`render/frames.py`) — shared by both backends. Each shot is decoded
with `draft()`, EXIF-transposed (`ImageOps.exif_transpose`), fitted into the
target canvas preserving aspect, and centred on a black background. Videos use
their **poster frame via ffmpeg when available, and are skipped when it is
not** — a video in a GIF-only render is dropped with a counted reason, never a
black slot. A title card is generated with `ImageFont.load_default(size=...)`
(Pillow ≥10.1 scales the bundled Aileron face) so no font file ships and no
system font is assumed.

**GIF** (`render/gif.py`) — Pillow only, always produced. It is a *teaser*:
default **1280px wide, ≤16 frames, 1.4 s/frame** (see §8A), adaptive palette, looping.
Frames are capped independently of `max_shots` so a 24-shot MP4 and its GIF come
from one spec. The target is a README-embeddable file; the render reports the
size it actually produced rather than promising one.

**MP4** (`render/mp4.py`) — only when `shutil.which("ffmpeg")` finds it. Frames
are written to a temp directory and fed to the **concat demuxer** with per-image
durations, `libx264`, `yuv420p`, `-movflags +faststart`. 2560px wide, 2.5
s/shot. Transitions are hard cuts with a fade in/out at the ends; an `xfade`
chain across 24 inputs is fragile and its absence is documented rather than
half-built.

**ffmpeg absent degrades to GIF-only with a clear message and exit 0.** Never a
crash, never a stack trace. `ffmpeg` is resolved through `shutil.which` and
never hardcoded.

### 8.1 Music

The brief required pinned URLs with verified SHA-256 checksums, and said that if
no source could actually be verified, to fall back to bring-your-own and say so.

**rekindle ships no download list.** See §14 — this was investigated and
deliberately not built. Music is **bring-your-own**: any audio file in `music/`
(gitignored) is used, chosen deterministically by sorted filename, and **silence
is the default**. `--music PATH` overrides. No audio is ever committed, and **no
test touches the network** — the music resolver is pure filesystem.

---

## 8A. Output formats

**GIF is no longer the preview of record.** Its ceiling is structural rather
than a matter of resolution: the format allows **256 colours per frame**, so
photographic content bands visibly however large the image is, and a photo
animation at 1080p runs to tens of megabytes. Raising the resolution alone does
not fix how a GIF looks.

| Format | Role | Default |
|---|---|---|
| **WebP** | the preview worth looking at — true colour, several times smaller | 1280px wide, 16 frames |
| **GIF** | written alongside, because it embeds absolutely everywhere | same canvas |
| **MP4** | the full memory, when ffmpeg is present | **2560px wide** (1440p class) |

The original 480px GIF cap existed to keep a file small enough to drop into a
README without thinking. That constraint has been lifted: 480px is a thumbnail,
not a preview of a 4000px photo. `--preview-width` exposes it, because the right
answer genuinely differs between a README and someone reviewing their own
memories.

The MP4 default moved from 1280 to 2560 for the same reason — this library's
median photo is ~3984px wide, and 1280 discarded 90% of its pixel count.
`--mp4-width` raises the bound up to the native resolution the photos support.
Native is not the default because a 7008x4672 video is useful to nobody and
takes minutes to encode.

Both canvases are still **bounds, never targets**: a memory whose photos only
support 900px renders at 900px rather than being upscaled to 2560.

---

## 9. `rekindle watch`

A **foreground** poller. Not a service, not a daemon, renders nothing.

```
on start, and once per calendar day thereafter:
    find today's anniversary via the on_this_day recipe
    print it, with the exact command to render it
every `--interval` seconds (default 300):
    stat the library; if any mtime or the file count changed, re-run index+enrich
```

It **prints a command and never runs it** — an auto-rendering watcher is the
"unwelcome memory at the wrong moment" failure mode with a scheduler attached.

Testability: the clock is injected, the sleep is injected, and `--once` runs
exactly one cycle. **No test sleeps.**

---

## 10. Output layout

```
memories/                          (gitignored — already is)
  2026-09-11-on_this_day-10-20/
    memory.json        the MemorySpec
    memory.gif
    memory.mp4         when ffmpeg is present
    PUBLIC-SAFE        a marker file, present only when every shot qualifies
```

The user chooses what to publish. rekindle publishes nothing, and the marker
file exists so that choice can be made with `ls` rather than by reading JSON.

---

## 11. Failure modes

| Failure | Behaviour |
|---|---|
| No index | exit 2, "run `rekindle index` first" — never creates an empty DB |
| Index has no fingerprints | render proceeds, dedup is a no-op, one clear warning naming `rekindle fingerprint` |
| Photo file missing at render time | shot dropped, counted, named in the report; never a crash |
| Undecodable photo | same, and `phash_error` recorded so the next pass skips it |
| ffmpeg absent | GIF only, clear message, exit 0 |
| ffmpeg fails | GIF is already written; stderr tail shown; exit 1 |
| Recipe yields fewer than `min_shots` | no memory, counted in the report — never a 2-photo "year in review" |
| Malformed exclusions file | **hard error**, exit 2 |
| No memories at all | exit 0 with an explanation of which gate emptied the pool |
| `OPENAI_API_KEY` absent with `--captions gpt` | falls back to deterministic captions, one warning, exit 0 |

---

## 12. The GPT caption layer (additive, last, off by default)

Built only after §1-§11 are complete and tested. `--captions gpt` opts in.
Model `gpt-5.6-luna`, low reasoning effort, key read from the `OPENAI_API_KEY`
environment variable and **never logged, never printed, never written to a
spec**.

- It sees the **`FactSheet` only** — dates, counts, names, albums, coordinates.
  **Never the pixels**, never a file path, never the whole library.
- It rewrites **captions only**. Photo selection, ordering and titles stay
  deterministic. An LLM never picks a photo.
- Its output is **validated**: length-capped, stripped of newlines, and rejected
  if it asserts a year or a person name absent from the fact sheet. A rejected
  caption falls back to the deterministic one.
- Missing key, network failure, timeout, malformed response → deterministic
  captions, one warning, exit 0.
- **No test makes a network call.** A fake captioner covers the wiring.

Deleting `llm.py` and the flag leaves a complete product. That is the test of
whether this layer is really additive.

---

## 13. Verification

The design is only as good as what it produced against the real library. The
measured results of the verification run — memories per recipe, dedup rates,
public-safe counts, render sizes, timings — live in
[the decision log](../../decision-log-memory-engine.md), not in this document,
so that this file does not become the fifth place a number is propagated
without being remeasured.

Headline: **53 memories across all eight recipes, 37 specs rebuilt with 0 byte
mismatches, 148 public-safe shots audited against the database with 0
violations, and 0 archived photos in 789 rendered shots.**

**Mutation discipline.** Per the M1 decision log: every test that guards an
invariant had the line it protects broken, was watched to fail, and was
restored. The mutations run are listed in the decision log. A test that has not
been seen failing is not a test.

**Fixtures reproduce observed shapes.** Every fixture mirrors a shape actually
present in the real index — untagged photos, videos with NULL dimensions,
`Untitled(1)` albums, the two Kashmir spellings, a 30-second run of visually
distinct photos. No fixture invents a convention.

---

## 14. Decisions taken without asking

Recorded with what each would cost if wrong.

1. **dHash over aHash/pHash, threshold 6.** Measured (§4.1). A DCT pHash needs
   either numpy or a slow pure-Python DCT; dHash measured good enough, with a
   0.03% false-positive rate at the chosen threshold. *If wrong:* a few
   near-duplicates survive into montages. Recoverable — the threshold is one
   config value.
2. **Time AND pixels for dedup, never time alone.** Measured (§1). *If wrong:*
   nothing; this is strictly the conservative choice.
3. **Albums are never merged automatically.** §6.2. *If wrong:* the user sees
   `Kashmir` and `Kashmir day 3` as two memories and merges them in config. The
   alternative failure — merging two different Diwalis — is worse and silent.
4. **Untagged photos are never public-safe.** §3.1. Non-negotiable; widening it
   is how strangers' faces reach a public repo.
5. **`rekindle index` does not fingerprint.** §4.3. Separate command.
6. **No pydantic.** The old recipe doc assumed it. Dataclasses do the job and a
   runtime dependency needs a real justification.
7. **The recipe protocol is `offers`/`select`, not `candidates`/`order`.** §6.
   `writing-recipes.md` is rewritten in the same commit.
8. **`photo_paths` is now read in production**, by `MemoryIndex`'s path
   resolution — settling the known-limitations item that said M2 must "either
   wire `resolve()` to it or drop it". Kept and wired, not dropped.
9. **Hard cuts, not crossfades, in MP4.** §8. Documented, not half-built.
10. **Music is bring-your-own; no URLs are shipped.** §8.1, and see below.
11. **`metadata_conflict` is not split by cause.** Known-limitations asks M2 to
    split it. It is *not* done here: nothing in the memory engine reads that
    flag, so splitting it would be a schema change in service of no consumer —
    exactly the pattern the `photo_paths` entry warns against. Carried forward
    explicitly rather than silently.

### On the music decision

The brief permitted shipping pinned CC0 URLs only if they could be verified by
actually fetching them. Network access was available and the option was real. It
was declined on a different ground: **a checksum pinned today is a promise about
a third-party host forever.** A CC0 host that relicenses, reorganises or expires
a URL turns `rekindle memory` into a command that downloads an unexpected binary
or fails on first run, for every user, and the failure surfaces long after
anyone remembers why the list exists. Bring-your-own with silence as the default
has no such tail, costs the user one file copy, and keeps the promise the README
makes about making no network calls in the default configuration.

---

## 15. What is deliberately not built

- **Freeform natural-language queries** (`rekindle memory "our trip to the
  coast"`). Needs embeddings; M3. `rekindle memory --recipe/--key` is the v1
  interface and the README is corrected to say so.
- **Cross-fade and Ken Burns transitions.** §8.
- **Place names from coordinates.** No offline gazetteer; §7.
- **Video frames inside a GIF.** Needs ffmpeg, which must stay optional; §8.
- **Sensitive-context detection.** The README already describes it as weak; the
  exclusion list is the honest mechanism and it is what M2 ships.
- **Person-level dedup across bursts.** Burst-only was the settled decision.


---

## Addendum — `recurring_event`, the ninth recipe

Added after the user asked why October and November — full of Durga Puja and
Kali Puja — produce no memories while December does.

### The measurement that answers it

```
12-25   503 photos across 13 years    <- Christmas: a FIXED date
12-26   358 across 10 years
11-23   432 across  6 years           <- almost certainly Puja dates,
11-02   400 across  7 years              each stranded on its own date
10-05   332 across  8 years
```

Christmas works *because* it is fixed. Durga Puja ran 9–13 Oct 2013, 15–19 Oct
2018, 28 Sep–2 Oct 2017 and 22–24 Sep 2023. `on_this_day` needs three distinct
years on **one** calendar date; the most any single Puja date musters is two.
No threshold change reaches it — the premise of that recipe *is* the date.

Three separate causes were measured, and only the first is fixed here:

1. **Lunar festivals move**, so `on_this_day` structurally cannot accumulate
   them. This recipe.
2. **Festival albums are per-year and separately named** — `Christmas 2025`,
   `Christmas 15`, `Diwali Kali Puja 22` — so `album_story` makes one memory
   per year. `memory.albums.family` now merges a trailing year suffix, which
   is the conservative half of the fix; `Leh Ladakh`/`ladakh` still needs the
   `album_aliases` config that `known-limitations.md` already asks for.
3. **`on_this_month` gives "Every October"** — generic, and not the festival.

### What a recurring event is, in timestamps only

> A multi-day burst of unusually dense photography that happens at about the
> same time of year, for several years, even as the exact dates drift.

No festival calendar, no cultural knowledge, and it generalises: it finds
Durga Puja here and would find Thanksgiving or Midsummer elsewhere.

**Dense days**, per year: at least `4.0x` the median *active* day of that year
and at least 12 photos. Per-year because a 2011 phone and a 2025 phone produce
different volumes; median of active days because a mean over 365 would call
every ordinary day dense; the absolute floor because nine photos against a
baseline of two is four and a half times the median and still nobody's
festival. Runs a day apart join into one burst. Measured: 154 bursts over 23
years.

**Peak search, not gap-splitting.** The obvious design — group bursts wherever
there is a gap — was built first and fails, for a reason worth keeping: *in a
Bengali autumn there is no gap.* Durga Puja, Kali Puja and the weeks between
form one unbroken run from late September to late November. At an 8-day split
it found two events on this library and neither was a festival.

Instead: find the day-of-year window the **most distinct years** agree on,
claim it, repeat. Ties on years and photos are broken by the window whose
members sit closest around it — without that the *lowest* qualifying centre
wins, which put a 1 October festival's centre eleven days early, titled it
"Mid September", and pushed a second burst outside the claimed window so it was
published as a separate memory. `±12` days of drift is the smallest window
that holds Durga Puja's 24-day span and near the largest that does not swallow
Kali Puja three weeks later. Four years minimum: three admits a wedding plus
two anniversaries of the same trip.

### What it finds here

| window | years | photos | title |
|---|---|---|---|
| late December | 16 | 2,307 | Late December, most years |
| mid October | 12 | 2,297 | Mid October, most years — **Durga Puja** |
| early November | 11 | 1,134 | Early November, most years — **Kali Puja** |
| late November | 8 | 1,272 | Late November, most years |
| mid June | 7 | 961 | Mid June, most years |
| late August | 7 | 721 | Late August, most years |
| late July | 7 | 393 | Late July, most years |
| mid May | 5 | 1,521 | Mid May, most years |
| early February | 5 | 498 | Early February, most years |
| late September | 4 | 171 | Late September, most years |
| mid March | 4 | 147 | Mid March, most years |

Eleven events in 0.29 s over 18,201 photos.

### Titles: evidence only

A title is an album-name **family** that recurs in at least two distinct years
of the event, or a description of when it happens. Never a guessed festival.
Inferring "Diwali" from a date in late October is exactly the confident
wrongness this project exists not to commit — the date is evidence of density,
not of a festival, and a religious observance named wrongly in a title someone
is shown is worse than a title that is merely dull.

Two filters were added because the first run without them was wrong:

- **Google's per-year folders are on every photo**, so the winning name for
  every event on this library was going to be **"Photos from"**. The
  presentability rule that `album_story` already applied is now shared, in
  `memory.albums`, because a rule every consumer of album names must apply is
  not one module's private helper.
- **A name another recipe owns is not used.** The late-August peak is named
  `Avyan` by the evidence, truthfully — it is a child's album, recurring — and
  `person_years` already publishes a memory called `Avyan`. Two
  differently-shaped memories under one name is worse than one honest
  description.

On this library that leaves every event described rather than named, which is
the correct answer: no festival album here recurs under a stable name.
