# Decision log — the memory engine (M2)

This is the record of how M2 was actually built: the judgement calls, the
things that were wrong and how they were caught, and the numbers the design
rests on.

It exists because the substantive outcomes live in the code, in
[the design spec](superpowers/specs/2026-09-11-memories-design.md) and in
[known-limitations.md](known-limitations.md), but the *reasoning* would
otherwise be lost.

## What the milestone does

`rekindle fingerprint` computes a perceptual hash, a focus measure and a
brightness measure for every photo, once. `rekindle memories` lists every
memory the library could produce. `rekindle memory` builds one — a GIF always,
an MP4 when ffmpeg is present. `rekindle dismiss` makes a memory never return.
`rekindle watch` polls the library and prints today's anniversary without
rendering anything.

Eight recipes, one guardrail chokepoint, no model in the default path.

---

## The measurements the design rests on

Every number here was measured against the real 19,480-row index, not carried
in from the brief. Two of them changed the design outright.

### A 30-second window is not a burst

| | |
|---|---|
| Consecutive live images within 30s of the previous one | **59.2%** |
| Images grouped into multi-photo runs by time alone | **14,425 of 18,201** |
| Largest such run | **80 photos spanning 40 minutes** |
| **Median dHash distance *within* a 30-second run** | **22** |

Photos taken seconds apart are usually *genuinely different photos*. A
time-only burst dedup — which is what "collapse near-identical frames captured
within ~30 seconds" reads like on first pass — would have destroyed the
library. Time is the gate that decides which pairs are worth comparing; the
pixels decide.

### dHash over aHash, threshold 6

Measured on 414 real photos drawn from real 30-second runs:

| Threshold | within-burst pairs ≤T | unrelated pairs ≤T |
|---|---|---|
| 4 | 15.99% | 0.000% |
| **6** | **18.37%** | **0.033%** |
| 8 | 22.79% | 0.067% |
| 12 | 29.25% | 0.133% |

aHash needed a 1.37% false-positive rate to reach comparable recall — 4-5x
worse. A DCT pHash would likely beat both but needs numpy (a new runtime
dependency for one function) or a pure-Python DCT costing more than the JPEG
decode it follows.

### `draft()` is not an optimisation, it is the pass

| | ms/photo | full pass |
|---|---|---|
| With `Image.draft()` | **18.6** | **5.6 min** |
| Without | 52.2 | 15.8 min |

The actual run: **18,363 images hashed, 0 failures, 1,117 videos skipped, in
5.61 minutes** — within 1% of the estimate.

### The quality thresholds, and the trap in them

Sharpness percentiles over 1,149 photos stratified across every year:
p1 = 1.37, p5 = 3.23, median = 9.62. But **per-year** 5th percentiles range
from **1.83 (2011)** and 2.28 (2023) to **6.18 (2020)**.

A threshold at the whole-library p5 would have deleted roughly a fifth of 2011,
2018 and 2023 while touching almost nothing in 2020 — silently gutting the
early years. **1.5 sits below every single year's p5.** Soft photos are not
dropped at all; sharpness is also the ranking signal, so they simply rank
lower.

### What the guardrails actually cost

Over the whole 19,318-photo live pool:

| Gate | Removed | % |
|---|---|---|
| archived (chokepoint) | 162 | 0.8% |
| video | 1,117 | 5.8% |
| screenshot / wallpaper | 401 | 2.1% |
| below 480px short edge | 303 | 1.6% |
| out of focus | 131 | 0.7% |
| too dark | 85 | 0.4% |
| extreme aspect (>2.5:1) | 30 | 0.2% |
| too bright | 5 | 0.03% |
| **remaining** | **17,246** | **89.3%** |

Burst dedup then collapses **1,824 more (10.58%)** across 1,505 multi-photo
bursts, leaving 15,422. Largest burst after the pixel test: **12 photos** —
against 80 under time alone.

Every one of those buckets is reported to the user with a reason and an
example filename. Nothing is dropped silently.

---

## The verification run

53 memories into a gitignored `memories/`, covering all eight recipes.

**Offers available on this library: 480** — album_story 35, on_this_day 192,
on_this_month 12, person_years 38, pair_years 137, then_and_now 44,
year_in_review 19, place_cluster 3.

**37 general memories built** (16 skipped, all for overlapping an
already-accepted memory — the `Leh Ladakh` / `ladakh` shape doing exactly what
it should). GIF median 1,262 KB, MP4 total 298 MB, 10.5s per memory.

**Determinism: 37 specs rebuilt from a fresh index, 0 byte mismatches.**

**16 public-safe memories** built from the restricted 959-photo pool.

### The public-safe audit

Read back from the rendered JSON and checked against the **database**, not
through the engine that produced them:

```
public-safe memories audited : 16
shots checked against the DB : 148
VIOLATIONS                   : 0

normal memories audited      : 37
shots checked                : 789
ARCHIVED photos surfaced     : 0
stray PUBLIC-SAFE markers    : none
```

**The number that matters: 959 of 19,318 live photos (4.96%) are public-safe.**
8,433 live photos (43.7%) carry no face tag at all and are therefore *never*
publishable — Google's face tags cover 55.9% of this library, so an untagged
photo may contain anyone. Widening the rule to admit untagged photos is the
single most dangerous change available in this codebase.

Note that **zero** of the 37 general memories qualified as public-safe. A
memory needs *every* shot to qualify, and a 24-shot memory drawn from the full
pool essentially never does. Publishable memories have to be built with
`--public-safe`, which restricts the pool before selection — which is why that
flag exists rather than being a post-filter.

---

## The one lesson, again

**Every genuine bug in this milestone was found by running code, not by reading
it.** The specific mechanisms were mutation testing and a conformance run
against the real index.

### Four tests that could not fail

1. **`assert "fingerprint" in result.output`.** pytest names its temp directory
   after the test — `test_an_unfingerprinted_index_warns_...` — and that path
   was echoed in the CLI output. The assertion passed with the warning deleted.
   Found by mutating the call away and watching the suite stay green.
2. **A "distinct" fixture that wasn't.** `1 << i` hashes are 2 bits apart, so
   the dedup test asserting an 80-photo run survives was pinning a fiction —
   collapsing them was correct. The replacement has a measured minimum pairwise
   distance of 19, reproducing the real library's median of 22, and a test now
   guards the fixture itself.
3. **`assert duration >= 1.0`** for the ffmpeg concat tail repeat. Measured:
   2.04s with the repeat, 1.48s without. Both satisfy 1.0.
4. **A key-leak test using a fake transport.** It never reached
   `http_transport`, so a mutation echoing the request headers — including the
   `Authorization` bearer token — into the error message passed every
   assertion.

Also: a 4000x200 fixture that the *aspect* rule rejected while the test claimed
to be testing the *resolution floor*, and a music-ordering test that could not
fail because NTFS happens to return sorted directory entries.

### Two bugs only real data could find

**`then_and_now` advertised memories it could not build.** Every other recipe
hands the engine a generous pool and lets composition drop what cannot be
shown. This one picks exactly two photos, so if either is dropped afterwards
the memory dies. On the real index the earliest photo of `Abhik Maiti` is a
6928x2309 panorama, which the aspect gate rejects — so `rekindle memories`
listed three memories `rekindle memory` returned nothing for. The recipe now
narrows to showable photos *before* choosing its ends.

**`collapse()` emitted a photo once per appearance in its input**, so a recipe
handing it duplicates got a memory showing the same photo twice with dedup
having "run". Found by a deliberately rogue test recipe.

### A latent M0 defect, surfaced by asking a new question

`read_exif` recorded `Image.size` without consulting the EXIF orientation tag.
A camera stores a portrait photo as landscape pixels plus a tag, so **roughly
2,475 of 18,201 live images (13.6% of a 456-file sample) were indexed with
width and height swapped.**

It survived two milestones because `w*h` is invariant under the swap — even the
dedup tiebreak that reads those columns could not notice. Only a rule that asks
"is this taller than it is wide?" exposes it, and M2 was the first code to ask.

Fixed at source. Existing indexes are repaired in place by the fingerprint
pass, which decodes every image anyway, so no user has to re-index. **Only
orientations 5-8 exchange the axes** — the tempting `!= 1` test corrupts the
180-degree and mirrored cases, and both readings are pinned.

### Mutation discipline

Roughly 80 mutations were run across the milestone, each seen failing and
restored. Five survived and were investigated rather than excused:

- Two were **redundancies**, not gaps, per M1's ruling that "a mutation that
  cannot fail because the mutated code was a no-op is a redundancy": swapping
  the chronological ordering branch for the as-given branch is a no-op for
  recipes whose selection is already chronological (removing *both* does fail),
  and `_ranked`'s file_hash tiebreak is pre-applied by `chronological()` in
  every shipped recipe. The latter is defence in depth for an `AS_GIVEN` recipe
  and is now pinned by a direct unit test.
- Three were **real test defects**, listed above, and the tests were fixed.

One guard was **removed rather than tested**: `_under`'s
`except (OSError, ValueError)`, which existed for "mismatched drives on
Windows". Measured on 3.12, `Path.is_relative_to` returns False rather than
raising, so the handler was unreachable on every supported Python and no
mutation of it could fail. An untestable guard is a line nobody can maintain.

---

## Rulings

Decisions taken without stopping to ask, each with what it would cost if wrong.

**Never invent a fact, even where inventing is convenient** (3 rulings).
Albums are never merged automatically — `Leh Ladakh` / `ladakh` are one trip
and the three Kashmir albums are another, but no metadata says so, and
`Diwali 25` / `Diwali Kali Puja 22` are *different years* under an equally
similar pair of names. A place memory never names a place; there is no offline
gazetteer, so the title is fixed and the coordinates live in the fact sheet
where they can be checked. `Untitled(1)` albums are not offered rather than
being given an invented title.

**Default deny on anything that could publish a face** (1 ruling). Public-safe
requires a *non-empty* tag set that is a *subset* of the allow-list. Both
halves were attacked by mutation; both fail loudly.

**Structure over convention** (2 rulings). The guardrail is not a filter each
recipe applies — it is the only way to obtain a photo, and the test enumerates
every public method via `inspect` so a query added later without filtering
fails automatically. Dedup, the cap, ranking and composition live in the engine
for the same reason: a recipe cannot forget them.

**Prefer the conservative failure** (3 rulings). Dedup at threshold 6 rather
than 12. Quality gates below every year's 5th percentile. A caption verifier
that over-rejects — "Sunlight on the water" is refused because nothing
substantiates "Sunlight", and a rejection costs a fallback to a caption that is
always correct.

**Defer when the fix is wider than the problem** (2 rulings).
`metadata_conflict` is *not* split by cause despite the M1 entry asking M2 to:
nothing in the memory engine reads it, so splitting it would be a schema change
in service of no consumer — the exact pattern the `photo_paths` entry warns
against. Crossfades are documented as absent rather than half-built.

**Decline a permitted option on a ground the brief did not raise** (1 ruling).
The brief allowed pinned CC0 music URLs with verified SHA-256 checksums, and
network access was available. Declined anyway: a checksum pinned today is a
promise about a third-party host *forever*, and a host that relicenses or
expires a URL turns `rekindle memory` into a command that downloads an
unexpected binary or fails on first run — long after anyone remembers why the
list exists. Bring-your-own has no such tail and keeps the README's promise
that the default configuration makes no network calls. A test asserts the music
module contains no URL at all.

**Correct the documentation in the commit that makes it false** (1 ruling).
`writing-recipes.md` described `pydantic` params and an `index.search(text=...)`
that never shipped. It is rewritten alongside the code, not left for later —
this project has shipped that bug once already.

---

## What contributors should take from this

- **Point new code at a real library before trusting it.** Two of this
  milestone's bugs were invisible to 700 passing tests and obvious within one
  conformance run. See `tests/test_memory_conformance.py`.
- **Mutation-test anything that guards an invariant**, and when a mutation
  survives, find out *why* before accepting it. Three of five survivors here
  were real test defects wearing the costume of a redundancy.
- **Measure the threshold you are about to hardcode, per stratum.** A sharpness
  floor that looks fine on the whole library deletes a fifth of 2011.
- **A fixture must reproduce a shape you have observed.** Inventing one and
  pinning it with a test remains the most expensive mistake available here.
