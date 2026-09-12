# Decision log — making `MemoryIndex` lazy

> *"The MemoryIndex materializes the entire library in memory and every recipe
> scans it several times. You justify it explicitly at 19,480 rows, and you're
> right at that size — but a 300k photo library is a different regime, and
> that's the first thing a heavy user will hit. The fix isn't SQL-per-recipe
> (you're correct that reintroduces the bypass problem); it's keeping the
> chokepoint and making it lazy behind the same interface."*

Correct on every clause, including the one that rules out the easy fix. This
is what was built, what it cost, and the numbers.

Everything here was measured on this machine with
`tests/bench_memory_index.py`, against two libraries: the reference index
(19,480 rows, 19,318 usable) and a **synthetic 300,000-row** library generated
by the same script. Re-measure with:

```
python tests/bench_memory_index.py --rows 300000 --db <scratch>/big.sqlite --generate
```

Memory is `tracemalloc` — the Python heap, which is the thing being measured.
Timings are from separate untraced runs, because `tracemalloc` costs roughly
2.5x wall-clock and reporting both from one run would overstate every time.

---

## 1. Where the bytes actually were

The first thing measured, before anything was designed, and it changed the
design. On the synthetic 300k library, with the materialised index:

| | |
|---|---|
| `Photo` objects for 297,585 usable rows | **681.2 MB** |
| ...per photo | **2,289 bytes** |
| The eight derived indexes over them (`_by_year`, `_by_album`, …) | **32.5 MB** |
| Total after `MemoryIndex.open` | **713.7 MB** |

**The photos are the problem; the indexing is not.** A design that compacted
the indexes and kept the photos would have bought 5%. So the indexes hold
SQLite **rowids** — eight bytes each, in `array("q")` — and `Photo` objects
are fetched back only when a caller reads one.

Per-photo, the 2,289 bytes break down as roughly 581 B of JSON-decoded lists
(people, albums, keywords, face regions, paths), 561 B of strings retained
from the row (the 64-character `file_hash` alone is 113 B), 404 B of
`Photo`/`PhotoMeta` object overhead, and 184 B of `Path` objects. There is no
single field to drop.

## 2. What was built

`_StoreSource` owns a **read-only, thread-local** SQLite connection and is the
only thing that turns a row into a `Photo`. It applies
`ExclusionPolicy.deny_reason` on the opening scan **and again on every later
fetch**. That second application is the load-bearing one: with the library no
longer materialised, "the index does not contain a rejected photo" is no
longer what makes the guarantee true. What makes it true now is that exactly
one function builds a `Photo` and it refuses.

The connection is its own rather than the store's for two reasons that are
both real: `web/library.py` closes the store the instant the index is built,
and the same index is then read from many request threads.

Four things had to change beyond the index itself:

* **`iter_all` / `iter_images`.** `rekindle memories` printed a warning
  computed by `sum(1 for p in index.all() if p.meta.phash is None)`. On a 300k
  library that is 690 MB paid before a single memory is built, to produce one
  integer. The streaming variants fetch in chunks and do not admit them to the
  cache. Three call sites moved over; `all()` and `images()` are unchanged and
  still return a real list.

* **`recurring_event`.** The one recipe that materialised the library from
  *inside* the guardrail, and therefore the one place laziness could not help:
  it called `index.images()` once per offer and once per select. It now reads
  `index.image_day_counts()` — a spine aggregate, no photographs — to find its
  bursts, and asks `by_date` only for the days those bursts cover.
  `recurring.bursts_from` / `events_from` exist for that: the density rule was
  always a question about a histogram.

* **The offers phase of five more recipes.** Measured after the first version
  of this change, `all_offers` at 300k had gone from 4.1 s to 185.9 s. Five
  recipes were asking for a slice of the library to compute a count and a set
  of distinct years and then discarding the photographs, so the index answers
  both off the spine now. See §4a — that is where the number and the reasoning
  live, because it is a trade, not a tidy-up.

* **`MemoryIndex.close()`.** Windows will not delete an open file, and a test
  that builds an index inside a `TemporaryDirectory` has no other way to let
  it be cleaned up. Reopening is transparent, so `close()` releases the file
  rather than ending the index.

**Per-recipe SQL was not considered.** The owner's brief ruled it out and was
right to: it is the "filter each recipe remembers to apply" the chokepoint
exists to prevent. The filter moved *closer to the data*; it did not multiply.

## 3. Memory: the index stopped scaling with the library

| | 19,480 rows (real) | | 300,000 rows (synthetic) | |
|---|---|---|---|---|
| | before | after | before | after |
| Peak during `MemoryIndex.open` | 45.4 MB | **3.2 MB** | 713.7 MB | **46.7 MB** |
| Resident after `open` | 45.3 MB | **2.1 MB** | 711.1 MB | **29.7 MB** |
| Peak over `open` + a full `--all-recipes` pass | 47.3 MB | **44.5 MB** | 731.8 MB | **469.6 MB** |
| Resident at the end of that pass | 46.5 MB | **43.9 MB** | 714.0 MB | **184.4 MB** |

**The index itself is 15.3x smaller at 300k and 14.3x smaller at 19k**, which
is the thing the observation was about. The whole pass is 1.56x smaller.

The §4a offers change, which removed roughly half the hydrations a full pass
does, moved the 300k full-pass peak by **1.2 MB** — 468.4 MB before it,
469.6 MB after. That is worth stating because it is the opposite of what one
would guess: the peak is not set by how many photographs the offers phase
loads, because it drops them as it goes. It is set by the build phase and by
the cache.

With the library no longer resident, the peak of a full pass is set by the
build phase — one `on_this_month` memory alone runs 24,800 freshly-created
photographs through compose, dedup and diversity — and by the 60,000-photo
cache. Neither is the index, which is why the whole-pass figure improves so
much less than `open` does.

The full-pass peak at 19k is 6% *lower* after the change, at 44.5 MB against
47.3 MB, and that is a smaller claim than it looks: the whole reference
library fits inside the default cache, so the pass holds much the same
photographs it always did, plus a 2 MB spine, minus the ones the offers phase
no longer loads at all. A library that fits in the cache keeps every bit of
the old behaviour. That is deliberate (see §5).

## 4. Time: `open` got faster, `all_offers` got slower

**Read the caveat in §4b before quoting any number in this section.**

`MemoryIndex.open`, three interleaved before/after rounds, medians:

| | before | after | |
|---|---|---|---|
| 19,480 rows | 0.914 s | **0.820 s** | 10% faster |
| 300,000 rows | 19.38 s | **13.14 s** | **32% faster** |

Opening is faster because nothing survives it. The scan still builds a `Photo`
for every row so that `deny_reason` can be applied to it, but it drops each
one immediately instead of threading 297,585 of them onto a structure the
garbage collector then has to trace for the rest of the run.

A full `--all-recipes --limit 1` pass, untraced:

| | 19,480 rows | | 300,000 rows | |
|---|---|---|---|---|
| | before | after | before | after |
| `MemoryIndex.open` | 1.17 s | 0.96 s | 14.99 s | **9.20 s** |
| `all_offers` | 0.96 s | 1.74 s | 4.06 s | **46.48 s** |
| `build_all` | 19.1 s | 16.8 s | 51.23 s | 59.40 s |
| **total** | **21.2 s** | **19.7 s** | **70.28 s** | **115.07 s** |

(The 300k columns are the mean of two consecutive rounds on a quiet machine —
before 70.00 / 70.55 s, after 114.61 / 115.53 s. See §4b for why the earlier
sequential numbers were larger and one of them was wrong.)

**At 19,480 rows the whole pass is 7% faster.** That is the case every user
has today, and it was not the goal.

**At 300,000 rows the pass is 1.64x slower, and all of it is one phase.**
`all_offers` goes from 4.1 s to 46.5 s while `open` gets 39% faster and
`build_all` is within 16%. That is the failure the brief warned about,
arriving in the one place the design did not anticipate it: the offers phase
asks every recipe for every slice of the library, uses each slice to compute a
*count* and a *set of years*, and throws the photographs away.

### 4a. What the offers phase does instead now

The first version of this change left that regression at **25x** (185.9 s).
Six of the nine recipes were fixed rather than documented, because the fix
is small and the measurement made it obvious:

* `recurring_event` called `index.images()` once per offer AND once per
  select. It reads `image_day_counts()` now and asks `by_date` only for the
  days its bursts cover.
* `on_this_day`, `on_this_month`, `person_years`, `pair_years` and
  `year_in_review` wanted nothing from their slice but a count and a set of
  distinct years, so the index answers both off the spine:
  `year_counts()`, `month_counts()`, `month_day_counts()`, `month_years()`,
  `month_day_years()`, `person_years()`, `pair_years()`, `album_years()`.
  Distinct years cost a four-byte-per-photo array aligned with the rowid
  spine and a bisect - 1.2 MB across a 300k library.

`year_in_review` is the case that makes the point, and it is worth its own
table because `--all-recipes` is the *worst* case for a lazy index and the
best case for a materialised one — so measuring only that answers a question
nobody asks. One recipe, one memory, on the 300k library, traced:

| | before | after, pre-fix | after |
|---|---|---|---|
| `offers()` | 0.014 s | **38.65 s** | **0.010 s** |
| total | 103.6 s | 134.7 s | **98.4 s** |
| peak | 714.9 MB | 212.0 MB | **63.4 MB** |

Before the fix, `offers()` spent **38.7 seconds hydrating 298,000 photographs
to produce 24 integers** `year_counts()` already had. After it, the run a
person actually types — `rekindle memory --recipe year_in_review --key 2017` —
is **faster than the materialised index was** and uses **11.3x less memory**.
That is the shape the whole change was for: you pay for what you touch.

**Three still load photographs to build an offer, and all three are named
rather than hand-waved.** `then_and_now` is the big one: it runs `compose()`
over every person and every album to find out which photographs would
survive the guardrails, because it picks exactly two and a dropped one kills
the memory. That needs width, height, sharpness, brightness and media type
per photograph — roughly 570 k hydrations at 300k, and most of the 46.5 s
that remains. `album_story` calls `captions.subtitle_for`, which wants a span
and a description of the people in the slice. `place_cluster` splits a GPS
cell into visits by timestamp, and is the smallest of the three because only
12% of a library has GPS. Serving the first means putting the composition
scalars on the spine, which is a bigger change than this one and is written
up in `known-limitations.md`.

### 4b. One of these numbers was wrong until it was measured properly

This machine was shared with other multi-gigabyte processes for most of this
work. The first, *sequential* sweep read `open` at 300k as 17.9 s before and
22.3 s after, and an earlier draft of this document said "25% slower" —
**which was contention, and the opposite of the truth.** Three interleaved
rounds of the same comparison give 19.4 s before and 13.1 s after; two clean
rounds later give 15.0 s and 9.0 s. A sequential sweep on a busy machine
measures drift as well as the change.

Every timing here is therefore either an interleaved median or a pair of
consecutive rounds on a quiet machine, and the ratios are more trustworthy
than the absolute seconds. Memory is unaffected by contention.

**A figure in this project has failed re-measurement twelve times; this is
the thirteenth, caught before it shipped.**

### 4c. It is not the synthetic library's fault — checked, and it isn't

The obvious excuse was that the synthetic library's capture dates are
uniformly random per row, so rowid order carries no information about date and
every date-keyed query scatters across the whole table — whereas on the
reference library **93.9% of adjacent rows are already in capture order**,
because `rekindle index` walks Takeout's per-year folders. That would make the
benchmark an unfair worst case.

It was measured rather than assumed. `--chronological` generates the realistic
shape; on a 300,000-row library built that way, **before the §4a fix** (so the
comparable random-order figures are 185.9 s and 303.4 s):

| | before | after, pre-§4a |
|---|---|---|
| `all_offers` | 5.6 s | **172.4 s** |
| total | 93.7 s | 293.8 s |

Ordering buys 7%. The excuse does not hold, and the paragraph that assumed it
would was wrong before this run replaced it. The cost is not locality: it is
that `then_and_now` sweeps every person and every album, and `by_person` and
`by_pair` scatter no matter how the table is ordered.

## 5. Why the cache is 60,000 photos

The cache is bounded in photographs, not bytes, and the default is 60,000 —
about 138 MB at the 2,289 bytes measured above.

It is chosen so that **every library up to 60,000 photographs behaves exactly
as the materialised index did**: one pass fills the cache and no query after
that touches SQLite. Making ordinary libraries slower in order to make a large
one possible would be a bad trade, and making it silently would be worse.
Above 60,000 the cache becomes a working set and the index starts re-reading;
a full re-read of 300,000 rows costs about 9.4 s.

Two properties keep that honest, and both have a test that fails without them:

* **Scan resistance.** A read larger than the cache is not admitted at all. An
  `index.all()` would otherwise evict every recipe's working set on the way
  past, in order to cache photographs it is about to drop.
* **Streaming does not cache.** `iter_all` and `iter_images` never admit,
  whatever the cache size.

## 6. What did not change, and how that was checked

**Byte for byte.** Every offer, every memory, every shot list, every index
key and every count, dumped as JSON and compared:

| library | memories | offers | result |
|---|---|---|---|
| 19,480 real rows | 27 | 490 | **identical** |
| 300,000 synthetic rows | 7 | 7,768 | **identical** |

Both dumps include `all_hashes` in scan order, `album_merges`, all four
`*_counts` maps, `dates()`, and the full shot list of every memory built. The
files are md5-identical between the pre-change tree and this one, and the
comparison was re-run after the §4a offers change - which touched six recipes
and therefore had every opportunity to move a subtitle or an offer order -
and was identical again.

**The guardrail-enumeration test still passes**, and got stronger. It sweeps
every public method, so `iter_all` and `iter_images` joined the sweep
automatically — but a *generator* was being treated as one opaque object, so
the sweep ran every assertion against the generator and passed without ever
producing a photograph. `_touch` now drains iterators. Proved, not assumed:
making `iter_all` bypass the guardrail is caught with the drain in place and
**survives without it**.

## 7. Mutations run

Every line below was broken, the tests run, and the line restored. Nineteen
of twenty were caught; the twentieth is a demonstration rather than a defect.
Mutations 1-13 are the lazy index, 14-20 the spine aggregates of §4a.

| # | mutation | caught by |
|---|---|---|
| 1 | `_StoreSource.fetch` stops applying the policy | `test_no_query_method_can_return_a_blocked_photo` |
| 2 | `_StoreSource.scan` stops applying the policy | `test_person_exclusion_is_honoured_by_every_index` |
| 3 | merged album lists are not sorted back into scan order | `test_a_merged_album_comes_back_in_the_order_the_library_is_scanned_in` |
| 4 | the scan-resistance guard is removed | `test_the_cache_is_bounded_and_queries_are_still_complete` |
| 5 | the LRU bound is removed | `test_the_cache_is_bounded_and_queries_are_still_complete` |
| 6 | `iter_all` admits its chunks to the cache | `test_streaming_never_admits_a_photo_to_the_cache` |
| 7 | the opening scan drops `ORDER BY rowid` | `test_a_merged_album_comes_back_in_the_order_the_library_is_scanned_in` |
| 8 | `image_day_counts` counts videos too | `test_image_day_counts_counts_images_only_and_matches_by_date` |
| 9 | `RecurringEvent._images_of` stops filtering to images | `test_a_video_inside_the_burst_never_reaches_the_memory` |
| 10 | `years_available` is computed from the event's own photos | `test_every_year_is_a_claim_about_the_library_not_about_the_event` |
| 11 | `MemoryIndex.close` does nothing | `test_config.py::…[selection.min_shots]` (Windows) |
| 12 | `iter_all` bypasses the guardrail | `test_no_query_method_can_return_a_blocked_photo` |
| 13 | …and the sweep stops draining iterators | **nothing — by design** |
| 14 | `_years_in` reads by rowid instead of by position | `test_no_query_method_can_return_a_blocked_photo` |
| 15 | `_years_in` is off by one | `test_every_spine_aggregate_agrees_with_the_photos_it_summarises` |
| 16 | a dateless row does not append to `_years`, so it drifts | `test_a_dateless_photo_does_not_misalign_the_year_array` |
| 17 | `month_day_counts` counts months instead of month-days | `test_every_spine_aggregate_agrees_with_the_photos_it_summarises` |
| 18 | `year_counts` counts the whole library for every year | `test_every_spine_aggregate_agrees_with_the_photos_it_summarises` |
| 19 | `year_in_review` reads the wrong count | `test_the_offer_count_is_the_real_number_of_photographs` |
| 20 | `person_years` goes back to hydrating the slice | `test_offers_does_not_hydrate_a_single_photograph[person_years]` |

Three of those found real defects in this work rather than confirming it:

* **Mutation 7 found dead code.** It was aimed at a `PhotoStore.iter_rows`
  that had been added for the scan and then never wired up, because
  `_StoreSource.scan` grew its own query. The method was deleted, the
  `ORDER BY` reasoning moved to where the query actually is, and the mutation
  re-aimed.
* **Mutation 10 is a bug this work would have introduced.** Handing
  `_title` only the event's own photographs makes `years_available` equal to
  `len(event.years)` by construction, so every recurring event would have been
  titled "every year" whether or not it was.
* **Mutation 3 and mutation 6 were both surviving** on the first run, and both
  because the test that should have caught them was satisfiable another way -
  a merged album whose two spellings did not interleave, and a cache too small
  for the admission rule to be what kept it empty. Both tests were rewritten
  to be capable of failing.
* **Mutations 16, 19 and 20 all survived their first run too**, and the three
  reasons are worth keeping apart. 16 looked like dead code - `deny_reason`
  rejects a dateless photograph so `MemoryIndex.open` can never produce one -
  until the direct constructor, which deliberately does not apply the policy,
  turned out to reach it; without the alignment line `album_years` returns
  the wrong years for everything after the gap rather than failing. 19 had no
  test because the count only reaches a subtitle and an offer's `size`, and
  nothing compared either against the real number. 20 is the interesting one:
  reverting `person_years` to hydrate its slice changes NO OUTPUT, so no
  determinism check and no conformance sweep can see it - the whole of §4a is
  invisible to every test that looks at what is produced. It needed a test
  that looks at what is *read*, which is
  `test_offers_does_not_hydrate_a_single_photograph`, with `album_story` as
  the negative control that must still hydrate.

### 7a. Six failures in `test_memory_cli.py` that are not this work

The suite ends 6 failed / 2,415 passed on a `uv sync` with no extras. All six
are in `test_memory_cli.py` and all six were checked against the pre-change
source before this was committed:

* Five are the semantic-extra tests that assume `onnxruntime` is present.
* The sixth, `test_the_weak_path_is_named_when_no_source_describes_the_prompt`,
  is **order-dependent**: it passes in a full-suite run and fails when
  `test_memory_cli.py` runs alone, at the pre-change source as well as at
  this one. Running the file by itself at `f8ae624` with none of this work's
  tests collected reproduces it, which is what rules this work out as the
  cause. Its prompt is refused by the judge rather than reaching the warning
  the test asserts on.

Recording that here rather than in the commit message because "six failures,
none of them mine" is a claim that needs its experiment written down; the
experiment is `pytest tests/test_memory_cli.py` on a clean checkout.

## 8. What this does not fix

* **The index is a snapshot.** It always was, but the failure mode changed: a
  stale materialised index showed old data, while a stale rowid could in
  principle address a *different* row, because `_insert` is INSERT OR REPLACE
  and REPLACE allocates a new rowid (verified, not assumed). Nothing in
  rekindle writes the `photos` table while an index is open — the fingerprint
  and orientation passes use targeted UPDATEs, which keep the rowid, and the
  web UI writes only to other tables — so this is a documented constraint
  rather than a live hazard.
* **`all_offers` is still 11x slower at 300k** — 46.5 s against 4.1 s — and
  it is the whole of the 1.64x on a full pass. Three recipes are left:
  `then_and_now`, which needs the composition scalars on the spine,
  `album_story`, which needs a spine subtitle, and `place_cluster`, which
  needs timestamps per GPS cell. §4a has the detail.
* **`open` could be about twice as fast again** with a narrowed `SELECT`, at
  the cost of a correctness hazard that needs its own test. See §4.
* **`images()` and `iter_images()` now have no production caller.** Fixing
  `recurring_event` took the last one. Both are part of the documented recipe
  API, both are swept by the guardrail test and both are covered — but
  mutation 7 in §7 is exactly the lesson that an addition nothing calls is
  worth naming rather than leaving to be discovered. `images()` predates this
  work; `iter_images()` does not, and it is here because shipping the
  streaming half of a documented pair while omitting the other half is a
  worse API than either.
* **`build_all` was always the other half of the bill and still is.** 51.2 s
  before this work and 59.4 s after at 300k, essentially all of it inside
  `diversity.pick`, which is O(slots x candidates) per memory and decodes a
  colour histogram from hex on every comparison. Independent of the index and
  largely untouched here.
