# M2 implementation plan — the memory engine

Design: [2026-09-11-memories-design.md](../specs/2026-09-11-memories-design.md).

Ordering is load-bearing. **T1-T12 are the deterministic engine and must be
complete and green before T13 (the GPT layer) starts.** If time runs out, stop
after T12 and ship: a complete deterministic v1 with no LLM layer is the good
outcome.

Baseline to hold at every step: **263 passed, 9 skipped**, `ruff check` and
`ruff format --check` clean. Every task ends with the suite green and a commit.

Each task names the mutations to run — break the line the test protects, watch
it fail, restore it. A test not seen failing is not a test.

---

## T1 — schema v3 and the migration ladder

`db.py`. Add `phash INTEGER`, `sharpness REAL`, `phash_error TEXT` to `photos`.
`SCHEMA_VERSION = 3`. Replace `_migrate`'s `!= 1` guard with a ladder
`{1: _to_v2, 2: _to_v3}` applied while the stored version is a known step; an
unknown version is still left untouched for `__init__` to reject. Extend
`_insert`, `_row_to_photo` and `PhotoMeta` with the three fields.

**Tests** (`tests/test_db_v3.py`)
- v1 database migrates all the way to v3 in one open (the ladder's reason to exist).
- v2 database migrates to v3; existing rows keep every prior column value.
- v3 database opens unchanged; a v4 database raises "upgrade rekindle".
- An unknown version (99) still raises rather than being silently stamped.
- Round-trip: `phash`/`sharpness`/`phash_error` survive insert → read.
- `merge_meta` preserves a stored `phash` when a folder re-scan supplies none.

**Mutations:** drop `phash` from `_V3_COLUMNS`; make the ladder stop after one
step; drop `phash` from `_insert`'s column list; drop it from `merge_meta`.

---

## T2 — fingerprints

`memory/fingerprint.py`: `dhash(img) -> int`, `sharpness(img) -> float`,
`fingerprint(path) -> Fingerprint | FingerprintError`, and a resumable
`run_fingerprints(store, *, progress)` that skips rows already carrying a
`phash` or a `phash_error`, marks videos `"video"`, and batches writes.

**Tests** (`tests/test_fingerprint.py`)
- dHash of a known synthetic gradient is a known constant (pins bit order).
- A horizontally mirrored image hashes differently (proves it reads gradients).
- Identical bytes → distance 0; a solid black vs solid white image → a stable value.
- A blurred copy scores lower `sharpness` than its sharp original.
- Sharpness is computed at a fixed size: the same image at 2x scale scores within tolerance.
- Undecodable file → `phash_error`, no exception, row still written.
- Videos are marked `"video"` and never decoded.
- A second run is a no-op (nothing re-decoded) — asserted by counting decode calls.
- `draft()` is actually called on JPEG input.

**Mutations:** invert the dHash comparison; remove the resume skip; remove the
video guard; swap `<` for `<=` in the resume predicate.

---

## T3 — dedup

`memory/dedup.py`: `hamming`, `bursts(photos, gap, threshold)`,
`collapse(photos, ...) -> (kept, DedupReport)`.

**Tests** (`tests/test_dedup.py`)
- Two identical phashes 5s apart → one survivor.
- Two identical phashes **3 hours** apart → **both survive** (the brief's explicit case).
- Two *different* phashes 5s apart → both survive (time alone must not collapse).
- An 80-photo run of distinct hashes 30s apart → 80 survivors (the real-library shape).
- Anchor semantics: A≈B, B≈C, A far from C → C starts a new burst.
- A photo with `phash=None` never joins and never absorbs.
- Winner is the sharpest; on equal sharpness the larger; on both equal the
  lexicographically smallest `file_hash`.
- NULL `width`/`height` (the real video/undecoded shape) sorts last, never crashes.
- Determinism: shuffling the input gives byte-identical output.

**Mutations:** drop the phash condition (leaving time only); drop the time
condition; compare to previous instead of anchor; remove the final hash tiebreak.

---

## T4 — policy and the public-safe rule

`memory/policy.py`: `ExclusionPolicy`, `deny_reason(photo)`, `is_public_safe`,
`load_policy(path)`, `append_exclusion(...)`.

**Tests** (`tests/test_policy.py`)
- Archived denied; trashed denied; **no config can re-admit either**.
- Person exclusion denies a photo where the excluded person is one of five.
- Date exclusion uses `taken_at_local`, is inclusive at both ends, and a photo
  one day outside survives.
- Album exclusion is casefolded; path exclusion matches a parent directory but
  not a sibling with a shared prefix (`/a/photos2` vs `/a/photos`).
- Public-safe: empty tags → **denied**; `{Abhik}` → allowed; `{Abhik, Paramita}`
  → denied; `{Paramita}` → denied.
- Missing config file → permissive default, no error. Malformed TOML → raises.
- `deny_reason` returns the *first* matching reason, in documented order.

**Mutations:** flip `bool(people)` off in `is_public_safe` (the dangerous
widening — must fail loudly); change `<=` to `&` (intersects); remove the
archived check; make the date bound exclusive.

---

## T5 — the MemoryIndex chokepoint

`memory/index.py`. Loads, filters once, exposes only query methods:
`all()`, `by_year()`, `by_month_day()`, `by_person()`, `by_pair()`,
`by_album()`, `by_gps_cell()`, `people_counts()`, `album_counts()`,
`resolve_path(photo)` (wired to `photo_paths`, settling the known-limitation).

**Tests** (`tests/test_memory_index.py`)
- **The guardrail test:** build a store containing an archived photo, an
  excluded person and a normal photo; assert *every public query method* returns
  only the normal photo. Written as a loop over `dir(MemoryIndex)` so a method
  added later without filtering fails this test automatically.
- `ExclusionReport` counts by first-matching reason and the counts sum to the
  number excluded.
- Public-safe mode removes non-qualifying photos from every query.
- An index over an empty store answers every query with an empty list.

**Mutations:** make one query method read the store directly (the loop must
catch it); drop the archived filter; drop the report increment.

---

## T6 — spec, facts, captions

`memory/spec.py` (`MemorySpec`, `Shot`, `FactSheet`, JSON round-trip),
`memory/captions.py` (all title/caption derivations from §7).

**Tests** (`tests/test_spec.py`, `tests/test_captions.py`)
- `MemorySpec` → JSON → `MemorySpec` round-trips exactly; JSON keys are stable.
- Two runs over the same input produce byte-identical JSON.
- `FactSheet` from photos with no GPS has **no place field at all** (not an
  empty one, not a guess).
- `FactSheet` never contains a filesystem path (asserted by scanning the
  serialised form for the drive/root prefix) — it is what the LLM sees.
- "6 years ago today" is arithmetic; at 1 year it says "1 year ago today".
- `public_safe` is true only when every shot qualifies; one bad shot flips it.

**Mutations:** emit a placeholder place name when GPS is absent; make
`public_safe` an `any()` instead of an `all()`.

---

## T7 — the eight recipes

`memory/recipes/` — one module per recipe plus `registry.py`.

**Tests** (`tests/test_recipes.py`, one class per recipe)
For every recipe: deterministic offer ordering; an empty index yields no offers;
`select` on a stale offer returns `None` rather than raising; thresholds are
respected at the boundary (n-1 excluded, n included).

Per-recipe specifics:
- `album_story`: auto `Photos from YYYY` excluded; `Untitled(1)` excluded;
  a name starting with `,` excluded; `Kashmir` and `ladakh` remain **separate**
  offers (the no-merge decision, pinned).
- `on_this_day`: only (m,d) in ≥3 years; a leap day is handled, not crashed.
- `person_years` / `pair_years`: pair key is order-independent; `A+B` and `B+A`
  produce exactly one offer.
- `then_and_now`: exactly 2 shots, ≥1 year apart, `ordering="as_given"`.
- `year_in_review`, `on_this_month`, `place_cluster`: cell rounding is stable at
  a negative longitude; a visit gap >14 days splits.

**Mutations:** remove the auto-album filter; make the pair key order-dependent;
drop the ≥3-year threshold; remove the `then_and_now` year separation.

---

## T8 — the engine

`memory/engine.py`: `build(index, offer)` → dedup → rank → cap → order →
caption → `MemorySpec`; `build_all`, `offers_for_today`.

**Tests** (`tests/test_engine.py`)
- **The pipeline guarantee:** a recipe that returns 100 photos including
  duplicates yields ≤ `max_shots` distinct shots — asserted for *every
  registered recipe* via the registry, so a new recipe cannot opt out.
- A recipe returning a blocked photo cannot: it has no way to obtain one
  (asserted by giving a fake recipe an index and checking the blocked photo is
  absent from what it can see).
- Ranking is deterministic under input shuffling.
- Chronological ordering is restored after the cap.
- Below `min_shots` → no memory, counted.
- Missing file at render time is dropped and counted, not raised.

**Mutations:** skip dedup in the pipeline; skip the cap; remove the re-order.

---

## T9 — rendering

`memory/render/frames.py`, `gif.py`, `mp4.py`, `music.py`.

**Tests** (`tests/test_render.py`)
- A GIF is produced from a 3-shot spec, opens in Pillow, has the expected frame
  count, is animated, and loops.
- Frame count is capped independently of shot count.
- Portrait and landscape inputs both fit the canvas; aspect is preserved;
  letterbox is black.
- EXIF orientation 6 is applied (a wide image becomes tall).
- A missing file is skipped, counted, and does not abort the render.
- Title card renders with no font file present.
- **ffmpeg absent (monkeypatched `which` → None): GIF only, message, exit 0, no
  exception.** This is the degradation guarantee.
- ffmpeg present but failing (fake binary, non-zero exit): GIF survives, stderr
  tail reported.
- `music.py`: empty `music/` → silence, no error; two files → the
  lexicographically first, deterministically; `--music` overrides. **No network.**
- The real-ffmpeg MP4 test is `skipif not shutil.which("ffmpeg")`.

**Mutations:** remove the `which` guard (must crash the absent-ffmpeg test);
remove the frame cap; remove the missing-file guard.

---

## T10 — CLI

`fingerprint`, `memories`, `memory`, `watch`, `exclude`. README and
`writing-recipes.md` rewritten to match reality in this task's commit.

**Tests** (`tests/test_memory_cli.py`)
- Each command on a missing index → exit 2, **no database file created**.
- `memories` lists offers grouped by recipe and exits 0 on an empty library.
- `memory --recipe X --key Y` writes `memory.json` + `memory.gif` to a temp dir.
- `--public-safe` writes the `PUBLIC-SAFE` marker only when it qualifies.
- `exclude --person NAME` appends and preserves hand-written keys in the file.
- `--help` for every command (catches a typer signature error at import).

**Mutations:** remove the index-existence check (must leave a stray DB and fail).

---

## T11 — watch

`memory/watch.py`, injected clock and sleep, `--once`.

**Tests** (`tests/test_watch.py`)
- `--once` runs exactly one cycle with a fake clock; **no test sleeps**.
- Today's anniversary is printed with the exact command to render it.
- It **renders nothing** — asserted by pointing it at a temp output dir and
  checking the dir stays empty.
- A changed mtime triggers a re-index; an unchanged library does not.
- The daily check fires once per calendar day, not once per poll (advance the
  fake clock 25 hours across three polls; assert exactly two checks).

**Mutations:** remove the once-per-day latch (the 25-hour test must fail).

---

## T12 — real-data verification

`tests/test_memory_conformance.py`, skipped without `REKINDLE_MEMORY_DB`, plus a
scratch verification run producing ~50 memories into gitignored `memories/`.

Measure and record in `docs/decision-log-memory-engine.md`: offers per recipe,
dedup rate, public-safe memory count, render sizes, timings. **Then re-read the
design doc and correct any number that moved.**

Also here: delete `llm.py` and the flag, run the suite, confirm green. That is
the proof the LLM layer is additive — run *before* it exists, as the empty case.

---

## T13 — the GPT caption layer (only after T12 is green)

`memory/llm.py`, `--captions gpt`, model `gpt-5.6-luna`, low reasoning effort,
key from `OPENAI_API_KEY`.

**Tests** (`tests/test_llm.py`) — **no network in any of them**
- Fact sheet in, caption out, via a fake transport.
- A caption asserting a year not in the fact sheet is **rejected**.
- A caption naming a person not in the fact sheet is **rejected**.
- Over-length and multi-line captions are normalised or rejected.
- Missing key → deterministic captions, warning, exit 0.
- Transport raising → deterministic captions, exit 0.
- **The key never appears** in any log line, spec, or exception message.
- Default (no flag) makes **zero** transport calls.

**Mutations:** remove the year validator; remove the key-absent fallback.
