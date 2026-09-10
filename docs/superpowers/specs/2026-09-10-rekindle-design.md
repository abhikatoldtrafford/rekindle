# rekindle — design spec

**Date:** 2026-09-10
**Status:** draft, pending review
**License:** MIT

## 1. Purpose

Turn a personal photo library into **memories**: short narrated montages of related photos, set to music. A memory is produced either on demand from a natural-language prompt ("Goa trip 2014", "me and Mom over the years") or automatically on a schedule (anniversaries, "N years ago today").

rekindle is a local-first, open-source tool. It must be useful to people other than its author, on machines without a GPU, without access to the author's photos, and without API keys.

### Non-goals

- Not a photo manager, editor, or backup tool.
- Not a cloud service. No hosted multi-tenant deployment in v1.
- Not a face-recognition system. Person identity comes from existing tags.

## 2. Constraints

### 2.1 Google's API shutdown (external, non-negotiable)

Google removed `photoslibrary.readonly`, `photoslibrary.sharing` and `photoslibrary` on 2025-04-01. No scope reads a user's existing library. Face groupings and content labels were never exposed by any API.

**Consequence:** Google Takeout is the primary ingestion path. The Picker API (`photospicker.mediaitems.readonly`) is a metadata-poor incremental supplement. See [connecting-google-photos.md](../../connecting-google-photos.md).

### 2.2 Open-source constraints

- **CI has no photos, no GPU and no API keys.** Every provider needs a deterministic fake, and test data must be synthetically generated.
- **Cross-platform**: Windows, macOS, Linux. `pathlib` throughout, discovered ffmpeg, no drive letters, CPU fallback for every model.
- **No hardcoded paths or personal assumptions.** All config via TOML + env.
- **Scale beyond the author's library.** Disk-backed index, not in-memory.

### 2.3 Reference hardware

Author's machine: RTX A4000 16GB, Ryzen 5 5600GT, 96GB RAM, Python 3.13, ffmpeg present. Library ~10–50k photos. This is the *upper* comfort target, not the minimum. Minimum target is a CPU-only laptop.

## 3. Architecture

```
sources/  ->  enrich/  ->  index/  ->  memory/  ->  render/
Takeout       junk         LanceDB    recipes     MemorySpec
Picker        embed                   router      web player
folder        caption                 narrate     ffmpeg mp4
              quality                 guards
```

Three protocols carry all extensibility:

| Protocol | Implementations | Contributor story |
|---|---|---|
| `Source` | takeout, picker, folder | Add Immich, Apple Photos, Nextcloud |
| `Embedder` / `Captioner` / `Narrator` | siglip-local, openai, ollama, fake | Add a model backend |
| `Recipe` | trip, anniversary, before_and_now, year_in_review, freeform | **Add a memory type** |

`Recipe` is the primary contribution surface and the one documented most carefully.

## 4. Data model

### `Photo`

Source-agnostic normalised record. Identity is the **content hash** (BLAKE2b of pixel bytes), which de-duplicates the same photo appearing in year and album folders, and makes indexing idempotent and resumable.

```
id (content hash), path, source, source_id
taken_at (UTC), taken_at_local, tz_source
gps (lat, lon, alt) | None
people[] (names, from Takeout - no bounding boxes)
description, favorite, albums[]
camera_make, camera_model, width, height
derived: junk_score, junk_reason, quality_score, sensitive_flags[]
         embedding_id, caption | None
```

### `MemorySpec` — the central contract

**The engine's output is a JSON document, not a video.** The web player and the ffmpeg exporter are independent consumers of the same document.

```
id, title, subtitle, recipe, params, created_at
shots[]      -> photo_id, start_ms, duration_ms, transition,
                ken_burns(from_rect -> to_rect), caption
music        -> track_id, in_point, beat_cues[]
narration[]  -> segment text + grounded_in[] (source fields)
provenance   -> inputs, filters applied, model + prompt versions
```

Why this matters:

- Memories are cacheable, diffable and re-renderable without re-running models.
- A TTS track is a new field, not a rewrite.
- Tests assert on a `MemorySpec` without encoding a frame of video.

### `FactSheet`

The **only** input to narration. A validated, pre-filtered set of facts extracted from selected photos. The LLM never sees the library, never chooses photos, and cannot assert anything absent from this structure.

## 5. Ingestion and enrichment

### 5.1 Takeout parser (highest-risk component)

Resolves sidecars through a fallback ladder: exact -> `supplemental-metadata` variant -> truncation-aware -> duplicate-counter permutations -> fuzzy. Reports coverage honestly; unmatched files are logged, never silently dropped.

Handles: `photoTakenTime` vs `creationTime`, `-edited` files without sidecars, album/year duplication, sidecars split across archives, zeroed `geoData` with populated `geoDataExif`, UTC epoch timestamps needing GPS-based timezone resolution.

Exposed as `rekindle doctor`, which validates an export before indexing.

**This component gets the heaviest test coverage in the project.**

### 5.2 Junk filtering

Runs before any expensive model. Typically removes 30-50% of a library and is the single biggest quality lever.

| Filter | Method |
|---|---|
| Screenshots | Exact device-resolution match + absent camera EXIF |
| Documents / receipts / memes | CLIP zero-shot against a small label set |
| Blur | Laplacian variance, threshold per resolution |
| Burst near-duplicates | Embedding cosine + capture-time proximity, keep sharpest |

### 5.3 Embedding

SigLIP via `transformers`, batched on GPU, smaller checkpoint auto-selected on CPU. Checkpointed and resumable, content-hash keyed. Roughly 1 hour for 40k photos on an A4000.

Captions are **not** generated here — only for photos that reach a memory.

### 5.4 Sensitive-context detection

Zero-shot pass against a separate label set (hospital, funeral, cemetery, accident). Flags, never deletes. Flagged photos require explicit confirmation before inclusion in a memory.

### 5.5 Storage

LanceDB — embedded, disk-backed, no server, scales past 400k photos.

## 6. Memory engine

**Approach: recipes as skeleton, LLM as router and writer.** Selection and ordering are deterministic Python. The LLM does two bounded jobs.

### 6.1 Router

Prompt -> recipe + typed params, via structured outputs. `"Goa trip 2014"` becomes `TripRecipe(place="Goa", year=2014)`. Low reasoning effort; falls back to `FreeformRecipe` on low confidence.

### 6.2 Recipes

```python
class Recipe(Protocol):
    name: str
    params_model: type[BaseModel]

    def candidates(self, params, index) -> list[Photo]: ...
    def arrange(self, photos, params) -> list[Shot]: ...
    def fact_sheet(self, shots, params) -> FactSheet: ...
```

v1 set: `TripRecipe` (GPS + time clustering), `AnniversaryRecipe` (date match across years), `BeforeAndNowRecipe` (same person, maximally separated in time), `YearInReviewRecipe`, `FreeformRecipe`.

### 6.3 FreeformRecipe (first-class, per decision)

For prompts no template matches. Hybrid retrieval (vector + metadata filters) returns roughly 200 candidates; the LLM curates and orders about 30. This is the one place the LLM chooses photos — so guardrails still run **after** its selection, and it may only choose from a pre-filtered candidate pool.

### 6.4 Narration

`FactSheet` -> prose. Each narration segment records `grounded_in[]`: the source fields that justify it. A validation pass rejects segments asserting anything not traceable to the fact sheet, and retries once before falling back to a template.

## 7. Guardrails (enforced in code, not prompts)

| Guardrail | Where | Mechanism |
|---|---|---|
| Junk filtering | enrich | Excluded at index time |
| Exclusion list | engine, pre-selection | Named people, albums, date ranges removed from the candidate pool before any prompt is built |
| Sensitive-context | engine, pre-selection | Flagged photos held back pending explicit user confirmation |
| Factual grounding | narrate, post-generation | Every claim must trace to a `FactSheet` field; untraceable segments rejected |

The ordering is the point: exclusions and sensitive filtering apply **before** the LLM sees anything, so they cannot be prompt-injected or ignored. Grounding applies **after**, as validation rather than instruction.

## 8. Render

`MemorySpec` feeds two independent consumers:

- **Web player** (static HTML/JS, served by FastAPI): CSS transitions, Ken Burns via transforms, Web Audio for the music bed. Fast iteration, no encoding.
- **ffmpeg exporter**: same spec, producing MP4 with pans, crossfades, music, burned-in captions.

Ken Burns targets come from **visual saliency**, not faces — Takeout gives no face locations.

Music: curated Creative Commons tracks tagged by mood/tempo/energy, selected to match the memory's mood. Track metadata includes attribution, surfaced in exports for license compliance.

## 9. Testing and CI

The binding constraint: **CI has no photos, no GPU, no API keys.**

- **Synthetic fixture generator** produces procedural images plus Takeout JSON sidecars reproducing every known quirk. This is a first-class component, not test scaffolding.
- **Fakes for every provider**: `FakeEmbedder` (deterministic hash vectors), `FakeCaptioner`, `FakeNarrator`. The full pipeline runs in CI in seconds.
- **Golden `MemorySpec` tests**: recipes assert on the spec document, not video.
- **CI matrix**: Windows / macOS / Linux, CPU only.

## 10. Milestones

**M1 — vertical slice.** Takeout parser + `doctor` + synthetic fixtures + local embeddings + LanceDB + `TripRecipe` + narration + web player. One real memory, end to end.

**M2 — quality.** Junk filtering, quality scoring, sensitive detection, exclusion list, grounding validation.

**M3 — breadth.** Remaining recipes, router, `FreeformRecipe`, ffmpeg export, beat-synced music.

**M4 — automation and polish.** Scheduler for auto-memories, Picker top-up, contributor docs, demo assets, public release.

## 11. Open risks

| Risk | Mitigation |
|---|---|
| Takeout quirks not yet seen in the wild | Honest coverage reporting; `doctor` before indexing; fixtures grow as quirks are found |
| Python 3.13 ML wheel availability | Pin a supported Python via `uv`; verify torch/transformers before M1 |
| `FreeformRecipe` quality below templated recipes | Guardrails run post-selection; measure and iterate |
| Narration blandness under strict grounding | Grounding restricts *claims*, not tone; tune voice separately |
| Music licensing | Curated CC set only; attribution carried in `MemorySpec` and exports |
