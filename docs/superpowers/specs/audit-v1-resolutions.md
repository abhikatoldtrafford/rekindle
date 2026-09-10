# Audit of design v1 — findings and resolutions

**Date:** 2026-09-10
**Reviewer:** independent adversarial review (automated, max effort)
**Outcome:** v1 revised to v2. Several v1 claims were factually wrong.

This document exists so contributors can see what was challenged, what changed,
and what remains a known limitation rather than a solved problem.

## Factual corrections

| v1 claimed | Correct | Source |
|---|---|---|
| Scopes removed **1 April 2025** | **After 31 March 2025** | [Google](https://developers.google.com/photos/support/updates) |
| Content labels **never exposed by any API** | Exposed as a search *filter* (`contentFilter`, 26 categories, never per-item). Only **face groupings** were never exposed | [Google](https://developers.google.com/photos/library/guides/apply-filters) |
| Takeout gives the **full library** | Only media **you** uploaded. Others' contributions to shared albums are silently excluded | [Metadata Fixer](https://metadatafixer.com/learn/download-shared-albums-google-photos) |
| Scheduled exports are recurring full snapshots | **Incremental since June 2026** — first full, rest deltas only | [9to5Google](https://9to5google.com/2026/06/01/google-photos-schedule-export/) |
| "Add to Drive" enables automatic pulls | Consumes Google storage quota (150–400 GB typical export vs 15 GB free), and rekindle has no Drive source. Recommendation withdrawn | — |
| `Expand-Archive` for 50 GB archives | Fails above ~2 GB. Use `tar` | — |
| Python 3.13 ML wheels are a risk | They aren't; **3.14** is (no CUDA wheels). Pin via `uv` | [pytorch#169929](https://github.com/pytorch/pytorch/issues/169929) |

## Critical findings

| # | Finding | Resolution |
|---|---|---|
| C1 | Shared-album photos absent from Takeout — worst possible gap for a memories product | Documented prominently; `doctor` flags suspected sparse albums. No code fix exists |
| C2 | Fuzzy sidecar matching is a known corruption vector — a mature tool produced ~34.5% wrong GPS this way | `sidecar_match` tier is a first-class field; heuristic tier **rejected by default**; narration may never assert a place or date from a low tier; `doctor` reports a histogram, not a coverage percentage |
| C3 | Grounding validator trusted `grounded_in[]` emitted by the narrating model — self-attested provenance is not provenance | Generator and verifier separated. The verifier extracts claims independently and checks field values itself. Explicit inference allowlist. **~100-triple eval set gates CI** |
| C4 | M1 shipped LLM narration with all four guardrails deferred to M2 | Re-sequenced: M0 parser, M1 pipeline with `TemplateNarrator` + exclusions + sensitive gating, M2 LLM + verifier. Guardrails now precede what they guard |

## Major findings

| # | Finding | Resolution |
|---|---|---|
| M5 | Face tags unavailable in Illinois/Texas and opt-in elsewhere, so "blocklist people" is not the guarantee the README promised | README states the limit; date-range and album exclusions lead as the reliable primitives; `doctor` reports coverage |
| M6 | `FreeformRecipe` contradicted the README's "selection is deterministic Python" claim, and is the *default* fallback path | README rewritten to be accurate; IDs hard-validated against the pool; serialized fields specified |
| M7 | Prompt injection via descriptions, filenames and album names reaches the narrator — and grounding *legitimizes* it | Fields classified trusted/untrusted; untrusted delimited, capped, screened, and **ineligible for grounding** |
| M8 | §5.2 claimed junk filtering ran "before any expensive model" while two filters needed models; CLIP and SigLIP both named | Split into prefilter (no models) and postfilter (post-embedding); one model family; **flag, never exclude at index time** |
| M9 | LanceDB solved a problem this scale doesn't have (50k × 768 fp32 = 154 MB), and no relational store was designed at all | SQLite as source of truth + `numpy` memmap `float16` vectors, behind an `Index` façade |
| M10 | Pixel-byte hashing forces a full decode pass and breaks on any Pillow upgrade; doesn't dedupe `-edited`; multi-album merge policy undefined | `file_hash` for identity, `phash` for near-dupes; explicit `-edited` linking; merge policy written down |
| M11 | Picker OAuth refresh tokens expire every 7 days in "Testing" status | Picker dropped from v1 — incremental Takeout supersedes it |
| M12 | Web player and ffmpeg exporter would diverge, with no test layer below `MemorySpec` | Timeline semantics made normative (normalized rects, post-rotation, explicit overlap convention) + renderer conformance suite |
| M13 | `arrange()` returning timings made beat-sync a breaking change to the published contributor API | Timing removed from recipes; a separate `Timeline` stage owns it |

## Minor findings addressed

Broken doc links (`writing-recipes.md`, `writing-sources.md` now exist);
suffix truncation modelled combinatorially rather than as a ladder; `-edited`
sidecar presence softened to "may"; local time prefers EXIF `OffsetTimeOriginal`
over GPS; `tar` replaces `Expand-Archive`; shell quoting fixed; Windows
`MAX_PATH` documented; unsourced "30–50% junk" demoted to a hypothesis;
Laplacian blur demoted to a near-duplicate tiebreaker; screenshot detection
requires two signals; config surface gained provider and base-URL variables;
`.gitignore` no longer blocks fixture archives; `SECURITY.md` added.

## Not previously considered, now specified

Video, motion photos, Live Photo pairs, HEIC (and `.heic` files that are
actually JPEG); **localized Takeout folder names**; accessibility (transcript,
`prefers-reduced-motion`, keyboard, no autoplay); thumbnail cache budget and
eviction; memory storage and schema versioning; error recovery beyond the
embedding pass; hosted-model cost estimation, caps and dry-run; music mood
selection as a deterministic mapping rather than an LLM choice.

## Known limitations we are choosing to live with

These are documented rather than solved. Contributions welcome.

1. **Shared-album photos cannot be recovered from Takeout.** No workaround
   exists short of re-saving them to your own library first.
2. **Person exclusions are best-effort**, bounded by face-tag coverage.
3. **Sensitive-context detection is weak on relational and temporal pain** — the
   cases that hurt most. Mitigated by onboarding, the feedback loop, and
   auto-memories defaulting off, not by detection.
4. **Grounding verification is a floor, not a proof.** It validates
   traceability, not truth, and cannot catch a claim derived from wrong
   metadata. This is why match tiers restrict what may be asserted.

## The open problem

Synthetic fixtures validate **plumbing, not selection quality**. A contributor
writing `EverySunset` cannot test whether retrieval actually finds sunsets, so
every recipe PR would arrive with green CI and unknown behaviour.

The planned answer (M3) is a small redistributable **CC0 image corpus** with
hand-authored Takeout sidecars, fetched on demand, plus a **recipe eval harness**
carrying human-labelled expectations. Until that exists, "add a recipe" is an
invitation contributors cannot fully accept — and reviewers must eyeball recipe
PRs against their own libraries.
