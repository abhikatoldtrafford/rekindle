# Decision log — Takeout enrichment

This is the record of how the Takeout enrichment milestone was actually built:
the judgement calls, the things that were wrong and how they were caught, and
the testing discipline the rest of the project is held to.

It is here because the substantive outcomes live in the code and in
[known-limitations.md](known-limitations.md), but the *reasoning* would
otherwise be lost — and most of what is worth knowing about this codebase is in
the reasoning.

## What the milestone does

`rekindle index` walks a photo folder and writes normalised records to SQLite.
`rekindle enrich` is a second pass that reads Google Takeout's per-photo
`.supplemental-metadata.json` sidecars and enriches those rows with face tags,
capture dates, GPS, descriptions, favourites and album titles.
`rekindle doctor --from-index` reports on the result.

Google Photos' API was shut down in April 2025 — `photoslibrary.readonly` is
gone, and face groupings were never exposed at all. A Takeout export is the only
route to this data, which means there is no second source to check the parse
against. That constraint shapes everything below.

Measured on a real 45,900-file export:

| | before `enrich` | after |
|---|---|---|
| with people | 0 | **10,887 (55.9%)** |
| with capture date | 16,323 | **19,149 (98.3%)** |
| with GPS | 1,993 | 2,330 |

Accounting: 18,306 matched · 3,364 superseded · 2,565 orphaned · 1 ambiguous ·
942 ties broken by directory preference · 3 cross-photo collisions refused.

## The one lesson

**Every genuine bug in this milestone was found by running against a real
export. Not one was found by reading.**

Reviews caught real things — but the defects that would have shipped bad data
were all found the same way: point the code at 45,900 real files and compare
what it claims against what it did.

Three recurring shapes:

**1. Tests that cannot fail.** A test asserting a match-rate *floor* passes
while 984 sidecars vanish. A test counting `update_many` *calls* passes a
three-commit implementation. Twice, an entire block was deleted and the suite
stayed green. This is the defect class the project's mutation discipline exists
for: **break the line a test protects, watch it fail, restore it.** A test you
have not seen fail is not a test.

**2. Fixtures certifying their own fiction.** Three separate times a fixture
invented a Takeout convention that does not exist — a sidecar naming scheme with
0 real instances, a `geoData` pairing with 0 real instances, a motion-photo name
that produced 0 pairs on real data — and a test pinned the invention as
canonical. One of them landed in the file eight downstream tasks treated as
ground truth. If you add a fixture case, it must come from a real export.

XMP is still in this state: every XMP test parses output from our own generator,
and the reference export reports `with_xmp = 0`. See
[known-limitations.md](known-limitations.md).

**3. Counts propagated without measuring.** Four figures were carried into the
spec or a brief from a reviewer's report without being re-measured, and all four
were wrong — a root-level `metadata.json` that does not exist, album counts of
4-differ/2-empty (actually 3 and 1), and a pre-enrichment `with_gps` of 1,844
(actually 1,993). Counts are exactly what fixtures get built against. Re-measure
them.

## The design corrections that mattered

**The original spec's central claim was false.** It said sidecars carry a
`title` field naming their own photo, derived from inspecting one sidecar.
Measured across all 24,248: the counter lives in the *filename*, not the title —
`DSC_0880.JPG.supplemental-metadata(1).json` belongs to `DSC_0880(1).JPG`.
Matching on `title` would have dropped 984 sidecars, left 967 photos unenriched
and made 963 mis-pairs. Matching on the sidecar filename also reduces
name collisions from 4,737 to 3,772.

**"Google's date always wins" would have corrupted local time.** `photoTakenTime`
is a UTC *instant*; the timestamp resolver expects naive wall-clock. Round-tripping
one through the other yields `local = utc` — 811 photos shifted by −330 minutes
and 442 correct `EXIF_OFFSET` values discarded. The fix is a derivation ladder:
EXIF `OffsetTimeOriginal` → GPS lookup → `exif_naive − google_utc` → UTC.
`TzSource.TAKEOUT` describes the *date's* provenance, not the zone's.

**The accounting mechanism was found defective four separate times** — a single
by-construction identity that could not fail; a refusal branch that was dead code
(942 photos silently took conflicting stories while the report said 2); a
`claimed` dict keyed by filename that ate 13 refusals through `dict.__setitem__`;
and nothing testing that `account()` reads `applied` at all, so `matched += 1`
passed all nine tests. It is now the most heavily mutation-tested code in the
repo.

**A fix for a misattribution bug introduced a silent one.** Three sidecars were
each being applied to two content-distinct photos that shared a filename across
`Photos from 2011` and `Photos from 2012` — found only by running the full
integration, never by review. The first fix refused them correctly but
*invisibly*: `photos_enriched` dropped from 18,309 to 18,306 with nothing in the
report naming why.

## Rulings

Sixteen decisions were made without stopping to ask, each recorded with what it
would cost if wrong. Grouped by what they have in common:

**Untested guards get tests, not arguments** (7 rulings). Every time a reviewer
demonstrated that deleting a line left the suite green, the ruling was to add the
guard rather than accept the reasoning that the line was correct. This covered
`photo_paths` write-path consistency, `_disagree`'s widened field set,
case-insensitive derivative pairing, two CLI foreign-root warnings, the
accounting mechanism's `applied` read, and — twice — the sticky refusal guard in
`enrich()`.

**Never fabricate a value** (2 rulings). `_geo` was synthesising
`Gps(5.0, 0.0, 0.0)` from a partial block, and later fabricating sea level for an
absent altitude while letting a malformed altitude reject a perfectly good
latitude and longitude. Both were latent — zero live instances — and both were
fixed anyway, because the constraint is binding regardless of whether today's
data happens to trip it.

**Latent divergence gets pinned, not changed** (1 ruling). `name.ext(N).json`
handling diverges from the old behaviour, but zero such sidecars exist in a real
export. Pinned with a test rather than "fixed" on speculation.

**Defer when the fix is wider than the problem** (3 rulings). Album collision
detection stays metadata-only. The dead `photo_paths` table stays, because
deleting it is a schema migration and shipping one to remove infrastructure a
milestone after adding it is worse than carrying it. An import-weight issue stays
untouched rather than churning files mid-wave. All three are recorded in
[known-limitations.md](known-limitations.md) rather than silently dropped.

**Overrule a reviewer when it is wrong** (1 ruling). A re-reviewer demanded that
`UnicodeDecodeError` be isolated from `except (OSError, ValueError,
UnicodeDecodeError)` because removing it broke no test. Correct observation,
wrong conclusion: `UnicodeDecodeError` *is* a `ValueError` subclass, so the entry
is redundant, not an unproven branch. A mutation that cannot fail because the
mutated code was a no-op is a redundancy, not a defect.

**Fix the whole thing even when the process says stop** (2 rulings). The final
review's fix wave took 12 of 14 findings; the six residuals that survived the
re-review were taken in a second wave rather than parked, because every one was
"a line you can delete with the suite green" — precisely the defect the milestone
was rebuilt to eliminate.

## Three instructions that were wrong, and refused

Workers disputed instructions four times and were right every time. This is
recorded because it is the intended behaviour, not a failure:

- An instruction to build `doctor.render_enrich` inside one task — it was a
  different task's separately-specified deliverable, with its own pseudocode.
- An instruction to assert literal `EnrichReport` equality in the idempotency
  test — it would **fail on correct code**, because 8 of the 24 fields are
  "changed relative to stored state" deltas whose guards are already false on a
  correct second run. The implementer's 16-equal/8-zero partition is strictly
  stronger.
- An instruction to file M1 bug writeups under a heading whose own text is
  M0-specific, and to apply a "correction" that a prior task had already fixed —
  applying it verbatim would have reintroduced a false statement.
- An instruction to hardcode one dataset's numbers into permanent user-facing
  warning text, which would have misled every other user.

One worker also went beyond its brief with evidence and was right to: a re-index
was raising `metadata_conflict` on 10,065 rows that `enrich` had left clear — a
pure timezone artefact, because `merge_meta` compared instants exactly with no
offset normalisation. Nothing in the brief mentioned it.

## What contributors should take from this

- Point new code at a real export before trusting it. See
  [CONTRIBUTING.md](../CONTRIBUTING.md) for how to run the conformance suite.
- Mutation-test anything that guards an invariant. If you cannot make your own
  test fail, you have not written one.
- Fixture cases come from real data. Inventing a convention and pinning it is the
  most expensive mistake available here.
- Count every file you skip, with a reason. Silently dropping input is a bug —
  the accounting identities in `EnrichReport` exist to make that impossible to
  hide.
