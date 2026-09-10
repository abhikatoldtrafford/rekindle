# Takeout enrichment — design spec

**Date:** 2026-09-10
**Revision:** v2, after an adversarial review that measured v1's claims against a
real 45,900-file export and falsified the central one.
**Status:** approved in shape, revised in substance
**Supersedes:** the sidecar-matching design in
[the main spec](2026-09-10-rekindle-design.md) §5.1

## 0. What v1 got wrong

v1 argued that every sidecar carries a `title` field naming its own photo, that
this gives an exact match key, and that the main spec's confidence-tier matching
ladder was therefore unnecessary machinery for a hazard removed by construction.

**That was false, and it was measured on a sample of one.**

Google puts the disambiguating counter in the sidecar's *filename*, not in
`title`. `DSC00107.JPG.supplemental-metadata(1).json` carries
`title: "DSC00107.JPG"` but belongs to `DSC00107(1).JPG`. Measured across all
24,248 sidecars in the reference export:

The right measure is **collision reduction**, not match count. Filename
derivation does not find more photos — measured over all 24,248 sidecars it
matches four *fewer*, because five `(N)` sidecars name a photo absent from this
export and correctly become orphans. What it does is stop two sidecars claiming
the same photo:

| Match key | Sidecars beyond the first claiming one photo |
|---|---|
| `title` (v1's design) | 4,737 |
| **sidecar filename** | **3,772** |

That −965 is the 963 mis-pairs below. An earlier draft of this section quoted
"19,358 vs 20,322 correct pairs"; those figures did not reproduce under the
definition the conformance test uses, and are corrected here rather than left
in the repo.

v1 would have silently dropped **984 sidecars**, left **967 photos** with
sidecars unenriched, and made **963 definite mis-pairs** — avoided in practice
only because `(` sorts before `.` in ASCII, so the correct sidecar happened to
overwrite the wrong one. Reverse the directory iteration order and a thousand
photos inherit another photo's date and face tags.

The failure mode is the one v1's own §7 warned about: a convention observed once,
generalised, and then relied on structurally. It is M0's motion-photo bug again.

**Corrected position:** match on the sidecar's own filename, which is what M0
already does in `_sidecar_target`. Use `title` only as a cross-check.

## 1. Purpose

Google Takeout ships a JSON sidecar beside almost every photo, carrying what EXIF
cannot: **face-tag names**, Google's capture time, descriptions, favourites, and
album membership.

Measured on the reference export: **24,248 sidecars, 59.6% carrying people across
40 distinct names**, all clean (no empty, whitespace-only, or case-colliding
values). Roughly **2,360 photos have no EXIF date at all** and a matched sidecar
fixes them. There is no other route to any of this.

## 2. What the real data actually looks like

Every claim below is measured across the whole export, not sampled.

```json
{
  "title": "PXL_20260321_083923528.MP.jpg",
  "description": "",
  "creationTime":   {"timestamp": "1774085809"},
  "photoTakenTime": {"timestamp": "1774082363"},
  "geoData":     {"latitude": 0.0, "longitude": 0.0, "altitude": 0.0},
  "geoDataExif": {"latitude": 0.0, "longitude": 0.0, "altitude": 0.0},
  "people": [{"name": "Paro r Baba"}, {"name": "Avyan"}],
  "url": "...", "googlePhotosOrigin": {...}
}
```

| Fact | Measurement |
|---|---|
| `creationTime` is upload time | 57 minutes after `photoTakenTime` on the sample above. Using it would date memories by migration date. |
| Sidecar filename schemes | Exactly two: `.supplemental-metadata` (23,255) and `.supplemental-metadata(N)` (993). **No truncated forms.** |
| `geoData` vs `geoDataExif` | Both non-zero on the same 2,532 records (10.4%); `geoDataExif` is key-absent on the rest. **Zero disagreements.** It never rescues a stripped photo — Takeout adds coordinates to 0.3% of photos. |
| Descriptions | Present as a key on 100%, non-empty on 145 (0.60%). |
| `favorited` | Absent when false; present on 7 sidecars, always `true`. |
| `archived` / `trashed` | 162 archived, 12 trashed. **User intent, sitting unused.** |
| Album `metadata.json` | 42 files; **3** titles differ from the folder name, **1** is empty, none are non-string. (An earlier review reported 4 and 2; re-measured directly — those figures do not reproduce. This is the second count from that review I propagated without checking.) Also present: `shared_album_comments.json` and `user-generated-memory-titles.json`, whose `title` is a **list**, and — reported by an earlier review but **not present in the reference export** (verified: no root-level `metadata.json` exists, and no `title` is null; one is an empty string). A null title is handled defensively, not because it was observed. |
| Album folders with zero media | **8**, holding hundreds of sidecars each. |
| `photoTakenTime` clustering | 20 timestamps shared by ≥10 sidecars, covering 907 records — one value on 618. These are Google's *guesses*, not shutter times. |

## 3. Shape: an enrichment pass, not a Source

```
rekindle index  <folder>    # pixels   -> Photo rows        (M0, unchanged)
rekindle enrich <folder>    # sidecars -> updates those rows (this spec)
```

New module `src/rekindle/enrich/takeout.py`; new CLI verb; `FolderSource`
untouched. It does **not** implement `Source`: that protocol returns
`(list[Photo], SourceReport)` for something producing records from pixels, and
this updates existing records addressed by content hash.

**But v1 dropped the one thing that protocol was enforcing** — the
count-everything contract — and that is how 984 vanishing sidecars became
possible. §7 restores it explicitly.

### Guards against undesigned states

- **`enrich` before `index`**: an empty store must abort with "run `rekindle index` first", not orphan 24,248 sidecars and fire doctor's INCOMPLETE EXPORT warning at 100%.
- **`enrich` against a foreign folder**: paths are stored absolute. The `meta` table records `enrich_root` and `index_root`; a mismatch warns rather than silently matching nothing.
- **Half-enriched index**: `meta` records `enriched_at`, so `sidecar_match = none` can be read as "enrich never ran" versus "enrich found nothing".

## 4. Matching

### Key: the sidecar's filename, not its title

Extend M0's `_sidecar_target` so it **relocates** the counter instead of
stripping it:

```
DSC00107.JPG.supplemental-metadata.json     -> DSC00107.JPG
DSC00107.JPG.supplemental-metadata(1).json  -> DSC00107(1).JPG      # relocated
IMG_1234.jpg.json                           -> IMG_1234.jpg
```

M0's current implementation returns `DSC00107.JPG` for both, which is the
collision. The change is confined to one function and is covered by the existing
`test_sidecar_target_*` tests plus new counter cases.

**`title` becomes a cross-check, not the key.** When the filename-derived target
and `title` disagree beyond the counter, that is a signal worth recording, not
ignoring.

### Index shape: global, with directory as a preference

v1 keyed on `(directory, title)`. That discards **3,927 sidecars (16.2%), 3,026
of them carrying people**, because Takeout writes a sidecar into every album a
photo belongs to without duplicating the pixels there — 1,204 of those photos
exist in the index and would have got nothing.

The index is therefore `dict[casefolded_filename, list[sidecar]]`, built over the
whole tree. Resolution for a photo:

1. A sidecar in the **same directory** as one of the photo's paths wins.
2. Otherwise, if exactly one candidate exists anywhere, use it.
3. If several remain and they **disagree** on capture time or people, **refuse to
   enrich** and count it. Never let dictionary insertion order decide.

Measured need for step 3: 833 colliding groups, 90 with >1 day of spread (max 332
days), 100 with differing people sets.

### Orphans: one definition, shared

M0's `FolderSource` already computes `orphan_sidecars` globally and
filename-derived (2,438). v1's per-directory title-keyed count gives 3,927 for
the same export. Two numbers for one concept, both rendered by `doctor`, is a
defect. **One function, used by both.**

`EXCLUDED_DIRS` is inherited from `folder.py` — the enricher must not parse
`Trash/`'s sidecars and count them against an index that correctly excludes them.

## 5. Merge semantics

### The instant: Google wins, with a bound

Measured on 1,317 comparable photos: 34.2% agree to the second; every sub-day
disagreement is **exactly −330 minutes** (IST), meaning Google's instant is right
and naive EXIF is wrong. Where they differ by more than a year, EXIF is a broken
camera clock and Google is right.

**Exception (v1 had none):** suppress the override when Google's timestamp
belongs to a large identical-timestamp cluster *and* EXIF carries a distinct real
date. 618 photos sharing one second did not fire the shutter together.

### Local time: derived, never round-tripped

**This is v1's most damaging error.** `photoTakenTime` is a UTC instant;
`timestamps.resolve()` expects a naive wall-clock value and stamps a zone onto
it. Feeding one to the other yields `local = utc`, so a photo taken at 21:00 IST
is relabelled 15:30 — and v1 did this to **every EXIF-dated photo**: 811 shifted,
and 442 more that already had a correct `EXIF_OFFSET` had it discarded.

Correct derivation of `taken_at_local`, in order:

1. EXIF `OffsetTimeOriginal` if present
2. GPS timezone lookup if coordinates exist
3. **`exif_naive − google_utc`**, which recovers +05:30 exactly
4. UTC

`tz_source` keeps `EXIF_OFFSET` when an offset is what resolved local time.
`TzSource.TAKEOUT` describes the *date's* provenance, not the *zone's*;
conflating them loses information.

### Field rules

| Field | Rule |
|---|---|
| `taken_at_utc` | Google's `photoTakenTime`, subject to the cluster bound above |
| `taken_at_local` | Derived as above — never via the naive ladder |
| `people` | `(existing − previous_takeout) + new_takeout`, tracked via `PhotoMeta.takeout_people` (see below) |
| `gps` | Fills only when absent; `(0,0)` treated as absent |
| `description` | Longest non-empty |
| `favorite` | From `favorited`; absent means false |
| `archived` / `trashed` | **New fields.** Archived means the user deliberately hid it — it must not surface in a montage |
| `albums` | Real title from album `metadata.json`, applied wherever the photo lives |
| `exif_taken_at_utc` | **New field.** Retains the displaced EXIF date |
| `metadata_conflict` | Set only when the two dates differ as *instants* after offset normalisation |

### Why replace-not-union on re-enrich

v1's union/OR merges are non-retractable: a face tag corrected in Google Photos,
or a photo un-favourited, could never be fixed by re-exporting. For a tool
holding 40 real people's names, an index with no way to retract a wrong
identification is a defect, not a simplification.

v1 said "union on first enrich, replace on re-enrich", which **is not
implementable against a flat list** — with only a merged `people` list there is
nothing to subtract, so a second run cannot tell which names it contributed
last time from which came from another source.

**`PhotoMeta.takeout_people` holds the last enrich's contribution**, making the
rule computable:

```
people = (existing − previous_takeout) + new_takeout
```

A name Google no longer reports is removed; a name from any other source
survives untouched.

### Why `metadata_conflict` changed

v1's rule fired on **65.8%** of comparable photos — a boolean true for the
majority conveys nothing, and it was almost entirely the −330 minute timezone
artefact rather than genuine disagreement. Comparing instants after
normalisation makes it mean "different day", which is what a reader would assume.

v1 also promised the record was "flagged rather than silently overwritten" while
`PhotoMeta` has exactly one date field — so it was flagged *and* overwritten.
`exif_taken_at_utc` makes the promise true.

### Idempotency

v1's conflict rule was not idempotent: after run 1 the stored `tz_source` is
`TAKEOUT` and the EXIF date is gone, so run 2 cannot recompute the flag.
Retaining `exif_taken_at_utc` and replacing rather than unioning makes a second
run produce identical output.

## 6. Derivatives inherit enrichment

Google writes **no sidecar for `-edited` files or `.MP` halves** — 177 and 692
respectively in the reference export, zero matched by any key. Each is a separate
`file_hash` and therefore a separate `Photo` row, so the version a user most
likely wants in a montage is the one with no date and no face tags.

M0 links `-edited` files to their originals via `edited_of`, which **is**
persisted. Its `.MP` pairing is **not** — `motion_pairs` is a report counter, an
integer, never an edge stored on any row. v1 asserted "M0 already links them"
about both; that is only half true, and it was asserted without checking.

Enrichment therefore propagates along `edited_of` where the edge exists, and
**re-derives** the motion-photo pairing from the naming rule
(`PXL_x.MP` ↔ `PXL_x.MP.jpg`) where it does not.

## 7. Reporting, with an enforced invariant

```
EnrichReport:
  sidecars_seen, matched, orphaned, ambiguous, excluded_dirs
  photos_enriched, derivatives_enriched
  people_added, dates_corrected, gps_added, albums_retitled
  conflicts, clustered_dates_suppressed
```

**Accounting-invariant tests, mirroring M0's.** Note there are *two* identities,
not one: v1 specified a single equation that **cannot hold**, because it had no
bucket for the 44 non-photo JSON files (`shared_album_comments.json`,
`user-generated-memory-titles.json`, the root `metadata.json`) or for parse
failures. An invariant that cannot balance is worse than none — it would have
been "fixed" by loosening it.

```
json_files_seen == sidecars_seen + album_metadata + other_json
                   + excluded_dirs + unparseable

sidecars_seen  == matched + orphaned + ambiguous
```

M0's equivalent test is what makes a whole class of silent drop impossible to
ship. v1 inherited the slogan "report, never silently drop" and none of the
machinery — this test alone would have caught v1's 984 vanishing sidecars.

### `doctor` must read the index

`doctor` currently runs a live `FolderSource().scan()` and never opens the
database, so after a successful enrich it would still print *"No person data
found in XMP sidecars"* and *"the Takeout parser … is not implemented yet"*
forever. `doctor` gains an index-reading path.

## 8. Store API and migration — both were missing from v1

v1's central loop — *"for each `Photo`, for each path it has"* — is
**unimplementable against the current `PhotoStore`**, which exposes only `count`,
`all_hashes`, `upsert_many`, `get` and `schema_version`. There is no way to
enumerate photos, no path lookup (`paths` is a JSON blob), and no writer that
does not route through `merge_meta`.

Required additions:

- `iter_photos()` — enumerate (M0 had one and it was deleted as dead code; it now has a consumer)
- a `photo_paths(file_hash, path)` table, indexed — the honest fix for path lookup
- `update_meta(file_hash, meta, ...)` — writes without merging

### Migration is mandatory, not optional

`db.py` raises on schema mismatch: *"no migration exists yet. Delete the file and
re-index."* Adding `sidecar_match`, `archived`, `trashed` and
`exif_taken_at_utc` bumps `SCHEMA_VERSION`, so **every existing user would lose
their index and re-hash 45,900 files** — while §3 justifies the separate verb
precisely by avoiding that.

A real migration ships with this: `ALTER TABLE` for the new columns, the new
table, and a version bump. Three lines, and the difference between an upgrade and
a data-loss event.

Note also that `_row_to_photo` does `TzSource(row["tz_source"])`, so a DB
containing `TAKEOUT` opened by an older rekindle raises `ValueError` from inside
deserialisation rather than the friendly schema error. The migration must bump
the version so the friendly path fires first.

## 9. Testing

### Fixtures from measured structure

Reproducing what was observed, not assumed: both sidecar filename schemes
including `(N)`, colliding `(dir, title)` pairs with differing people, an album
folder containing sidecars and **zero** media, `geoDataExif` key-absent rather
than zeroed, `favorited` absent when false, `archived`/`trashed`, a
`metadata.json` with a null title, a `user-generated-memory-titles.json` whose
`title` is a list, and `-edited` / `.MP` files with no sidecar.

### Conformance against a real export

`tests/test_takeout_conformance.py` runs only when `REKINDLE_TAKEOUT_DIR` is set
and skips otherwise.

**v1's version would not have caught v1's bug.** It asserted "matched photos
exceed a floor", and a floor of 80% passes while 984 sidecars vanish. It must
assert the §7 accounting identity per-sidecar instead.

The same treatment remains owed to XMP, which still has no real-world coverage.

## 10. Sequencing risk

Person data arrives **before** the exclusion list that governs it (main spec
§7.2, currently M2). No memory engine exists, so nothing surfaces a person
unprompted and today's risk is nil. **Exclusions must land before anything
auto-triggers.**

## 11. Out of scope

- `url`, `googlePhotosOrigin`, `appSource` — read nothing we do not use.
- Deletion reconciliation. `last_seen` stands.
- Shared-album photos: Takeout does not export other people's uploads. (Note 6
  sidecars do carry `sharedAlbumComments`; the main spec's claim that no shared
  data is exported is narrowly wrong, though the pixels genuinely are absent.)
- Rejoining albums split across export parts (`Dida` / `Dida(1)`), and album
  titles that collide after substitution (`Untitled`, `Untitled(1)`,
  `Untitled(3)` → one name). Detected and reported; not merged.
