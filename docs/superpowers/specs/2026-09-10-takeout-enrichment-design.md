# Takeout enrichment — design spec

**Date:** 2026-09-10
**Status:** approved
**Supersedes:** the sidecar-matching design in
[the main spec](2026-09-10-rekindle-design.md) §5.1

## 1. Purpose

Google Takeout ships a JSON sidecar beside almost every photo. Those sidecars
carry what EXIF cannot: **face-tag names**, Google's authoritative capture time,
descriptions, favourites, and coordinates for photos whose EXIF was stripped
during re-compression.

M0 indexes the pixels and deliberately ignores the JSON. This adds a second pass
that reads it and enriches the rows M0 already created.

Measured on a real 45,900-file export: **24,280 sidecars, 59% of them carrying
people, across 31 distinct names.** Person-based memories are impossible without
this and straightforward with it.

## 2. What the real data looks like

Every design decision below is grounded in an inspected export rather than in
documentation or folklore.

A per-photo sidecar:

```json
{
  "title": "PXL_20260321_083923528.MP.jpg",
  "description": "",
  "creationTime":   {"timestamp": "1774085809"},
  "photoTakenTime": {"timestamp": "1774082363"},
  "geoData":     {"latitude": 0.0, "longitude": 0.0, "altitude": 0.0},
  "geoDataExif": {"latitude": 0.0, "longitude": 0.0, "altitude": 0.0},
  "people": [{"name": "Paro r Baba"}, {"name": "Avyan"}],
  "url": "https://photos.google.com/photo/...",
  "googlePhotosOrigin": {"mobileUpload": {"deviceType": "ANDROID_PHONE"}}
}
```

Observed facts that shape the design:

- **`title` holds the exact target filename.** The sidecar names its own photo.
- **`creationTime` is upload time.** On the sample above it is 57 minutes after
  `photoTakenTime`. Using it would date every memory by when the user migrated
  to Google Photos.
- **`geoData` is frequently zeroed** while `geoDataExif` carries the real
  coordinates. In a 400-sidecar sample both were populated on exactly the same
  47 records, so they agree when present.
- **Descriptions and favourites are rare** — 2 descriptions and 0 favourites in
  400 sidecars. Worth reading, not worth optimising for.
- **Album folders carry `metadata.json` with the album's true title**, which is
  the pre-sanitisation name: folder `Abhirup Birthday- Sudipta Saad` has title
  `Abhirup Birthday/ Sudipta Saad`. 3 of 41 albums differ; some titles are empty.

## 3. Shape: an enrichment pass, not a Source

```
rekindle index  <folder>    # pixels  -> Photo rows          (M0, unchanged)
rekindle enrich <folder>    # sidecars -> updates those rows  (this spec)
```

New module `src/rekindle/enrich/takeout.py`; new CLI verb; `FolderSource`
untouched.

It deliberately does **not** implement the `Source` protocol. `Source.scan`
returns `(list[Photo], SourceReport)` for something that walks pixels and
produces records. This walks JSON and *updates* records addressed by content
hash. Forcing it into that signature would misdescribe it. It gets its own
narrow interface:

```python
class TakeoutEnricher:
    name = "takeout"

    def enrich(self, root: Path, store: PhotoStore) -> EnrichReport: ...
```

Re-running is idempotent: the same sidecars produce the same values.

### Why not fold it into FolderSource

That would make the generic folder source Google-aware, which is the vendor
coupling v1 was scoped to avoid (main spec §3.1). Keeping enrichment separate
also means a user can re-run it after adding archive parts without re-hashing
45,900 files.

## 4. Matching — exact, or not at all

Build an index `{(folder, title): sidecar}` from every per-photo JSON, keyed on
the sidecar's own `title` field. For each `Photo`, for each path it has, look up
`(path.parent, path.name)`. A photo filed in two albums gets two chances.

**No fuzzy fallback. No filename munging. No confidence ladder.**

The main spec §5.1 specified a five-rung matching ladder with tiers, written
before we knew `title` existed. That design existed to manage the risk that
aggressive filename matching mis-pairs sidecars — a mature tool shipped exactly
that and produced roughly **a third of its GPS pointing at the wrong place**.

With an exact key that risk does not arise, so the mitigation is unnecessary.
§7's guardrail — *narration may never assert a place or date derived from a
heuristic match* — is satisfied **by construction** rather than by enforcement.

`Photo.sidecar_match` is recorded as `exact | none` so `doctor` can report
coverage. Two states, not five. **Building the tier ladder now would be
speculative machinery for a hazard this design has removed.**

Unmatched sidecars are counted and reported. They are never force-matched, and
a sidecar whose `title` names a file that is not present is an *orphan* — the
existing signal that archive parts are missing.

### Case sensitivity

Lookups casefold the filename, matching the folder source's own convention.
Directory identity is the `Path` itself.

## 5. Merge semantics

Enrichment does **not** route through `models.merge_meta`. That function's
"earliest real date wins" is a tiebreak between sources of *equal* authority.
Here they are not equal, and smuggling source priority into a general-purpose
merge would hide the distinction.

| Field | Rule |
|---|---|
| `taken_at_utc` | **Google's `photoTakenTime` wins**, unconditionally. New `TzSource.TAKEOUT` records the provenance. |
| `taken_at_local` | Recomputed from the new UTC value through the existing ladder |
| `people` | Union, order-preserving, de-duplicated |
| `gps` | Fills only when absent. Prefers whichever of `geoDataExif`/`geoData` is non-zero; a (0,0) pair is treated as absent |
| `description` | Longest non-empty wins |
| `favorite` | Logical OR |
| `albums` | The real title from album `metadata.json` replaces the sanitised folder name. An empty title leaves the folder name in place |
| `metadata_conflict` | Set when EXIF carried a **real** date (not the mtime fallback) that disagrees with Google's |

### Why Google's date wins

It is Google's own record of when the shutter fired, it survives the
re-compression that strips EXIF, and it is present on essentially every photo.
On the reference library **3,012 of 19,480 photos currently have only a
filesystem-mtime date** — this fixes all of them that have a sidecar.

Where EXIF exists and agrees, nothing changes. Where it disagrees, the record is
flagged rather than silently overwritten.

## 6. Reporting

`EnrichReport`, rendered by `doctor` and by `enrich` itself:

```
sidecars_seen, matched, orphaned
photos_enriched
people_added, dates_corrected, gps_added, descriptions_added,
favourites_added, albums_retitled
conflicts            # EXIF vs Google disagreements
```

`doctor` gains a person-coverage line once enrichment has run.

**Report, never silently drop** applies unchanged: every sidecar is either
matched, orphaned, or counted with a reason.

## 7. Testing

### Fixtures built from observed structure

M0's motion-photo bug came from a fixture inventing a naming convention Google
does not use. These fixtures reproduce **inspected** structure: the real field
names above, zeroed `geoData` beside populated `geoDataExif`, the
counter-inside-suffix form `DSC_0880.JPG.supplemental-metadata(1).json`, album
`metadata.json` with both differing and empty titles, and `title` values that
contain `.MP.jpg`.

### A conformance test against a real export

`tests/test_takeout_conformance.py` runs only when `REKINDLE_TAKEOUT_DIR` is
set, and skips otherwise. It asserts structural invariants against a real
library — every sidecar parses, `title` is present and non-empty, matched
photos exceed a floor — without committing anyone's personal data.

This is the answer to *our fixtures certify our own fiction*. CI never needs an
export; a contributor who has one gets real verification. **The same treatment
is owed to XMP**, which still has no real-world coverage (see
[known-limitations.md](../../known-limitations.md)).

## 8. Sequencing risk

This delivers person data **before** the exclusion list that governs it (main
spec §7.2, currently M2). No memory engine exists yet, so nothing can surface a
person unprompted and the risk today is nil.

**Exclusions must land before anything auto-triggers.** Recorded here so the
ordering is deliberate rather than accidental.

## 9. Out of scope

- The `url` and `googlePhotosOrigin` fields. Read nothing we do not use.
- Deletion reconciliation. An incremental export cannot express deletion; the
  main spec's `last_seen` policy stands.
- Shared-album photos. Takeout does not export other people's uploads, and no
  parser can recover them.
