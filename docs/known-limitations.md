# Known limitations and deferred work

Every item here was raised by a code reviewer during M0's implementation, then
triaged and consciously deferred. None is a correctness bug in shipped
behaviour; the ones that were are already fixed. This file exists so the
reasoning is not lost, and so a contributor looking for a first PR can find one.

Items are tagged by the task that surfaced them.

## Deferred

- **T2** `old.x or new.x` scalar fallback treats a legitimate 0 as absent (width/height/gps). Irrelevant for photo dimensions; flag if fields are reused.
- **T2** no test for the both-sides-present tie-break on scalar fallback fields.
- **T2** max(descs, key=len) tie-break on equal-length differing descriptions is undocumented and untested.
- **T3** sniff() collapses OSError into (UNKNOWN,"unknown"), losing the "could not open" vs "not media" distinction; untested branch.
- **T3** file_hash has no error handling for missing/unreadable files (raises uncaught); task-3-report overstates error coverage.
- **T3** two RIFF-prefixed branches repeat the startswith check.
- **T4** upsert_many does one SELECT + full deserialization per photo to decide insert-vs-merge (plan-mandated); consider preloading all_hashes() for large batches.
- **T4** `iter_photos()` was deleted during M0's final review as dead code. The Takeout enrichment spec now needs it, so it returns with a consumer.
- **T4** no test for two PhotoStore instances open concurrently; SQLite default 5s lock timeout is unconfigured.
- **T4** untested whether a mid-batch exception rolls back cleanly.
- **T5** test_make_xmp_sidecar_writes_parseable_xml does substring checks, never parses the XML it claims to validate.
- **T5** xmlns:stDim declared in the XMP template but unused.
- **T5** _deg_to_dms can emit 60.00 seconds instead of carrying into the next minute; latent, not hit by any current coordinate.
- **T6** read_exif never forces a pixel decode (only header/EXIF parsing), so a file with a VALID header but corrupt pixel data reports decode_ok=True. "Corrupt file" is therefore narrower than it sounds — it means "fails to open". CARRY TO TASK 9: its unreadable report inherits this limit.
- **T6** the dedicated DecompressionBombError catch has no test; deleting the clause would fail nothing.
- **T6** no test for HEIF_AVAILABLE; an `isinstance(..., bool)` check would at least catch the name being undefined on one branch.
- **T7** no tests for a non-Face Type region, a Type-absent region (MWG-legal, must be kept), an empty Name, or a non-numeric coordinate. All handled correctly in code; none pinned by a test.
- **T7** read_xmp never-raises contract untested against a directory path (covered by the broad except).
- **T8** naive (tzinfo-less) file_mtime branch exists and works but no test exercises it.
- **T8** an aware non-UTC file_mtime is returned with its original offset rather than normalised via .astimezone(UTC) — same instant, but the tzinfo is not UTC as the name implies.
- **T9** first_seen is not narrowed on duplicates while last_seen is — folded into fix as it is a one-line asymmetry.
- **T9** unopenable file reported as "not_media", a false claim about its content.
- **T9** excluded_dirs counts files, not directories, despite the name.
- **T9** stem key computed twice via two independent normalisations, the exact drift Ruling 1 exists to prevent.
- **T9** scan() is ~150 lines; the post-walk block would read better extracted.
- **T10** INCOMPLETE EXPORT warning uses the same [yellow]! prefix as every other warning; "prominent" would justify a distinct style. Inherited from the brief.
- **T10** no CLI-level test for an empty/zero-media folder (covered at unit level).

## The one that matters most

**No real XMP sidecar has ever passed through `read_xmp`.** Every XMP test
parses output from our own `make_xmp_sidecar`, which invents its packet layout,
and a run against a real 44,000-file Takeout export reported `with_xmp = 0`.

That is exactly the setup that produced M0's motion-photo bug: a fixture
certifying a convention nobody had checked against reality. XMP is the only
source of person data in the design and it backs `doctor`'s headline
"no person data" warning.

**Before M1 leans on person data, commit one real sidecar** — exported from
Lightroom, digiKam or osxphotos — as a test asset.

XMP still has no real-world coverage. Takeout now does, via
`tests/test_takeout_conformance.py` — the same treatment is still owed to XMP.

## Fixed during M0, recorded so they are not reintroduced

- Motion-photo pairing matched `{stem}.jpg`; Google names the still
  `{name}.jpg` (`PXL_x.MP` pairs with `PXL_x.MP.jpg`). Found only by running
  against a real export — the synthetic fixture had invented the convention.
- A damaged EXIF rational reached SQLite as `Gps(lat=nan, ...)`. Pillow returns
  it as an `IFDRational` that floats to NaN, not a zero-denominator tuple.
- `.xmp`/`.aae`/`.thm` files were counted in no bucket at all, so `doctor`'s
  numbers did not add up. There is now an accounting-invariant test asserting
  `files_seen == media_indexed + duplicates_merged + json_sidecars +
  excluded_dirs + total_skipped`.
- An unopenable file was reported as `not_media` — a claim about content that
  was never read.
- `.gitignore` matched `takeout/` but not `Takeout/`, so on a case-sensitive
  filesystem a contributor's `git add .` would have committed a personal photo
  library to a public repo.

## Fixed during M1, recorded so they are not reintroduced

*These five bugs were found and fixed during M1 (the Takeout enrichment
milestone this document otherwise predates), not M0 - the "Fixed during M0"
heading above already means something specific (M0 task numbers T2-T10), so
these get their own heading rather than being misfiled under it.*

- **Matching a Takeout sidecar on its `title` field.** `title` omits the
  disambiguating counter that Google puts in the sidecar's *filename*:
  `DSC00107.JPG.supplemental-metadata(1).json` says `title: "DSC00107.JPG"` but
  belongs to `DSC00107(1).JPG`. Measured across 24,248 real sidecars, the
  property is COLLISION REDUCTION, not a higher raw match count: `title` leaves
  4,737 sidecars beyond the first claiming one photo, the filename leaves 3,772.
  Raw matches actually fall by 4, because five `(N)` sidecars name a photo in an
  un-extracted archive part and correctly become orphans. The design that used
  `title` would have dropped 984 sidecars silently. Match on the filename via
  `rekindle.sidecars.sidecar_target`; `title` is a cross-check.
- **Passing `photoTakenTime` to `timestamps.resolve`.** It is a UTC instant;
  `resolve` expects naive wall-clock time and stamps a zone onto it, so the
  result is `local == utc` and a 21:00 IST photo is relabelled 15:30. Use
  `timestamps.from_takeout`, which derives the offset instead.
- **A `PhotoMeta` field not added to `merge_meta`.** That function builds a new
  record from an explicit field list, so a field left out is destroyed by the
  next `rekindle index`.
- **Crediting `matched` with every sidecar claiming a photo.** A photo with
  three candidates contributes one application and two discards; counting all
  three inflated `matched` by 3,358 on a real export and hid the discards. The
  accounting identity cannot catch this on its own - it partitions a set built
  by the same function - so `matched == photos_enriched` is asserted separately.
- **Reading a low `ambiguous` count as "no conflicts".** Every disagreeing
  sidecar group in the reference export has its candidates in different
  directories, so the same-directory preference resolves 942 of them and the
  refusal branch fires twice. Both numbers are reported for that reason.

## Carried into M2: person data has arrived before the rules that govern it

`rekindle enrich` now writes 40 real people's names into the index. The person
exclusion list that is supposed to govern them (main spec §7.2) is M2 work and
does not exist.

Today's risk is nil: there is no memory engine, so nothing can surface a person
unprompted. **The exclusion list must land before anything auto-triggers.** This
is recorded so the ordering stays deliberate rather than accidental.

## Carried into M2: `photo_paths` is written on every run and read by nobody

The `photo_paths` table, its `idx_photo_paths_name` index and
`PhotoStore.hashes_for_filename` have **zero production consumers**.
`TakeoutEnricher.enrich()` materialises `list(store.iter_photos())` once and
builds its lookups in memory; the only callers of `hashes_for_filename` are
tests. The schema comment that justified the table — "the enricher must look a
photo up by filename 20,000 times" — was simply false, and has been corrected
in place.

It still costs roughly **19k deletes and 24k inserts per `enrich` run** on the
reference export (`_insert` clears and rewrites a photo's rows every time),
for no current reader.

It is deliberately **not** removed here: deleting it is a v3 schema migration,
and shipping a second migration to delete infrastructure one milestone after
adding it is worse than carrying it one more milestone. **M2 must either wire
`resolve()` to it or drop it** — carrying it a third milestone is not the
intent of this entry.

## Carried into M2: `metadata_conflict` is one boolean for two causes

`Photo.metadata_conflict` is set by two different comparisons:
`models.merge_meta` raises it when two sightings of the same bytes disagree on
a real capture date **or carry different non-empty descriptions**, and
`enrich.apply_sidecar` raises it when Google's `photoTakenTime` disagrees with
the displaced EXIF instant.

M1 made the enrichment verdict **retractable** — a date the user corrects in
Google Photos now clears the flag, where `or` had made it permanent. Because
there is one boolean and no record of which comparison set it, that recompute
also clears a *description* conflict `merge_meta` had raised on a photo
enrichment then dates. The window is narrow (0 rows on the reference export
have any XMP description at all) but reachable two ways, not one: two copies
of one file carrying different XMP descriptions (`FolderSource`'s same-bytes-
in-two-folders case), **or a single copy re-indexed**: after `enrich` writes
Google's description onto the row, the next `index` calls `merge_meta(stored,
freshly-scanned)`, whose description clause compares that just-written value
against the file's own (unchanged) XMP description and raises the flag, and
the following `enrich` clears it — no second copy involved at all. The
alternative to retracting — never doing so — was the worse bug. **M2 should
split the flag by cause** rather than widening either side of this trade.

Residual 3 fixed a narrower, separate defect on the same flag: `conflicts`/
`conflicts_retracted` (the *counters* `doctor` prints, not the flag itself)
used to read this shared, cause-blind boolean directly, so a description-
caused flip of it was mis-credited as a date conflict retracting (or a
genuine new date conflict on an already-flagged row was silently dropped).
The counters now re-derive the prior DATE-only verdict from the fields that
comparison actually reads, so they credit only what their labels say. The
flag itself is unchanged by that fix and still carries both causes as
described above.

## Album collision detection is metadata-only, not folder-name-complete

`SidecarIndex.albums` (the `taken` set in `album_renames`) is built only from
folders that have a parseable, non-empty `metadata.json` — a folder without
one is invisible to collision detection in both directions: it can neither
block a rename nor have its own title recovered. A photo set split across
export parts (`Dida` / `Dida(1)`, produced when Google splits one album across
multiple Takeout archives) is detected as a collision only when *both* halves
carry their own `metadata.json`; if one part lost its metadata file (or hasn't
been extracted yet), the split is silently invisible to `album_renames`. Not
observed live on the reference export — recorded so it is not rediscovered by
a future contributor staring at an unrenamed pair of album folders.

## Settled by M2 (the memory engine)

These entries were carried into M2 as open questions. Each is now closed.

- **`photo_paths` is now READ in production.** `MemoryIndex.resolve_path`
  answers "which copy of this photo actually exists on disk?" at render time -
  a real question, because a photo is routinely the same bytes in two folders
  and a library can be partially mounted. The entry said M2 must "either wire
  `resolve()` to it or drop it"; it is wired, not dropped.
- **The person exclusion list exists.** `rekindle exclude --person NAME` and
  `rekindle dismiss` both write to one persisted store, merged into the
  `ExclusionPolicy` at `MemoryIndex.open` - the single chokepoint every recipe
  passes through. The entry's requirement ("must land before anything
  auto-triggers") is met: nothing auto-triggers at all, and the exclusion list
  landed first regardless.

## Fixed during M2, recorded so they are not reintroduced

- **Stored `width`/`height` ignored the EXIF orientation tag.** A camera writes
  a portrait photo as landscape pixels plus a tag, and `Image.size` is the raw
  stored size. Roughly 2,475 of 18,201 live images (13.6% of a 456-file sample)
  were indexed with the two swapped. Latent through M0 and M1 because `w*h` is
  invariant under the swap, so even the dedup tiebreak that reads those columns
  could not notice; only a rule asking "is this taller than it is wide?" exposes
  it. Fixed in `meta.exif.read_exif`; existing indexes are repaired in place by
  the fingerprint pass, which decodes every image anyway. **Only orientations
  5-8 exchange the axes** - the tempting `!= 1` test corrupts the 180-degree and
  mirrored cases.
- **A 64-bit perceptual hash overflows SQLite's signed INTEGER.** Roughly half
  of all dHashes set the top bit, so this was the ordinary case, not an exotic
  one. The value is unsigned in memory (a negative int makes
  `bin(a ^ b).count("1")` silently wrong) and reinterpreted as signed at the
  storage boundary only.
- **A `CREATE INDEX` on a new column cannot live in `_SCHEMA`.** That script
  runs before `_migrate`, and on an old database `CREATE TABLE IF NOT EXISTS` is
  a no-op that leaves the old column set - so indexing `photos(phash)` raised
  "no such column" and made every v1 database unopenable. Late indexes are
  created after migration instead.
- **`collapse()` emitted a photo once per appearance in its input.** A recipe
  handing it a list containing duplicates got a memory showing the same photo
  twice, with dedup having "run".

## Carried into M3

- **A photographed document is undetectable.** Screenshots are excluded from
  metadata evidence (filename convention, or a screen-size match with no camera
  metadata and no face tags), but a *photo of* a receipt, a whiteboard or a
  form carries ordinary camera metadata and ordinary dimensions. Telling them
  apart needs a model v1 deliberately does not have. Roughly 93 WhatsApp images
  are also caught by the size branch, which is the false-positive cost of the
  rule.
- **Videos never appear in memories.** All 1,117 rows have no stored
  dimensions, no perceptual hash, and rendering one needs ffmpeg - and
  including them only when ffmpeg happens to be installed would make the
  `MemorySpec` depend on the machine. 5.8% of the library is therefore
  unreachable by any memory.
- **`place_cluster` keys embed a visit's start date**, so adding a photo
  *earlier than the first photo of an existing visit* changes that visit's key
  and a dismissal of it stops applying. Every other recipe's key is derived
  from a subject that library growth cannot move. Not observed live; recorded
  so it is not rediscovered.
- **`metadata_conflict` is still one boolean for two causes.** The M1 entry
  asked M2 to split it. It is deliberately NOT split: nothing in the memory
  engine reads that flag, so splitting it would be a schema change in service
  of no consumer - exactly the pattern the `photo_paths` entry warns against.
  Carried forward explicitly rather than silently.
- **A sharp subject smaller than one tile is still scored as blurred.**
  Sharpness is now a reblur ratio over ~128px tiles of a 1024px copy, scored
  on the *sharpest* tile, which is what keeps an ordinary shallow-depth-of-
  field portrait: measured retention 0.800 against 0.006 for a percentile
  form. But a face that occupies less than one tile — a group shot, a distant
  subject — cannot drive the score. Only face boxes close this, and the
  detector is M3's. The whole-image alternative is strictly worse, so this is
  a residual rather than a regression.
- **The measure reads texture, so a flat, hard-edged subject scores low
  whether or not it is in focus.** The absolute gradient across a step edge
  does not change when the edge is spread over three pixels, so a picture made
  only of hard borders is nearly invisible to it — a 40px checkerboard scores
  0.043 sharp and 0.009 grossly blurred. Photographs are texture nearly
  everywhere and this is what makes the measure contrast-invariant, but a
  photographed sign or document sits low in the distribution on merit it does
  not lack. `is_screenshot` removes the common case and the gate sits at the
  1st percentile; the rest is accepted.
- **Sharpness still cannot be compared across libraries.** It is far more
  stable than the measure it replaced — per-year 5th percentiles span 1.71x
  where the old one spanned 4.70x, and 1.16x across resolution decades — but
  `MIN_SHARPNESS = 0.24` was derived from *this* library. A library of
  scanned film or of screenshots would need it re-derived, and nothing in the
  code detects that.
- **Album merging is manual.** `Leh Ladakh` / `ladakh` and the three Kashmir
  albums are each one trip, but no metadata says so, and `Diwali 25` /
  `Diwali Kali Puja 22` are different years under an equally similar pair of
  names. An `album_aliases` config table lets the user say so; the default
  merges nothing. The overlap cap catches the resulting redundancy at build
  time.
