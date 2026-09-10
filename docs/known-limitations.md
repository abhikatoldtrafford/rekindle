# Known limitations and deferred work (M0)

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
