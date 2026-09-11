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

## M3 (semantic): what was deferred, and one label that must not be trusted

All measured; the evidence is in [audit-m3-semantic.md](audit-m3-semantic.md).

### The cluster labels must not be wired to sensitive-context detection

`cluster.SCENE_VOCABULARY` names a cluster by the cosine between its centroid
and 32 candidate phrases. The label is documented in code as cosmetic, and on
the reference library it mostly is — #31 really is green countryside, #50
really is food, #56 really is dogs.

**But cluster #16 (270 photos of institutional buildings with gardens — a
campus, balcony views, an old colonial building) was labelled
"a hospital or a clinic", and contains no hospital.** In this project a
hospital is a *sensitive context* (main spec §7.2). A cosmetic label that
lands on a sensitive-context word stops being cosmetic the moment anything
reads it. Nothing does today. **Nothing may, until the labels are either
calibrated against a hand-checked set or replaced by a classifier with a
confidence a caller can act on.** Label scores range 0.127–0.282 and 12 of 57
clusters were labelled "a religious ceremony or temple", so the vocabulary is
absorbing whatever is nearest rather than recognising anything.

### Clusters are slices of a cone, not islands

48 of 57 clusters have a neighbouring centroid closer to them than their own
members are (median cohesion 0.834, median nearest-centroid 0.894). CLIP
embeddings occupy a narrow cone, so k-means partitions one continuous region
rather than finding natural groups. The partition is stable, reproducible and
useful — but `k` is a choice, not a discovery, and "57 scene types" is not a
claim this milestone makes. A density-based clustering (HDBSCAN) would be the
principled fix and would bring a "this photo is in no scene" answer that
`min_cosine` only approximates. Deferred: it is another dependency with two
parameters to tune per library.

### The face gate's numbers rest on 19 positives

Measured recall is 1.000 (19 of 19, zero misses) and precision 0.633 on 64
hand-checked photos. **Nineteen consecutive successes bound the miss rate below
about 15% at 95% confidence and no tighter.** The cases most likely to be
missed — a face at 20 px, deep shade, a profile behind a shoulder — are
under-represented in 64 random photos. The gate therefore proposes and never
publishes; there is no code path from `Verdict.ELIGIBLE` to a published file.
Before anyone relies on it, re-measure on several hundred hand-checked photos.

Nine of the eleven false positives are statues, painted idols and printed
portraits — this library is full of Durga Puja. Whether a statue should block
publication is a policy question for the user, not a detector defect, and it
is why the gate reports boxes and counts rather than a verdict alone.

### The ONNX CPU path is 49× slower than CUDA

Measured on 32 real photos: **1.04 img/s** (ONNX, fp32, CPU) against
**51.1 img/s** (torch, fp16, CUDA). The whole library would take about five
hours on CPU against five and a half minutes on the GPU. Agreement is good —
mean cosine 0.9966 against the torch vectors, text vectors identical to five
decimal places, and 3 of 4 test queries return a byte-identical top-5 — so the
fallback is correct, just slow. `Xenova/clip-vit-large-patch14` also ships
`vision_model_uint8.onnx` and `_fp16` variants that would be several times
faster; they are not pinned because their agreement with the fp32 path has
not been measured, and an unmeasured quantisation is exactly the kind of
silent quality loss this milestone's storage design exists to prevent.

### Deferred, with reasons

- **No ANN index.** 18,201 × 768 float32 is 55.9 MB and a query measures 31 ms
  end to end. The design spec's escalation gate for vector search has not been
  tripped.
- **Cluster ids are not written back to the index.** That is a schema change,
  and the whole storage design exists to avoid making one while M2 is making
  one. Re-running `spherical_kmeans` at a fixed seed reproduces them exactly.
- **`semantic-gpu` pins the cu126 wheel index for win32/linux only.** macOS
  gets the PyPI wheel (Metal). An AMD/ROCm user must install torch themselves;
  `resolve_device` will report CPU and warn only if `nvidia-smi` sees a GPU,
  so a ROCm machine gets a silent CPU fallback. Not a regression — there was
  no GPU support at all before — but it is a gap.
- **The aesthetic head is a model of average human preference**, not this
  user's. It likes sunsets, bokeh and symmetry and undervalues a blurry photo
  of someone who matters. It ranks within a candidate set and never filters
  across the library.

### Two silent CPU fallbacks in the accelerated path

Both measured in [audit-m3-semantic.md](audit-m3-semantic.md) §12–13. Neither
is a crash, neither logs anything a user reads, and the only symptom of each is
being slower — which is exactly the shape the PyPI CPU-only torch wheel taught
this milestone to distrust.

- **`torchvision` is absent, so transformers silently falls back from
  `CLIPImageProcessor` to `CLIPImageProcessorPil`.** CLIP preprocessing then
  runs at 117 img/s on one CPU thread while the ViT-L/14 forward it feeds
  sustains 236 img/s — **66% of the encode is CPU preprocessing**, which is why
  raising the batch changes nothing and why the fix was to run that half in the
  decode pool. Installing torchvision would change the pixel values the model
  sees, so it cannot be done without re-measuring agreement and re-embedding
  the library. Not a dependency line; a migration.
- **`FaceDetector(prefer_gpu=True)` does nothing.** The installed onnxruntime
  is the CPU build, whose `get_available_providers()` offers only
  `AzureExecutionProvider` and `CPUExecutionProvider`, so the CUDA branch is
  unreachable and the detector reports `('CPUExecutionProvider',)` without
  complaint. The pinned graph is fixed batch 1 as well. Reaching the GPU means
  `onnxruntime-gpu`, which *replaces* `onnxruntime` in the same import
  namespace and would put the verified ONNX CPU fallback at risk for a stage
  that is not CLIP. Threading brought the library scan to about 11 minutes
  instead of 26, which made the swap not worth its risk — **but `prefer_gpu`
  should say that it could not be honoured rather than quietly returning CPU.**

### The batch size changes the stored vectors

fp16 reduction order depends on batch shape. Against batch 32, batch 16 is
bit-identical, while batches 64 and 128 differ at a minimum cosine of 0.99998.
Irrelevant at the 0.25–0.29 cosines search and clustering work with, but **a
store should be filled with one batch size throughout**: the reference store was
built at 64 and a rebuild at the default 32 reproduced only 25 of 18,201 vectors
bit-for-bit (worst cosine 0.99902). Nothing detects this, and nothing needs to;
it is recorded so the next person measuring agreement does not chase it.
