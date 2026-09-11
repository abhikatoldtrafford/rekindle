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
- **The measure is unreliable on heavily compressed sub-megapixel photos**,
  for the same reason as the entry above: JPEG blocking at 800x600 is hard
  edges, and hard edges are what it cannot read. Those files are 53% of 2011
  against 7.5% of the library, so a gate calibrated on library-wide
  percentiles lands disproportionately on the early years — measured, a 0.24
  gate took 2.25% of 2008–2013 against 0.83% of 2020–2026. `MIN_SHARPNESS` was
  lowered to 0.12 for this reason, which flattens it to 0.15% against 0.16%.
  The underlying weakness is unfixed; only the gate was moved out of its way.
- **Sharpness still cannot be compared across libraries.** It is far more
  stable than the measure it replaced — per-year 5th percentiles span 1.75x
  where the old one spanned 3.84x, and 1.16x across resolution decades — but
  `MIN_SHARPNESS = 0.12` was derived from *this* library, twice, and the
  second derivation needed a person looking at the photographs it rejected.
  A library of scanned film or of screenshots would need it re-derived, and
  nothing in the code detects that.
- **A `recurring_event` key can move if its peak sits on a fortnight
  boundary.** The key is the centre day-of-year quantised to a half-month, so
  that adding a year does not change a memory's id and silently un-dismiss it.
  Adding a year moves a decade-old peak by a day or two, which the bucket
  absorbs unless the centre already sat that close to the 15th or the 1st.
  Same shape as `place_cluster`'s entry above; the alternative, keying on the
  discovered dates, breaks on *every* new year rather than rarely.
- **Two `recurring_event` peaks in one fortnight: the weaker is dropped.**
  Two peaks thirteen days apart normally merge, because one window covering
  both outscores either alone - measured, a collision occurs in about 1 random
  burst layout in 16,000. When it does happen the stronger event is offered
  and the weaker becomes unreachable, because publishing both would give two
  different memories one id and dismissing either would dismiss both. The
  dropped one is not reported to the user.
- **`recurring_event` finds the festival and cannot name it.** On this library
  it discovers Durga Puja (12 years, 2,297 photos) and Kali Puja (11 years,
  1,134) from timestamps alone, and titles them `Mid October, most years` and
  `Early November, most years`, because naming requires an album name that
  recurs across years and this library's festival albums do not
  (`Mahasaptami, 2013` and `Durga Puja 25` are one year each). Guessing the
  festival from the date is deliberately not done: a religious observance
  named wrongly in a title someone is shown is worse than a description that
  is merely dull. An `album_aliases` config would fix the naming, and is the
  same mechanism the entry below asks for.
- **Album merging is manual, except for a year on the end.** `Christmas 2025`
  and `Christmas 15` are now one `Christmas` album, because stripping a
  year-like suffix is the one merge no metadata can contradict. It is applied
  ONLY where two names share a family: measured on this library, stripping the
  suffix everywhere merges exactly one pair and renames seven more albums that
  have no partner (`Durga Puja 25` to `Durga Puja`, `Puri 25` to `Puri`), and
  each of those renames changes a memory id so every dismissal of one stops
  applying. The merge is printed by `rekindle memories` and overridable with
  `album_aliases`.
- **Everything else about album merging is manual.** `Leh Ladakh` / `ladakh` and the three Kashmir
  albums are each one trip, but no metadata says so, and `Diwali 25` /
  `Diwali Kali Puja 22` are different years under an equally similar pair of
  names. An `album_aliases` config table lets the user say so; the default
  merges nothing. The overlap cap catches the resulting redundancy at build
  time.

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


## M4 — prompt memories

### The refusal gate is shipped disabled, and the reason is a measurement

`prompt.tag_agreement` is computed, printed, and **used for nothing**. It was
the eighth statistic tested as a refusal signal on this library and the eighth
to fail: over 16 concepts the library holds and 16 it does not, the
distributions overlap almost completely (present 0.10–0.86 median 0.49; absent
0.05–0.81 median 0.41), and the best available threshold refuses 7 of 16 absent
concepts while wrongly refusing 2 of 16 present ones — 66% accuracy against a
50% base rate. Among concepts described by three or more tags the direction
reverses.

`tests/test_prompt_real.py::test_tag_agreement_does_not_separate_present_from_absent`
re-measures it and fails if it ever starts working, which is the point: that
would be news, not a green light. **Do not add a threshold here without
repeating the 16-vs-16 evaluation and writing the numbers down.**

Pool size separates the two sets at 88% accuracy and is *not* shipped: the
threshold falls exactly on the largest present value with zero margin, and a
raw pool size is a library-scale quantity, which is the absolute-magnitude trap
that killed the earlier seven signals in a different costume.

### Deferred, with the measurement that deferred it

- **No place filter and no gazetteer.** GeoNames `cities1000` would name 11 of
  this library's 12 GPS clusters and resolve the user's own spelling
  "midnapur", but the payoff for the query that motivates it is 8 photos from
  one morning of 2019 — about 2 after burst dedup, below `MIN_SHOTS`. It needs
  a fetch command, a checksum, a CC BY 4.0 attribution obligation, a reverse
  geocode at index time and a `places` table, and it is its own milestone. The
  CLI says "this word narrowed nothing" instead. **CLIP is worse than a
  constant predictor at place** — 46.0% on 7-way classification against a 62.5%
  majority baseline — so do not try to substitute it.
- **Tag consensus is a small step BACKWARDS on a concept CLIP already
  resolves.** `durga puja` straight to the encoder gives 23 of 24 correct
  shots; the best hand-written tag set alone gives 17. The corpus month window
  is what recovers it to 24. Nothing detects which case a prompt is in, and
  nothing chooses between them.
- **Day expansion assumes a capture day is one coherent event.** A day holding
  a pandal visit *and* an unrelated lunch pulls the lunch in. The shipped Durga
  Puja memory contains two shots of a college lawn for exactly this reason.
  **Nobody has measured how often it happens.**
- **`MIN_SEEDS` is nearly inert** now that the day quorum exists, and is kept
  at 2 only because it is free. Do not describe it as load-bearing; the plan
  did, and measurement disagreed.
- **`SEED_K`, `TAG_K` and the quorum were not swept.** Two tag sets, three
  prompts, three values of `min_seeds`. The right values may differ by prompt
  shape.
- **The LLM plausibility gate is unmeasured.** It is implemented, biased hard
  towards accepting, and refuses nothing on an unparseable reply — but it needs
  an API key and the session that wrote it had none. It has never been run
  against a real model.
- **`minority_orientation` is an untaxed cost on this path.** A 15-year
  semantic pool is near a coin-flip on orientation while an album is one shoot,
  so `composition.py` discards 20-30% of a prompt pool wholesale: 349 photos on
  the shipped Durga memory, 235 on the Kali one. It did not starve any measured
  prompt; it is not proven safe on a thin one.
- **No video, ever.** All 1,117 videos are unembedded, so the search cannot see
  them. The CLI says so on every build rather than leaving it here.
- **The tag cache has no eviction, no versioning and no size bound.** A user
  who types a thousand prompts gets a thousand entries in
  `data/prompt_tags.json`. Fine at the scale anyone will reach by hand; not
  fine if prompts are ever generated.

## Fixed after M3: the semantic pipeline decoded every photo raw

**`semantic.embed._decode` and `semantic.faces._examine` never applied the
EXIF orientation tag.** Both opened the file, called `draft`, and converted to
RGB — the exact idiom the renderer and the fingerprint pass had already been
fixed away from. 2,503 of the library's 18,363 images (13.6%) carry a
90/270-degree tag, and another 35 carry a 180-degree one, so the encoder and
the face detector were looking at 13.8% of the library on its side.

Measured with the real `yolov11n-face` detector on 200 of those photos:

- **14.5% of gate verdicts change** once the transpose is applied.
- **9.0% were judged too permissively** — the old decode called them safer
  than they are. `DSC_0436.JPG` shows the detector **zero** faces sideways and
  **eight** upright; `IMG_20201114_200059981_HDR.jpg` goes 0 → 4. A gate whose
  stated failure mode is "a stranger's face on the internet" was failing open
  on roughly 225 of the 2,503 rotated photos.
- It cost accuracy in the other direction too:
  `IMG_20161030_193606253_HDR(1).jpg` scores **ten** faces sideways and zero
  upright. The gallery review sheet's "one photo tagged 'Abhik Maiti' came
  back with ten faces" — 82 of 189 frames excluded at that step — is that
  photo.

**The renderer was never affected, and a bug report said it was.** The gallery
review recorded six files as "renders 90° rotated" and dropped them. All six
carry orientation 6 or 8; re-rendering `on_this_month --key 08` and
`year_in_review --key 2017` against the same index draws every one of them
upright, and the WebP is byte-identical before and after this change. What was
sideways was the **contact sheet they were reviewed on**, which decoded with
`Image.open` + `draft` + `convert` and no transpose — the same idiom that was
genuinely still shipping in `semantic`. A review harness that does not share
the renderer's decode path is a review harness that can condemn a good frame,
and four of those six were dropped for a defect that was never in the render.

And with the real CLIP ViT-L/14 encoder on 60 of them: the sideways vector has
a **median cosine of 0.935** against the upright one, where two entirely
unrelated photos of this library sit at **0.553** (max 0.866 over 190 pairs).
The worst case, 0.811, is further from its own upright vector than some
unrelated pairs are from each other. The aesthetic head scores those same
vectors, so it inherited the error.

**Why M2's fix and M2's tests did not prevent this.** M2 corrected the stored
`width`/`height` and shipped tests for them — and those tests assert NUMBERS.
A 400x300 file tagged 90 degrees is 300x400 whichever direction the rotation
went, and orientations 2, 3 and 4 do not change the size at all, so half the
tag values are invisible to any dimension assertion. The index was right and
two decode sites were still wrong, for a whole milestone.

`meta.exif.open_upright` is now the single place pixels are turned the right
way up, and all four decode sites go through it. `tests/test_orientation.py`
asserts the PIXELS at every one of them, using a four-colour quadrant marker
that distinguishes all eight orientations from each other; the fixture is
itself checked against Pillow's `exif_transpose` rather than against our own
code, and against its own ability to tell a mirror from a rotation.

### Carried: 2,503 stored vectors were computed from sideways pixels

The fix corrects future decodes. It does not touch the 18,201 vectors already
in `data/semantic/clip-vit-l14/`, and `embed_photos` skips any hash the store
already holds, so **13.8% of the live embedding store stays wrong until it is
rebuilt** — semantic search, clustering and every aesthetic score inherit it.
There is no targeted re-embed: `EmbeddingStore` has no invalidation path, and
`model_revision` raises on a mismatch rather than expiring rows. The recovery
today is to delete the store directory and re-run `rekindle semantic embed` -
about five and a half minutes on the GPU, five hours on CPU.

**A `--redo` flag taking a set of hashes is the right fix**, and is not built
here: it is a new CLI surface, and this change is deliberately confined to the
decode.

**Resolved.** The store was deleted and the library re-embedded upright
(18,201 vectors). Measured against the preserved sideways store, 2,534 of the
18,201 vectors changed, the changed ones have a median self-cosine of 0.9352
against their upright replacement, and two unrelated photos of this library
sit at 0.549. `--redo` is still the right fix and still is not built.

### Some of this library's orientation tags are stale, and applying them is what turns the photo sideways

Found by opening the shipped `kalipuja diwali celebration` memory after the
re-embed, rather than by any number. **Three of its 24 shots render on their
side, and they are sideways *because* the fix is applied, not despite it.**
`DSC01306.jpg1.jpg`, `DSC01319.jpg2.jpg` and `DSC01320.jpg1.jpg` are stored
portrait, carry EXIF orientation 6, and their raw pixels are already upright:
some earlier tool rotated the pixels and left the tag behind. `open_upright`
does exactly what the tag says and rotates a correct photograph 90 degrees.

`open_upright` is not wrong. `ImageOps.exif_transpose` is the standard
behaviour, and any viewer, Pillow-based or not, shows these three files the
same way. The files are lying, and nothing in EXIF distinguishes a stale tag
from a live one.

**The population is small and is not quantified.** The obvious signature —
stored portrait plus a 90/270-degree tag — matches 222 of 18,201 images, but
it is *not* a detector: 18 of those were opened both ways by hand and only 5
were actually stale, the other 13 being ordinary rotated photographs that the
tag fixes correctly. So the real figure is on the order of tens of files,
concentrated in particular shoots (all three in the Kali memory come from one
2013 Diwali evening), and a reliable count needs content, not metadata.

Consequences, in the order they bite:

* the renderer draws these sideways, which is user-visible;
* the embedder and the face gate see them sideways, so they are now the
  photographs the *old* store happened to get right;
* `meta.width`/`meta.height` are stored post-rotation, so the index agrees
  with the wrong answer and no dimension assertion can see it.

**Not fixed here, and it should not be fixed by loosening the rule.** A
heuristic that second-guesses the tag whenever the stored pixels are portrait
would break 13 files for every 5 it repaired, on this library's own numbers.
The honest fixes are a per-file override the user can set, or an
orientation-detection model, and both are new surface area. Recorded so the
next person who opens a montage and sees a photograph on its side does not go
looking for the bug in `open_upright`.

**Mostly resolved, by content.** `rekindle semantic orient` (`meta.orientation`)
now proves staleness from the picture instead of from the metadata, using the
face detector that already ships. `open_upright` is untouched in its logic: the
pass records, per file, that a tag is stale, and the chokepoint stops applying
that one tag. It can only ever DISABLE a tag the file already carries — it never
invents a rotation for a file whose tag says upright, and never proposes 180
degrees. Measured on the reference library:

| | |
|---|---|
| images examined | 18,363 (1,117 videos are out of scope) |
| carrying a 90/270-degree tag | 2,503 (13.6%) |
| **proved stale, now decoded without the tag** | **212 (1.15% of images, 8.5% of the suspects)** |
| tag trusted — the picture agrees with it | 1,463 |
| **no face at any rotation — UNREACHABLE** | **828 (33.1% of the suspects)** |
| unreadable | 0 |
| whole library, 6 threads | 222 s |

Precision was hand-checked by opening 90 of the proposals and looking at them:
at the shipped margin of 0.35, **two independent samples of 30 found 57 right,
0 wrong and 3 unsure** (a macro flower and two pets, where no orientation is
objectively right). In the band just below it, 0.20–0.35, 30 files gave 21
right and 4 genuinely wrong, which is why the default sits where it does;
`--margin` exposes the knob.

Compare the dimension-based detector this replaces: 5 right for 13 wrong.

**The three files that started this are NOT among the 212, and that is the
most useful thing this section can tell you.** `DSC01306.jpg1.jpg`,
`DSC01319.jpg2.jpg` and `DSC01320.jpg1.jpg` are genuinely stale — opening them
both ways confirms it — and the detector agrees, but only by +0.233, +0.098
and +0.106. They are night shots at a crowded Diwali event, where the faces
are small and dim and the detector is weak in BOTH orientations, so the
evidence points the right way and never becomes decisive. Lowering the margin
far enough to catch them means entering the band measured at 44% precision,
which would turn more correct photographs sideways than it repaired.

So they go to `rekindle semantic orient --review`, which ranks every
sub-threshold file that leans against its tag, using the evidence already
stored — no detector, no decode, so it runs on a machine with no model at all.
That queue is **282 files long, and the three sit at ranks 51, 150 and 160**.
They are not at the top of it and nothing about the evidence puts them there:
a human would have to work a fair way down. That is the honest shape of the
result — the queue is a place to look, not a shortlist.

The queue mixes the two ways a file can fall short, and labels which: 220
failed the margin, 62 had no detection confident enough to count as a face at
all. Neither is evidence the tag is right.

**The unreachable third is the real limit, and it is not going away.** A
rotated photograph with no face in it carries no evidence this method can
read, so 828 files keep whatever their tag says. Nothing proposes them for
review either — a queue of 828 photographs nobody will ever work through is
not a fix.

**Invalidation, which did not previously exist.** Recording a stale tag
changes the pixels `open_upright` returns, so `PhotoStore.set_orientations`
clears that photo's `phash`, `sharpness`, `brightness`, `colour` and
`phash_error`, which puts it back in `rekindle fingerprint`'s queue (and the
re-run repairs the stored `width`/`height`, which were swapped). **Embeddings
are still not invalidated** — they live in a separate store with no delete
path, which is exactly how 2,503 sideways vectors survived the earlier fix.
The pass prints the count and says so; `--redo` remains the right fix and
still is not built.
