"""`rekindle embed`: fill an embedding store from the photo index.

MEASURED FIRST, THEN WRITTEN
----------------------------
The naive loop - open, convert, hand to the processor, encode - runs at
**6.1 images/sec** on this machine with CLIP ViT-L/14 on an RTX A4000. The GPU
is idle for 93% of that: 81.9 of the 88.4 seconds spent on 540 photos was
JPEG decode and resize on one core. Buying a faster GPU would have changed
nothing.

Three fixes, all measured on real photos from the reference library:

  * **`Image.draft()`** asks libjpeg to decode at 1/2, 1/4 or 1/8 scale using
    the DCT coefficients directly. A 4000x3000 photo destined for a 224x224
    crop is decoded at 500x375 - the same pixels the resize would have thrown
    away, never materialised. JPEG only; it is a no-op on PNG and is safe to
    call unconditionally.
  * **A decode thread pool.** Pillow releases the GIL inside the C decoder, so
    threads (not processes) get real parallelism with no pickling of images
    across a process boundary.
  * **Preprocessing in that same pool, genuinely running ahead of the GPU.**
    This is the third fix and it was missing. The previous loop called
    `pool.map` and then blocked on its result, so the pool was idle for every
    second of the encode and the encode was idle for every second of the
    decode - the docstring claimed "one batch ahead" and the code did not do
    it. Worse, 66% of the encode was itself single-threaded CPU work: the
    Hugging Face image processor runs at 117 img/s while the ViT-L/14 forward
    it feeds sustains 236 img/s.

    `prepare_images`/`encode_prepared` (see `PreparingEncoder`) split the
    encoder so the CPU half joins the pool and the GPU half stays on one
    thread. Batches are now submitted `PREFETCH_BATCHES` ahead and consumed in
    order. Output vectors are IDENTICAL, not approximately equal.

Both numbers are re-measured in docs/audit-m3-semantic.md; do not trust the
paragraph above over the table there.

ACCOUNTING
----------
`EmbedReport` obeys
`considered == already + embedded + missing + unreadable + deferred`,
asserted by a test. M1's decision log is unambiguous about why: a pipeline that
silently drops input is a bug, and the accounting identity is what makes the
drop impossible to hide. `deferred` is new and is the reason the loop no longer
stops early on `--limit`: see WHAT `--limit` NOW MEANS below.

STALENESS: WHAT "THE PIXELS CHANGED" ACTUALLY KEYS ON
-----------------------------------------------------
`embed_photos` used to skip any hash the store already held, full stop. That
is how **2,503 vectors stayed sideways through the fix that was meant to
repair them** (docs/known-limitations.md, "Carried: 2,503 stored vectors were
computed from sideways pixels"). Recovery cost two whole re-embeds, and a
re-embed is three minutes on this machine's GPU and **five hours** on a
contributor's CPU.

The trap in fixing it is choosing the wrong key. A content hash of the file
does not move when an **orientation verdict** changes - and an orientation
verdict is exactly what caused the incident. The bytes are byte-for-byte
identical; what changed is how `open_upright` decodes them. Key on the file
and the feature silently does nothing, which is worse than not shipping it.

So `decode_key` names the DECODE INPUTS, and only the ones the file hash does
not already pin down:

  * **the orientation verdict** - `raw` when `meta.orientation` has proved
    this file's EXIF tag stale and `open_upright` is therefore handing back
    the stored pixels, `exif` when the tag is being applied. This is the term
    that varies at RUNTIME, per file, without any code changing, and it is the
    one the incident needed.
  * **`target_px`** - the decode is drafted to a multiple of it, so a
    different request size is a different decode.
  * **`DECODE_REVISION`** - the blunt instrument, for a change to `_decode`
    itself that no other term captures. Bumping it makes every vector stale.

The EXIF orientation TAG is deliberately NOT in the key, and its absence is
not an oversight. The tag lives in the file's bytes, `file_hash` is BLAKE2b of
those bytes, and the store is already keyed by it - so `(file_hash, verdict)`
determines the applied transform exactly. Putting the tag in as well would
record a value that cannot vary independently, at the cost of an
`Image.open()` per photo at planning time on 18,363 images.

MEASURED ON THE REFERENCE LIBRARY
---------------------------------
18,201 images (the live index minus 162 archived), against the real
`clip-vit-l14` store of 18,201 vectors, on a copy so the live store was not
touched. Every vector was stamped with the key it would have had if it had
been embedded BEFORE `rekindle semantic orient` ran, and then re-planned:

    iterate the index                 0.23 s
    `decode_key` for all 18,201       2.67 s
    `EmbeddingStore.plan`             0.01 s
    -> stale                            212
    -> fresh                         17,989

212 is exactly the set the index records as `orient_ignore_exif = 1`, verified
by set equality against the database and not by count alone. Under the old
code all 18,201 of those would have been skipped as "already embedded".

The 2.67 s is `Path.resolve()` inside `orientation._key`, which is a real
filesystem call per path. It is paid only on a library that HAS overrides:
`ignores_exif` short-circuits on the empty table, so a library the orientation
pass has never run on pays nothing. Against a re-embed of three minutes on
this machine's GPU and five hours on a contributor's CPU, it is not a cost
worth optimising.

WHAT `--limit` NOW MEANS
------------------------
It used to break out of the scan, so a photo past the limit was never even
counted. It now caps the WORK: every photo is examined and classified, and
anything over the cap is reported as `deferred`. That is what the numbers
above buy - being TOLD that 212 vectors are stale even on a run that has time
to redo only fifty of them.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rekindle.semantic.encoder import ImageTextEncoder, Vector
from rekindle.semantic.photos import IndexedPhoto
from rekindle.semantic.store import EmbeddingStore

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image

#: Batch handed to the model at once.
#:
#: THIS IS NOT A VRAM KNOB, AND SIZING IT FROM VRAM WOULD BE CARGO CULT. The
#: obvious criticism of a hardcoded 32 is that it was tuned for a laptop card
#: and wastes a 16 GB A4000. It was measured instead, on 512 real photos,
#: pre-decoded so the GPU was the only variable:
#:
#:     batch    img/s   peak VRAM
#:         8     75.1       899 MiB
#:        16     71.2       966 MiB
#:        32     79.4     1,099 MiB
#:        64     79.3     1,365 MiB
#:       128     76.5     1,900 MiB
#:       256     75.4     2,963 MiB
#:
#: A 32x range of batch sizes spans 71-79 img/s: no trend, all noise. Even
#: batch 256 uses 18% of the card. The batch is flat because the stage it
#: sizes is not the bottleneck - the encode is 66% CPU preprocessing (see the
#: module docstring), and a bigger batch makes the serial CPU half bigger in
#: exact proportion to the GPU half it is trying to fill.
#:
#: So 32 stays, for the reasons that actually apply: it bounds the decoded
#: images held in memory per in-flight batch, and it keeps the progress
#: counter responsive. The lever is PREFETCH_BATCHES, not this.
DEFAULT_BATCH = 32
#: Decode + preprocess workers. Six physical cores. Measured 9.6 -> 25.3 img/s
#: on the face scan at four and falling off at eight, because onnxruntime and
#: torch both run their own intra-op pools across the same six cores.
DEFAULT_WORKERS = 6
#: How many batches of decode+preprocess may be in flight ahead of the device.
#:
#: THIS is the lever the batch size is not. Median of five interleaved rounds
#: over 512 real photos, orders rotated per round:
#:
#:     prefetch    median   observed range
#:            1      25.4     18 - 29
#:            2      52.3     30 - 53
#:            4      78.7     52 - 83
#:            6      60.1     58 - 93
#:           12      93.4     67 - 103
#:           24      70.8     68 - 98
#:
#: Read the ranges, not only the medians. A prefetch of 1 - which is what the
#: old `pool.map`-and-block loop effectively was - is 3-4x slower than
#: anything from 4 up, and that gap is far outside the noise. Between 4 and 24
#: this machine CANNOT TELL THEM APART: a second agent was working in the main
#: worktree throughout, and the same setting measured 17.9 and 106.8 img/s
#: twenty minutes apart. Claiming a winner among those four would be reading
#: someone else's build as signal.
#:
#: 12 is chosen as 2 x DEFAULT_WORKERS - every worker holding a batch with one
#: queued behind it, so a worker that finishes early never waits for the main
#: thread to come back round. Memory cost is PREFETCH_BATCHES x DEFAULT_BATCH
#: decoded images: 384 at roughly 500x375 RGB, about 215 MB.
PREFETCH_BATCHES = 12

#: Bumped when `_decode` changes what pixels it produces in a way no other
#: term of `decode_key` captures. **Every stored vector becomes stale.** It is
#: the blunt instrument, so prefer adding a term that names the actual input.
#:
#: 1 - `open_upright` with a 2x draft hint, converted to RGB. The state of the
#:     code as of the orientation fix; earlier vectors were computed without
#:     `open_upright` at all, and no store still holds any (the live store was
#:     deleted and rebuilt) so there is no revision 0 in the wild to catch.
DECODE_REVISION = 1


def decode_key(paths: Iterable[Path], *, target_px: int, revision: int = DECODE_REVISION) -> str:
    """What a vector for this photo would be computed FROM, as a short string.

    See STALENESS in the module docstring for what is in it and, just as
    importantly, what is left out and why.

    Takes every path the photo is known at, not just the one that will be
    decoded. `meta.orientation` records its verdict against a file HASH and
    installs it for every path that hash has, so all of a photo's paths agree
    - but a library re-indexed from a second folder can briefly know a path
    the override table does not, and `any` fails towards "the verdict
    applies", which redoes a vector unnecessarily rather than keeping a wrong
    one. Pure dict work: no file is opened, no EXIF is read.
    """
    from rekindle.meta import orientation

    ignore = any(orientation.ignores_exif(p) for p in paths)
    return f"v{revision}|px{target_px}|{'raw' if ignore else 'exif'}"


@dataclass
class EmbedReport:
    considered: int = 0
    already_embedded: int = 0
    embedded: int = 0
    missing_file: int = 0
    unreadable: int = 0
    #: Work `--limit` cut. Not "skipped": the run knows about these and chose
    #: not to do them, which is a different thing from never having looked.
    deferred: int = 0
    elapsed_s: float = 0.0
    model_key: str = ""
    model_revision: str = ""
    runtime: str = ""
    device: str = ""
    failures: list[tuple[Path, str]] = field(default_factory=list)

    # -- what the plan found, before `--limit` and `--redo` touched it. These
    # -- are DIAGNOSTICS, not buckets: they overlap `already_embedded` and
    # -- `embedded`, and are outside the accounting identity on purpose.
    #: Held a vector whose recorded decode key matches the current one.
    fresh: int = 0
    #: Held a vector computed from pixels this build no longer produces.
    stale: int = 0
    #: Held a vector written before the store recorded provenance.
    unverified: int = 0
    #: The same hash offered twice by the reader (two paths, one file).
    duplicates: int = 0
    #: Of `embedded`: how many had no vector at all beforehand.
    newly_embedded: int = 0
    #: Of `embedded`: how many replaced a vector that was stale, unverified or
    #: named on `--redo`.
    recomputed: int = 0
    #: `--redo` hashes the reader never offered, so nothing could be done
    #: about them. Reported rather than swallowed - a typo in a hash is
    #: otherwise indistinguishable from a hash that was already fresh.
    redo_unknown: list[str] = field(default_factory=list)

    @property
    def images_per_s(self) -> float:
        return self.embedded / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def accounted(self) -> bool:
        """Every considered photo landed in exactly one bucket."""
        return self.considered == (
            self.already_embedded
            + self.embedded
            + self.missing_file
            + self.unreadable
            + self.deferred
        )


def _decode(path: Path, target: int) -> Image:
    """Open `path` UPRIGHT and as RGB, decoded no smaller than `target`.

    `draft` (inside `open_upright`) only ever picks a scale whose result is
    still at least as large as the requested size, so the subsequent
    high-quality resize has the pixels it needs. It is the single biggest
    speed lever in this module.

    ORIENTATION IS PART OF THE DECODE, not a rendering concern. 2,503 of this
    library's 18,363 images (13.6%) carry a 90/270-degree EXIF tag, and this
    function used to hand every one of them to the encoder on its side: the
    vector for a sideways photo has a median cosine of 0.935 against the
    upright one, where two entirely unrelated photos sit at 0.553. The
    aesthetic head reads those same vectors, so it inherited the error.
    """
    from rekindle.meta.exif import open_upright

    # A generous multiple of the target: draft rounds DOWN to a power-of-two
    # scale, and asking for exactly `target` can land on a scale whose output
    # is slightly smaller than the crop, which upsamples.
    return open_upright(path, draft=(target * 2, target * 2)).convert("RGB")


def _batches(items: Sequence, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def embed_photos(
    photos: Iterable[IndexedPhoto],
    encoder: ImageTextEncoder,
    store: EmbeddingStore,
    *,
    batch_size: int = DEFAULT_BATCH,
    workers: int = DEFAULT_WORKERS,
    prefetch: int = PREFETCH_BATCHES,
    target_px: int = 224,
    limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    runtime: str = "",
    device: str = "",
    redo: Iterable[str] = (),
    redo_unverified: bool = False,
    decode_revision: int = DECODE_REVISION,
) -> EmbedReport:
    """Embed every photo with no vector, plus every photo whose vector is stale.

    Resumable and safe to re-run: a second run with nothing to do writes
    nothing and reports `fresh == considered`.

    `redo` forces specific hashes whatever the plan says - the escape hatch
    for a photo somebody knows is wrong for a reason this code cannot see.
    `redo_unverified` additionally recomputes vectors written before the store
    recorded provenance; it is opt-in because on a pre-existing store that is
    every vector, and a full re-embed is five hours on CPU.
    """
    report = EmbedReport(
        model_key=store.model_key,
        model_revision=store.model_revision,
        runtime=runtime,
        device=device,
    )
    by_hash: dict[str, IndexedPhoto] = {}
    wanted: list[tuple[str, str]] = []
    keys: dict[str, str] = {}
    for photo in photos:
        report.considered += 1
        # `decode_paths`, not `paths`: a video's pixels come from its cached
        # still, so that is the file whose orientation verdict matters.
        key = decode_key(photo.decode_paths, target_px=target_px, revision=decode_revision)
        keys.setdefault(photo.file_hash, key)
        by_hash.setdefault(photo.file_hash, photo)
        wanted.append((photo.file_hash, key))

    plan = store.plan(wanted)
    report.fresh = plan.fresh
    report.stale = len(plan.stale)
    report.unverified = len(plan.unverified)
    report.duplicates = plan.duplicates

    forced = list(dict.fromkeys(redo))
    report.redo_unknown = [h for h in forced if h not in by_hash]
    todo = list(
        dict.fromkeys(
            [h for h in forced if h in by_hash] + plan.to_embed(include_unverified=redo_unverified)
        )
    )
    if limit is not None:
        report.deferred = max(0, len(todo) - max(0, limit))
        todo = todo[: max(0, limit)]
    # Everything the scan saw that this run will not touch. Derived, not
    # counted, so a bucket cannot be forgotten in a branch.
    report.already_embedded = report.considered - len(todo) - report.deferred
    new_hashes = set(plan.missing)
    candidates: list[IndexedPhoto] = [by_hash[h] for h in todo]

    if not candidates:
        return report

    total = len(candidates)
    done = 0
    prepares = _prepare_fn(encoder)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        pending: deque[Future[_BatchWork]] = deque()
        batches = _batches(candidates, batch_size)
        depth = max(1, prefetch)

        def submit_more() -> None:
            while len(pending) < depth:
                batch = next(batches, None)
                if batch is None:
                    return
                pending.append(pool.submit(_prepare_batch, batch, target_px, prepares))

        submit_more()
        while pending:
            work = pending.popleft().result()
            # Top the pool back up BEFORE the encode call, not after: that
            # call is the only thing on this thread that takes real time, and
            # it is exactly when the workers should be busy.
            submit_more()

            # ACCOUNT on this thread. `report.unreadable += 1` from six
            # workers is a lost-update race: attribute += is load/add/store,
            # not atomic, and the count it corrupts is the one the accounting
            # identity rests on. The workers therefore mutate nothing shared;
            # they return their failures and this thread counts them.
            report.missing_file += len(work.missing)
            report.failures.extend(work.missing)
            report.unreadable += len(work.unreadable)
            report.failures.extend(work.unreadable)

            if work.photos:
                vectors = _encode(encoder, work)
                store.add_many(
                    [
                        (photo.file_hash, vec)
                        for photo, vec in zip(work.photos, vectors, strict=True)
                    ],
                    # The provenance is written in the SAME transaction as the
                    # vector. Stamping it afterwards would leave a window in
                    # which a crash produces the exact state this feature
                    # exists to detect - a vector nobody can date - and would
                    # do it on the crash path, which is the one nobody tests.
                    decode_keys={p.file_hash: keys[p.file_hash] for p in work.photos},
                )
                report.embedded += len(work.photos)
                for photo in work.photos:
                    if photo.file_hash in new_hashes:
                        report.newly_embedded += 1
                    else:
                        report.recomputed += 1
            work.close()
            done += work.size
            if progress is not None:
                progress(min(done, total), total)
    report.elapsed_s = time.perf_counter() - started
    return report


@dataclass
class _BatchWork:
    """One batch after decode and preprocessing, ready for the device.

    Carries its own failures rather than counting them, because it is built on
    a worker thread and the report counters are not atomic.
    """

    size: int
    photos: list[IndexedPhoto] = field(default_factory=list)
    images: list[Image] = field(default_factory=list)
    prepared: object = None
    missing: list[tuple[Path, str]] = field(default_factory=list)
    unreadable: list[tuple[Path, str]] = field(default_factory=list)

    def close(self) -> None:
        for image in self.images:
            image.close()
        self.images = []


def _prepare_fn(encoder: ImageTextEncoder):
    """The encoder's threadable CPU half, or None if it has no such half.

    Resolved once per run rather than per batch, and by `getattr` on BOTH
    names: an encoder that grew only one of the pair is a half-finished
    migration, and using it would run `encode_prepared` on data that was never
    prepared. See `PreparingEncoder` in encoder.py.
    """
    prepare = getattr(encoder, "prepare_images", None)
    finish = getattr(encoder, "encode_prepared", None)
    return prepare if callable(prepare) and callable(finish) else None


def _prepare_batch(batch: Sequence[IndexedPhoto], target_px: int, prepares) -> _BatchWork:
    """Runs on a WORKER THREAD. Touches no shared state, never raises."""
    work = _BatchWork(size=len(batch))
    for photo in batch:
        path = photo.existing_path()
        if path is None:
            work.missing.append((photo.path, "file not found"))
            continue
        image, error = _safe_decode(path, target_px)
        if image is None:
            work.unreadable.append((path, error or "decode failed"))
            continue
        work.photos.append(photo)
        work.images.append(image)
    if work.photos and prepares is not None:
        try:
            work.prepared = prepares(work.images)
        except Exception:  # noqa: BLE001 - the encoder family is wide
            # Preprocessing off-thread is an optimisation and must never be a
            # new way to fail. Whatever it raised will be raised again by
            # `encode_images` on the main thread, where the run-ending error
            # handling already lives and where the traceback is the user's.
            work.prepared = None
    return work


def _encode(encoder: ImageTextEncoder, work: _BatchWork) -> list[Vector]:
    if work.prepared is not None:
        return encoder.encode_prepared(work.prepared)  # type: ignore[attr-defined]
    return encoder.encode_images(work.images)


def _safe_decode(path: Path, target: int) -> tuple[Image | None, str]:
    """`(image, "")` or `(None, reason)`. Never raises, never mutates shared state.

    One unreadable file among 18,000 must not end the run, and it must not
    vanish either: its reason is returned to the calling thread, which counts
    it - that is what makes the accounting identity hold without a lock.
    """
    try:
        return _decode(path, target), ""
    except (OSError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
