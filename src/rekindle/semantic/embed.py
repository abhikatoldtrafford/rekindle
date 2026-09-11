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
`EmbedReport` obeys `considered == already + embedded + missing + unreadable`,
asserted by a test. M1's decision log is unambiguous about why: a pipeline that
silently drops input is a bug, and the accounting identity is what makes the
drop impossible to hide.
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


@dataclass
class EmbedReport:
    considered: int = 0
    already_embedded: int = 0
    embedded: int = 0
    missing_file: int = 0
    unreadable: int = 0
    elapsed_s: float = 0.0
    model_key: str = ""
    model_revision: str = ""
    runtime: str = ""
    device: str = ""
    failures: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def images_per_s(self) -> float:
        return self.embedded / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def accounted(self) -> bool:
        """Every considered photo landed in exactly one bucket."""
        return self.considered == (
            self.already_embedded + self.embedded + self.missing_file + self.unreadable
        )


def _decode(path: Path, target: int) -> Image:
    """Open `path` as RGB, decoded no smaller than `target` on the short side.

    `draft` mutates the image in place and only ever picks a scale whose
    result is still at least as large as the requested size, so the subsequent
    high-quality resize has the pixels it needs. It is the single biggest
    speed lever in this module.
    """
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        # A generous multiple of the target: draft rounds DOWN to a power-of-two
        # scale, and asking for exactly `target` can land on a scale whose
        # output is slightly smaller than the crop, which upsamples.
        im.draft("RGB", (target * 2, target * 2))
        return im.convert("RGB")


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
) -> EmbedReport:
    """Embed every photo that has no vector yet. Resumable; safe to re-run."""
    report = EmbedReport(
        model_key=store.model_key,
        model_revision=store.model_revision,
        runtime=runtime,
        device=device,
    )
    candidates: list[IndexedPhoto] = []
    have = store.hashes()
    seen: set[str] = set()
    for photo in photos:
        report.considered += 1
        if photo.file_hash in have or photo.file_hash in seen:
            report.already_embedded += 1
            continue
        seen.add(photo.file_hash)
        candidates.append(photo)
        if limit is not None and len(candidates) >= limit:
            break

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
                    ]
                )
                report.embedded += len(work.photos)
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
