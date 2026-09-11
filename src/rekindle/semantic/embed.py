"""`rekindle embed`: fill an embedding store from the photo index.

MEASURED FIRST, THEN WRITTEN
----------------------------
The naive loop - open, convert, hand to the processor, encode - runs at
**6.1 images/sec** on this machine with CLIP ViT-L/14 on an RTX A4000. The GPU
is idle for 93% of that: 81.9 of the 88.4 seconds spent on 540 photos was
JPEG decode and resize on one core. Buying a faster GPU would have changed
nothing.

Two fixes, both in `_decode`:

  * **`Image.draft()`** asks libjpeg to decode at 1/2, 1/4 or 1/8 scale using
    the DCT coefficients directly. A 4000x3000 photo destined for a 224x224
    crop is decoded at 500x375 - the same pixels the resize would have thrown
    away, never materialised. JPEG only; it is a no-op on PNG and is safe to
    call unconditionally.
  * **A decode thread pool.** Pillow releases the GIL inside the C decoder, so
    threads (not processes) get real parallelism with no pickling of images
    across a process boundary. The pool runs one batch ahead of the GPU.

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
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rekindle.semantic.encoder import ImageTextEncoder
from rekindle.semantic.photos import IndexedPhoto
from rekindle.semantic.store import EmbeddingStore

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image

#: Batch handed to the model at once. 64 x 224 x 224 x fp16 through ViT-L/14
#: peaks at about 1.2 GB of VRAM - comfortable on a 16 GB card and large
#: enough that per-call overhead has stopped mattering.
DEFAULT_BATCH = 32
#: Decode workers. Six physical cores; more threads than cores makes decode
#: contend with the main thread's tensor marshalling for no gain.
DEFAULT_WORKERS = 6


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
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for batch in _batches(candidates, batch_size):
            usable: list[IndexedPhoto] = []
            paths: list[Path] = []
            for photo in batch:
                path = photo.existing_path()
                if path is None:
                    report.missing_file += 1
                    report.failures.append((photo.path, "file not found"))
                    continue
                usable.append(photo)
                paths.append(path)
            if not usable:
                continue

            # Decode in the pool, ACCOUNT on this thread. `report.unreadable
            # += 1` from six workers is a lost-update race: attribute += is
            # load/add/store, not atomic, and the count it corrupts is exactly
            # the one the accounting identity rests on.
            decoded = list(pool.map(lambda p: _safe_decode(p, target_px), paths))
            ok: list[tuple[IndexedPhoto, Image]] = []
            for photo, path, (image, error) in zip(usable, paths, decoded, strict=True):
                if image is None:
                    report.unreadable += 1
                    report.failures.append((path, error or "decode failed"))
                    continue
                ok.append((photo, image))
            if not ok:
                done += len(batch)
                if progress is not None:
                    progress(min(done, total), total)
                continue
            vectors = encoder.encode_images([im for _, im in ok])
            store.add_many(
                [(photo.file_hash, vec) for (photo, _), vec in zip(ok, vectors, strict=True)]
            )
            report.embedded += len(ok)
            for _, im in ok:
                im.close()
            done += len(batch)
            if progress is not None:
                progress(min(done, total), total)
    report.elapsed_s = time.perf_counter() - started
    return report


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
