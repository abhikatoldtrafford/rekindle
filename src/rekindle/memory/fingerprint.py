"""Perceptual fingerprints: one decode per photo, two numbers out.

Burst dedup (`memory.dedup`) needs to know whether two photos taken seconds
apart show the same thing. That question cannot be answered from metadata, so
it needs the pixels - and decoding 18,363 images is far too expensive to do on
every render. This module computes the fingerprint ONCE and the store keeps it.

Why dHash and not aHash or a DCT pHash, measured on 414 real photos drawn from
real 30-second runs in the reference library:

    threshold   within-burst pairs <= T     unrelated pairs <= T
        4              15.99%                     0.000%
        6              18.37%                     0.033%
        8              22.79%                     0.067%
       12              29.25%                     0.133%

aHash reached comparable recall at 4-5x worse specificity (at T=12: 42.18%
within-burst but 1.37% of unrelated pairs). A DCT pHash would likely be better
still, but needs either numpy - a new runtime dependency for one function - or
a pure-Python DCT that costs more than the JPEG decode it follows. dHash is
cheap, dependency-free, and measured good enough.

The number that shaped the whole design: the MEDIAN dHash distance between two
photos in the same 30-second run is 22. Photos taken seconds apart are usually
genuinely different photos, so time alone can never be the dedup test.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageChops, ImageStat

from rekindle.db import PhotoStore
from rekindle.models import MediaType, Photo

# The hash reads an 8x8 grid of horizontal gradients, so it needs 9 columns.
_HASH_W, _HASH_H = 9, 8
# Sharpness is measured at a FIXED size so that a 4000px and a 1200px photo
# produce comparable numbers. Without this, resolution alone would decide
# every burst.
_SHARP = 128
# Hint to libjpeg that we only need a small image. Measured on real files:
# 18.6 ms/photo with draft() against 52.2 ms without - a 2.9x speedup, and the
# difference between a 5.6-minute and a 16-minute pass over this library.
# libjpeg only scales by 1/2, 1/4, 1/8, so the exact value here matters far
# less than calling it at all.
_DRAFT = (64, 64)

# Values for `phash_error`. A recorded reason is FINISHED work: the resume
# predicate skips these rows, so a permanently undecodable file is not
# re-decoded on every run.
ERR_VIDEO = "video"
ERR_UNREADABLE = "unreadable"
ERR_UNDECODABLE = "undecodable"
ERR_MISSING = "missing"


def dhash(image: Image.Image) -> int:
    """64-bit difference hash: is each pixel brighter than the one to its right?

    Gradients rather than absolute levels, which is what makes it survive a
    brightness or exposure shift between two frames of one burst while still
    separating two different scenes.

    Bit order is most-significant-first, row-major from the top-left. It is
    arbitrary but it is PERSISTED, so changing it silently invalidates every
    stored hash; `test_fingerprint.py` pins it against a known constant.
    """
    small = image.convert("L").resize((_HASH_W, _HASH_H), Image.Resampling.BILINEAR)
    px = small.tobytes()
    bits = 0
    for y in range(_HASH_H):
        row = y * _HASH_W
        for x in range(_HASH_W - 1):
            bits = (bits << 1) | (1 if px[row + x] > px[row + x + 1] else 0)
    return bits


def sharpness(image: Image.Image) -> float:
    """Mean absolute horizontal gradient of a fixed-size grayscale copy.

    A RELATIVE focus measure. It is meaningful for ranking frames of one burst
    shot on one camera - which is the only thing dedup asks of it - and it is
    NOT an absolute quality score: a busy scene out of focus can beat a plain
    scene in focus, and two cameras are not comparable. Nothing in this
    codebase may use it to claim a photo is good, only that it is the sharper
    of two near-identical frames.

    Note also that measuring at 128x128 discards the fine detail where mild
    blur actually lives. Within a burst, where the alternative is ranking by
    file size, it is still the better signal - but it will not detect blur the
    way a full-resolution variance-of-Laplacian would.

    Done with ImageChops/ImageStat rather than a Python loop over 16k pixels:
    both run in C, and the loop version measurably dominated the JPEG decode.
    """
    gray = image.convert("L").resize((_SHARP, _SHARP), Image.Resampling.BILINEAR)
    left = gray.crop((0, 0, _SHARP - 1, _SHARP))
    right = gray.crop((1, 0, _SHARP, _SHARP))
    return float(ImageStat.Stat(ImageChops.difference(left, right)).mean[0])


@dataclass(frozen=True)
class Fingerprint:
    phash: int | None
    sharpness: float | None
    error: str | None

    @property
    def ok(self) -> bool:
        return self.phash is not None


def fingerprint_file(path: Path) -> Fingerprint:
    """Decode once, return both numbers. Never raises.

    A library of 19,480 files reliably contains a few that are truncated,
    locked, or not really images. One of them must not abort a six-minute
    pass, so every failure becomes a recorded `error` instead - which is also
    what stops the next run retrying it forever.
    """
    try:
        with Image.open(path) as im:
            # draft() is a no-op on formats that do not support it, so it is
            # safe to call unconditionally; on JPEG it is the whole speedup.
            im.draft("L", _DRAFT)
            im.load()
            return Fingerprint(phash=dhash(im), sharpness=sharpness(im), error=None)
    except FileNotFoundError:
        return Fingerprint(None, None, ERR_MISSING)
    except OSError:
        # Pillow raises OSError for both "cannot open" and "truncated file".
        # UnidentifiedImageError is an OSError subclass, so it lands here too.
        return Fingerprint(None, None, ERR_UNDECODABLE)
    except (ValueError, MemoryError, Image.DecompressionBombError):
        # DecompressionBombError is a subclass of Exception, not OSError, and
        # a malformed header claiming a 50-gigapixel image is exactly the kind
        # of file that reaches this code. Caught by name so the intent is
        # visible rather than relying on a broad except.
        return Fingerprint(None, None, ERR_UNDECODABLE)


@dataclass
class FingerprintReport:
    """What a pass did. Every photo it touched lands in exactly one bucket.

    `considered == hashed + failed + skipped_video` is asserted by the tests:
    silently dropping input is the bug the accounting identities in this
    project exist to prevent.
    """

    considered: int = 0
    hashed: int = 0
    failed: int = 0
    skipped_video: int = 0
    errors: dict[str, int] = field(default_factory=dict)

    def _fail(self, reason: str) -> None:
        self.errors[reason] = self.errors.get(reason, 0) + 1

    @property
    def accounted(self) -> bool:
        return self.considered == self.hashed + self.failed + self.skipped_video


def _rows(
    photos: list[Photo],
) -> Iterator[tuple[str, int | None, float | None, str | None]]:
    for photo in photos:
        if photo.media_type is MediaType.VIDEO:
            yield (photo.file_hash, None, None, ERR_VIDEO)
            continue
        yield _row_for(photo)


def _row_for(photo: Photo) -> tuple[str, int | None, float | None, str | None]:
    """First readable path wins.

    A photo can be the same bytes in two folders, and on a partially mounted
    or partially extracted library one copy may be gone while the other is
    fine. Trying only paths[0] would record a spurious `missing`.
    """
    last = Fingerprint(None, None, ERR_MISSING)
    for path in photo.paths:
        last = fingerprint_file(path)
        if last.ok:
            break
    return (photo.file_hash, last.phash, last.sharpness, last.error)


def run_fingerprints(
    store: PhotoStore,
    *,
    batch_size: int = 500,
    on_progress: Callable[[int, int], None] | None = None,
) -> FingerprintReport:
    """Fingerprint every photo that has not been attempted yet.

    Resumable by construction: `iter_unfingerprinted` excludes both rows that
    succeeded and rows that failed, so an interrupted pass resumes where it
    stopped rather than restarting. Writes in batches so that an interruption
    loses at most `batch_size` photos of work, not the whole run - a
    single-transaction-per-run rule is right for `index` (seconds) and wrong
    here (minutes).

    Videos are never decoded. Getting a frame out of one needs ffmpeg, which
    must stay optional, so they are marked and skipped - which also means they
    never participate in dedup, exactly as designed.
    """
    report = FingerprintReport()
    # Materialised: iter_unfingerprinted yields from a live cursor and this
    # function writes through the same connection.
    todo = list(store.iter_unfingerprinted())
    total = len(todo)

    batch: list[tuple[str, int | None, float | None, str | None]] = []
    for index, row in enumerate(_rows(todo), start=1):
        _, phash, _sharp, error = row
        report.considered += 1
        if error == ERR_VIDEO:
            report.skipped_video += 1
        elif phash is None:
            report.failed += 1
            report._fail(error or ERR_UNDECODABLE)
        else:
            report.hashed += 1
        batch.append(row)
        if len(batch) >= batch_size:
            store.set_fingerprints(batch)
            batch.clear()
            if on_progress:
                on_progress(index, total)
    if batch:
        store.set_fingerprints(batch)
    if on_progress:
        on_progress(total, total)
    return report
