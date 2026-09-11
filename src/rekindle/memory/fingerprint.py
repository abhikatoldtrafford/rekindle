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

from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat

from rekindle.db import FingerprintRow, PhotoStore
from rekindle.models import MediaType, Photo

# The hash reads an 8x8 grid of horizontal gradients, so it needs 9 columns.
_HASH_W, _HASH_H = 9, 8
# Brightness is measured at a FIXED small size: it is a mean, so detail adds
# nothing and a 128px copy is far cheaper than the decoded image.
_BRIGHT = 128

# Sharpness works on an aspect-preserving copy whose LONG edge is this, cut
# into tiles of roughly this many pixels. See `sharpness` for why 1024 rather
# than the 128 this shipped with.
_SHARP_LONG_EDGE = 1024
_SHARP_TILE = 128
# How hard the reference reblur is. 1.0px is deliberately close to the
# sampling limit: the question the measure asks is whether the image still
# HAS detail at the finest scale it can represent.
_SHARP_REBLUR = 1.0
# A tile below this is too small for its gradient statistics to mean
# anything, so a small image is measured as a single tile instead.
_SHARP_MIN_TILE = 16

# Hint to libjpeg how much image we need. Measured on 123 real files:
#
#     draft           ms/photo   long edge (median)
#     ("L", 64)          13.6            486        <- what M2 shipped
#     ("RGB", 64)        14.5            486
#     ("RGB", 1024)      24.6           1984        <- now
#
# Two changes, both deliberate. The SIZE, because `sharpness` cannot see mild
# blur in a 486px copy of a 4000px photo (measured: AUC 0.52 - a coin flip).
# The MODE, because "L" makes libjpeg decode only the luma plane, so
# `colour_signature` was running on a grey image and every stored histogram
# was a luminance histogram: measured over 2,000 real rows, a median of 42 of
# the 64 bins were exactly zero and 55% of the mass sat on the four grey bins.
#
# libjpeg only scales by 1/2, 1/4, 1/8, so the result is the smallest of those
# still at least 1024 on the long edge - never smaller than the request.
_DRAFT = (1024, 1024)

# Values for `phash_error`. A recorded reason is FINISHED work: the resume
# predicate skips these rows, so a permanently undecodable file is not
# re-decoded on every run.
ERR_VIDEO = "video"
ERR_UNREADABLE = "unreadable"
ERR_UNDECODABLE = "undecodable"
ERR_MISSING = "missing"

# Shared with meta.exif: only these EXIF orientations exchange the two axes.
_ORIENTATION = 0x0112
_SWAPS_AXES = frozenset({5, 6, 7, 8})


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
    """How much of the finest detail survives a one-pixel reblur, 0.0 to 1.0.

    For each tile: reblur it by one pixel and report the FRACTION of the local
    gradient energy that the reblur destroys. An image that still carries
    detail at the pixel scale loses most of it; an image already blurred has
    little left to lose, so it barely changes. The image's score is its
    SHARPEST tile.

    Three properties, each measured rather than assumed:

    **It sees mild blur, which the 128x128 measure this replaces could not.**
    Over 123 real photos spread across every year, each also rendered with a
    mild blur (radius = long edge / 1200) and a gross one (/250), the
    probability that a sharp photo outscores a blurred one:

        measure                          sharp/mild   sharp/gross
        mean gradient at 128x128 (old)      0.519        0.704
        max tile gradient at 1024           0.793        0.991
        this one                            0.903        1.000

    0.519 is a coin flip. Downscaling a 4000px photo to 128px low-pass filters
    away exactly the detail that mild blur removes, so the old measure was
    reading a signal that had already been destroyed.

    On 44 photos hand-graded at 100% zoom - drawn deliberately from the low
    tail of BOTH measures, so these are the hard cases - sharp against mildly
    blurred was 0.682 for the old measure and 0.793 for this one, and sharp
    against grossly blurred 0.600 against 0.800.

    **It is a RATIO, so scene contrast cancels.** This is what makes a single
    threshold safe across a library spanning 2000-2026. Measured per-year 5th
    percentiles over 2,240 photos, 120 per year: they span 4.70x for the old
    measure and 4.44x for an unnormalised gradient at 1024, but only 1.71x for
    this one. A low-contrast 2011 photo is no longer indistinguishable from a
    blurred one - which is exactly the failure that forced the old gate to sit
    far out in the tail to be safe.

    **The sharpest tile, not the mean, so shallow depth of field survives.** A
    portrait with a sharp face and a deliberately blurred background is often
    the best photo in the set and scores badly on any whole-image measure.
    Measured on a synthetic shallow-DoF version of each of the 123 photos
    (sharp centre, blurred surround), the median retention of the fully-sharp
    score is 0.800 taking the max tile, 0.330 at the 90th percentile and 0.006
    at the 75th - the choice of MAX is doing almost all of the work here, not
    the tiling. A mitigation, not a fix: a subject smaller than one tile is
    still missed, and only M3's face boxes can close that.

    **What it is blind to, found by a test that failed.** The gradient sum
    across an isolated step edge does not change when the edge is spread over
    three pixels instead of one, so a picture made only of hard edges is very
    nearly invisible to this: a 40px checkerboard measures 0.043 sharp and
    0.009 after a radius-4 blur, both far below the gate. What the measure
    reads is the loss of fine TEXTURE. That is the right thing for a
    photograph, which is texture nearly everywhere, and it is why the measure
    is contrast-invariant at all - but a subject that is genuinely all flat
    regions and hard borders (a sign, a screenshot, a document) scores low
    whether or not it is in focus. `is_screenshot` already removes the common
    case; the rest are caught by the gate sitting at the 1st percentile.

    Still a RELATIVE measure and still not a quality score. A grossly blurred
    photo of a high-contrast scene can outscore a sharp photo of a soft one -
    measured, it happens - which is why the gate in `memory.composition` sits
    at the 1st percentile rather than anywhere near the middle.

    ImageChops/ImageStat rather than a Python loop: both run in C, and the
    loop version measurably dominated the JPEG decode.
    """
    gray = _downscale(image.convert("L"), _SHARP_LONG_EDGE)
    soft = gray.filter(ImageFilter.GaussianBlur(_SHARP_REBLUR))
    best = 0.0
    for box in _tiles(gray.size):
        crisp = _gradient_energy(gray.crop(box))
        if crisp <= 0.0:
            # A flat tile has no detail to lose. Skipping it is right: it
            # carries no evidence either way, and because the score is a
            # maximum, a flat region cannot drag a sharp one down.
            continue
        best = max(best, min(1.0, (crisp - _gradient_energy(soft.crop(box))) / crisp))
    return best


def _downscale(gray: Image.Image, longest: int) -> Image.Image:
    """Aspect-preserving, and it NEVER upscales.

    Aspect-preserving because the old square resize stretched a 16:9 photo,
    which moves its horizontal and vertical detail into different bands.
    Never upscaling because an enlarged copy is smooth at the pixel scale by
    construction, so upscaling a small photo would score it as blurred.
    """
    width, height = gray.size
    longest_edge = max(width, height)
    if longest_edge <= longest:
        return gray
    scale = longest / longest_edge
    return gray.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.BILINEAR,
    )


def _tiles(size: tuple[int, int]) -> Iterator[tuple[int, int, int, int]]:
    """A grid of roughly `_SHARP_TILE`-pixel boxes covering the whole image.

    The COUNT adapts rather than the tile size: a fixed 8x8 grid would cut a
    320px image into 40px tiles whose gradient statistics are mostly noise,
    and a maximum over 64 noisy tiles biases the score upward - the one
    direction a blur gate must not be biased. Edges are computed from the
    index so the tiles tile exactly, with no seam and no overlap.
    """
    width, height = size
    count = max(1, min(8, max(width, height) // _SHARP_TILE))
    if width // count < _SHARP_MIN_TILE or height // count < _SHARP_MIN_TILE:
        count = 1
    for ty in range(count):
        for tx in range(count):
            yield (
                width * tx // count,
                height * ty // count,
                width * (tx + 1) // count,
                height * (ty + 1) // count,
            )


def _gradient_energy(tile: Image.Image) -> float:
    """Mean absolute gradient, both axes.

    Both axes because a one-axis measure is blind to an edge parallel to it,
    and motion blur - the most common mild blur in this library - is
    directional by nature.
    """
    width, height = tile.size
    if width < 2 or height < 2:
        return 0.0
    horizontal = ImageChops.difference(
        tile.crop((0, 0, width - 1, height)), tile.crop((1, 0, width, height))
    )
    vertical = ImageChops.difference(
        tile.crop((0, 0, width, height - 1)), tile.crop((0, 1, width, height))
    )
    return float(ImageStat.Stat(horizontal).mean[0] + ImageStat.Stat(vertical).mean[0])


# A 4x4x4 RGB histogram: coarse enough that a slight exposure shift does not
# move a photo between bins, fine enough to separate a red shirt from a blue
# one. 64 bins is also small enough to store as 128 hex characters per row.
_COLOUR_BINS = 4
_COLOUR_SIZE = 32


def colour_signature(image: Image.Image) -> str:
    """A coarse colour distribution, hex-encoded. See memory.diversity.

    Position-independent by design: the perceptual hash already encodes
    layout, so the complementary information is WHAT colours are present -
    the light, the room, what people are wearing - rather than where.

    Computed at 32x32 because the distribution, not the detail, is the point,
    and because it is another pass over an image already decoded.
    """
    small = image.convert("RGB").resize((_COLOUR_SIZE, _COLOUR_SIZE), Image.Resampling.BILINEAR)
    counts = [0.0] * (_COLOUR_BINS**3)
    step = 256 // _COLOUR_BINS
    # tobytes() rather than getdata(): getdata() is deprecated in Pillow 14,
    # and a flat RGB byte string is both faster and stable across versions.
    raw = small.tobytes()
    for offset in range(0, len(raw), 3):
        index = (
            (raw[offset] // step) * _COLOUR_BINS * _COLOUR_BINS
            + (raw[offset + 1] // step) * _COLOUR_BINS
            + (raw[offset + 2] // step)
        )
        counts[index] += 1.0

    counts = _smooth(counts)
    # Normalised against the PEAK bin, not the total: a photo dominated by one
    # colour would otherwise quantise every other bin to zero and lose the
    # detail that distinguishes it from another photo of the same wall.
    peak = max(counts) or 1.0
    return "".join(f"{min(255, round(255 * c / peak)):02x}" for c in counts)


def _smooth(counts: list[float], keep: float = 0.55) -> list[float]:
    """Diffuse each bin into its face-adjacent neighbours.

    Hard binning makes the signature brittle exactly where it matters. With
    four bins per channel, (200,40,40) and (150,30,30) - the same red shirt
    under slightly different light - fall in different bins, and for an image
    dominated by one colour that puts ALL the mass in disjoint bins and reads
    as maximally different. Measured before smoothing: those two scored 1.000,
    the same as red against blue.

    Diffusing means neighbouring colours partially overlap, so a small
    exposure or white-balance shift moves the signature a little rather than
    completely. `keep` is the share a bin retains; the rest is split evenly
    among its up-to-six neighbours.
    """
    bins = _COLOUR_BINS
    out = [0.0] * len(counts)
    for r in range(bins):
        for g in range(bins):
            for b in range(bins):
                index = (r * bins + g) * bins + b
                value = counts[index]
                if value == 0.0:
                    continue
                neighbours = []
                for dr, dg, db in (
                    (1, 0, 0),
                    (-1, 0, 0),
                    (0, 1, 0),
                    (0, -1, 0),
                    (0, 0, 1),
                    (0, 0, -1),
                ):
                    nr, ng, nb = r + dr, g + dg, b + db
                    if 0 <= nr < bins and 0 <= ng < bins and 0 <= nb < bins:
                        neighbours.append((nr * bins + ng) * bins + nb)
                out[index] += value * keep
                if neighbours:
                    share = value * (1.0 - keep) / len(neighbours)
                    for n in neighbours:
                        out[n] += share
                else:  # pragma: no cover - a 1-bin histogram has no neighbours
                    out[index] += value * (1.0 - keep)
    return out


def brightness(image: Image.Image) -> float:
    """Mean luminance, 0-255.

    Drives the near-black and blown-out gates in `memory.composition`.
    Measured over 1,149 real photos stratified across every year in the
    library: the 1st percentile is 33 and the median 113, so the gates sit far
    out in the tails deliberately.
    """
    gray = image.convert("L").resize((_BRIGHT, _BRIGHT), Image.Resampling.BILINEAR)
    return float(ImageStat.Stat(gray).mean[0])


@dataclass(frozen=True)
class Fingerprint:
    phash: int | None
    sharpness: float | None
    error: str | None
    brightness: float | None = None
    colour: str | None = None
    # Post-rotation, i.e. what a viewer actually sees. See `fingerprint_file`.
    width: int | None = None
    height: int | None = None

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
            im.draft("RGB", _DRAFT)
            im.load()
            # ORIENTATION FIRST. Everything below measures the image a viewer
            # sees, not the bytes on disk: a phone stores a portrait photo as
            # landscape pixels plus a tag, so measuring before transposing
            # classifies it as landscape and sizes the canvas wrongly. It also
            # REPAIRS the index - M0 stored the raw size, leaving ~2,475 rows
            # with width and height swapped (see meta.exif.read_exif).
            #
            # draft() may already have scaled the image down, so `size` here
            # is NOT the native resolution; `_native_size` re-reads that from
            # the header, which is what the resolution floor needs.
            upright = ImageOps.exif_transpose(im) or im
            width, height = _native_size(path, upright)
            return Fingerprint(
                phash=dhash(upright),
                sharpness=sharpness(upright),
                brightness=brightness(upright),
                colour=colour_signature(upright),
                width=width,
                height=height,
                error=None,
            )
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


def _native_size(path: Path, upright: Image.Image) -> tuple[int | None, int | None]:
    """The full-resolution, post-rotation size.

    `Image.draft()` scales the decode down by up to 8x, so the loaded image is
    not the native size and using it would record a 4000px photo as 500px -
    which the resolution floor would then reject. Re-open the header (cheap:
    no pixel decode) and apply the same axis swap the transpose applied.
    """
    try:
        with Image.open(path) as fresh:
            width, height = fresh.size
            if fresh.getexif().get(_ORIENTATION) in _SWAPS_AXES:
                width, height = height, width
            return width, height
    except (OSError, ValueError):
        # The pixels decoded but the header re-read did not. Fall back to what
        # we have rather than dropping the photo.
        return upright.size


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


def _rows(photos: list[Photo]) -> Iterator[FingerprintRow]:
    for photo in photos:
        if photo.media_type is MediaType.VIDEO:
            yield FingerprintRow(file_hash=photo.file_hash, error=ERR_VIDEO)
            continue
        yield _row_for(photo)


def _row_for(photo: Photo) -> FingerprintRow:
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
    return FingerprintRow(
        file_hash=photo.file_hash,
        phash=last.phash,
        sharpness=last.sharpness,
        brightness=last.brightness,
        colour=last.colour,
        width=last.width,
        height=last.height,
        error=last.error,
    )


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

    batch: list[FingerprintRow] = []
    for index, row in enumerate(_rows(todo), start=1):
        report.considered += 1
        if row.error == ERR_VIDEO:
            report.skipped_video += 1
        elif row.phash is None:
            report.failed += 1
            report._fail(row.error or ERR_UNDECODABLE)
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
