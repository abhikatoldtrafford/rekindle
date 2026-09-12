"""Choosing which of somebody's photographs to ask about.

SELECTING GOOD BORDERLINE EXAMPLES IS MOST OF THE WORK
------------------------------------------------------
A calibration that shows obvious cases learns nothing. Ask "is this too
blurry?" about the sharpest photograph in a library and the answer carries no
information about where the line is; ask about the blurriest and neither does
the answer. Every question here is aimed at a target value, and the photograph
shown is the one whose measured value sits closest to it.

The target comes from `judge.next_probe`, which bisects whatever interval the
user's answers have not yet resolved. So the sequence converges: the first
question is somewhere in the middle of the offered band and each subsequent
one lands inside the gap the previous answers left.

THE BAND IS THE LIBRARY'S, NOT THE DEFAULT'S
---------------------------------------------
The range of values a step explores is read off the user's own distribution -
percentiles of their sharpness scores, of their own within-burst distances -
and not from a fixed window around rekindle's shipped number. Anchoring on the
shipped value would make it the prior, and the entire point of this exercise
is that the shipped value has met exactly one library.

EVERYTHING HERE IS DETERMINISTIC
--------------------------------
No random draws. Ties break on file hash. The same library asked the same
questions in the same order gives the same photographs, which is what makes a
sitting resumable and a bug reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rekindle.models import MediaType, Photo

#: The band a blind-judgement step explores, as percentiles of the library's
#: own values. Wide enough to contain obvious cases at both ends - the user
#: needs to be able to say a confident no somewhere - and narrow enough that
#: bisection reaches the interesting part in a handful of questions.
DEFAULT_LOW_PERCENTILE = 0.5
DEFAULT_HIGH_PERCENTILE = 25.0


@dataclass(frozen=True)
class Example:
    """One thing to show, and the number behind it."""

    value: float
    #: What identifies it on resume. One hash, or two joined by "+".
    subject: str
    #: The files to show. One for a quality question, two for a sameness one.
    paths: tuple[Path, ...]
    #: A short factual line - never the value, which would stop the judgement
    #: being blind.
    caption: str = ""


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile over a sorted copy.

    Written here rather than imported: `numpy` is behind an optional extra and
    the whole calibration path has to run on a plain `uv sync`, which is four
    packages. Eleven lines is cheaper than making the first-run experience
    depend on a 120 MB download.
    """
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (q / 100.0) * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def band(
    values: list[float],
    *,
    low_q: float = DEFAULT_LOW_PERCENTILE,
    high_q: float = DEFAULT_HIGH_PERCENTILE,
) -> tuple[float, float]:
    """The interval a step explores, read off the user's own distribution."""
    low = percentile(values, low_q)
    high = percentile(values, high_q)
    if high <= low:
        # A degenerate distribution - every photo scoring the same. Open the
        # band by a hair so bisection has somewhere to go rather than
        # dividing by zero and asking the same question forever.
        high = low + max(abs(low), 1.0) * 0.01
    return low, high


def measured(photos: list[Photo], attribute: str) -> list[tuple[float, Photo]]:
    """(value, photo) for every photo that HAS the measurement, sorted.

    Photos with no value are dropped rather than defaulted. A library that has
    not been fingerprinted has no sharpness scores at all, and asking someone
    to judge blur against a `None` treated as zero would derive a threshold
    from nothing.
    """
    out = [
        (float(getattr(p.meta, attribute)), p)
        for p in photos
        if p.media_type is MediaType.IMAGE and getattr(p.meta, attribute, None) is not None
    ]
    out.sort(key=lambda pair: (pair[0], pair[1].file_hash))
    return out


def nearest(
    pool: list[tuple[float, Photo]],
    target: float,
    *,
    used: set[str] = frozenset(),  # type: ignore[assignment]
) -> Example | None:
    """The unshown photograph whose value is closest to `target`.

    Ties break on file hash, so two runs of the same sitting ask about the
    same photographs in the same order.
    """
    best: tuple[float, str, float, Photo] | None = None
    for value, photo in pool:
        if photo.file_hash in used:
            continue
        candidate = (abs(value - target), photo.file_hash, value, photo)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        return None
    _, _, value, photo = best
    return Example(value=value, subject=photo.file_hash, paths=(photo.paths[0],))


def pairs_within(
    photos: list[Photo],
    *,
    gap_seconds: float,
    distance: "callable",  # noqa: UP037 - a plain callable, kept untyped for the stdlib-only import graph
    limit: int = 4000,
) -> list[tuple[float, Photo, Photo]]:
    """Consecutive pairs taken within `gap_seconds`, with their distance.

    CONSECUTIVE, not all pairs. Dedup only ever compares a photo to the anchor
    of a run it is already time-adjacent to, so a calibration that showed the
    user arbitrary pairs would be asking them to judge a question the code
    never asks. It is also what keeps this O(n) on a 19,480-photo library
    instead of O(n^2).

    `limit` bounds the work on a library where every photo is inside somebody's
    burst; the cap is applied AFTER sorting by distance-free file order, so it
    is the same cap on every run.
    """
    dated = [
        p for p in photos if p.media_type is MediaType.IMAGE and p.meta.taken_at_utc is not None
    ]
    dated.sort(key=lambda p: (p.meta.taken_at_utc, p.file_hash))
    out: list[tuple[float, Photo, Photo]] = []
    for a, b in zip(dated, dated[1:], strict=False):
        gap = abs((b.meta.taken_at_utc - a.meta.taken_at_utc).total_seconds())
        if gap > gap_seconds:
            continue
        value = distance(a, b)
        if value is None:
            continue
        out.append((float(value), a, b))
        if len(out) >= limit:
            break
    out.sort(key=lambda t: (t[0], t[1].file_hash, t[2].file_hash))
    return out


def nearest_pair(
    pool: list[tuple[float, Photo, Photo]],
    target: float,
    *,
    used: set[str] = frozenset(),  # type: ignore[assignment]
) -> Example | None:
    best: tuple[float, str, float, Photo, Photo] | None = None
    for value, a, b in pool:
        subject = f"{a.file_hash}+{b.file_hash}"
        if subject in used:
            continue
        candidate = (abs(value - target), subject, value, a, b)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        return None
    _, subject, value, a, b = best
    seconds = abs((b.meta.taken_at_utc - a.meta.taken_at_utc).total_seconds())
    return Example(
        value=value,
        subject=subject,
        paths=(a.paths[0], b.paths[0]),
        caption=f"taken {seconds:.0f}s apart",
    )
