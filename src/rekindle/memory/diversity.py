"""Diversity: stop a memory showing the same picture twice.

Burst dedup removes near-identical frames taken within ~30 seconds. It is
working as specified and does not touch this problem: photos minutes or hours
apart are correctly outside its window. The gap was that **selection had no
diversity constraint at all** - it ranked by quality and took the top N, so
three good photos of the same child on the same afternoon each won on their
own merits and the viewer saw the same picture three times.

Measured on the rendered output before this module existed:

    album_story-wedding_arnab_pics   24 shots over 7 days,
                                     12 of them from 2016-06-24 alone
    album_story-avyan                24 shots over 18 days,
                                     3 of Avyan on 2025-09-17

**The criterion is content, not the calendar.** A per-day cap was the first
design here and it was wrong in both directions: a single-day album has every
right to spend all 24 slots on that day as long as they are genuinely
different moments, and two shots of the same person in the same room in the
same clothes are redundant even four hours apart. So the day is not capped at
all; what is measured is how different two photos LOOK.

## The signal interface

`DissimilaritySignal` is a deliberate seam. Today's implementation is
deterministic pixel statistics already stored in the index; a semantic
embedding distance is strictly better at this and is being built in another
milestone. Substituting it should be an addition, not a rewrite - so anything
answering this protocol can be dropped into `CompositeSignal`:

    class DissimilaritySignal(Protocol):
        name: str
        weight: float
        def between(self, a: Photo, b: Photo) -> float | None: ...

`between` returns 0.0 for indistinguishable, 1.0 for unrelated, and **None
when it cannot judge** - a missing fingerprint, a missing embedding. None is
not 0 and not 1: it means this signal abstains and the others decide, which is
what stops one un-fingerprinted photo suppressing its neighbours.

## What this cannot do

"Different dress or location" is only partly resolvable from pixel statistics.
The same outfit in two rooms with similar lighting will fool a colour
histogram, and a perceptual hash will call two framings of one scene different
when a viewer would call them the same photo. Semantic redundancy - six
restaurant tables on six different days in six different places - is invisible
to all of this: different pixels, same idea. That is embedding territory and
it is deliberately left to the milestone building them.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Protocol, runtime_checkable

from rekindle.memory.dedup import hamming
from rekindle.models import Photo

# How much a candidate is pushed down for looking like something already
# chosen. Quality is normalised to (0, 1], so 0.5 means a photo indis-
# tinguishable from a pick gives up half the entire quality range - enough to
# lose to a clearly worse but different photo, not enough to be erased.
LAMBDA = 0.5

# Hamming distance at which two photos are perceptually unrelated.
#
# Far looser than dedup's threshold of 6, and with no time gate. Dedup asks
# "is this the same frame?" and must almost never say yes wrongly. This asks
# "does this LOOK like one I already picked?", where a false positive costs a
# slightly worse photo rather than a deleted memory. Measured on this library:
# the median distance within a 30-second run is 22 and between unrelated
# photos is 32, so 24 catches same-scene pairs and leaves unrelated ones be.
PHASH_RADIUS = 24

# Weak, and weak on purpose. Two photos an hour apart are marginally more
# likely to be redundant than two a year apart, but the calendar is a proxy
# for content and a bad one - it is a tiebreak between otherwise equal
# candidates, never a cap.
TIME_TIEBREAK = 0.05
TIME_SATURATION = timedelta(days=1)

REJECT_TOO_SIMILAR = "too_similar_to_a_chosen_shot"

# Below this dissimilarity a candidate is refused outright rather than merely
# penalised. Only near-identity qualifies: the MMR penalty does the graded
# work and this catches the case where a photo is so close to one already
# chosen that showing both is simply a mistake.
HARD_FLOOR = 0.08


@runtime_checkable
class DissimilaritySignal(Protocol):
    """How different two photos are. See the module docstring.

    Implementations must be deterministic, must never raise, and must return
    None rather than guessing when they lack the data to judge.
    """

    name: str
    weight: float

    def between(self, a: Photo, b: Photo) -> float | None: ...


@dataclass(frozen=True)
class PerceptualSignal:
    """Structural similarity from the stored 64-bit dHash.

    Catches same framing, same composition, same room from the same angle -
    the strongest deterministic signal available, and it is already in the
    index for every image.
    """

    name: str = "phash"
    weight: float = 0.6

    def between(self, a: Photo, b: Photo) -> float | None:
        left, right = a.meta.phash, b.meta.phash
        if left is None or right is None:
            return None
        distance = hamming(left, right)
        return min(1.0, distance / PHASH_RADIUS)


@dataclass(frozen=True)
class ColourSignal:
    """Colour distribution from the stored 4x4x4 RGB histogram.

    Position-independent, which is exactly what makes it complementary to the
    perceptual hash: the hash already encodes layout, so a histogram adds
    genuinely new information about the light, the room and what people are
    wearing. Two shots of the same person in the same outfit against the same
    wall have near-identical distributions; change room or change shirt and
    they separate.

    Distance is L1 over the normalised histogram, halved so that two
    completely disjoint distributions give exactly 1.0.
    """

    name: str = "colour"
    weight: float = 0.4

    def between(self, a: Photo, b: Photo) -> float | None:
        left, right = decode_colour(a.meta.colour), decode_colour(b.meta.colour)
        if left is None or right is None:
            return None
        total = sum(abs(x - y) for x, y in zip(left, right, strict=True))
        return min(1.0, total / 2.0)


@dataclass(frozen=True)
class CompositeSignal:
    """Weighted mean of whichever signals can judge this pair.

    Signals that abstain are dropped from the mean rather than counted as
    "different", so a photo missing a colour histogram is still judged on its
    perceptual hash. When NOTHING can judge, the pair is treated as fully
    dissimilar - unknown is not similar, the same rule dedup follows, and for
    the same reason.
    """

    signals: tuple[DissimilaritySignal, ...] = (PerceptualSignal(), ColourSignal())
    name: str = "composite"
    weight: float = 1.0

    def between(self, a: Photo, b: Photo) -> float | None:
        total = 0.0
        weight = 0.0
        for signal in self.signals:
            value = signal.between(a, b)
            if value is None:
                continue
            total += value * signal.weight
            weight += signal.weight
        if weight == 0.0:
            return None
        return total / weight


DEFAULT_SIGNAL = CompositeSignal()


def _colour_hex(values: list[float]) -> str:
    return "".join(f"{min(255, max(0, round(v * 255))):02x}" for v in values)


def encode_colour(values: list[float]) -> str:
    """A normalised histogram as hex. 64 bins, one byte each, 128 characters.

    Quantising to a byte loses precision that the L1 comparison does not need
    and keeps the column at 128 bytes a row - 2.5 MB across this library.
    """
    return _colour_hex(values)


def decode_colour(raw: str | None) -> list[float] | None:
    """Hex back to a normalised histogram, or None if absent or malformed.

    Malformed returns None rather than raising: a corrupt value should make
    this signal abstain, not take down a render.
    """
    if not raw or len(raw) % 2:
        return None
    try:
        values = [int(raw[i : i + 2], 16) / 255.0 for i in range(0, len(raw), 2)]
    except ValueError:
        return None
    total = sum(values)
    if total <= 0:
        return None
    return [v / total for v in values]


@dataclass
class DiversityReport:
    """What diversity refused, and why.

    A user who wonders where their favourite photo went deserves an answer, so
    this is counted and surfaced like every other guardrail.
    """

    considered: int = 0
    picked: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    # Picks that took a similarity penalty but were still chosen - how much
    # graded work the signal did, as opposed to outright refusals.
    penalised: int = 0
    # Refused shots that came back because the memory would otherwise be short.
    restored: int = 0

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    @property
    def total_rejected(self) -> int:
        return sum(self.rejected.values())

    def merge(self, other: DiversityReport) -> None:
        self.considered += other.considered
        self.picked += other.picked
        self.penalised += other.penalised
        self.restored += other.restored
        for reason, count in other.rejected.items():
            self.rejected[reason] = self.rejected.get(reason, 0) + count


def _time_gap(a: Photo, b: Photo) -> float:
    """Normalised time distance in [0, 1], saturating after a day."""
    left, right = a.meta.taken_at_utc, b.meta.taken_at_utc
    if left is None or right is None:
        return 1.0
    gap = abs((left - right).total_seconds())
    return min(1.0, gap / TIME_SATURATION.total_seconds())


def pick(
    candidates: list[Photo],
    slots: int,
    *,
    rank: Callable[[list[Photo]], list[Photo]],
    already: list[Photo] | None = None,
    signal: DissimilaritySignal = DEFAULT_SIGNAL,
    lam: float = LAMBDA,
) -> tuple[list[Photo], DiversityReport]:
    """Greedy maximal-marginal-relevance pick.

        value = quality - lambda * similarity_to_the_nearest_already_picked
                        + a weak bonus for being far apart in time

    Deterministic throughout: quality comes from the caller's ranking (itself
    a total order), the penalty is arithmetic over stored values, and ties
    break on ranked position. The same library produces the same memory, byte
    for byte.

    `already` is what other buckets have contributed so far, so diversity
    applies ACROSS a memory and not merely within one temporal bucket -
    otherwise filling a year's slot could still return three near-identical
    shots from one day of that year.
    """
    report = DiversityReport(considered=len(candidates))
    if slots <= 0 or not candidates:
        return [], report

    ordered = rank(candidates)
    # Quality by RANK POSITION, not the raw score: the raw score ranges over
    # 0..10 depending on which metadata a library happens to have, so lambda
    # would mean something different in every library.
    size = len(ordered)
    quality = {p.file_hash: (size - i) / size for i, p in enumerate(ordered)}
    position = {p.file_hash: i for i, p in enumerate(ordered)}

    chosen: list[Photo] = list(already or [])
    picked: list[Photo] = []
    refused: list[Photo] = []
    remaining = list(ordered)

    while remaining and len(picked) < slots:
        best: Photo | None = None
        best_value = -math.inf
        best_dissimilarity = 1.0
        for photo in remaining:
            dissimilarity, spacing = _closest(photo, chosen, signal)
            value = quality[photo.file_hash] - lam * (1.0 - dissimilarity) + TIME_TIEBREAK * spacing
            # Strictly greater, so ties fall to the earlier ranked position -
            # `remaining` is in ranked order.
            if value > best_value:
                best, best_value, best_dissimilarity = photo, value, dissimilarity

        assert best is not None
        remaining.remove(best)

        if chosen and best_dissimilarity < HARD_FLOOR:
            # Near-identical to something already chosen. Showing both is not
            # a judgement call.
            report.reject(REJECT_TOO_SIMILAR)
            refused.append(best)
            continue

        if chosen and best_dissimilarity < 1.0:
            report.penalised += 1
        picked.append(best)
        chosen.append(best)

    # A memory must not end up SHORT because of diversity. If slots remain,
    # take the refused shots back best-first: a slightly repetitive memory
    # beats a truncated one, and their rejection stays in the report so the
    # restoration is visible rather than silent.
    if len(picked) < slots and refused:
        for photo in sorted(refused, key=lambda p: position[p.file_hash]):
            if len(picked) >= slots:
                break
            picked.append(photo)
            report.restored += 1

    report.picked = len(picked)
    picked.sort(key=lambda p: position[p.file_hash])
    return picked, report


def _closest(photo: Photo, chosen: list[Photo], signal: DissimilaritySignal) -> tuple[float, float]:
    """(dissimilarity to the most similar chosen photo, time gap to it).

    1.0 when nothing has been chosen yet, so the first pick is decided purely
    on quality.
    """
    if not chosen:
        return 1.0, 1.0
    best = 1.0
    spacing = 1.0
    for other in chosen:
        value = signal.between(photo, other)
        if value is None:
            # This signal cannot judge the pair. Unknown is not similar.
            continue
        if value < best:
            best = value
            spacing = _time_gap(photo, other)
    return best, spacing
