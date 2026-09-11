"""Stratified selection: spreading a memory across the dimension it is about.

Without this, selection is top-N by quality score with no temporal constraint,
so shots cluster wherever the strongest-scoring run happens to sit. Measured on
the real 19,480-photo library before this module existed:

    16 of 37 rendered memories were confined to a SINGLE YEAR
    10 of 37 were confined to a single MONTH
    on_this_day:12-22   24 shots, all from 2019, while 2020 and 2022 had
                        photos available and unused
    year_in_review:2021 24 shots, all from December
    album_story:Avyan   24 shots, all from one August, from an album
                        spanning that child's life

An "on this day" showing one year defeats the entire concept of the recipe -
the point is the same date ACROSS years. The cause was structural rather than a
tuning problem: nothing fed the temporal spread back as a constraint on the
choice.

So each recipe declares the dimension its memory is *about*, slots are
allocated across the buckets of that dimension, and each bucket is then filled
best-first by the existing quality ranking.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from rekindle.memory.diversity import DiversityReport, pick
from rekindle.models import Photo

# The dimension a recipe stratifies over.
BY_YEAR = "year"
BY_MONTH = "month"
BY_DAY = "day"
# Adaptive: pick the finest granularity the memory's own span supports. For an
# album, "across its date span" means days for a one-week trip and months for
# a two-year album - a fixed choice is wrong for one of them.
BY_SPAN = "span"
# No stratification. `then_and_now` is the opposite case: it wants the
# extremes, not the spread, and stratifying it would defeat the recipe.
NONE = None

_LEVELS = (BY_DAY, BY_MONTH, BY_YEAR)


def bucket_key(photo: Photo, level: str) -> tuple:
    """A photo's bucket at a fixed granularity. Local time, always.

    A user thinks in the dates they lived, and 13,116 rows in the reference
    library have a non-UTC local zone - bucketing on the UTC instant would put
    a 00:30 photo in the previous day.
    """
    local = photo.meta.taken_at_local
    if local is None:  # pragma: no cover - the policy rejects dateless photos
        return ()
    if level == BY_YEAR:
        return (local.year,)
    if level == BY_MONTH:
        return (local.year, local.month)
    return (local.year, local.month, local.day)


def resolve_level(photos: list[Photo], level: str | None, slots: int) -> str | None:
    """Turn BY_SPAN into a concrete granularity.

    Picks the FINEST level whose bucket count still fits in the available
    slots, so every bucket can receive at least one shot. A 510-photo album
    shot over eight days in one month has one year-bucket and one month-bucket
    - only days separate it - while a 19-month album has too many days and
    wants months.

    Falls back to the coarsest level when even years outnumber the slots (a
    26-year span into 24 shots); the allocator then keeps the largest buckets
    and reports the rest as unrepresented rather than silently dropping them.
    """
    if level != BY_SPAN:
        return level
    for candidate in _LEVELS:
        if len({bucket_key(p, candidate) for p in photos}) <= max(slots, 1):
            return candidate
    return BY_YEAR


@dataclass
class StratumReport:
    """What the spread actually was, and what could not be represented.

    A bucket can be empty for a legitimate reason - every photo in that year
    failed the composition or quality gates - and that is a correct outcome.
    But it must be VISIBLE, or a lopsided memory looks like a bug.
    """

    memory_id: str = ""
    dimension: str | None = None
    # Buckets in the recipe's candidate pool, before any gate ran.
    offered: int = 0
    # Buckets still present after composition and dedup.
    surviving: int = 0
    # Buckets that actually received at least one shot.
    used: int = 0
    keys_lost_to_gates: list[str] = field(default_factory=list)
    keys_without_slots: list[str] = field(default_factory=list)
    # What the diversity caps and the perceptual penalty refused while filling
    # these buckets. Reported like every other guardrail.
    diversity: DiversityReport = field(default_factory=DiversityReport)

    @property
    def lost_to_gates(self) -> int:
        return self.offered - self.surviving

    @property
    def unslotted(self) -> int:
        return self.surviving - self.used


def allocate(sizes: dict[tuple, int], slots: int) -> dict[tuple, int]:
    """How many shots each bucket gets. Proportional, with a floor of one.

    Why not an equal split: on the real `on_this_day:12-22`, the surviving
    buckets are 2019 with 126 photos, 2020 with 2 and 2022 with 1. An equal
    split gives the year that actually has a story the same eight slots as the
    year with one photo, and two of those slots cannot even be filled.

    Why not pure proportional: that is what the unstratified code effectively
    did - 2019 takes 23 of 24 and the thin years vanish.

    So: **every non-empty bucket gets one slot before any bucket gets a
    second**, and the remainder is distributed by weight using the largest-
    remainder method. Deterministic throughout - buckets are processed in
    sorted key order and ties break on the key, never on dict iteration order.

    When there are more buckets than slots, the LARGEST buckets win the slots;
    the rest are reported as unrepresented rather than silently dropped.
    """
    present = {key: n for key, n in sizes.items() if n > 0}
    if not present or slots <= 0:
        return {}

    ordered = sorted(present, key=lambda k: (-present[k], k))
    if len(ordered) > slots:
        ordered = ordered[:slots]

    allocation = dict.fromkeys(ordered, 1)
    remaining = slots - len(ordered)

    # Largest-remainder (Hare quota) over the buckets that can still take more.
    while remaining > 0:
        hungry = [k for k in ordered if allocation[k] < present[k]]
        if not hungry:
            break
        total = sum(present[k] for k in hungry)
        exact = {k: remaining * present[k] / total for k in hungry}
        floor = {k: int(exact[k]) for k in hungry}
        # Never award more than the bucket can fill, or slots evaporate.
        floor = {k: min(floor[k], present[k] - allocation[k]) for k in hungry}
        handed = sum(floor.values())
        for key, extra in floor.items():
            allocation[key] += extra
        remaining -= handed

        if handed == 0:
            # Every exact share rounded below 1. Hand the leftovers out one at
            # a time, largest fractional remainder first, so the loop always
            # terminates and the richest bucket is favoured.
            for key in sorted(hungry, key=lambda k: (-exact[k], k)):
                if remaining == 0:
                    break
                if allocation[key] < present[key]:
                    allocation[key] += 1
                    remaining -= 1
            if remaining > 0 and all(allocation[k] >= present[k] for k in ordered):
                break
    return allocation


def stratify(
    photos: list[Photo],
    *,
    level: str | None,
    slots: int,
    rank: Callable[[list[Photo]], list[Photo]],
    offered: list[Photo] | None = None,
) -> tuple[list[Photo], StratumReport]:
    """Choose `slots` photos spread across `level`, best-first within each.

    `rank` is injected rather than imported so this module never has to know
    how a photo is scored - the spread and the ranking are separate decisions
    and stay separately testable.

    `offered` is the pre-gate candidate pool, used only to report which
    buckets the guardrails removed entirely.
    """
    report = StratumReport(dimension=level)
    if not photos:
        return [], report

    resolved = resolve_level(photos, level, slots)
    report.dimension = resolved

    if resolved is None:
        # Unstratified: the recipe wants its own choice. `then_and_now` is the
        # only such recipe and asks for exactly two shots, so diversity is a
        # no-op there - but it is applied rather than skipped, so a future
        # unstratified recipe does not silently opt out of it.
        chosen, div = pick(photos, slots, rank=rank)
        report.diversity.merge(div)
        report.offered = report.surviving = report.used = 1
        return chosen, report

    buckets: dict[tuple, list[Photo]] = {}
    for photo in photos:
        buckets.setdefault(bucket_key(photo, resolved), []).append(photo)
    report.surviving = len(buckets)

    if offered is not None:
        offered_keys = {bucket_key(p, resolved) for p in offered}
        report.offered = len(offered_keys)
        report.keys_lost_to_gates = [_label(k) for k in sorted(offered_keys - set(buckets))]
    else:
        report.offered = len(buckets)

    allocation = allocate({k: len(v) for k, v in buckets.items()}, slots)
    report.used = len(allocation)
    report.keys_without_slots = [_label(k) for k in sorted(set(buckets) - set(allocation))]

    chosen: list[Photo] = []
    for key in sorted(allocation):
        # `already=chosen` is what makes diversity apply ACROSS the memory
        # rather than only within one bucket. The ALLOCATION is binding
        # though: diversity chooses which photo fills a bucket's slot, never
        # whether that bucket gets one. Stratification decides the shape of
        # the memory; diversity decides what goes in each slot.
        taken, div = pick(buckets[key], allocation[key], rank=rank, already=chosen)
        report.diversity.merge(div)
        chosen.extend(taken)
    return chosen, report


def _label(key: tuple) -> str:
    """A bucket key as a user-readable period."""
    if len(key) == 1:
        return f"{key[0]}"
    if len(key) == 2:
        return f"{key[0]}-{key[1]:02d}"
    return f"{key[0]}-{key[1]:02d}-{key[2]:02d}"
