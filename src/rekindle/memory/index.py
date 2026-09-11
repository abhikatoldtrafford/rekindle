"""The chokepoint. The only way a recipe can obtain a photo.

> "Exclusions must be configurable and honoured everywhere: one chokepoint
> that every recipe goes through - not a filter each recipe remembers to
> apply. If a recipe can bypass the guardrail, the guardrail does not exist."

So the guardrail is not a filter. `MemoryIndex.open` loads every row, applies
the policy ONCE, and keeps only the survivors. No method on this class can
return a photo the policy rejected, because no rejected photo is ever stored
on the object.

A recipe is handed a `MemoryIndex` and nothing else - no `PhotoStore`, no
path, no connection. Bypassing the guardrail requires importing `PhotoStore`
yourself, which is exactly the reviewable act it should be. `tests/
test_memory_index.py` enumerates every public method and asserts none of them
leaks, so a query added later without filtering fails automatically.
"""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from rekindle.db import PhotoStore
from rekindle.memory.policy import ExclusionPolicy, is_public_safe
from rekindle.models import MediaType, Photo

# Coarse enough that one town is one cell, fine enough that two towns are not.
# The reference library's 2,318 GPS photos fall into 19 cells at this size.
GPS_CELL = 0.25


@dataclass
class ExclusionReport:
    """What was withheld, counted by FIRST matching reason.

    Counts only - never the photos themselves, and never their paths. This is
    what `rekindle memories` prints, and a report that named the excluded
    photos would defeat the exclusion.
    """

    total: int = 0
    allowed: int = 0
    by_reason: dict[str, int] = field(default_factory=dict)

    @property
    def excluded(self) -> int:
        return self.total - self.allowed

    @property
    def accounted(self) -> bool:
        return self.excluded == sum(self.by_reason.values())


def gps_cell(lat: float, lon: float, size: float = GPS_CELL) -> tuple[float, float]:
    """Snap a coordinate to a grid cell.

    Floor division, not `round`: rounding puts the boundary in the middle of a
    cell, so two photos 10 metres apart across a rounding boundary land in
    different cells while the cell CENTRES stay stable. Floor also behaves
    correctly for negative coordinates, where `int()` truncates towards zero
    and would fold the southern and northern hemispheres together at 0.
    """
    return (
        round((lat // size) * size, 6),
        round((lon // size) * size, 6),
    )


class MemoryIndex:
    """Guardrailed, in-memory view of the library.

    Everything is materialised and indexed up front. The whole library is
    19,480 rows and a few tens of megabytes of metadata; every recipe scans it
    several times, and doing that in SQL would mean each recipe writing its own
    WHERE clause - which is precisely the "filter each recipe remembers to
    apply" that this class exists to prevent.
    """

    def __init__(self, photos: list[Photo], policy: ExclusionPolicy, report: ExclusionReport):
        # Private and a tuple: a recipe holding a reference cannot append a
        # photo that never passed the policy.
        self._photos: tuple[Photo, ...] = tuple(photos)
        self._policy = policy
        self.report = report
        self._by_hash = {p.file_hash: p for p in self._photos}
        self._build_indexes()

    @classmethod
    def open(cls, store: PhotoStore, policy: ExclusionPolicy | None = None) -> MemoryIndex:
        policy = policy or ExclusionPolicy()
        report = ExclusionReport()
        allowed: list[Photo] = []
        for photo in store.iter_photos():
            report.total += 1
            reason = policy.deny_reason(photo)
            if reason is None:
                allowed.append(photo)
            else:
                report.by_reason[reason] = report.by_reason.get(reason, 0) + 1
        report.allowed = len(allowed)
        return cls(allowed, policy, report)

    def _build_indexes(self) -> None:
        self._by_year: dict[int, list[Photo]] = defaultdict(list)
        self._by_month: dict[int, list[Photo]] = defaultdict(list)
        self._by_month_day: dict[tuple[int, int], list[Photo]] = defaultdict(list)
        self._by_person: dict[str, list[Photo]] = defaultdict(list)
        self._by_pair: dict[tuple[str, str], list[Photo]] = defaultdict(list)
        self._by_album: dict[str, list[Photo]] = defaultdict(list)
        self._by_cell: dict[tuple[float, float], list[Photo]] = defaultdict(list)

        aliases = self._policy.album_aliases
        for photo in self._photos:
            local = photo.meta.taken_at_local
            # `deny_reason` already rejected a dateless photo, so this is a
            # belt-and-braces narrowing for the type checker rather than a
            # reachable branch.
            if local is None:  # pragma: no cover
                continue
            self._by_year[local.year].append(photo)
            self._by_month[local.month].append(photo)
            self._by_month_day[(local.month, local.day)].append(photo)

            people = sorted({p for p in photo.meta.people if p})
            for person in people:
                self._by_person[person].append(photo)
            # Sorted pairs, so "A and B" and "B and A" are one key. Without
            # this the pair recipe offers every pair twice.
            for pair in itertools.combinations(people, 2):
                self._by_pair[pair].append(photo)

            for album in photo.albums:
                self._by_album[aliases.get(album, album)].append(photo)

            if photo.meta.gps is not None:
                self._by_cell[gps_cell(photo.meta.gps.lat, photo.meta.gps.lon)].append(photo)

    # ---- queries. Every one reads self._photos or an index derived from it.

    def all(self) -> list[Photo]:
        return list(self._photos)

    def count(self) -> int:
        return len(self._photos)

    def get(self, file_hash: str) -> Photo | None:
        return self._by_hash.get(file_hash)

    def by_year(self, year: int) -> list[Photo]:
        return list(self._by_year.get(year, ()))

    def by_month(self, month: int) -> list[Photo]:
        return list(self._by_month.get(month, ()))

    def by_month_day(self, month: int, day: int) -> list[Photo]:
        return list(self._by_month_day.get((month, day), ()))

    def by_person(self, person: str) -> list[Photo]:
        return list(self._by_person.get(person, ()))

    def by_pair(self, a: str, b: str) -> list[Photo]:
        return list(self._by_pair.get(tuple(sorted((a, b))), ()))  # type: ignore[arg-type]

    def by_album(self, album: str) -> list[Photo]:
        return list(self._by_album.get(album, ()))

    def by_gps_cell(self, cell: tuple[float, float]) -> list[Photo]:
        return list(self._by_cell.get(cell, ()))

    def images(self) -> list[Photo]:
        return [p for p in self._photos if p.media_type is MediaType.IMAGE]

    # ---- aggregate views, for building offers cheaply

    def years(self) -> list[int]:
        return sorted(self._by_year)

    def months(self) -> list[int]:
        return sorted(self._by_month)

    def month_days(self) -> list[tuple[int, int]]:
        return sorted(self._by_month_day)

    def people_counts(self) -> Counter[str]:
        return Counter({name: len(v) for name, v in self._by_person.items()})

    def pair_counts(self) -> Counter[tuple[str, str]]:
        return Counter({pair: len(v) for pair, v in self._by_pair.items()})

    def album_counts(self) -> Counter[str]:
        return Counter({name: len(v) for name, v in self._by_album.items()})

    def gps_cells(self) -> Counter[tuple[float, float]]:
        return Counter({cell: len(v) for cell, v in self._by_cell.items()})

    def years_present(self, photos: list[Photo]) -> set[int]:
        return {p.meta.taken_at_local.year for p in photos if p.meta.taken_at_local}

    # ---- public-safe

    def is_public_safe(self, photo: Photo) -> bool:
        return is_public_safe(photo, self._policy.public_safe_allow)

    @property
    def public_safe_allow(self) -> frozenset[str]:
        return self._policy.public_safe_allow

    # ---- paths

    def resolve_path(self, photo: Photo) -> Path | None:
        """The first path of this photo that exists on disk.

        A photo routinely has the same bytes in two folders, and a library can
        be partially mounted or partially extracted, so `paths[0]` is not
        reliably the one that is there. Returns None when none of them are,
        which the renderer reports as a dropped shot rather than crashing.
        """
        for path in photo.paths:
            try:
                if path.is_file():
                    return path
            except OSError:
                # A path too long for the platform, or an unreachable network
                # share. Not a reason to abort a render.
                continue
        return None

    def earliest(self, photos: list[Photo]) -> Photo | None:
        return min(photos, key=_chrono) if photos else None

    def latest(self, photos: list[Photo]) -> Photo | None:
        return max(photos, key=_chrono) if photos else None


def _chrono(photo: Photo) -> tuple[datetime, str]:
    """Chronological, with file_hash breaking ties.

    1,430 timestamps in the reference library are shared by more than one
    photo (one by 40), so a bare date key is not a total order and
    `min()`/`max()` would depend on input order.
    """
    assert photo.meta.taken_at_utc is not None  # guaranteed by deny_reason
    return (photo.meta.taken_at_utc, photo.file_hash)
