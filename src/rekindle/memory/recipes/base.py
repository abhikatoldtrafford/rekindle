"""The recipe protocol and the shared pieces every recipe returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from rekindle.memory import strata
from rekindle.memory.index import MemoryIndex
from rekindle.memory.spec import FactSheet
from rekindle.models import Photo

# Ordering modes a recipe may ask the engine for.
CHRONOLOGICAL = "chronological"
AS_GIVEN = "as_given"

# Below this, a memory is not worth rendering. Three is low on purpose: the
# reference library contains `Durga Puja 25` (6 photos) and `Puri 25` (3), and
# the user named the former as an expected output. A floor of 8 would have
# silently deleted it.
MIN_SHOTS = 3


@dataclass(frozen=True)
class Offer:
    """One memory this recipe could build. Cheap to produce, no pixels read.

    `key` is the STABLE subject of the memory - an album title, a person's
    name, a month-day. Combined with the recipe name it forms the memory id
    that dismissal and the cooldown are keyed on, so it must not depend on
    which photos happen to be in the library today. See memory.history.
    """

    recipe: str
    key: str
    title: str
    subtitle: str = ""
    size: int = 0

    @property
    def memory_id(self) -> str:
        return f"{self.recipe}:{self.key}"


@dataclass(frozen=True)
class Selection:
    """A recipe's answer for one offer: photos, in order, plus the facts.

    Generous and pre-cap. The engine applies composition guardrails, dedup,
    ranking and the cap afterwards, so a recipe returning 400 photos is
    normal and correct.
    """

    photos: list[Photo]
    facts: FactSheet
    ordering: str = CHRONOLOGICAL
    captions: dict[str, str] = field(default_factory=dict)
    # A recipe whose whole form is a fixed, smaller number of shots declares
    # it here. `then_and_now` is exactly two, and the engine's default floor
    # of three would otherwise reject every one of them - which it silently
    # did until a test caught it.
    min_shots: int | None = None
    # The dimension this memory is ABOUT, across which its shots are spread.
    # See memory.strata: without it, selection is top-N by quality and shots
    # cluster wherever the strongest-scoring run happens to sit - which on the
    # real library confined 16 of 37 memories to a single year.
    #
    # None means "do not stratify", which `then_and_now` needs: it wants the
    # extremes, not the spread.
    stratify: str | None = strata.BY_SPAN


@runtime_checkable
class Recipe(Protocol):
    name: str
    title: str

    def offers(self, index: MemoryIndex) -> list[Offer]: ...

    def select(self, index: MemoryIndex, offer: Offer) -> Selection | None: ...


def chronological(photos: list[Photo]) -> list[Photo]:
    """Sorted by capture time, with file_hash breaking ties.

    The tiebreak is not optional: 1,430 timestamps in the reference library
    are shared by more than one photo (one by 40), because Takeout dates are
    second-granular. Without it, `sorted` is stable with respect to an input
    order that comes from a SQLite scan, and the engine's byte-for-byte
    promise quietly depends on the database's physical layout.
    """
    return sorted(photos, key=lambda p: (p.meta.taken_at_utc, p.file_hash))


def years_of(photos: list[Photo]) -> set[int]:
    return {p.meta.taken_at_local.year for p in photos if p.meta.taken_at_local}
