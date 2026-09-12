"""Showing the consequence before writing the number.

"This keeps 412 more photographs and drops 89 you currently keep" - with
examples of both - and only then write. A number changed without seeing its
effect is a number changed blind, which is the failure this whole feature
exists to fix; reproducing it inside the fix would be embarrassing.

THREE KINDS OF CONSEQUENCE, AND THE HONEST ONE IS USED FOR EACH
----------------------------------------------------------------
Some thresholds are per-photo GATES. `min_sharpness`, `min_brightness`,
`max_brightness`, `min_short_edge` and `max_aspect` each answer yes or no
about one photograph, so their consequence is countable directly off the
index: how many photographs cross the line, and which ones.

The dedup thresholds are DELETIONS. `phash_distance` and `gap_seconds` do not
change whether a photograph is usable; they change whether it survives its own
burst. That still has a hard number - how many photographs get collapsed away
- and it needs one badly.

  Measured on the reference library while calibrating it: four honest blind
  judgements about real pairs derived a `phash_distance` of 21, and 21
  collapses 5,341 photographs where the shipped 6 collapses 2,077. An extra
  3,264 photographs - 18% of the library - would have been deleted from every
  memory on the strength of four answers. The number was not wrong about those
  four pairs; it was catastrophic as a threshold, and the only thing that
  could have caught it before it was written is this count. So it exists.

The rest are neither. `lambda_penalty` reorders a pick and `max_shots`
truncates one; neither has a "photos kept" number, and inventing one would be
worse than having none. Their consequence is the memory-level diff in
`impact.py`, which is a real answer to a real question.

`countable` says which is which, and nothing here guesses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rekindle import config
from rekindle.memory.composition import compose
from rekindle.models import MediaType, Photo

#: Settings whose effect is a per-photo verdict, and can therefore be counted
#: straight off the index without building anything.
GATES = (
    "composition.min_sharpness",
    "composition.min_brightness",
    "composition.max_brightness",
    "composition.min_short_edge",
    "composition.max_aspect",
)

#: Settings that decide how many photographs are collapsed away as duplicates.
#: Counted differently, and counted BECAUSE they delete.
DEDUP = (
    "dedup.phash_distance",
    "dedup.gap_seconds",
)

COUNTABLE = GATES + DEDUP

#: How many example photographs to carry back for each direction. Enough to
#: look at, few enough to show at once.
EXAMPLES = 6


def countable(setting: str) -> bool:
    return setting in COUNTABLE


@dataclass
class Consequence:
    """What moving one threshold does to this library."""

    setting: str
    current: float
    proposed: float
    kept_now: int = 0
    kept_then: int = 0
    #: Photographs the new value would ADMIT that the current one rejects.
    newly_kept: list[Photo] = field(default_factory=list)
    #: Photographs the new value would REJECT that the current one keeps. The
    #: ones that matter: this is the direction that deletes something.
    newly_dropped: list[Photo] = field(default_factory=list)
    considered: int = 0

    @property
    def gained(self) -> int:
        return max(0, self.kept_then - self.kept_now)

    @property
    def dropped(self) -> int:
        return max(0, self.kept_now - self.kept_then)

    @property
    def unchanged(self) -> bool:
        return self.kept_now == self.kept_then and not self.newly_dropped

    @property
    def deletes(self) -> bool:
        """Does moving this collapse photographs away, rather than gate them?"""
        return self.setting in DEDUP

    @property
    def collapsed_now(self) -> int:
        return self.considered - self.kept_now

    @property
    def collapsed_then(self) -> int:
        return self.considered - self.kept_then

    def sentence(self) -> str:
        if self.proposed == self.current:
            return "That is the value you already have; nothing changes."
        if self.deletes:
            return self._deletion_sentence()
        if self.unchanged:
            return (
                f"Moving this from {self.current:g} to {self.proposed:g} changes "
                f"nothing at all on your library - the same {self.kept_now:,} "
                f"photographs stay usable."
            )
        bits = []
        if self.gained:
            bits.append(f"keeps {self.gained:,} more photograph{'s' if self.gained != 1 else ''}")
        if self.dropped:
            bits.append(
                f"drops {self.dropped:,} you currently keep"
                if self.dropped != 1
                else "drops one you currently keep"
            )
        return f"This {' and '.join(bits)}."

    def _deletion_sentence(self) -> str:
        """Phrased as a deletion, because it is one.

        "Keeps 3,264 fewer photographs" reads like a filter. "Collapses 3,264
        MORE photographs away" reads like what actually happens to them, which
        is the difference between a number someone glances at and a number
        that stops them.
        """
        extra = self.collapsed_then - self.collapsed_now
        if extra == 0:
            return (
                f"Moving this from {self.current:g} to {self.proposed:g} collapses "
                f"exactly the same photographs on your library."
            )
        if extra > 0:
            return (
                f"This collapses {extra:,} MORE photographs away as duplicates - "
                f"{self.collapsed_then:,} of your {self.considered:,} instead of "
                f"{self.collapsed_now:,}. Only one photograph of each collapsed "
                f"group ever reaches a memory."
            )
        return (
            f"This keeps {-extra:,} photographs that are currently collapsed away "
            f"as duplicates - {self.collapsed_then:,} collapsed instead of "
            f"{self.collapsed_now:,}."
        )


def consequence(
    photos: list[Photo],
    setting: str,
    proposed: float,
    *,
    settings: config.Settings | None = None,
) -> Consequence:
    """Count what changes, and keep a few of each to show.

    Runs the REAL rule - `compose` for a gate, `collapse` for a dedup
    threshold - twice, rather than reimplementing the comparison. A preview
    that reimplements the rule it is previewing is a preview of a different
    rule, and the two would drift the first time somebody touched either.
    """
    base = settings or config.active()
    current = base.get(setting)
    after = base.with_values({setting: proposed})
    survivors = _dedup_survivors if setting in DEDUP else _gate_survivors

    with config.using(base):
        kept_now = survivors(photos)
    with config.using(after):
        kept_then = survivors(photos)

    now_hashes = {p.file_hash for p in kept_now}
    then_hashes = {p.file_hash for p in kept_then}
    by_hash = {p.file_hash: p for p in photos}

    gained = sorted(then_hashes - now_hashes)
    lost = sorted(now_hashes - then_hashes)
    return Consequence(
        setting=setting,
        current=current,
        proposed=proposed,
        kept_now=len(kept_now),
        kept_then=len(kept_then),
        newly_kept=[by_hash[h] for h in gained[:EXAMPLES]],
        newly_dropped=[by_hash[h] for h in lost[:EXAMPLES]],
        considered=len(photos),
    )


def _gate_survivors(photos: list[Photo]) -> list[Photo]:
    usable, _ = compose(list(photos), enforce_orientation=False)
    return usable


def _dedup_survivors(photos: list[Photo]) -> list[Photo]:
    """What is left after burst collapse.

    No embedding is handed in, so this counts what the PIXELS would collapse.
    That understates the effect on an embedded library, and it understates it
    in the direction that matters: the real number of deletions is never
    smaller than the one shown.
    """
    from rekindle.memory.dedup import collapse

    kept, _ = collapse([p for p in photos if p.media_type is MediaType.IMAGE])
    return kept


def example_paths(photos: list[Photo]) -> list[Path]:
    return [p.paths[0] for p in photos if p.paths]
