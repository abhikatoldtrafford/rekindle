"""The exit question: which of your memories would actually change?

THE NAIVE VERSION IS A FAILURE
------------------------------
"You changed a threshold. Rebuild all 189 memories? See you in three hours."
That is the wrong answer twice over. A user who tightened blur by a hair
should not be told to rebuild their whole library, and one whose change
quietly altered 140 memories deserves to know before it happens rather than
after.

SELECTION IS CHEAP; RENDERING IS EXPENSIVE
-------------------------------------------
Everything that decides WHICH photographs are in a memory - the composition
gates, dedup, the diversity pick, the cap - is arithmetic over rows already in
the index. Everything expensive - decoding, letterboxing, motion, encoding
WebP and GIF and MP4 - happens afterwards and is a pure function of the shot
list. So the answer to "did my change matter?" is available for the price of
re-running selection, and re-running selection over a whole library costs
seconds rather than hours.

So: re-run `engine.build` for every memory already on disk, diff the resulting
list of file hashes against the one recorded in its `memory.json`, and say
something true and specific.

    Your changes affect 14 of 189 memories. 11 gain photos, 3 lose one.

If a change affects nothing, that is said too, and it is worth knowing. It is
also the honest test of whether a threshold change mattered at all - the kind
of thing this project has been wrong about before.

WHAT CANNOT BE RE-CHECKED IS SAID, NOT GUESSED
-----------------------------------------------
A prompt memory's shot list came from a CLIP query, and re-running it needs
the semantic extra and an embedded library. When those are absent the memory
is reported as UNCHECKABLE rather than as unchanged - reporting "no change"
for something that was never re-run is exactly the silent-lie shape this
project keeps finding in its own tests.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from rekindle import config
from rekindle.memory import engine
from rekindle.memory.index import MemoryIndex
from rekindle.memory.recipes import Offer, registered
from rekindle.memory.spec import MemorySpec

SPEC_NAME = "memory.json"

#: One memory's verdict.
SAME = "same"
GAINED = "gained"
LOST = "lost"
SWAPPED = "swapped"
GONE = "gone"
UNCHECKABLE = "uncheckable"


@dataclass(frozen=True)
class Change:
    """What one existing memory would become under the new configuration."""

    memory_id: str
    title: str
    folder: Path
    verdict: str
    before: int = 0
    after: int = 0
    gained: int = 0
    lost: int = 0
    #: Why it could not be re-checked, when `verdict` is UNCHECKABLE.
    note: str = ""

    @property
    def changed(self) -> bool:
        return self.verdict in (GAINED, LOST, SWAPPED, GONE)


@dataclass
class Impact:
    """The whole answer, and the sentence to show for it."""

    changes: list[Change] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.changes)

    @property
    def affected(self) -> list[Change]:
        return [c for c in self.changes if c.changed]

    @property
    def uncheckable(self) -> list[Change]:
        return [c for c in self.changes if c.verdict == UNCHECKABLE]

    def count(self, verdict: str) -> int:
        return sum(1 for c in self.changes if c.verdict == verdict)

    def sentence(self) -> str:
        """What to say, in the words the brief asked for.

        Written as one function so the CLI and the web UI cannot phrase the
        same finding two different ways.
        """
        if not self.total:
            return "You have no built memories yet, so there is nothing to rebuild."

        affected = self.affected
        checked = self.total - len(self.uncheckable)
        noun = "memory" if checked == 1 else "memories"
        if not affected:
            base = (
                f"Your changes affect none of your {checked} {noun} - "
                f"every one of them would be built from exactly the same photographs."
            )
            return base + self._unchecked_clause()

        parts = []
        gained, lost, swapped, gone = (
            self.count(GAINED),
            self.count(LOST),
            self.count(SWAPPED),
            self.count(GONE),
        )
        if gained:
            parts.append(f"{gained} gain photo{'s' if gained != 1 else ''}")
        if lost:
            parts.append(f"{lost} lose {'photos' if lost != 1 else 'one'}")
        if swapped:
            parts.append(f"{swapped} swap photos in and out")
        if gone:
            parts.append(f"{gone} would no longer be built at all")
        detail = ", ".join(parts[:-1]) + (" and " + parts[-1] if len(parts) > 1 else parts[-1])
        return (
            f"Your changes affect {len(affected)} of {checked} {noun}. "
            f"{detail[0].upper() + detail[1:]}." + self._unchecked_clause()
        )

    def _unchecked_clause(self) -> str:
        n = len(self.uncheckable)
        if not n:
            return ""
        return (
            f" {n} more could not be re-checked - they are listed, and they are "
            f"not counted as unchanged."
        )


def find_specs(root: Path) -> list[Path]:
    """Every `memory.json` under `root`, in a stable order.

    Sorted, so two runs list the same memories in the same order and a diff of
    two impact reports is a diff of the findings rather than of the walk.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(root.rglob(SPEC_NAME))


def _offer_for(spec: MemorySpec) -> Offer:
    """Reconstruct the offer that produced this memory.

    A memory's identity is `recipe:key` and both are recorded in the spec, so
    this is a lookup rather than a guess. That is the same property dismissal
    relies on, and it is why an impact report can be computed at all: a memory
    can be re-selected without knowing which photographs used to be in it.
    """
    return Offer(recipe=spec.recipe, key=spec.key, title=spec.title, subtitle=spec.subtitle)


def _recipe_names() -> set[str]:
    return {r.name for r in registered()}


def _verdict(before: list[str], after: list[str] | None) -> tuple[str, int, int]:
    if after is None:
        return GONE, 0, 0
    old, new = set(before), set(after)
    gained = len(new - old)
    lost = len(old - new)
    if not gained and not lost:
        return SAME, 0, 0
    if gained and lost:
        return SWAPPED, gained, lost
    return (GAINED if gained else LOST), gained, lost


def analyse(
    index: MemoryIndex,
    memories_root: Path,
    *,
    settings: config.Settings | None = None,
    semantic=None,
    progress=None,
) -> Impact:
    """Re-run selection for every built memory and diff the shot lists.

    `settings` is the configuration to test. None means whatever is active,
    which is what the CLI wants after a calibration has been applied; the web
    preview passes a candidate so it can answer "what WOULD this do?" without
    writing anything.

    No frames are decoded and nothing is encoded. On the reference library -
    19,480 photos, 189 memories - this is the difference between seconds and
    the three hours the naive version would have cost.
    """
    impact = Impact()
    known = _recipe_names()
    paths = find_specs(memories_root)

    ctx = config.using(settings) if settings is not None else _null_context()
    with ctx:
        for i, path in enumerate(paths, start=1):
            if progress is not None:
                progress(i, len(paths), path)
            change = _one(index, path, known=known, semantic=semantic)
            if change is not None:
                impact.changes.append(change)
    return impact


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _one(index: MemoryIndex, path: Path, *, known: set[str], semantic) -> Change | None:
    try:
        spec = MemorySpec.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError, UnicodeDecodeError, KeyError):
        # A spec this build cannot read is not a memory this build can
        # rebuild. Reported, never skipped silently.
        return Change(
            memory_id=path.parent.name,
            title=path.parent.name,
            folder=path.parent,
            verdict=UNCHECKABLE,
            note="its memory.json could not be read by this version",
        )

    before = [s.file_hash for s in spec.shots]
    memory_id = f"{spec.recipe}:{spec.key}"

    if spec.recipe not in known:
        # A prompt memory, or one from a recipe this build no longer has.
        # Its shot list came from a CLIP query and cannot be reproduced from
        # the index alone.
        return Change(
            memory_id=memory_id,
            title=spec.title,
            folder=path.parent,
            verdict=UNCHECKABLE,
            before=len(before),
            note=(
                f"`{spec.recipe}` memories are built from a search, not from the "
                f"index alone, so a threshold change cannot be diffed against them"
            ),
        )

    rebuilt = engine.build(index, _offer_for(spec), semantic=semantic)
    after = [s.file_hash for s in rebuilt.shots] if rebuilt is not None else None
    verdict, gained, lost = _verdict(before, after)
    return Change(
        memory_id=memory_id,
        title=spec.title,
        folder=path.parent,
        verdict=verdict,
        before=len(before),
        after=len(after) if after is not None else 0,
        gained=gained,
        lost=lost,
    )


def rebuildable(impact: Impact) -> Iterator[Change]:
    """The memories worth re-rendering, in the order they should be done.

    Biggest change first: if someone stops the rebuild half way, the memories
    that moved most are the ones already redone.
    """
    yield from sorted(
        impact.affected,
        key=lambda c: (-(c.gained + c.lost), c.memory_id),
    )
