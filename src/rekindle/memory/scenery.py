"""Scenery memories: a memory whose subject is a SCENE rather than a person.

Every recipe in `recipes/builtin.py` keys off a person, an album, a date or a
GPS cell. Measured on the reference library, **6,419 photos - 33.2% - have
none of the first three**: no face tag, no album anyone named, no
coordinates. They can appear inside `year_in_review` and `on_this_day`, where
the subject is the calendar, and they can never be what a memory is ABOUT.

This module makes them the subject, from a checked-in list of concepts
(`corpus/scenery.toml`) and the retrieval path prompt memories already use.

## What this reuses, and why it must

`prompt.consensus` is the retrieval path, unchanged and not wrapped: each
visual description is searched independently and a photo ranks by HOW MANY
DISTINCT DESCRIPTIONS put it near the top, over ranks and never over raw
cosine. That is not a stylistic echo of the prompt module - it is the fix for
a measured failure, and it took festival bleed from 9 of 24 shots on the
wrong festival to 0 of 24. A second retrieval path here would be a second
place for that failure to come back.

## What it deliberately does NOT reuse

**Seed-day expansion.** `prompt.build_selection` finds the DAYS a concept
happened on and then takes those days whole, because a festival memory that
is only the top-ranked frames is a wall of idols with no family in it - 4 of
24 shots carried a face tag before the expansion and 22 to 24 after.

A scenery memory wants the opposite. Its subject is the scene, and a third of
the photographs it exists to reach have no people in them at all. Expanding
"the sea" to every photograph taken on a day someone went to the sea drags in
the hotel room, the lunch and the drive back - the day is a coherent event,
which is exactly why it is the wrong unit when the memory is about one thing
seen on many days. So the consensus ranking IS the selection here, and the
engine's stratifier spreads it across years.

## What it does not claim

`min_votes` is a PRECISION gate and it is not a presence test. Requiring two
independent descriptions to agree keeps a single description's private idea
of the concept from filling a memory. It does not tell you whether the
concept is in the library: `prompt.tag_agreement` documents the eighth
attempt at that signal and its failure, measured over 16 concepts this
library has and 16 it does not, and nothing here does better. A scenery
memory is built and shown; the verifier is the person looking at it.

Nothing here imports `rekindle.semantic`. Retrieval arrives as an injected
callable, exactly as in `prompt`, so the tests run with no torch, no
embedding store and no network.
"""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rekindle.memory import strata
from rekindle.memory.albums import presentable
from rekindle.memory.festivals import CorpusError
from rekindle.memory.index import MemoryIndex
from rekindle.memory.prompt import TAG_K, Retriever, consensus
from rekindle.memory.recipes import Selection
from rekindle.memory.recipes.base import chronological
from rekindle.memory.spec import build_fact_sheet
from rekindle.models import Photo

CORPUS_PATH = Path(__file__).with_name("corpus") / "scenery.toml"

#: The recipe name a scenery memory records. There is NO registered recipe by
#: this name and `tests/test_engine.py` asserts there never is, for the same
#: reason as `prompt.RECIPE`: a registered recipe would need a retriever
#: inside `select()`, which the `Recipe` protocol does not have and which CI
#: - no models, no embeddings - could not supply.
RECIPE = "scenery"

# `TAG_K` - how deep each individual description is searched - is imported
# from `prompt` rather than re-declared: it is a property of the retrieval
# path, not of the caller, and two copies of it would drift.

#: How many distinct descriptions must agree before a photo may enter the
#: pool. Two, when the concept has two or more descriptions.
#:
#: Measured on the reference library: over the seven shipped concepts this
#: takes the candidate pool from 205-358 photographs down to 32-101, and the
#: photographs it removes are the one-vote tail, which is where every
#: hand-graded miss sat. It matters because the engine's stratifier fills a
#: year-bucket best-first by QUALITY SCORE and not by consensus rank, so a
#: sparse year represented only by one-vote junk would otherwise get a slot.
#:
#: It is not a refusal signal. See the module docstring.
MIN_VOTES = 2

#: The fewest distinct years a scenery memory must span.
#:
#: A scenery memory is about a thing seen repeatedly, so one afternoon at the
#: beach is a different and lesser memory than the beach across a decade -
#: and, unlike `on_this_day`, nothing in the title would tell the viewer which
#: they were looking at. Two rather than three: `night` and `rain` genuinely
#: have thin years in this library, and refusing them outright would say less
#: than showing them short.
MIN_YEARS = 2


@dataclass(frozen=True)
class Concept:
    """One entry of the corpus. Pure data, read from TOML."""

    key: str
    title: str
    names: tuple[str, ...]
    tags: tuple[str, ...]
    months: tuple[int, ...] = ()
    note: str = ""


def _fold(text: str) -> str:
    from rekindle.memory.prompt import normalise

    return normalise(text)


@lru_cache(maxsize=1)
def load(path: Path | None = None) -> tuple[Concept, ...]:
    """Read the corpus. Cached, and validated hard.

    Raises `CorpusError` - the same class `festivals.load` raises, so a caller
    that guards one guards both - naming the file and the entry. A corpus
    typo must never reach a user as a silent empty memory.
    """
    target = path or CORPUS_PATH
    try:
        raw = tomllib.loads(target.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CorpusError(f"{target} could not be read: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise CorpusError(f"{target} is not valid TOML: {exc}") from exc

    out: list[Concept] = []
    seen: set[str] = set()
    for entry in raw.get("concept", []):
        key = entry.get("key")
        if not key:
            raise CorpusError(f"{target}: a [[concept]] entry has no `key`.")
        if key in seen:
            raise CorpusError(
                f"{target}: two [[concept]] entries share the key {key!r}. A key "
                "is half of a memory id, so a duplicate would make one memory "
                "shadow another and a dismissal of one silently dismiss both."
            )
        seen.add(key)
        title = entry.get("title")
        if not title:
            raise CorpusError(
                f"{target}: concept {key!r} has no `title`. The title is shown "
                "over the photographs and there is nothing to derive it from - "
                "deriving one from the key would be inventing it."
            )
        tags = tuple(entry.get("tags") or ())
        if not tags:
            raise CorpusError(
                f"{target}: concept {key!r} has no `tags`. A concept with no "
                "visual description cannot be searched for, and its name alone "
                "is exactly what does not work - see festivals.toml."
            )
        months = tuple(int(m) for m in entry.get("months") or ())
        if any(not 1 <= m <= 12 for m in months):
            raise CorpusError(f"{target}: concept {key!r} has a month outside 1-12.")
        out.append(
            Concept(
                key=key,
                title=title,
                names=tuple(entry.get("names") or ()),
                tags=tags,
                months=months,
                note=entry.get("note", ""),
            )
        )
    if not out:
        raise CorpusError(f"{target}: no [[concept]] entries.")
    return tuple(out)


def all_concepts(path: Path | None = None) -> tuple[Concept, ...]:
    return load(path)


def get(key: str, path: Path | None = None) -> Concept | None:
    for concept in load(path):
        if concept.key == key:
            return concept
    return None


def match(subject: str, path: Path | None = None) -> Concept | None:
    """The concept a user's word names, or None.

    Longest name first and on whole words of the normalised subject, exactly
    as `festivals.match` does, so `the sea` beats `sea` and `sea` does not
    match inside `season`.
    """
    folded = f" {_fold(subject)} "
    best: tuple[int, Concept] | None = None
    for concept in load(path):
        for name in (concept.key, *concept.names):
            candidate = _fold(name)
            if f" {candidate} " in folded and (best is None or len(candidate) > best[0]):
                best = (len(candidate), concept)
    return best[1] if best else None


# --------------------------------------------------------------------------
# selection


@dataclass(frozen=True)
class SceneryBuild:
    """Everything the CLI needs to explain what it did."""

    concept: Concept
    pool: int
    resolved: int
    reached: int
    votes: dict[int, int]
    years: tuple[int, ...]
    selection: Selection | None

    @property
    def title(self) -> str:
        return self.concept.title


def orphan(photo: Photo) -> bool:
    """A photo no other recipe can be ABOUT.

    No face tag, no album a person named (Google's own `Photos from 2019` is
    on every photo and is not a title - see `albums.presentable`), and no
    coordinates. This is the population scenery memories exist for, and it is
    reported per memory so the claim can be checked rather than repeated.
    """
    return (
        not any(photo.meta.people)
        and not any(presentable(a) for a in photo.albums)
        and photo.meta.gps is None
    )


def build_selection(
    index: MemoryIndex,
    concept: Concept,
    retrieve: Retriever,
    *,
    tag_k: int = TAG_K,
    min_votes: int = MIN_VOTES,
    min_years: int = MIN_YEARS,
) -> SceneryBuild:
    """Concept -> Selection, by consensus over its visual descriptions.

    The index is the chokepoint: `resolve_many` drops any hash the exclusion
    policy refused, and it runs BEFORE the vote gate and before the year
    count, so an excluded person's photographs cannot influence either.
    """
    result = consensus(concept.tags, retrieve, tag_k=tag_k)
    resolved = index.resolve_many(result.order)

    floor = min_votes if len(concept.tags) >= min_votes else 1
    kept = [p for p in resolved if result.votes.get(p.file_hash, 0) >= floor]
    if concept.months:
        allowed = set(concept.months)
        kept = [p for p in kept if p.meta.taken_at_local.month in allowed]

    votes: dict[int, int] = {}
    for photo in kept:
        n = result.votes.get(photo.file_hash, 0)
        votes[n] = votes.get(n, 0) + 1
    years = tuple(sorted({p.meta.taken_at_local.year for p in kept}))

    build = SceneryBuild(
        concept=concept,
        pool=len(result.order),
        resolved=len(resolved),
        reached=len(kept),
        votes=votes,
        years=years,
        selection=None,
    )
    if not kept:
        return build

    photos = chronological(kept)
    selection = Selection(
        photos=photos,
        facts=build_fact_sheet(
            photos,
            title=concept.title,
            recipe=RECIPE,
            # The title is written in the corpus by a person and describes the
            # CONTENT of the memory, not a proper noun about it. "The sea" and
            # "The mountains" substantiate no name, no year and no place, and
            # `festivals.toml`'s rule against place names applies here too -
            # so treating the title's words as substantiated costs nothing and
            # keeps the flag meaning what it means everywhere else.
            title_substantiated=True,
        ),
        # A scenery memory is about a thing seen again and again, so it is
        # spread across YEARS and `min_strata` is counted in years.
        #
        # `BY_SPAN` was tried and measured on the reference library, where it
        # resolves to DAYS for every concept here: it moved one shot in `sea`,
        # one in `mountains` and none in `rain`. The two rules agree on this
        # library, so the choice was made on what the memory PROMISES - a
        # scene seen again over time - rather than on a difference in output.
        # A day-stratified memory that happened to span one year would satisfy
        # `min_strata` while being the single-trip memory this recipe exists
        # not to build.
        stratify=strata.BY_YEAR,
        min_strata=min_years,
    )
    return SceneryBuild(
        concept=concept,
        pool=build.pool,
        resolved=build.resolved,
        reached=build.reached,
        votes=build.votes,
        years=build.years,
        selection=selection,
    )


def describe_votes(votes: dict[int, int]) -> str:
    """`3 tags: 12, 2 tags: 40` - the agreement spread, most agreement first."""
    return ", ".join(f"{n} tags: {c}" for n, c in sorted(votes.items(), reverse=True))


def concept_keys(names: Sequence[str] | None = None, path: Path | None = None) -> list[str]:
    """Every concept key, or the subset named. Unknown names raise KeyError."""
    known = {c.key for c in load(path)}
    if not names:
        return [c.key for c in load(path)]
    out = []
    for name in names:
        concept = get(name, path) or match(name, path)
        if concept is None:
            raise KeyError(
                f"{name!r} is not a concept in the scenery corpus. Known: "
                + ", ".join(sorted(known))
            )
        out.append(concept.key)
    return out
