"""The caption pipeline: three layers, each able to veto.

    deterministic  the year, the date, or nothing. Ships today, always works.
         clip      + what CLIP recognised, from a closed vocabulary.
          gpt      + a language model phrasing those terms and the fact sheet.
                     It never sees the pixels.

Each layer is a strict superset of the one above and each one degrades into
it. No embeddings means `clip` is `deterministic`. No API key means `gpt` is
`clip`. No semantic extra at all means `deterministic`, which is what CI runs
and what most installs run.

**Reject rather than repair.** Every gate in this path drops rather than
rewrites: a term below its percentile is dropped, a facet with no clear winner
contributes nothing, a caption the index cannot substantiate is discarded in
favour of the deterministic one. A dropped caption is a blank; a repaired one
is a guess wearing a citation.

**A GPT caption is generated once per photograph, then never again.** It is
written to `photo_captions` in the index the first time it is produced and
read back forever after, so the same library produces the same memory byte for
byte - the promise the whole engine rests on, and the one thing a language
model in the loop would otherwise break. That cache is personal data and lives
in the index, which is gitignored.

The CLIP layer has NO cache and needs none. It is a lookup in a vocabulary
against scores the embedding store already holds; the line it produces ends
with the shot's deterministic caption, which differs between recipes, so a
cached line would be wrong in the next memory anyway. There used to be a write
here with no reader; see `GroundingReport`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from rekindle.memory import vocab
from rekindle.memory.index import MemoryIndex
from rekindle.memory.llm import ShotFacts
from rekindle.memory.spec import MemorySpec, Shot

MODE_DETERMINISTIC = "deterministic"
MODE_CLIP = "clip"
MODE_GPT = "gpt"
MODES = (MODE_DETERMINISTIC, MODE_CLIP, MODE_GPT)

SOURCE_CLIP = "clip"
SOURCE_GPT = "gpt"

#: Bumped when `caption_vocab.toml` changes in a way that changes captions.
#: Cached rows carrying a different version are treated as absent, so an
#: edited vocabulary regenerates rather than mixing two vocabularies inside
#: one memory.
#:
#: v2 removed three subject terms - a boat, a vehicle, a bird - after grading
#: the six photographs the library scores highest for each and finding 2, 1
#: and about 3 of 6. Any caption cached under v1 may name one of them.
#:
#: v3 removed "at a temple", which broke the vocabulary's OWN rule 4 - "not a
#: place of worship named as such" - written eighteen lines above the term in
#: the same file, and reached 429 photographs in this library. The test that
#: exists to enforce rule 4 listed church, mosque and synagogue but not
#: temple, so the rule, the file and the test disagreed for an entire release;
#: that list now covers the traditions this library actually contains. The
#: version moved rather than the term merely being deleted because a caption
#: cached under v1 or v2 can still name a temple.
VOCAB_VERSION = 3


class VisionSupport(Protocol):
    """Percentile scores for one photograph, or None when it has no vector.

    Handed in rather than imported, exactly like `diversity.SemanticSupport`:
    `rekindle.memory` never imports `rekindle.semantic`, so every rule in this
    file is tested in CI with no model and no photographs.
    """

    def scores_for(self, file_hash: str) -> dict[str, float] | None: ...


@dataclass
class GroundingReport:
    """What the CLIP layer said, and how often it declined to say anything."""

    requested: int = 0
    grounded: int = 0
    unembedded: int = 0
    silent: int = 0
    facets: dict[str, int] = field(default_factory=dict)

    #: There is no `cached` here, deliberately, and there was one that could
    #: never be non-zero.
    #:
    #: `apply_clip` wrote a `source='clip'` row for every grounded shot and
    #: NOTHING ever read one back: the only `_StoreCache` built anywhere is
    #: `gpt_cache`, pinned to `SOURCE_GPT`. A counter documented as existing
    #: so that a dead cache shows up as a zero was itself the dead thing.
    #:
    #: The write is gone rather than the read being added, because a cache
    #: here cannot pay for itself. What `phrase` returns has the shot's
    #: DETERMINISTIC caption appended - a year for one recipe, a full date for
    #: another - so a line cached from one memory is wrong in the next, and
    #: what a hit would save is `vocab.choose` over scores already in hand.
    #: The expensive half is the embedding, and that has its own store.

    @property
    def accounted(self) -> bool:
        return self.requested == self.grounded + self.unembedded + self.silent


@dataclass
class _StoreCache:
    """`PhotoStore` as an `llm.CaptionCache`, pinned to one source and model."""

    store: object
    source: str
    model: str = ""
    vocab_version: int = VOCAB_VERSION
    terms: tuple[str, ...] = ()

    def get(self, file_hash: str) -> str | None:
        found = self.store.caption_get(  # type: ignore[attr-defined]
            file_hash, self.source, vocab_version=self.vocab_version, model=self.model
        )
        return found[0] if found else None

    def put(self, file_hash: str, caption: str) -> None:
        self.store.caption_put(  # type: ignore[attr-defined]
            file_hash,
            self.source,
            caption,
            terms=self.terms,
            vocab_version=self.vocab_version,
            model=self.model,
        )


def shot_facts(index: MemoryIndex, shot: Shot, terms: tuple[str, ...] = ()) -> ShotFacts:
    """What is true of ONE photograph. Read from the index, never from a model.

    A shot whose hash the index no longer holds - excluded since the spec was
    written - yields an EMPTY ShotFacts rather than None, so the verifier
    narrows to "nothing is substantiated about this photograph" instead of
    silently widening back to the memory. A caption under a photo the index
    cannot vouch for should be refused, and an empty ShotFacts refuses every
    name and every year.
    """
    photo = index.get(shot.file_hash)
    if photo is None:
        return ShotFacts(terms=terms)
    local = photo.meta.taken_at_local
    return ShotFacts(
        year=local.year if local else None,
        people=tuple(sorted({p for p in photo.meta.people if p})),
        has_gps=photo.meta.gps is not None,
        terms=terms,
    )


def ground(
    spec: MemorySpec,
    vision: VisionSupport,
    *,
    store=None,
    min_percentile: float = vocab.MIN_PERCENTILE,
    min_margin: float = vocab.MIN_MARGIN,
) -> tuple[dict[str, vocab.Grounding], GroundingReport]:
    """What CLIP is willing to say about each shot. No sentence is built here.

    Returns `file_hash -> Grounding`, including empty ones: a photograph CLIP
    declined to describe is a fact the caller should be able to report, and
    dropping it from the mapping would make "declined" and "not asked"
    indistinguishable.
    """
    report = GroundingReport()
    out: dict[str, vocab.Grounding] = {}
    for shot in spec.shots:
        report.requested += 1
        scores = vision.scores_for(shot.file_hash)
        if scores is None:
            report.unembedded += 1
            out[shot.file_hash] = vocab.Grounding(terms=())
            continue
        grounding = vocab.choose(scores, min_percentile=min_percentile, min_margin=min_margin)
        out[shot.file_hash] = grounding
        if grounding:
            report.grounded += 1
            for term in grounding.terms:
                report.facets[term.facet] = report.facets.get(term.facet, 0) + 1
        else:
            report.silent += 1
    return out, report


def apply_clip(
    spec: MemorySpec,
    groundings: dict[str, vocab.Grounding],
) -> MemorySpec:
    """Replace each caption with its grounded phrase, where there is one.

    A shot CLIP declined to describe keeps its deterministic caption. The
    deterministic caption is appended to the grounded one when it fits - it is
    a year or a date, which is a fact and worth keeping.

    NO `store` PARAMETER, and there was one. It wrote a `source='clip'` row
    per grounded shot that nothing ever read; see `GroundingReport` for why a
    cache at this layer cannot pay for itself.
    """
    shots = []
    for shot in spec.shots:
        grounding = groundings.get(shot.file_hash)
        line = vocab.phrase(grounding, shot.caption) if grounding else ""
        if not line:
            # CLIP declined. The deterministic caption stands, untouched -
            # ONE branch, not a fallback inside a rebuilt Shot as well. Two
            # guards on the same condition let either be deleted with nothing
            # failing, which a mutation run found here.
            shots.append(shot)
            continue
        shots.append(
            Shot(
                file_hash=shot.file_hash,
                caption=line,
                taken_at_local=shot.taken_at_local,
                public_safe=shot.public_safe,
            )
        )
    return MemorySpec(
        recipe=spec.recipe,
        key=spec.key,
        title=spec.title,
        subtitle=spec.subtitle,
        shots=tuple(shots),
        facts=spec.facts,
        public_safe=spec.public_safe,
    )


def gpt_cache(store, model: str) -> _StoreCache:
    """The GPT layer's cache. Keyed on the model, so changing model regenerates."""
    return _StoreCache(store=store, source=SOURCE_GPT, model=model)


def describe_grounding(report: GroundingReport) -> str:
    """One line for the CLI. Says what was NOT described as well as what was."""
    facets = ", ".join(f"{n} {f}" for f, n in sorted(report.facets.items()))
    parts = [f"{report.grounded}/{report.requested} shots grounded"]
    if facets:
        parts.append(f"({facets})")
    if report.silent:
        parts.append(f"{report.silent} cleared no term")
    if report.unembedded:
        parts.append(f"{report.unembedded} have no embedding")
    return "CLIP captions: " + ", ".join(parts)
