"""Grounded captions: the closed vocabulary, and the rule that picks from it.

**A wrong caption on a photograph of someone's family is worse than no
caption.** Three layers stand between a photograph and a sentence under it,
and every one of them can veto:

1. **CLIP grounds it.** This module. What is visually present, scored against
   the fixed list in `corpus/caption_vocab.toml`. There is no free-text
   guessing about content anywhere in the caption path.
2. **The language model phrases it** (`memory/llm.py`). It receives the terms
   this module chose and the fact sheet. **It never sees the pixels.** It
   turns "at the sea, at sunset" plus "October 2019" into a sentence; it does
   not decide what is in the photograph.
3. **The index vetoes it** (`llm.substantiated`). Anything asserted that the
   index cannot support - a year this photo does not have, a person not in
   its own tags - is rejected before it renders.

**Reject rather than repair.** Nothing here rewrites a doubtful answer into a
safer one. A term that does not clear the gate is dropped, a facet with no
term contributes nothing, and a photograph that clears nothing gets no
caption. A dropped caption is a blank; a repaired one is a guess wearing a
citation.

## Why percentiles and not scores

The scores this module reads are **percentiles of each probe's own
distribution over the whole embedded library**, computed in
`semantic/describe.py`. Raw cosines are not usable for this and the project
has the measurement to prove it: `prompt.py` records that `scuba diving
underwater` scores a higher z@1 against this library than `durga puja` does,
with zero scuba photographs in it. Comparing "a plate of food" against "a
sandy beach" on one image is exactly that error - two different queries, two
different score scales - and picking the larger picks whichever phrase CLIP
is warm on generally. A percentile is comparable between probes because every
probe is measured against the same 18,201 photographs.

This module imports nothing from `rekindle.semantic` and reads no vector. It
receives a mapping of probe to percentile and applies the rules, so every rule
here is tested in CI with no model, no GPU and no photographs.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rekindle.memory.festivals import CorpusError

CORPUS_PATH = Path(__file__).with_name("corpus") / "caption_vocab.toml"

#: The facets, and the order they compose in. A noun phrase, then where it
#: was, then what the light was doing - which is the order that produces
#: English rather than a list.
SUBJECT = "subject"
SETTING = "setting"
LIGHT = "light"
FACETS = (SUBJECT, SETTING, LIGHT)

#: How far into a probe's own distribution a photograph must sit before that
#: probe may describe it, PER FACET.
#:
#: 0.97 means: of the 18,201 embedded photographs, this one is in the top 3%
#: for that description.
#:
#: **The floors differ because the claims differ, and that was measured rather
#: than assumed.** A first pass used one floor of 0.97 everywhere and was
#: hand-graded on 36 random photographs. It captioned 12 of them, and every
#: clear error was in the SUBJECT facet: "A vehicle by the water" on a lake
#: shore with no vehicle, "A vehicle in the hills" on a bare quarry face, "A
#: statue indoors" on a framed print. The settings and the light were right
#: almost everywhere - "In the hills", "In open country", "Trees in the
#: hills".
#:
#: The reason is structural, not a bad threshold. A setting is supported by
#: the whole frame; an OBJECT claim says a particular thing is present, and
#: the top 3% of "a car, a bus or a truck" in a library with few vehicles is
#: not vehicles, it is roads and outdoor scenes - whatever is nearest, which
#: is the same failure `known-limitations.md` records for the cluster labels.
#: So naming an object needs the top half per cent and naming a place needs
#: the top three.
#:
#: Re-graded at these floors on the same 36 photographs: see
#: `tests/test_caption_real.py`, which re-measures rather than trusting this
#: comment.
FACET_PERCENTILE = {
    SUBJECT: 0.995,
    SETTING: 0.97,
    LIGHT: 0.98,
}

#: The floor for a facet not listed above. Also the value callers pass when
#: they want one number for everything, and what the tests sweep.
MIN_PERCENTILE = 0.97

#: How far clear of the runner-up in its own facet the winner must be.
#:
#: Two settings both at the 99th percentile is CLIP saying it cannot tell a
#: garden from open country, and picking the larger of two numbers that close
#: is picking noise. Measured: 0.015 removes the confusable pairs this library
#: actually produces (garden/open country, street/building) while leaving a
#: genuine beach, which clears its runner-up by 0.03 and more.
MIN_MARGIN = 0.015

#: The caption ceiling, shared with `llm.MAX_CAPTION_CHARS`. A caption sits
#: under a photograph in a montage; anything longer is unreadable.
MAX_CHARS = 64


@dataclass(frozen=True)
class Term:
    """One vocabulary entry. `probe` goes to CLIP, `says` reaches the caption."""

    facet: str
    probe: str
    says: str


@lru_cache(maxsize=1)
def load(path: Path | None = None) -> tuple[Term, ...]:
    """Read the vocabulary. Cached, and validated hard.

    Raises `CorpusError` - the same class the festival and scenery corpora
    raise - naming the file and the entry. A typo here would silently narrow
    what can be said about a photograph, which is the kind of failure nobody
    notices.
    """
    target = path or CORPUS_PATH
    try:
        raw = tomllib.loads(target.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CorpusError(f"{target} could not be read: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise CorpusError(f"{target} is not valid TOML: {exc}") from exc

    out: list[Term] = []
    probes: set[str] = set()
    for entry in raw.get("term", []):
        facet = entry.get("facet")
        probe = entry.get("probe")
        says = entry.get("says")
        if facet not in FACETS:
            raise CorpusError(
                f"{target}: term {probe!r} has facet {facet!r}, which is not one "
                f"of {', '.join(FACETS)}. A facet decides where the fragment "
                "lands in the sentence, so an unknown one cannot be placed."
            )
        if not probe or not says:
            raise CorpusError(f"{target}: a [[term]] entry is missing `probe` or `says`.")
        if probe in probes:
            raise CorpusError(
                f"{target}: two [[term]] entries share the probe {probe!r}. Scores "
                "are keyed on the probe, so one would shadow the other."
            )
        probes.add(probe)
        out.append(Term(facet=facet, probe=probe, says=says))
    if not out:
        raise CorpusError(f"{target}: no [[term]] entries.")
    return tuple(out)


def probes(path: Path | None = None) -> tuple[str, ...]:
    """Every probe, in file order. What the scorer has to embed."""
    return tuple(t.probe for t in load(path))


def by_facet(facet: str, path: Path | None = None) -> tuple[Term, ...]:
    return tuple(t for t in load(path) if t.facet == facet)


# --------------------------------------------------------------------------
# choosing


@dataclass(frozen=True)
class Grounding:
    """What CLIP was willing to say about one photograph, and what it refused.

    `terms` is in facet order and is what reaches both the caption and the
    language model. `rejected` names the facets that had a candidate and
    refused it, so a run can report how often the gates fired rather than
    leaving a blank caption looking like a missing feature.
    """

    terms: tuple[Term, ...]
    rejected: tuple[str, ...] = ()

    @property
    def says(self) -> tuple[str, ...]:
        return tuple(t.says for t in self.terms)

    def __bool__(self) -> bool:
        return bool(self.terms)


REJECT_BELOW_FLOOR = "below_percentile"
REJECT_TOO_CLOSE = "no_margin"


def choose(
    scores: Mapping[str, float],
    *,
    min_percentile: float | None = None,
    min_margin: float = MIN_MARGIN,
    path: Path | None = None,
) -> Grounding:
    """The per-facet winner, where there is one. Pure, and the whole rule.

    `scores` maps a probe to its percentile in [0, 1]. A probe missing from
    the mapping simply does not compete - which is what happens when the
    vocabulary has grown since the scores were cached, and is why a stale
    cache degrades to a shorter caption rather than to a wrong one.

    `min_percentile` overrides `FACET_PERCENTILE` with ONE floor for every
    facet. That is not the shipped configuration - see `FACET_PERCENTILE` for
    why the floors differ - and exists so a sweep can measure one number.
    """
    chosen: list[Term] = []
    rejected: list[str] = []
    for facet in FACETS:
        floor = (
            min_percentile
            if min_percentile is not None
            else FACET_PERCENTILE.get(facet, MIN_PERCENTILE)
        )
        # A TIE IS BROKEN BY CORPUS ORDER, and that is deterministic because
        # `by_facet` reads a checked-in file and `sorted` is stable.
        #
        # This key carried `(-score, probe)` until a mutation run showed the
        # probe could be removed with nothing failing. It was there to stop
        # the answer depending on dict iteration order - but the iteration is
        # over `by_facet`, not over `scores`, so dict order was never in play.
        # An alphabetical tiebreak would also have OVERRIDDEN corpus order,
        # taking away the one way a maintainer can say which of two equally
        # scoring phrases they prefer. Removed, and the rule written down.
        ranked = sorted(
            ((scores[t.probe], t) for t in by_facet(facet, path) if t.probe in scores),
            key=lambda st: -st[0],
        )
        if not ranked:
            continue
        best, term = ranked[0]
        if best < floor:
            rejected.append(REJECT_BELOW_FLOOR)
            continue
        runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
        if best - runner_up < min_margin:
            rejected.append(REJECT_TOO_CLOSE)
            continue
        chosen.append(term)
    return Grounding(terms=tuple(chosen), rejected=tuple(rejected))


def phrase(grounding: Grounding, existing: str = "", *, max_chars: int = MAX_CHARS) -> str:
    """The terms as one caption line. Deterministic, and never a sentence.

    Assembly, in facet order:

        subject + setting          "A flower in a garden"
        setting alone, capitalised "At the sea"
        subject alone              "Clouds"

    plus the light fragment and then `existing` - the deterministic caption,
    which is a year or a date and is therefore a FACT - appended while they
    fit. Dropping the tail rather than truncating mid-word is deliberate: a
    caption that ends "at the sea, at su" reads as a bug, and the fragments
    are independently true so losing one loses nothing but detail.

    Returns "" when there is nothing grounded to say, even if `existing` is
    set: this function composes a GROUNDED caption, and a caller that wants
    the bare deterministic one already has it.
    """
    if not grounding.terms:
        return ""
    facets = {t.facet: t.says for t in grounding.terms}
    head = facets.get(SUBJECT, "")
    setting = facets.get(SETTING, "")
    light = facets.get(LIGHT, "")
    if head and setting:
        line = f"{head} {setting}"
    elif setting:
        line = _sentence(setting)
    elif head:
        line = head
    else:
        # Light alone. It has to lead, or the caption opens with a comma -
        # which is what this did until a rendered memory was looked at and
        # four of its twenty-four captions read ", at sunset".
        line = _sentence(light)
        light = ""
    for tail in (light, existing):
        if not tail:
            continue
        candidate = f"{line}, {tail}"
        if len(candidate) <= max_chars:
            line = candidate
    return line if len(line) <= max_chars else ""


def _sentence(fragment: str) -> str:
    """A prepositional fragment made to lead a caption. Capital, nothing else."""
    return fragment[0].upper() + fragment[1:] if fragment else ""
