"""The optional GPT caption layer. Additive, off by default, deletable.

This module is the ONLY place in rekindle that touches a network, and nothing
reaches it unless the user passes `--captions gpt`. Delete this file and the
flag and v1 is still a complete product - that is the test of whether the
layer is really additive, and the engine is built so that it passes.

What it does NOT do:

* It never picks a photo. Selection, ordering, dedup, the cap and every
  guardrail are deterministic and have already run by the time a caption is
  written. An LLM rewrites words and nothing else.
* It never sees a pixel. Not a thumbnail, not a path, not a filename.
* It never sees the library. Only the `FactSheet` for ONE memory - the dates,
  counts, names, albums and coordinates that the engine already derived from
  stored metadata.
* It never decides what is true. Every caption it returns is validated against
  the same fact sheet, and anything asserting a year or a person the sheet
  does not contain is rejected in favour of the deterministic caption.

The key is read from the `OPENAI_API_KEY` environment variable and is never
logged, never printed, never written to a spec, and never included in an
exception message. `test_llm.py` asserts that last one directly.

No new runtime dependency. The `openai` SDK would be installed for every user
including the overwhelming majority who never enable this, to save one
`urllib.request` POST. The transport is injectable, so tests use a fake and
NEVER touch the network.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from rekindle.memory.spec import FactSheet, MemorySpec, Shot

API_URL = "https://api.openai.com/v1/responses"
MODEL = "gpt-5.6-luna"
REASONING_EFFORT = "low"
TIMEOUT_SECONDS = 30.0
ENV_KEY = "OPENAI_API_KEY"

# A caption sits under a photo in a 480px-wide GIF. Anything longer is
# unreadable, and a model asked for "short" will still occasionally write a
# paragraph.
MAX_CAPTION_CHARS = 64

REJECT_TOO_LONG = "too_long"
REJECT_UNKNOWN_YEAR = "unsubstantiated_year"
REJECT_UNKNOWN_PERSON = "unsubstantiated_person"
REJECT_EMPTY = "empty"
#: A cached caption that the memory around it no longer substantiates.
REJECT_STALE_CACHE = "cache_no_longer_substantiated"

_YEAR = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
# Capitalised words that look like a name or a place.
#
# Deliberately crude AND deliberately strict: ANY capitalised word that is
# neither ordinary English prose (_COMMON) nor present in the fact sheet is a
# rejection, including the first word of the caption. That over-rejects -
# "Sunlight on the water" is refused because nothing substantiates
# "Sunlight" - and that is the right direction to be wrong in. A rejection
# costs one fallback to a deterministic caption, which is always correct; an
# acceptance of "Sunset over Srinagar" puts an invented place name under
# someone's photo. The measured acceptance rate is in the milestone report.
_NAME = re.compile(r"\b[A-Z][a-z]{2,}\b")

# Words that are capitalised in ordinary prose and are not claims about who
# was there. Without this, "December" and "Then" get rejected as unknown
# people and the layer falls back on almost every caption.
_COMMON = frozenset(
    {
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "Then",
        "Now",
        "Today",
        "This",
        "That",
        "The",
        "And",
        "Our",
        "Your",
        "Their",
        "Here",
        "There",
        "When",
        "Where",
        "What",
        "One",
        "Two",
        "Three",
        "Four",
        "Five",
        "Six",
        "Seven",
        "Eight",
        "Nine",
        "Ten",
        "Years",
        "Year",
        "Ago",
        "Photos",
        "Photo",
        "Place",
        "Every",
        "Before",
        "After",
        "First",
        "Last",
        "Summer",
        "Winter",
        "Spring",
        "Autumn",
        "Together",
        "Again",
        "Still",
        "Back",
    }
)

#: A caption naming a year the MEMORY has but THIS PHOTOGRAPH does not.
#:
#: Separate from `REJECT_UNKNOWN_YEAR` because it is a different failure and
#: the counts must not be merged: the memory-level check catches a model
#: inventing 1999 out of nothing, and this one catches it printing 2019 under
#: a 2011 photograph in a memory that spans both. The second is far likelier
#: and far harder for a viewer to spot.
REJECT_WRONG_SHOT_YEAR = "year_not_this_photo"
#: A caption naming someone tagged in ANOTHER shot of the same memory.
REJECT_WRONG_SHOT_PERSON = "person_not_in_this_photo"


@dataclass(frozen=True)
class ShotFacts:
    """What is true of ONE photograph, as opposed to the memory around it.

    The fact sheet describes the whole memory, and until this existed every
    caption was checked against it. That is the right check for the memory's
    title and the wrong one for a caption under a single photograph: a memory
    spanning 2011 to 2019 substantiated "2019" under its 2011 shots, and a
    memory of four people substantiated all four names under a photograph
    containing one of them. Both are exactly the failure this layer exists to
    prevent - a confident, checkable-looking, wrong sentence under someone's
    family - and both passed the memory-level verifier.

    Built at caption time from the index and never serialised into the spec.
    Adding it to `Shot` would put per-photograph name lists into a JSON file
    the user might share, and would bump `SPEC_VERSION`, invalidating every
    spec already on disk - for information the caption layer needs for the
    length of one call.

    `terms` is what CLIP was willing to say about the photograph, from the
    closed vocabulary in `memory/vocab.py`. It is the ONLY channel by which
    anything about the pixels reaches the language model.
    """

    year: int | None = None
    people: tuple[str, ...] = ()
    has_gps: bool = False
    terms: tuple[str, ...] = ()


_PROMPT = """You write one very short caption for a single photo in a personal \
photo montage.

You cannot see the photo. You are given facts the index holds about the memory \
it belongs to and, when available, a short list of things an image model \
recognised in this one frame, chosen from a fixed vocabulary. That list is the \
ONLY thing you know about what the photo shows.

Rules, all binding:
- At most {max_chars} characters. One line. No quotation marks.
- Use ONLY the facts given and the recognised list. Do not invent a place, an \
event, a relationship or a mood.
- Do not add to what the recognised list says is visible. If it says "at the \
sea", you may not say who was there, why, or what kind of day it was.
- Never name a city, country or landmark. The coordinates, if any, are not a \
place name.
- Do not mention a year that is not listed for THIS photo, or a person who is \
not listed for THIS photo.
- Plain, warm, factual. No exclamation marks. No emoji.

If you have nothing substantiated to add, repeat the existing caption exactly.
"""


@dataclass
class CaptionReport:
    """What the layer did. Counted, and surfaced by the CLI."""

    requested: int = 0
    accepted: int = 0
    #: Answered from the cache without a call. Counted apart from `accepted`
    #: so a user can see that a second run of the same memory cost nothing,
    #: and so a cache that is silently never hit is visible as a zero.
    cached: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    @property
    def total_rejected(self) -> int:
        return sum(self.rejected.values())

    @property
    def accounted(self) -> bool:
        return self.requested == self.accepted + self.cached + self.total_rejected


class CaptionCache(Protocol):
    """One caption per photograph, forever. Implemented by `PhotoStore`.

    A Protocol rather than the store itself, because `memory.llm` must stay a
    module you can delete: it knows about a fact sheet and a shot, and adding
    a database import here would make the optional layer structural.
    """

    def get(self, file_hash: str) -> str | None: ...

    def put(self, file_hash: str, caption: str) -> None: ...


def _recaptioned(shot: Shot, caption: str) -> Shot:
    """A shot with a new caption and nothing else touched."""
    return Shot(
        file_hash=shot.file_hash,
        caption=caption,
        taken_at_local=shot.taken_at_local,
        public_safe=shot.public_safe,
    )


class LLMUnavailable(RuntimeError):
    """No key, no network, or the service refused. Never fatal to a render."""


Transport = Callable[[dict[str, Any], str], dict[str, Any]]


def http_transport(payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    """One POST, stdlib only.

    `api_key` is a parameter rather than read here so that the only place the
    environment is consulted is `captioner_from_env` - one read site is one
    thing to audit.
    """
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # The body may echo request details. The STATUS is the useful part and
        # cannot contain a credential, so that is all that propagates - an
        # `Authorization` header echoed into an exception message is exactly
        # how a key ends up in a log file.
        raise LLMUnavailable(f"the captioning service returned HTTP {exc.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LLMUnavailable(
            f"could not reach the captioning service: {type(exc).__name__}"
        ) from None
    except json.JSONDecodeError:
        raise LLMUnavailable("the captioning service returned a malformed response") from None


def extract_text(response: dict[str, Any]) -> str:
    """Pull the assistant text out of a Responses-API payload.

    Tolerant of shape: `output_text` when the service provides the
    convenience field, otherwise the first text part of the first message in
    `output`. A shape this code does not recognise returns "" and the caller
    falls back to the deterministic caption - which is the correct behaviour
    for every failure in this module.
    """
    direct = response.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    if isinstance(direct, list) and direct:
        joined = "".join(part for part in direct if isinstance(part, str))
        if joined.strip():
            return joined.strip()
    for item in response.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for part in item.get("content", []) or []:
            if (
                isinstance(part, dict)
                and isinstance(part.get("text"), str)
                and part["text"].strip()
            ):
                return part["text"].strip()
    return ""


def substantiated(caption: str, facts: FactSheet, shot: ShotFacts | None = None) -> str | None:
    """Return a rejection reason, or None when the caption is allowed.

    This is the verifier. It is deliberately conservative: every check
    compares against the SAME facts the model was given, so a rejection means
    the model asserted something it was not told.

    With `shot`, the year and person checks narrow from the memory to the
    PHOTOGRAPH. Without it they fall back to the memory-wide check, which is
    what every caller did before `ShotFacts` existed and is still correct for
    a caller that genuinely has no per-shot context - it is simply weaker, and
    the two reasons are counted apart so a report can say which one fired.
    """
    text = " ".join(caption.split())
    if not text:
        return REJECT_EMPTY
    if len(text) > MAX_CAPTION_CHARS:
        return REJECT_TOO_LONG

    allowed_years = {str(y) for y in facts.years}
    for year in _YEAR.findall(text):
        if year not in allowed_years:
            return REJECT_UNKNOWN_YEAR
        if shot is not None and year != str(shot.year):
            return REJECT_WRONG_SHOT_YEAR

    # Names the sheet contains, split so "Abhik Maiti" admits "Abhik" - or,
    # when the shot is known, only the people tagged in THAT photograph.
    people = shot.people if shot is not None else tuple(facts.people)
    allowed_names = {part for person in people for part in person.split() if part}
    allowed_names |= {part for album in facts.albums for part in album.split() if part}
    # A recipe's title is itself a fact - an album name, a person's name, a
    # month - so its words are substantiated. A PROMPT's title is the user's
    # query, which may contain anything at all: whitelisting it would let
    # `christmas in midnapur` substantiate "Midnapur" and put an unresolvable
    # place name under a photograph.
    if facts.title_substantiated:
        allowed_names |= set(facts.title.split())
    for candidate in _NAME.findall(text):
        if candidate in _COMMON or candidate in allowed_names:
            continue
        if shot is not None and any(
            candidate == part for person in facts.people for part in person.split()
        ):
            # A real name from this memory, printed under a photograph that
            # does not contain that person. Counted apart from an invented
            # name: the model did not make this one up, it misplaced it, and
            # a report that merged the two would hide which happened.
            return REJECT_WRONG_SHOT_PERSON
        return REJECT_UNKNOWN_PERSON
    return None


def build_payload(
    facts: FactSheet, shot: Shot, existing: str, context: ShotFacts | None = None
) -> dict[str, Any]:
    """The request. Note what is in it: a fact sheet, a caption, and - when
    CLIP ran - a handful of phrases from a closed vocabulary.

    Nothing else from the library exists at this point in the program. Not a
    pixel, not a thumbnail, not a path, not a filename. `test_spec.py` asserts
    the fact sheet carries no path; `test_llm.py` asserts this payload carries
    nothing beyond the fact sheet, the caption and the vocabulary terms.
    """
    body: dict[str, Any] = {
        "facts": facts.to_json(),
        "existing_caption": existing,
        "photo_taken": shot.taken_at_local,
    }
    if context is not None:
        # Absent rather than empty when CLIP said nothing: an empty list reads
        # as "the model looked and found nothing in the frame", which is a
        # claim, and a model handed it writes around it.
        if context.terms:
            body["recognised_in_this_photo"] = list(context.terms)
        if context.people:
            body["people_in_this_photo"] = list(context.people)
        if context.year is not None:
            body["year_of_this_photo"] = context.year
    return {
        "model": MODEL,
        "reasoning": {"effort": REASONING_EFFORT},
        "input": [
            {
                "role": "system",
                "content": _PROMPT.format(max_chars=MAX_CAPTION_CHARS),
            },
            {
                "role": "user",
                "content": json.dumps(body, sort_keys=True, ensure_ascii=False),
            },
        ],
    }


class GptCaptioner:
    """Rewrites captions on a MemorySpec. Never raises at the call site."""

    def __init__(self, api_key: str, transport: Transport = http_transport) -> None:
        # Held, never logged. `__repr__` is overridden below so that a
        # traceback or a debug print of this object cannot leak it.
        self._api_key = api_key
        self._transport = transport

    def __repr__(self) -> str:
        return "GptCaptioner(api_key=<redacted>)"

    def caption(self, facts: FactSheet, shot: Shot, context: ShotFacts | None = None) -> str | None:
        payload = build_payload(facts, shot, shot.caption, context)
        response = self._transport(payload, self._api_key)
        text = extract_text(response)
        return text or None


def captioner_from_env(transport: Transport = http_transport) -> GptCaptioner:
    """The ONE place the environment is read. Raises when the key is absent."""
    key = os.environ.get(ENV_KEY)
    if not key:
        raise LLMUnavailable(
            f"{ENV_KEY} is not set, so the deterministic captions were used. "
            "This is a supported configuration, not an error."
        )
    return GptCaptioner(key, transport=transport)


def apply_captions(
    spec: MemorySpec,
    captioner: GptCaptioner,
    context: Callable[[Shot], ShotFacts] | None = None,
    *,
    cache: CaptionCache | None = None,
) -> tuple[MemorySpec, CaptionReport]:
    """Rewrite a spec's captions. Returns a NEW spec and a report.

    A failure anywhere - a rejected caption, an unreachable service, a
    malformed response - leaves the deterministic caption in place for that
    shot. The shot list, its order, and every other field are untouched: this
    function can only change the `caption` string of a `Shot`.

    `context` supplies the per-photograph facts the verifier narrows on and
    the CLIP terms the model is grounded by. Omitting it is supported and is
    the weaker configuration; see `substantiated`.

    `cache` makes this DETERMINISTIC. Without it the same photograph in two
    memories gets two calls and can get two different sentences, which breaks
    the engine's byte-for-byte promise the moment a caption is involved. With
    it a photograph is captioned once, ever.
    """
    report = CaptionReport()
    shots: list[Shot] = []
    for shot in spec.shots:
        report.requested += 1
        facts = context(shot) if context is not None else None
        cached = cache.get(shot.file_hash) if cache is not None else None
        if cached is not None:
            # A cached caption was verified before it was written. Re-verifying
            # is not paranoia about the cache: the memory around a photograph
            # changes, and a caption naming a person who is still in this
            # photograph but no longer in this memory's fact sheet must not
            # reappear because it was once allowed.
            if substantiated(cached, spec.facts, facts) is None:
                report.cached += 1
                shots.append(_recaptioned(shot, cached))
                continue
            report.reject(REJECT_STALE_CACHE)
            shots.append(shot)
            continue
        try:
            candidate = captioner.caption(spec.facts, shot, facts)
        except LLMUnavailable as exc:
            # The whole run is over, not just this shot. Keep every remaining
            # deterministic caption and report once.
            report.error = str(exc)
            shots.extend(spec.shots[len(shots) :])
            break
        if candidate is None:
            report.reject(REJECT_EMPTY)
            shots.append(shot)
            continue
        reason = substantiated(candidate, spec.facts, facts)
        if reason is not None:
            report.reject(reason)
            shots.append(shot)
            continue
        text = " ".join(candidate.split())
        report.accepted += 1
        if cache is not None:
            cache.put(shot.file_hash, text)
        shots.append(_recaptioned(shot, text))

    rewritten = MemorySpec(
        recipe=spec.recipe,
        key=spec.key,
        title=spec.title,
        subtitle=spec.subtitle,
        shots=tuple(shots),
        facts=spec.facts,
        public_safe=spec.public_safe,
    )
    return rewritten, report
