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
from typing import Any

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

_PROMPT = """You write one very short caption for a single photo in a personal \
photo montage.

Rules, all binding:
- At most {max_chars} characters. One line. No quotation marks.
- Use ONLY the facts given. Do not invent a place, an event, a relationship, \
a mood, or anything about what the photo shows - you cannot see it.
- Never name a city, country or landmark. The coordinates, if any, are not a \
place name.
- Do not mention a year that is not listed, or a person who is not listed.
- Plain, warm, factual. No exclamation marks. No emoji.

If you have nothing substantiated to add, repeat the existing caption exactly.
"""


@dataclass
class CaptionReport:
    """What the layer did. Counted, and surfaced by the CLI."""

    requested: int = 0
    accepted: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    @property
    def total_rejected(self) -> int:
        return sum(self.rejected.values())

    @property
    def accounted(self) -> bool:
        return self.requested == self.accepted + self.total_rejected


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


def substantiated(caption: str, facts: FactSheet) -> str | None:
    """Return a rejection reason, or None when the caption is allowed.

    This is the verifier. It is deliberately conservative: every check
    compares against the SAME fact sheet the model was given, so a rejection
    means the model asserted something it was not told.
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

    # Names the sheet contains, split so "Abhik Maiti" admits "Abhik".
    allowed_names = {part for person in facts.people for part in person.split() if part}
    allowed_names |= {part for album in facts.albums for part in album.split() if part}
    allowed_names |= set(facts.title.split())
    for candidate in _NAME.findall(text):
        if candidate in _COMMON or candidate in allowed_names:
            continue
        return REJECT_UNKNOWN_PERSON
    return None


def build_payload(facts: FactSheet, shot: Shot, existing: str) -> dict[str, Any]:
    """The request. Note what is in it: a fact sheet and a caption. Nothing
    else from the library exists at this point in the program."""
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
                "content": json.dumps(
                    {
                        "facts": facts.to_json(),
                        "existing_caption": existing,
                        "photo_taken": shot.taken_at_local,
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                ),
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

    def caption(self, facts: FactSheet, shot: Shot) -> str | None:
        payload = build_payload(facts, shot, shot.caption)
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


def apply_captions(spec: MemorySpec, captioner: GptCaptioner) -> tuple[MemorySpec, CaptionReport]:
    """Rewrite a spec's captions. Returns a NEW spec and a report.

    A failure anywhere - a rejected caption, an unreachable service, a
    malformed response - leaves the deterministic caption in place for that
    shot. The shot list, its order, and every other field are untouched: this
    function can only change the `caption` string of a `Shot`.
    """
    report = CaptionReport()
    shots: list[Shot] = []
    for shot in spec.shots:
        report.requested += 1
        try:
            candidate = captioner.caption(spec.facts, shot)
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
        reason = substantiated(candidate, spec.facts)
        if reason is not None:
            report.reject(reason)
            shots.append(shot)
            continue
        report.accepted += 1
        shots.append(
            Shot(
                file_hash=shot.file_hash,
                caption=" ".join(candidate.split()),
                taken_at_local=shot.taken_at_local,
                public_safe=shot.public_safe,
            )
        )

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
