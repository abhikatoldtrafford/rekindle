"""Prompt -> visual tags, and the two gates around that.

Three sources, tried in this order, and every one of them is optional except
the last:

1. **The festival corpus** (`festivals.toml`) - checked in, human-written,
   carries a month window as well as tags.
2. **The tag cache** - a committed starter file plus a per-library cache
   written by the model path. Keyed on the NORMALISED PROMPT, so the same
   prompt yields the same tags forever: a memory built today rebuilds
   identically next year, and a dismissal keyed on it keeps working.
3. **The language model** - `gpt-5.6-luna`, off unless `OPENAI_API_KEY` is
   set, asked for visual descriptions and nothing else.

With none of the three, the prompt itself is used as a single tag and the CLI
says plainly that it is doing so. That path is the measured-bad one: it is
what put nine of twenty-four shots of a Kali Puja memory on a Durga Puja day.

This module also does one job that is not about tags at all: given the
prompt and the titles the user gave their own albums, it asks the model which
of those albums the prompt is ABOUT. That is here rather than in `prompt`
because it is the model layer, it shares this module's cache discipline, and
because `prompt` must stay a module with no network in it. It runs only where
exact token matching already found nothing, it is cached with a digest of the
album list it was chosen from, and a title the library does not have is
dropped rather than searched for.

**What the model is allowed to know.** The tag generator is told the prompt
and nothing else. It never sees the library, never sees a photo, and is
instructed that it is supplying world knowledge about what a thing looks
like. The plausibility judge sees the library's own VOCABULARY - album
titles, the year range, how many people are tagged, how many distinct GPS
cells there are - and never any photo content, never a name it could repeat,
and it is never asked whether photos exist. It is asked whether the QUERY is
coherent. Both replies are validated before use, by the same principle as
`llm.substantiated`: anything the model asserts that it was not told is
rejected and the deterministic path is used instead.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rekindle.memory import festivals
from rekindle.memory.llm import (
    ENV_KEY,
    MODEL,
    REASONING_EFFORT,
    LLMUnavailable,
    Transport,
    extract_text,
    http_transport,
)
from rekindle.memory.prompt import PromptQuery, normalise

#: The committed starter cache. Ships with rekindle, contains no personal
#: data, and is what lets a contributor with no API key - and CI - exercise
#: the whole path.
STARTER_CACHE = Path(__file__).with_name("corpus") / "prompt_tags.json"

#: Where the model path writes what it generated. Inside the gitignored data
#: directory, because it is derived from the user's own prompts.
USER_CACHE_NAME = "prompt_tags.json"

#: Where an album shortlist is remembered. There is deliberately NO committed
#: starter for this one: a shortlist is a mapping from someone's prompt to
#: THEIR OWN album titles, so a checked-in file would ship one person's
#: private labelling to every user of the package.
ALBUM_CACHE_NAME = "prompt_albums.json"

#: How many album titles are ever put in one request. The reference library
#: has 37 named albums; a library with thousands would otherwise send them
#: all. Most-populous first, so the cut falls on the albums least likely to be
#: anyone's answer.
MAX_ALBUM_TITLES = 200

CACHE_VERSION = 1

SOURCE_CORPUS = "corpus"
SOURCE_CACHE = "cache"
SOURCE_MODEL = "model"
SOURCE_PROMPT = "prompt"
#: No tags were needed: an album of the user's own answered the prompt.
SOURCE_ALBUM = "album"

#: Tags are short visual phrases. A model asked for "short" writes a sentence
#: often enough that this has to be enforced rather than requested.
MAX_TAG_CHARS = 90
MAX_TAGS = 8
MIN_TAGS = 2

REJECT_TOO_FEW = "too_few_tags"
REJECT_TOO_LONG = "tag_too_long"
REJECT_NAMES_A_YEAR = "tag_names_a_year"
REJECT_NAMES_A_PERSON = "tag_names_a_person"
REJECT_EMPTY = "empty_response"
#: The model returned an album title this library does not have.
REJECT_NOT_AN_ALBUM = "album_not_in_library"

_YEAR = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")

_TAG_PROMPT = """You are given a short phrase describing a kind of photograph \
someone wants to find. Reply with visual descriptions of what such a \
photograph would LOOK like.

Rules, all binding:
- Reply with {min_tags} to {max_tags} descriptions separated by semicolons. \
Nothing else: no numbering, no preamble, no explanation.
- Each description is at most {max_chars} characters and describes what is \
VISIBLE in the frame - objects, colours, shapes, materials, light.
- Each description must be DISTINCTIVE: something you would rarely photograph \
at anything else. A description that fits any crowd, any celebration or any \
gathering is worse than useless and will drag in unrelated photographs.
- Do not use the name of the event, festival, place or holiday. Describe it \
instead. If the phrase names a festival, describe the idol, the decoration, \
the food or the clothing that belongs to that festival and no other.
- Never mention a year, a date, a season or a month.
- Never name a person, a city, a country or a landmark.
- You know nothing about the photo library this will be searched against. Do \
not refer to it, do not guess what is in it, and do not say whether anything \
will be found.

Phrase: {subject}
"""

_ALBUM_PROMPT = """You are given a short phrase somebody typed to find their \
own photographs, and the list of titles they gave their own photo albums.

Say which of those albums, if any, are ABOUT the same thing as the phrase.

Rules, all binding:
- Reply with album titles copied EXACTLY as they appear in the list, one per \
line. Nothing else: no numbering, no bullets, no quotation marks, no \
preamble, no explanation.
- Reply with the single word NONE when no album is about the phrase. NONE is \
the right answer more often than not.
- Never write a title that is not in the list. Do not fix a typo in it, do \
not translate it, do not shorten it, do not tidy its capitalisation: copy the \
characters.
- Include an album when the phrase names the same subject under a different \
spelling, a different transliteration, a misspelling, or another name for the \
same place or event. That is what you are here for.
- Do NOT include an album that merely shares a word with the phrase, or that \
is merely related to it. Two different festivals are two different subjects \
even when their names share a word.
- You cannot see a single photograph and you know nothing about what these \
albums contain beyond their titles. Do not guess at their contents.

Phrase: {subject}

Albums:
{titles}
"""


_JUDGE_PROMPT = """You are judging whether a SEARCH QUERY is coherent, and \
whether it is consistent with the kind of photo library described below.

You are NOT being asked whether matching photographs exist. You cannot see \
the library's photographs and you must not guess at them. You are given only \
the library's vocabulary: the titles its owner gave their albums, the years \
it spans, how many people are tagged in it and how many distinct places it \
has coordinates for.

Reply with exactly one line:
  OK
or
  REFUSE: <one short sentence, at most 120 characters>

Refuse only when one of these is true:
- the query is not language: keyboard mashing, random characters, gibberish.
- the query describes something that could not plausibly be photographed by \
the owner of a library like this one - not merely something you doubt is \
there.

When in doubt, reply OK. A wrong refusal is worse than a wrong acceptance, \
because the person can look at what was built and a refusal gives them \
nothing to look at.

Library vocabulary:
{vocabulary}

Query: {query}
"""


@dataclass(frozen=True)
class Vocabulary:
    """What the plausibility judge is allowed to see. No photo content."""

    albums: tuple[str, ...]
    year_from: int | None
    year_to: int | None
    people_count: int
    place_count: int

    def to_json(self) -> dict[str, Any]:
        return {
            "album_titles": list(self.albums),
            "years": [self.year_from, self.year_to],
            "tagged_people": self.people_count,
            "distinct_places_with_coordinates": self.place_count,
        }


def vocabulary_of(index) -> Vocabulary:
    """The library's own words, and four counts. Never a photo, never a path."""
    years = index.years()
    return Vocabulary(
        albums=tuple(sorted(index.album_counts())),
        year_from=years[0] if years else None,
        year_to=years[-1] if years else None,
        people_count=len(index.people_counts()),
        place_count=len(index.gps_cells()),
    )


@dataclass(frozen=True)
class Resolution:
    """Where a prompt's tags came from, and what else that source knew."""

    tags: tuple[str, ...]
    source: str
    festival: str | None = None
    months: tuple[int, ...] = ()
    note: str = ""
    rejected: tuple[str, ...] = field(default_factory=tuple)


# -------------------------------------------------------------------- cache


def _read_cache(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # A corrupt cache is not a reason to fail a render; it is a reason to
        # behave as though it were empty and let the other sources answer.
        return {}
    entries = raw.get("entries")
    if not isinstance(entries, dict):
        return {}
    return {
        normalise(k): [str(t) for t in v.get("tags", [])]
        for k, v in entries.items()
        if isinstance(v, dict) and v.get("tags")
    }


def read_caches(data_dir: Path | None = None) -> dict[str, list[str]]:
    """The starter cache, with the user's own cache layered over it."""
    merged = _read_cache(STARTER_CACHE)
    if data_dir is not None:
        merged.update(_read_cache(data_dir / USER_CACHE_NAME))
    return merged


def write_cache_entry(data_dir: Path, prompt: str, tags: Sequence[str]) -> Path:
    """Record what the model said, so it is never asked twice.

    Sorted keys and a trailing newline, so the file is stable under diff and a
    user who wants to contribute their entry back can paste it into the
    starter cache without reformatting anything.
    """
    path = data_dir / USER_CACHE_NAME
    existing: dict[str, Any] = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = {}
    entries = existing.get("entries")
    if not isinstance(entries, dict):
        entries = {}
    entries[normalise(prompt)] = {"tags": list(tags), "source": SOURCE_MODEL}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"version": CACHE_VERSION, "entries": entries},
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


# ------------------------------------------------------- the album shortlist


def library_fingerprint(titles: Sequence[str]) -> str:
    """A short digest of the album titles a shortlist was chosen from.

    Stored beside every cached shortlist and compared before it is trusted. A
    cached answer to "which of these albums is about `ladhak`" is only an
    answer to that question while the list of albums is the same one; after
    the user adds or renames an album it is a stale answer that would silently
    keep a new album out of every memory.
    """
    joined = "\n".join(sorted(titles))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def read_album_cache(data_dir: Path | None) -> dict[str, dict[str, Any]]:
    if data_dir is None:
        return {}
    path = data_dir / ALBUM_CACHE_NAME
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    entries = raw.get("entries")
    if not isinstance(entries, dict):
        return {}
    return {
        normalise(k): v
        for k, v in entries.items()
        if isinstance(v, dict) and isinstance(v.get("albums"), list)
    }


def write_album_cache_entry(
    data_dir: Path, prompt: str, fingerprint: str, names: Sequence[str]
) -> Path:
    """Remember a shortlist, INCLUDING an empty one.

    An empty answer is an answer and caching it is what stops a prompt that
    names no album from paying for a call on every run. Determinism is the
    real reason: with the cache, the same prompt against the same albums picks
    the same albums forever, which is the same guarantee `write_cache_entry`
    gives the tags.
    """
    path = data_dir / ALBUM_CACHE_NAME
    existing: dict[str, Any] = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = {}
    entries = existing.get("entries")
    if not isinstance(entries, dict):
        entries = {}
    entries[normalise(prompt)] = {"albums": list(names), "library": fingerprint}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"version": CACHE_VERSION, "entries": entries},
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


@dataclass(frozen=True)
class Shortlist:
    """Album titles a model proposed for a prompt, and where they came from."""

    names: tuple[str, ...] = ()
    source: str = ""
    rejected: tuple[str, ...] = field(default_factory=tuple)


def parse_album_reply(text: str, titles: Sequence[str]) -> tuple[list[str], list[str]]:
    """A model reply -> titles this library actually has.

    Every line is looked up in the list the model was given, matched on the
    NORMALISED title so a difference of case or of Unicode composition is not
    a rejection, and the library's own spelling is what comes back. A line
    that matches nothing is dropped and counted. **The model cannot introduce
    an album**; at worst it fails to name one.
    """
    real = {normalise(t): t for t in titles}
    out: list[str] = []
    rejected: list[str] = []
    for line in text.replace(";", "\n").splitlines():
        candidate = line.strip().strip("-*•").strip().strip('"').strip()
        if not candidate or candidate.strip(".").upper() == "NONE":
            continue
        actual = real.get(normalise(candidate))
        if actual is None:
            rejected.append(REJECT_NOT_AN_ALBUM)
            continue
        if actual not in out:
            out.append(actual)
    return out, rejected


def shortlist_albums(
    query: PromptQuery,
    titles: Sequence[str],
    *,
    data_dir: Path | None = None,
    generator: TagGenerator | None = None,
) -> Shortlist:
    """Which of the user's albums a model thinks this prompt is about.

    Consulted only where exact token matching already failed, so this can add
    an album and can never take one away. With no key and no cached answer it
    returns nothing, which leaves the caller exactly where it was before this
    function existed - the whole feature degrades to token matching rather
    than to an error.

    `titles` should arrive most-significant first; only the first
    `MAX_ALBUM_TITLES` are ever sent.
    """
    sent = tuple(titles)[:MAX_ALBUM_TITLES]
    if not sent:
        return Shortlist()
    fingerprint = library_fingerprint(sent)
    cached = read_album_cache(data_dir).get(query.text)
    if cached is not None and cached.get("library") == fingerprint:
        return Shortlist(names=tuple(str(n) for n in cached["albums"]), source=SOURCE_CACHE)
    if generator is None:
        return Shortlist()
    names, rejected = generator.albums_for(query.subject, sorted(sent))
    if data_dir is not None:
        write_album_cache_entry(data_dir, query.text, fingerprint, names)
    return Shortlist(names=tuple(names), source=SOURCE_MODEL, rejected=tuple(rejected))


# ------------------------------------------------------------------ parsing


def parse_tags(text: str, *, people: Sequence[str] = ()) -> tuple[list[str], list[str]]:
    """Split a model reply into tags, dropping anything unsubstantiated.

    Returns `(tags, rejections)`. The validation mirrors `llm.substantiated`:
    the model was told not to name a year or a person, so a reply that does
    either is a reply that ignored its instructions, and the safe response is
    to drop that tag rather than search for it.
    """
    rejected: list[str] = []
    out: list[str] = []
    lowered_people = {p.casefold() for name in people for p in name.split() if len(p) > 2}
    for chunk in text.replace("\n", ";").split(";"):
        tag = " ".join(chunk.strip().strip("-*.").split())
        if not tag:
            continue
        if len(tag) > MAX_TAG_CHARS:
            rejected.append(REJECT_TOO_LONG)
            continue
        if _YEAR.search(tag):
            rejected.append(REJECT_NAMES_A_YEAR)
            continue
        if any(word in lowered_people for word in tag.casefold().split()):
            rejected.append(REJECT_NAMES_A_PERSON)
            continue
        out.append(tag)
        if len(out) == MAX_TAGS:
            break
    if len(out) < MIN_TAGS:
        rejected.append(REJECT_TOO_FEW)
        return [], rejected
    return out, rejected


# -------------------------------------------------------------------- model


class TagGenerator:
    """The optional model layer. Never raises at the call site except
    `LLMUnavailable`, which the caller turns into a printed line."""

    def __init__(self, api_key: str, transport: Transport = http_transport) -> None:
        self._api_key = api_key
        self._transport = transport

    def __repr__(self) -> str:
        return "TagGenerator(api_key=<redacted>)"

    def _ask(self, system: str) -> str:
        payload = {
            "model": MODEL,
            "reasoning": {"effort": REASONING_EFFORT},
            "input": [{"role": "system", "content": system}],
        }
        return extract_text(self._transport(payload, self._api_key))

    def tags_for(self, subject: str, *, people: Sequence[str] = ()) -> tuple[list[str], list[str]]:
        text = self._ask(
            _TAG_PROMPT.format(
                subject=subject,
                min_tags=MIN_TAGS,
                max_tags=MAX_TAGS,
                max_chars=MAX_TAG_CHARS,
            )
        )
        if not text.strip():
            return [], [REJECT_EMPTY]
        return parse_tags(text, people=people)

    def albums_for(self, subject: str, titles: Sequence[str]) -> tuple[list[str], list[str]]:
        """Album titles from `titles` that are about `subject`.

        The model sees the phrase and the album titles - the same titles the
        plausibility judge is already given, so this opens no new channel out
        of the library. It sees no photograph, no path, no date, no name and
        no count. Its reply is looked up in `titles` and anything else is
        discarded; see `parse_album_reply`.
        """
        text = self._ask(_ALBUM_PROMPT.format(subject=subject, titles="\n".join(titles)))
        if not text.strip():
            return [], [REJECT_EMPTY]
        return parse_album_reply(text, titles)

    def plausible(self, query: str, vocabulary: Vocabulary) -> str | None:
        """None when the query is coherent, else the model's one-line reason."""
        text = self._ask(
            _JUDGE_PROMPT.format(
                query=query,
                vocabulary=json.dumps(vocabulary.to_json(), sort_keys=True, ensure_ascii=False),
            )
        ).strip()
        if not text or text.upper().startswith("OK"):
            return None
        if text.upper().startswith("REFUSE"):
            reason = text.split(":", 1)[1].strip() if ":" in text else ""
            return reason[:120] or "the query does not look like something this library holds"
        # Any other shape is a reply this code does not understand, and an
        # unparsed reply must never become a refusal.
        return None


def generator_from_env(transport: Transport = http_transport) -> TagGenerator:
    key = os.environ.get(ENV_KEY)
    if not key:
        raise LLMUnavailable(
            f"{ENV_KEY} is not set, so no visual tags could be generated for "
            "this prompt. Add an entry to the tag cache, name a festival the "
            "corpus knows, or set the key."
        )
    return TagGenerator(key, transport=transport)


# ------------------------------------------------------------------ resolve


def resolve(
    query: PromptQuery,
    *,
    data_dir: Path | None = None,
    generator: TagGenerator | None = None,
    people: Sequence[str] = (),
) -> Resolution:
    """Tags for a prompt, from the first source that has them.

    The corpus comes first because it carries a month window as well as tags,
    and because it is the only source a user can inspect and correct.
    """
    festival = festivals.match(query.subject)
    if festival is not None:
        return Resolution(
            tags=festival.tags,
            source=SOURCE_CORPUS,
            festival=festival.key,
            months=festival.months,
            note=festival.note,
        )

    cached = read_caches(data_dir).get(query.text)
    if cached:
        return Resolution(tags=tuple(cached), source=SOURCE_CACHE)

    if generator is not None:
        tags, rejected = generator.tags_for(query.subject, people=people)
        if tags:
            return Resolution(tags=tuple(tags), source=SOURCE_MODEL, rejected=tuple(rejected))
        return Resolution(tags=(query.subject,), source=SOURCE_PROMPT, rejected=tuple(rejected))

    return Resolution(tags=(query.subject,), source=SOURCE_PROMPT)
