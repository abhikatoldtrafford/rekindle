"""The MemorySpec: the durable, deterministic description of one memory.

A spec is JSON and is the artefact. Renderers are pure functions of it plus the
photo bytes, so the same index always produces the same spec and the same spec
always produces the same GIF.

Two rules shape what goes in here:

**No filesystem paths in the FactSheet.** The fact sheet is the ONLY thing the
optional GPT layer ever sees, and a path leaks a username, a drive layout and
often a person's name. `test_spec.py` asserts this by scanning the serialised
form rather than by trusting the field list.

**No invented facts anywhere.** Every string is copied from stored metadata or
computed arithmetically from it. There is no gazetteer in this project, so a
memory never names a place - not even when it has coordinates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from rekindle.models import Photo

SPEC_VERSION = 1


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


@dataclass(frozen=True)
class Shot:
    """One photo in a memory.

    Carries a file_hash, not a path. The renderer resolves the path through
    the index, which is what keeps a spec portable between machines and stops
    an absolute path to someone's photo library ending up in a JSON file.
    """

    file_hash: str
    caption: str
    taken_at_local: str | None
    public_safe: bool

    def to_json(self) -> dict:
        return {
            "file_hash": self.file_hash,
            "caption": self.caption,
            "taken_at_local": self.taken_at_local,
            "public_safe": self.public_safe,
        }

    @classmethod
    def from_json(cls, raw: dict) -> Shot:
        return cls(
            file_hash=raw["file_hash"],
            caption=raw["caption"],
            taken_at_local=raw["taken_at_local"],
            public_safe=raw["public_safe"],
        )


@dataclass(frozen=True)
class FactSheet:
    """Everything substantiated about a memory, and nothing else.

    This is the complete input to the optional GPT caption layer. A field
    added here is a field an LLM may assert, so anything speculative is a
    licence to hallucinate rather than a hint that produces better prose.

    `places` holds COORDINATES only. rekindle has no offline gazetteer, so a
    city name would be an invention, and "never invent a fact" is binding even
    where inventing would read better.
    """

    title: str
    recipe: str
    photo_count: int
    date_from: str | None = None
    date_to: str | None = None
    years: tuple[int, ...] = ()
    per_year: dict[str, int] = field(default_factory=dict)
    people: dict[str, int] = field(default_factory=dict)
    albums: tuple[str, ...] = ()
    places: tuple[tuple[float, float], ...] = ()
    video_count: int = 0
    # Is the TITLE itself a fact?
    #
    # For every recipe it is: an album title the user typed, a person's name
    # from their own tags, a month. `llm.substantiated` therefore treats the
    # words of the title as substantiated proper nouns, which is right for
    # `album_story:Kashmir`.
    #
    # For a prompt memory it is NOT. The title is the user's QUERY, and a
    # query may contain anything - `christmas in midnapur` would whitelist
    # "Midnapur" as a proper noun the caption layer may then print under a
    # photograph, defeating both REJECT_UNKNOWN_PERSON and the rule that a
    # memory never names a place.
    title_substantiated: bool = True

    def to_json(self) -> dict:
        out = {
            "title": self.title,
            "recipe": self.recipe,
            "photo_count": self.photo_count,
            "years": list(self.years),
            "per_year": self.per_year,
            "people": self.people,
            "albums": list(self.albums),
            "video_count": self.video_count,
        }
        # Absent when true, so every existing spec on disk round-trips
        # byte-for-byte and only a prompt memory carries the flag.
        if not self.title_substantiated:
            out["title_substantiated"] = False
        # Absent rather than null. An LLM handed `"date_from": null` will
        # cheerfully write around it; a missing key is unambiguous, and the
        # same reasoning applies to `places` - see the class docstring.
        if self.date_from:
            out["date_from"] = self.date_from
        if self.date_to:
            out["date_to"] = self.date_to
        if self.places:
            out["places"] = [list(p) for p in self.places]
        return out

    @classmethod
    def from_json(cls, raw: dict) -> FactSheet:
        return cls(
            title=raw["title"],
            recipe=raw["recipe"],
            photo_count=raw["photo_count"],
            date_from=raw.get("date_from"),
            date_to=raw.get("date_to"),
            years=tuple(raw.get("years", ())),
            per_year=dict(raw.get("per_year", {})),
            people=dict(raw.get("people", {})),
            albums=tuple(raw.get("albums", ())),
            places=tuple(tuple(p) for p in raw.get("places", ())),  # type: ignore[misc]
            video_count=raw.get("video_count", 0),
            title_substantiated=raw.get("title_substantiated", True),
        )


@dataclass(frozen=True)
class MemorySpec:
    recipe: str
    key: str
    title: str
    subtitle: str
    shots: tuple[Shot, ...]
    facts: FactSheet
    public_safe: bool

    @property
    def slug(self) -> str:
        return f"{self.recipe}-{safe_slug(self.key)}"

    def to_json(self) -> dict:
        return {
            "spec_version": SPEC_VERSION,
            "recipe": self.recipe,
            "key": self.key,
            "title": self.title,
            "subtitle": self.subtitle,
            "public_safe": self.public_safe,
            "shots": [s.to_json() for s in self.shots],
            "facts": self.facts.to_json(),
        }

    def dumps(self) -> str:
        """Deterministic JSON.

        `sort_keys` and a fixed separator so that two runs over the same index
        produce byte-identical output - the promise the whole engine is built
        on, and one that a dict iteration order change would quietly break.
        `ensure_ascii=False` so a Bengali album title stays readable.
        """
        return json.dumps(self.to_json(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    @classmethod
    def loads(cls, text: str) -> MemorySpec:
        raw = json.loads(text)
        if raw.get("spec_version") != SPEC_VERSION:
            raise ValueError(f"unsupported spec_version {raw.get('spec_version')!r}")
        return cls(
            recipe=raw["recipe"],
            key=raw["key"],
            title=raw["title"],
            subtitle=raw["subtitle"],
            shots=tuple(Shot.from_json(s) for s in raw["shots"]),
            facts=FactSheet.from_json(raw["facts"]),
            public_safe=raw["public_safe"],
        )


_SAFE = "abcdefghijklmnopqrstuvwxyz0123456789-_"


def safe_slug(value: str) -> str:
    """A filename-safe form of a memory key.

    Album titles in a real library contain commas, slashes and non-ASCII
    ("Mahasaptami, 2013", "Abhirup Birthday/ Sudipta Saad"), and a key becomes
    a directory name. Everything outside a conservative ASCII set becomes a
    hyphen, which can collide - so the CALLER addresses memories by the real
    key and this is used only for display on disk.
    """
    lowered = value.casefold()
    out = [c if c in _SAFE else "-" for c in lowered]
    slug = "".join(out).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "memory"


def build_fact_sheet(
    photos: list[Photo],
    *,
    title: str,
    recipe: str,
    albums: tuple[str, ...] = (),
    title_substantiated: bool = True,
) -> FactSheet:
    """Facts derived from the photos that actually made it into the memory.

    Derived from the FINAL shot list, not the candidate pool: a fact sheet
    describing 400 candidates while the memory shows 24 of them would let the
    narrator assert things the viewer cannot see.
    """
    from rekindle.models import MediaType

    dated = [p for p in photos if p.meta.taken_at_local]
    ordered = sorted(dated, key=lambda p: (p.meta.taken_at_local, p.file_hash))

    per_year: dict[str, int] = {}
    people: dict[str, int] = {}
    places: list[tuple[float, float]] = []
    for photo in ordered:
        year = str(photo.meta.taken_at_local.year)
        per_year[year] = per_year.get(year, 0) + 1
        for person in photo.meta.people:
            if person:
                people[person] = people.get(person, 0) + 1
        if photo.meta.gps is not None:
            places.append((round(photo.meta.gps.lat, 4), round(photo.meta.gps.lon, 4)))

    return FactSheet(
        title=title,
        recipe=recipe,
        photo_count=len(photos),
        date_from=_iso(ordered[0].meta.taken_at_local) if ordered else None,
        date_to=_iso(ordered[-1].meta.taken_at_local) if ordered else None,
        years=tuple(sorted(int(y) for y in per_year)),
        per_year=dict(sorted(per_year.items())),
        # Sorted by count then name, so the order is stable and the most
        # present person leads.
        people=dict(sorted(people.items(), key=lambda kv: (-kv[1], kv[0]))),
        albums=albums,
        places=tuple(sorted(set(places))),
        video_count=sum(1 for p in photos if p.media_type is MediaType.VIDEO),
        title_substantiated=title_substantiated,
    )


def contains_path_like(text: str) -> bool:
    """Heuristic used by the tests: does this look like it carries a path?

    Lives here rather than in the test file so the rule is visible beside the
    FactSheet it constrains.
    """
    lowered = text.lower()
    return (
        "\\\\" in text
        or ":/" in lowered
        or ":\\\\" in lowered
        or any(f"{d}:" in lowered for d in "cdefgh")
        or str(Path.home()).lower() in lowered
    )
