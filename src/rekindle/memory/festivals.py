"""The festival corpus: checked-in data that lets rekindle NAME a festival.

`recurring_event` already finds these clusters from timestamps alone - Durga
Puja shows up as twelve years and 2,297 photos in the reference library, Kali
Puja as eleven years and 1,134 - and it deliberately refuses to name them,
because naming a festival from a date is a guess and the whole project is
built on not guessing.

**The corpus is what makes naming evidence-based rather than a guess.** It is
a file a human wrote, can read, and can correct, and it carries two things a
timestamp cannot supply: what the festival LOOKS like, and roughly when in
the year it falls. Where it has no entry, the refusal to guess stands.

Nothing here asserts a date. A lunisolar festival moves by weeks between
years, so the corpus holds a MONTH WINDOW and the shipped code never claims a
photograph was taken on a festival - it only declines to look for Kali Puja
in June.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

CORPUS_PATH = Path(__file__).with_name("corpus") / "festivals.toml"


class CorpusError(ValueError):
    """The corpus file is malformed. Always names the file and the entry."""


@dataclass(frozen=True)
class Festival:
    key: str
    names: tuple[str, ...]
    months: tuple[int, ...]
    tags: tuple[str, ...]
    note: str = ""

    @property
    def window(self) -> tuple[int, ...]:
        return self.months


def _fold(text: str) -> str:
    # Imported lazily to keep this module importable from `prompt` without a
    # cycle; `normalise` is the one definition of what a prompt string is.
    from rekindle.memory.prompt import normalise

    return normalise(text)


@lru_cache(maxsize=1)
def load(path: Path | None = None) -> tuple[Festival, ...]:
    """Read the corpus. Cached, because it is read once per prompt."""
    target = path or CORPUS_PATH
    try:
        raw = tomllib.loads(target.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CorpusError(f"{target} could not be read: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise CorpusError(f"{target} is not valid TOML: {exc}") from exc

    out: list[Festival] = []
    for entry in raw.get("festival", []):
        key = entry.get("key")
        if not key:
            raise CorpusError(f"{target}: a [[festival]] entry has no `key`.")
        names = tuple(entry.get("names") or ())
        months = tuple(int(m) for m in entry.get("months") or ())
        tags = tuple(entry.get("tags") or ())
        if not names:
            raise CorpusError(f"{target}: festival {key!r} has no `names` to match on.")
        if not tags:
            raise CorpusError(
                f"{target}: festival {key!r} has no `tags`. A festival with no "
                "visual description cannot be searched for, and its name alone "
                "is exactly what does not work."
            )
        if any(not 1 <= m <= 12 for m in months):
            raise CorpusError(f"{target}: festival {key!r} has a month outside 1-12.")
        out.append(
            Festival(key=key, names=names, months=months, tags=tags, note=entry.get("note", ""))
        )
    return tuple(out)


def all_festivals(path: Path | None = None) -> tuple[Festival, ...]:
    return load(path)


def match(subject: str, path: Path | None = None) -> Festival | None:
    """The corpus entry this subject names, or None.

    Longest name first, so `kalipuja diwali` wins over `diwali` when both are
    in the same entry and, more importantly, so a two-word festival is never
    shadowed by a one-word one that happens to be a substring of it. Matching
    is on whole words of the normalised subject, so "pujo" does not match
    inside "pujor".
    """
    folded = f" {_fold(subject)} "
    best: tuple[int, Festival] | None = None
    for festival in load(path):
        for name in festival.names:
            candidate = _fold(name)
            if f" {candidate} " in folded and (best is None or len(candidate) > best[0]):
                best = (len(candidate), festival)
    return best[1] if best else None


def iter_names(path: Path | None = None) -> Iterator[tuple[str, str]]:
    """`(festival key, name)` for every name the corpus will match. For docs."""
    for festival in load(path):
        for name in festival.names:
            yield festival.key, name
