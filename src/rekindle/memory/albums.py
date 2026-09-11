"""Which album names may be shown to a person, and which are the same album.

Lifted out of `recipes.builtin` when a second consumer appeared. It moved
because leaving it there produced a real defect within minutes: `recurring`
counted album names as naming evidence, Google's per-year folders are on every
single photo, and every festival it discovered was about to be titled
**"Photos from"**. A rule that every consumer of album names must apply is not
a private helper of one module.
"""

from __future__ import annotations

import re

# Google's own per-year albums. Every photo is in one, so they carry no
# information a recipe could use, and "Photos from 2019" is not a title anyone
# wants to see. 23 of the 66 albums in the reference library.
_AUTO_ALBUM = re.compile(r"^Photos from \d{4}$")

# Album titles that are real metadata but not presentable. Google writes
# "Untitled", "Untitled(1)", "Untitled(3)" for albums the user never named -
# five such albums here - and a handful begin with a stray comma
# (", Abhirup, sudipta"). Neither can be shown as a memory title, and
# inventing a better one would be inventing a fact.
_UNTITLED = re.compile(r"^Untitled(\(\d+\))?$", re.IGNORECASE)

# An album name ending in a year-like suffix: "Christmas 2025", "Diwali 25",
# "Mahasaptami, 2013". Two digits or four, optionally after a comma, space,
# underscore or hyphen, optionally apostrophised.
_YEAR_SUFFIX = re.compile(r"[\s,_-]*'?\d{2}(?:\d{2})?$")


def presentable(name: str) -> bool:
    if not name or _AUTO_ALBUM.match(name) or _UNTITLED.match(name):
        return False
    # A title that opens with punctuation is a Google export artefact.
    return name[0].isalnum()


def family(name: str) -> str:
    """An album name with a trailing year-like suffix removed.

    `Christmas 2025` and `Christmas 15` are the same recurring event under two
    names, and `album_story` makes them two unrelated memories.

    Deliberately conservative: it strips a SUFFIX and never matches on a
    prefix or a shared word. `Diwali 25` and `Diwali Kali Puja 22` are
    different years of related-but-distinct festivals and must stay apart,
    which a shared-prefix rule would not manage. The reference library also
    holds `Leh Ladakh` and `ladakh` - the same trip under two names, which
    this does NOT merge, because no honest automatic rule can tell that apart
    from two different places with similar names. That case stays what
    `known-limitations.md` already says it is: one for an explicit alias
    config, not for cleverness.

    A stripped name shorter than three characters, or with no letter left in
    it, is returned unchanged: `Puri 25` is a place and a year, but `2015 2`
    is not a name with a year on the end.
    """
    stripped = _YEAR_SUFFIX.sub("", name).strip()
    if len(stripped) < 3 or not any(ch.isalpha() for ch in stripped):
        return name.strip()
    return stripped
