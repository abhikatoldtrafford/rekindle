"""Deterministic titles and captions.

Every string a memory shows is copied from stored metadata or computed
arithmetically from it. There is no other source. The complete vocabulary is
the table in the design doc, and the rule behind it is the one the brief calls
out: **never invent a fact.**

The sharpest instance: rekindle has no offline gazetteer, so a memory NEVER
names a place. Not a city, not a country, not "the coast". A place memory says
"A place you kept coming back to", and the coordinates live in the fact sheet
where they can be checked. A plausible-sounding city name derived from a
lat/lon we cannot resolve is exactly the failure this rule exists to prevent.
"""

from __future__ import annotations

import unicodedata
from datetime import datetime

from rekindle.models import Photo

# Said when a memory has GPS but, necessarily, no place name.
PLACE_TITLE = "A place you kept coming back to"

_MONTHS = (
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
)


def month_name(month: int) -> str:
    """Month names from a literal table, not `strftime`.

    `%B` is locale-dependent, so the same index would produce different
    memories on a machine with a different LC_TIME - and byte-for-byte
    reproducibility is the promise the whole engine rests on.
    """
    return _MONTHS[month - 1]


def month_year(moment: datetime) -> str:
    return f"{month_name(moment.month)} {moment.year}"


def day_month_year(moment: datetime) -> str:
    return f"{moment.day} {month_name(moment.month)} {moment.year}"


def years_ago(then: int, now: int) -> str:
    """Arithmetic, with the singular spelled correctly.

    "1 years ago today" is the kind of thing that makes a memory feel
    machine-made at exactly the wrong moment.
    """
    delta = now - then
    if delta <= 0:
        return "today"
    if delta == 1:
        return "1 year ago today"
    return f"{delta} years ago today"


def person_title(person: str) -> str:
    return f"{person} over the years"


def pair_title(a: str, b: str) -> str:
    """Two names, in the order given. The caller has already sorted them, so
    "A and B" and "B and A" never both appear."""
    return f"{a} and {b}"


def span_subtitle(photos: list[Photo]) -> str:
    """ "October 2024" for one month, "October 2024 - January 2025" across two.

    Empty when nothing is dated, never a guess.

    SORTED ON THE WALL CLOCK, not on the instant. `taken_at_local` is stored
    with its offset attached, so a bare `sorted()` compares absolute moments -
    and a photograph taken at 00:00:54 on 1 January 2019 in IST is the instant
    18:30:54 on 31 December 2018 UTC, which sorts BEFORE one taken at 18:39:15
    UTC that same evening.

    Found on the real library: the album `Photos from 2018` ends with New
    Year's Eve photographs whose local clock reads January 2019, and this
    function called the span "August 2018 - December 2018". The field is named
    `local` because the wall clock is the thing the person lived through; that
    is what a subtitle should name.
    """
    dated = sorted(
        p.meta.taken_at_local.replace(tzinfo=None) for p in photos if p.meta.taken_at_local
    )
    if not dated:
        return ""
    first, last = dated[0], dated[-1]
    if (first.year, first.month) == (last.year, last.month):
        return month_year(first)
    return f"{month_year(first)} - {month_year(last)}"


def count_subtitle(photos: list[Photo]) -> str:
    n = len(photos)
    return "1 photo" if n == 1 else f"{n} photos"


def subtitle_for(photos: list[Photo]) -> str:
    span = span_subtitle(photos)
    count = count_subtitle(photos)
    return f"{count}, {span}" if span else count


def span_subtitle_of(span: tuple[tuple[int, int], tuple[int, int]] | None) -> str:
    """`span_subtitle` from two `(year, month)` pairs instead of photographs.

    The same three cases and the same words - one month, two months, or
    nothing at all - so a caller that already knows the span need not load a
    library to get it. `MemoryIndex.album_span` reads it off the spine.

    Both functions, one wording: `test_captions.py` asserts they agree on the
    real shapes, because a subtitle that differs depending on which route
    computed it is a determinism bug that would only show up as a diff.
    """
    if span is None:
        return ""
    (first_year, first_month), (last_year, last_month) = span
    if (first_year, first_month) == (last_year, last_month):
        return f"{month_name(first_month)} {first_year}"
    return f"{month_name(first_month)} {first_year} - {month_name(last_month)} {last_year}"


def subtitle_of(count: int, span: tuple[tuple[int, int], tuple[int, int]] | None) -> str:
    """`subtitle_for` from a count and a span. See `span_subtitle_of`."""
    counted = "1 photo" if count == 1 else f"{count} photos"
    rendered = span_subtitle_of(span)
    return f"{counted}, {rendered}" if rendered else counted


def year_caption(photo: Photo) -> str:
    """The year a shot was taken. Used by the over-the-years recipes, where
    the passage of time IS the content."""
    local = photo.meta.taken_at_local
    return str(local.year) if local else ""


def date_caption(photo: Photo) -> str:
    local = photo.meta.taken_at_local
    return day_month_year(local) if local else ""


def anniversary_caption(photo: Photo, today_year: int) -> str:
    local = photo.meta.taken_at_local
    if local is None:
        return ""
    return years_ago(local.year, today_year)


def describe_people(photos: list[Photo], *, limit: int = 3) -> str:
    """ "Paramita, Abhik Maiti and Avyan" - the most-present people, by count.

    Returns an empty string when nobody is tagged, rather than "nobody" or
    "unknown". 43.7% of live photos carry no face tag, so silence is the
    honest output and it is very common.
    """
    counts: dict[str, int] = {}
    for photo in photos:
        for person in photo.meta.people:
            if person:
                counts[person] = counts.get(person, 0) + 1
    if not counts:
        return ""
    ranked = sorted(counts, key=lambda name: (-counts[name], name))[:limit]
    if len(ranked) == 1:
        return ranked[0]
    return f"{', '.join(ranked[:-1])} and {ranked[-1]}"


# ---------------------------------------------------------------------------
# What the renderer can actually draw
# ---------------------------------------------------------------------------
#
# rekindle ships no font file. `render.frames` draws with Pillow's bundled
# Aileron, which is a SUBSET face: it has no glyph for any dash but the ASCII
# hyphen, no accented Latin letter, and no non-breaking space. A character it
# lacks is not skipped - FreeType draws `.notdef`, a filled rectangle, so the
# caption shows a tofu box where the character should be.
#
# Nothing rekindle computes can produce one. The deterministic captions are
# dates and names copied from metadata, the CLIP vocabulary is ASCII, and this
# library's 19,480 rows carry no non-ASCII character in any album, person or
# keyword. The GPT layer can and does: 3 of 167 cached captions came back with
# an en or em dash, including `9 August 2025 - Paramita`, which rendered a box
# in the middle of the line.
#
# The table is MEASURED, not guessed - `test_captions.py` renders every key
# through the bundled font and asserts it draws `.notdef`, and renders every
# value and asserts it does not. That test fails if Pillow's bundled face ever
# changes in either direction, which is the only way to keep this honest.
#
# Some characters that look like they belong here deliberately do not: the
# ellipsis, the smart quotes, the middle dot, the degree sign and `©` all draw
# correctly, so folding them would lose typography for nothing.
_FOLD = {
    "‐": "-",  # hyphen
    "‑": "-",  # non-breaking hyphen
    "‒": "-",  # figure dash
    "–": "-",  # en dash        <- the one that shipped
    "—": "-",  # em dash
    "―": "-",  # horizontal bar
    "‚": "'",  # single low-9 quote
    "„": '"',  # double low-9 quote
    "′": "'",  # prime
    "″": '"',  # double prime
    "•": "·",  # bullet -> middle dot, which draws
    "×": "x",  # multiplication sign
    " ": " ",  # no-break space
    " ": " ",  # figure space
    " ": " ",  # thin space
    " ": " ",  # narrow no-break space
}


def renderable(text: str) -> str:
    """`text` with characters the bundled font cannot draw folded into ones it
    can, where an exact equivalent exists.

    Two passes. The table above handles punctuation and spacing, where the
    fold is exact - an en dash between two facts means the same thing as a
    hyphen. Then any remaining character with a canonical decomposition is
    stripped of its combining marks, so an accented Latin letter degrades to
    the letter (`Jose`, not `Jos` plus a box).

    **Anything else is returned unchanged.** A Bengali or Chinese title still
    draws as boxes, and that is the honest outcome: this function folds
    characters, it cannot invent glyphs. Transliterating a script would be
    inventing a name, which is the one thing captions may never do.
    """
    if text.isascii():
        return text
    folded = "".join(_FOLD.get(ch, ch) for ch in text)
    if folded.isascii():
        return folded
    out = []
    for ch in folded:
        if ch.isascii():
            out.append(ch)
            continue
        decomposed = unicodedata.normalize("NFD", ch)
        stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
        # Only accept the decomposition when it lands entirely in ASCII;
        # a partial fold is a different word, not a degraded one.
        out.append(stripped if stripped and stripped.isascii() else ch)
    return "".join(out)
