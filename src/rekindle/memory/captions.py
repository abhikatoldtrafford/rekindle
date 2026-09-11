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
    """
    dated = sorted(
        (p.meta.taken_at_local for p in photos if p.meta.taken_at_local),
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
