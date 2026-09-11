"""Recurring events discovered from timestamps alone.

Fixture shapes come from the reference library, where the question was asked:
Durga Puja ran 9-13 Oct 2013, 15-19 Oct 2018, 28 Sep-2 Oct 2017 and 22-24 Sep
2023 - a drift of 24 days in day-of-year - while Kali Puja follows about three
weeks later, so the two must separate without either being named.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rekindle.memory import albums, recurring
from rekindle.memory.recipes.builtin import OnThisDay
from rekindle.models import MediaType, Photo, PhotoMeta

# Durga Puja's real start dates in this library's years, to the day.
DURGA = {
    2011: (10, 3),
    2013: (10, 6),
    2014: (9, 30),
    2016: (10, 7),
    2017: (9, 28),
    2018: (10, 15),
    2019: (10, 4),
    2023: (9, 30),
}
# Kali Puja, roughly three weeks after each.
KALI = {2011: (10, 26), 2013: (11, 2), 2014: (11, 14), 2016: (10, 29), 2017: (11, 19)}


def _photo(h: str, when: datetime, albums_=()) -> Photo:
    return Photo(
        file_hash=h,
        paths=[Path(f"/lib/{h}.jpg")],
        media_type=MediaType.IMAGE,
        albums=list(albums_),
        meta=PhotoMeta(taken_at_utc=when.replace(tzinfo=UTC), taken_at_local=when),
        first_seen=when.replace(tzinfo=UTC),
        last_seen=when.replace(tzinfo=UTC),
    )


def _library(events, *, per_day=20, days=3, baseline_per_week=2, albums_of=None) -> list[Photo]:
    """A synthetic library: a quiet baseline plus dense multi-day events.

    The baseline matters. Density is measured against the median ACTIVE day of
    the same year, so a library with nothing but the event has no baseline and
    every rule degenerates.
    """
    out: list[Photo] = []
    n = 0
    years = sorted({y for y in events})
    for year in years:
        for week in range(52):
            day = datetime(year, 1, 1) + timedelta(weeks=week)
            for _ in range(baseline_per_week):
                n += 1
                out.append(_photo(f"b{n:05d}", day + timedelta(hours=n % 12)))
    for year, (month, day) in events.items():
        names = (albums_of or {}).get(year, ())
        for offset in range(days):
            when = datetime(year, month, day) + timedelta(days=offset)
            for _ in range(per_day):
                n += 1
                out.append(_photo(f"e{n:05d}", when + timedelta(minutes=n % 600), names))
    return out


# --------------------------------------------------------------------------
# bursts


def test_a_dense_run_of_days_becomes_one_burst():
    photos = _library({2020: (10, 5)})
    found = [b for b in recurring.bursts(photos) if b.year == 2020]
    assert len(found) == 1
    assert (found[0].start.month, found[0].start.day) == (10, 5)
    assert found[0].end.day == 7


def test_an_ordinary_day_is_not_a_burst():
    """Density is relative to the year's median active day. Without that, a
    library that simply took more photos in 2025 would show a burst every
    week."""
    photos = _library({}, baseline_per_week=2)
    photos += _library({2020: (10, 5)}, per_day=3, days=1, baseline_per_week=0)[0:]
    assert recurring.bursts(photos) == []


def test_a_thin_year_produces_no_bursts_at_all():
    """2005 holds one photo and 2007 holds four. A median over four active
    days would call a two-photo day dense."""
    photos = [_photo(f"t{i}", datetime(2005, 3, i + 1)) for i in range(4)]
    assert recurring.bursts(photos) == []


def test_two_separate_weekends_are_two_bursts_not_one():
    photos = _library({2020: (10, 5)}) + _library({2020: (10, 20)}, baseline_per_week=0)
    starts = sorted(b.start.day for b in recurring.bursts(photos) if b.year == 2020)
    assert starts == [5, 20]


def test_a_day_that_beats_the_ratio_but_is_still_tiny_is_not_a_burst():
    """The ratio alone is not enough. Against a quiet baseline of two photos a
    day, nine photos is four and a half times the median and still nobody's
    festival - and the early years of this library are exactly that quiet, so
    without an absolute floor they would manufacture events out of ordinary
    afternoons."""
    photos = _library({2020: (10, 5)}, per_day=9, days=3, baseline_per_week=2)
    assert recurring.bursts(photos) == []
    # One more photo a day and it clears both tests.
    photos = _library({2020: (10, 5)}, per_day=13, days=3, baseline_per_week=2)
    assert len(recurring.bursts(photos)) == 1


def test_a_year_with_almost_no_active_days_produces_no_bursts():
    """2000, 2005, 2006 and 2007 hold two, one, two and four photos. A median
    taken over three active days describes nothing, and the one day that has
    any photos at all becomes a "burst" forty times the median."""
    photos = [_photo(f"x{i}", datetime(2005, 3, 4, 9, i)) for i in range(40)]
    photos += [_photo("y1", datetime(2005, 7, 1)), _photo("y2", datetime(2005, 9, 1))]
    assert recurring.bursts(photos) == []


def test_density_is_judged_against_the_photos_own_year():
    """A 2011 phone and a 2025 phone produce wildly different volumes. Judged
    against the whole library, 2011's biggest week is quieter than 2025's
    ordinary Tuesday and no early year could ever show an event."""
    quiet = _library({2011: (10, 5)}, per_day=15, days=3, baseline_per_week=1)
    loud = []
    n = 0
    for week in range(52):
        day = datetime(2025, 1, 1) + timedelta(weeks=week)
        for _ in range(40):
            n += 1
            loud.append(_photo(f"L{n:05d}", day + timedelta(minutes=n % 600)))

    assert [b.year for b in recurring.bursts(quiet + loud)] == [2011]


# --------------------------------------------------------------------------
# events: the thing on_this_day structurally cannot do


def test_a_lunar_festival_is_found_although_no_date_repeats():
    """The whole point. Durga Puja's real dates here span 22 September to 19
    October - `on_this_day` sees eight unrelated dates and this sees one
    event."""
    photos = _library(DURGA)
    found = recurring.events(photos)

    assert found, "no recurring event found"
    top = found[0]
    assert set(top.years) == set(DURGA)

    # ...and `on_this_day` structurally could not have found it. Its bar is
    # three distinct years on ONE calendar date; the most any single date here
    # musters is two, because 2014 and 2023 happen to share 30 September.
    # (An earlier version of this test asserted that no date repeated at all,
    # which the fixture's own dates disprove - the claim that matters is not
    # "all different" but "never enough in one place".)
    from collections import Counter

    per_date = Counter(DURGA.values())
    assert max(per_date.values()) < OnThisDay.min_years


def test_a_one_off_is_not_called_recurring():
    """A wedding is dense and multi-day and happens once. Three years would
    admit a wedding plus two anniversaries of the same trip."""
    photos = _library({2016: (6, 21), 2017: (6, 21), 2018: (6, 21)})
    assert recurring.events(photos) == []
    # One more year and it is a pattern.
    photos = _library({2016: (6, 21), 2017: (6, 21), 2018: (6, 21), 2019: (6, 21)})
    assert len(recurring.events(photos)) == 1


def test_two_festivals_three_weeks_apart_stay_separate():
    """Durga Puja and Kali Puja. A window wide enough to absorb lunar drift is
    nearly wide enough to merge them, so this is the assertion that pins the
    half-window at the value it has."""
    photos = _library(DURGA) + _library(KALI, baseline_per_week=0)
    found = recurring.events(photos)

    assert len(found) == 2, [(e.centre, e.years) for e in found]
    centres = sorted(e.centre for e in found)
    assert centres[1] - centres[0] >= 14, "the two peaks were merged into one"


def _busy_october(year):
    """One year with four separate dense weekends spread across October.

    Five days apart, not two: `bursts` joins runs a day apart, so weekends any
    closer merge into a single burst and the test would pass for the wrong
    reason. An earlier version of this test did exactly that and survived the
    mutation it was written to catch.
    """
    out = _library({year: (10, 1)})
    for day in (8, 15, 22):
        out += _library({year: (10, day)}, baseline_per_week=0)
    return out


def test_one_year_contributes_at_most_one_burst_of_evidence():
    """A year with four dense weekends in October must not read as four years
    of recurrence. Four bursts, one year, and the bar is four YEARS."""
    photos = _busy_october(2020)
    assert len(recurring.bursts(photos)) == 4
    assert recurring.events(photos) == []


def test_a_claimed_window_is_not_mined_twice_for_the_same_event():
    """A festival with a big weekend and a smaller one a week later, four
    years running. Only one burst per year is used as evidence - so claiming
    just the bursts it used would leave the smaller weekends lying inside the
    same window, and the next pass would rediscover them as a second event
    with the same years, a week apart, and no way for a reader to tell the two
    memories apart."""
    years = (2018, 2019, 2020, 2021)
    photos = _library({y: (10, 1) for y in years}, per_day=20)
    photos += _library({y: (10, 8) for y in years}, per_day=15, baseline_per_week=0)

    assert len(recurring.bursts(photos)) == 8, "fixture no longer has two bursts a year"
    found = recurring.events(photos)
    assert len(found) == 1, [(e.centre, e.years) for e in found]
    # The big weekend is the one it kept.
    assert all(b.start.day == 1 for b in found[0].bursts)


def test_an_empty_library_is_not_an_error():
    assert recurring.events([]) == []


def test_the_same_library_always_gives_the_same_events():
    photos = _library(DURGA)
    first = recurring.events(photos)
    shuffled = list(reversed(photos))
    assert [(e.centre, e.years) for e in recurring.events(shuffled)] == [
        (e.centre, e.years) for e in first
    ]


# --------------------------------------------------------------------------
# keys: a memory id may not move when the library grows


def test_the_key_does_not_change_when_another_year_is_added():
    """Dismissal and the cooldown are keyed on the memory id. A key derived
    from the discovered dates would break on every new year."""
    before = recurring.events(_library(DURGA))[0].key
    grown = dict(DURGA)
    grown[2024] = (10, 9)
    after = next(e for e in recurring.events(_library(grown)) if set(e.years) >= set(DURGA)).key
    assert after == before


def test_the_key_is_a_fortnight_not_a_day():
    photos = _library(DURGA)
    key = recurring.events(photos)[0].key
    assert key in {f"{m:02d}-{half}" for m in range(1, 13) for half in "ab"}


# --------------------------------------------------------------------------
# titles: evidence only, never a guess


def test_an_event_nothing_names_is_described_not_named():
    """Inferring "Diwali" from a date in late October is the confident
    wrongness this project exists not to commit."""
    photos = _library(DURGA)
    event = recurring.events(photos)[0]

    assert recurring.naming_evidence(event, photos) is None
    title = recurring.describe(event, years_available=len(DURGA) + 4)
    assert "Diwali" not in title and "Puja" not in title
    assert "October" in title


def test_a_name_that_recurs_across_years_is_used():
    photos = _library(
        {2020: (12, 24), 2021: (12, 24), 2022: (12, 24), 2023: (12, 24)},
        albums_of={
            2020: ["Christmas 2020"],
            2021: ["Christmas 21"],
            2022: ["Christmas 2022"],
            2023: ["Christmas 23"],
        },
    )
    event = recurring.events(photos)[0]
    assert recurring.naming_evidence(event, photos) == "Christmas"


def test_a_name_appearing_in_only_one_year_is_not_used():
    """One big album in one year is a thing that happened once, not the name
    of a recurrence. Measured on the reference library, this is why the
    12-year October peak has no name: `Mahasaptami, 2013` and `Durga Puja 25`
    are different families and each appears once."""
    photos = _library(DURGA, albums_of={2013: ["Mahasaptami, 2013"], 2025: ["Durga Puja 25"]})
    event = recurring.events(photos)[0]
    assert recurring.naming_evidence(event, photos) is None


def test_googles_year_folders_can_never_name_an_event():
    """`Photos from 2019` is on EVERY photo. Without the presentable filter
    the winning name for every event on the real library was "Photos from" -
    which is what this caught."""
    photos = _library(DURGA, albums_of={y: [f"Photos from {y}"] for y in DURGA})
    event = recurring.events(photos)[0]
    assert recurring.naming_evidence(event, photos) is None


def test_a_name_another_recipe_owns_is_not_used():
    """`person_years` already publishes a memory called `Avyan`. Two
    differently-shaped memories under one title is worse than one honest
    description of when the thing happens."""
    photos = _library(
        {2020: (8, 20), 2021: (8, 22), 2022: (8, 19), 2023: (8, 21)},
        albums_of={y: ["Avyan"] for y in (2020, 2021, 2022, 2023)},
    )
    event = recurring.events(photos)[0]

    assert recurring.naming_evidence(event, photos) == "Avyan"
    assert recurring.naming_evidence(event, photos, reserved=frozenset({"avyan"})) is None


def test_the_description_only_claims_every_year_when_it_is_every_year():
    photos = _library({2020: (10, 5), 2021: (10, 9), 2022: (10, 1), 2023: (10, 12)})
    event = recurring.events(photos)[0]
    assert recurring.describe(event, years_available=4).endswith("every year")
    assert recurring.describe(event, years_available=9).endswith("most years")


# --------------------------------------------------------------------------
# album families


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Christmas 2025", "Christmas"),
        ("Christmas 15", "Christmas"),
        ("Mahasaptami, 2013", "Mahasaptami"),
        ("Diwali 25", "Diwali"),
        ("Eden 26", "Eden"),
        ("Kashmir", "Kashmir"),
    ],
)
def test_a_year_suffix_is_stripped(name, expected):
    assert albums.family(name) == expected


def test_two_related_but_distinct_festivals_are_not_merged():
    """`Diwali 25` and `Diwali Kali Puja 22` are different years of
    related-but-distinct festivals. A shared-prefix or shared-word rule would
    merge them; a suffix rule does not, and conservative beats clever."""
    assert albums.family("Diwali 25") != albums.family("Diwali Kali Puja 22")


def test_the_leh_ladakh_case_is_left_alone():
    """The same trip under two names. No honest automatic rule separates that
    from two different places with similar names, so it stays what
    known-limitations.md says it is: one for an explicit alias config."""
    assert albums.family("Leh Ladakh") != albums.family("ladakh")


def test_only_a_year_like_suffix_is_stripped_never_a_word():
    """A rule that took the last token instead would turn `Leh Ladakh` into
    `Leh` and merge it with any other album beginning `Leh`. The whole value
    of this being conservative is that it strips a YEAR, not a word."""
    assert albums.family("Leh Ladakh") == "Leh Ladakh"
    assert albums.family("Diwali Kali Puja") == "Diwali Kali Puja"
    assert albums.family("Wedding_arnab_pics") == "Wedding_arnab_pics"


def test_a_name_that_is_only_a_number_is_not_stripped_to_nothing():
    assert albums.family("2015 2") == "2015 2"
    assert albums.family("25") == "25"


# --------------------------------------------------------------------------
# the recipe


def _index(tmp_path, photos):
    from rekindle.db import PhotoStore
    from rekindle.memory.index import MemoryIndex

    store = PhotoStore(tmp_path / "db.sqlite")
    store.upsert_many(photos)
    return store, MemoryIndex.open(store)


def test_the_recipe_spans_years_as_a_hard_requirement(tmp_path):
    """The premise is that this happens EVERY year. A memory of one year's
    festival is `album_story`, and a 24-shot `recurring_event` that is
    secretly one afternoon in 2019 is worse than none - it looks fine."""
    from rekindle.memory import strata
    from rekindle.memory.recipes.registry import get

    store, index = _index(tmp_path, _library(DURGA))
    try:
        recipe = get("recurring_event")
        offers = recipe.offers(index)
        assert offers, "the recipe offered nothing for a library built from a festival"
        selection = recipe.select(index, offers[0])
        assert selection is not None
        assert selection.stratify == strata.BY_YEAR
        assert selection.min_strata >= 2
    finally:
        store.close()


# Two peaks that land in one fortnight bucket. NOT hand-designed: two peaks
# thirteen days apart usually merge, because a single window covering both
# scores higher than either alone. This layout was found by searching 20,000
# random burst arrangements (121 of them collided, 0.6%) and then minimised by
# dropping bursts while the collision held. Day-of-year and photo count both
# matter - the count decides which burst represents its year, and that decides
# where the peak lands.
_COLLIDING = [
    # (year, month, day, photos that day)   -> one peak in late December
    (2007, 12, 12, 110),
    (2011, 12, 8, 222),
    (2012, 12, 4, 299),
    (2019, 12, 9, 74),
    (2009, 12, 28, 282),
    # -> and a second at the turn of the year, which quantises into the same
    #    fortnight bucket as the first
    (2005, 12, 31, 29),
    (2011, 12, 31, 179),
    (2019, 12, 31, 44),
    (2022, 1, 7, 132),
]


def _explicit(spec) -> list[Photo]:
    """Photos placed on exact days, with a quiet baseline in each year."""
    out: list[Photo] = []
    n = 0
    for year in sorted({row[0] for row in spec}):
        for week in range(52):
            day = datetime(year, 1, 1) + timedelta(weeks=week)
            for _ in range(2):
                n += 1
                out.append(_photo(f"q{n:05d}", day + timedelta(hours=n % 12)))
    for year, month, day, count in spec:
        for _ in range(count):
            n += 1
            out.append(_photo(f"c{n:05d}", datetime(year, month, day) + timedelta(minutes=n % 600)))
    return out


def test_two_peaks_can_land_in_one_fortnight():
    """The premise of the test below, asserted separately so that a change
    making collisions impossible shows up as this failing rather than as the
    guard silently becoming dead code."""
    events = recurring.events(_explicit(_COLLIDING))
    assert len(events) == 2, [(e.centre, e.years) for e in events]
    assert events[0].key == events[1].key == "12-b"


def test_the_captions_do_not_claim_the_photos_share_a_date(tmp_path):
    """`anniversary_caption` renders "14 years ago today", which is true for
    `on_this_day` and false here by up to three weeks - this recipe exists
    because the date MOVES. The full date is used instead, which is both true
    and the interesting thing to show."""
    from rekindle.memory.recipes.registry import get

    store, index = _index(tmp_path, _library(DURGA))
    try:
        recipe = get("recurring_event")
        selection = recipe.select(index, recipe.offers(index)[0])
    finally:
        store.close()

    captions = list(selection.captions.values())
    assert captions
    assert not any("today" in c for c in captions), captions
    # ...and the drift is visible: more than one calendar day is named.
    days = {c.split()[0] for c in captions if c}
    assert len(days) > 1, captions


def test_two_peaks_in_one_fortnight_do_not_offer_the_same_id_twice(tmp_path):
    """The key is quantised to a fortnight so that it survives the library
    growing, which means two peaks CAN land in one bucket. Offering both would
    publish two different memories under one id - and dismissing one would
    dismiss the other, silently."""
    from rekindle.memory.recipes.registry import get

    photos = _explicit(_COLLIDING)
    store, index = _index(tmp_path, photos)
    try:
        keys = [o.key for o in get("recurring_event").offers(index)]
    finally:
        store.close()
    assert keys, "the recipe offered nothing"
    assert len(keys) == len(set(keys)), f"duplicate memory ids offered: {keys}"
