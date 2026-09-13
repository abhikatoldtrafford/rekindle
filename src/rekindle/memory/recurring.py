"""Recurring events, discovered from timestamps alone.

The user asked why October and November - full of Durga Puja and Kali Puja -
produce no memories while Christmas does. Measured, the answer is structural:

    12-25   503 photos across 13 years    <- fixed date, `on_this_day` works
    11-23   432 across  6 years           <- Puja, in one year
    11-02   400 across  7 years           <- Puja, in a different year
    10-05   332 across  8 years           <- Puja, in a third

Christmas works *because* it is fixed. The Bengali festival calendar is lunar,
so Durga Puja was 9-13 Oct in 2013, 28 Sep-2 Oct in 2025, 15-19 Oct in 2018.
No single calendar date ever accumulates it, and no amount of tuning
`on_this_day` can change that: the premise of that recipe is the date.

So this module asks a different question, and one that needs no festival
calendar and no cultural knowledge:

    A recurring event is a multi-day burst of unusually dense photography
    that happens at about the same time of year, for several years, even as
    the exact dates drift.

That signature is in the timestamps. It finds Durga Puja on this library and
would find Thanksgiving or Midsummer on someone else's, with nothing
hardcoded. **It does not and must not NAME what it finds** - see `title_for`.

Why a peak search rather than clustering the dense days
-------------------------------------------------------

The obvious approach - group dense clusters wherever there is a gap between
them - was built first and fails on the real library, for a reason worth
keeping. In a Bengali autumn there is no gap: Durga Puja, Kali Puja and the
weeks between them form one unbroken run of dense days from late September to
late November. Gap-splitting either merges the whole autumn into a single
60-day "event" or, with a tighter gap, shatters all of it. Measured: at an
8-day split it found two events on this library and neither was a festival.

Instead, find the day-of-year window that the MOST DISTINCT YEARS agree on,
claim it, and repeat on what is left. Durga Puja and Kali Puja separate
because each has its own peak - not because anything here knows their names.
Measured on the reference library, the first four peaks are 12 years around
early October, 16 around late December, 11 around late October, and 7 around
late November.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date

from rekindle.memory import albums
from rekindle.models import Photo

# A day is "dense" when it holds this many times the median ACTIVE day of its
# own year. Per-year, because a 2011 phone and a 2025 phone produce wildly
# different volumes; median of active days rather than a mean over 365,
# because a library with 200 empty days would otherwise call every ordinary
# day dense. Measured: 4.0 yields 154 dense clusters over 23 years, which is
# a few per year. At 2.0 it is 400+ and ordinary weekends qualify; at 8.0 the
# thinner early years stop contributing at all.
DENSITY = 4.0

# ...and at least this many photos, whatever the year's median. Without it,
# 2005 (one photo) and 2007 (four) generate "dense" days of two photos.
MIN_DAY_PHOTOS = 12

# Days this far apart still belong to the same burst. A festival has quiet
# mornings; 1 bridges them without joining two separate weekends.
JOIN_GAP = 1

# How far the same event may drift between years, either side of the peak.
# Durga Puja's centre moves from day-of-year 266 to 290 across the years in
# this library - 24 days, so +/-12 is the smallest window that holds it. It is
# also close to the largest that can: Kali Puja follows about 20 days later,
# and a wider window swallows both into one event.
HALF_WINDOW = 12

# Years that must show the event before it is called recurring. Three would
# admit a wedding plus two anniversaries of visiting the same place; four is
# where the reference library stops producing coincidences.
MIN_YEARS = 4


@dataclass(frozen=True)
class Burst:
    """One year's run of dense days."""

    year: int
    start: date
    end: date
    count: int

    @property
    def doy(self) -> int:
        """Day-of-year of the burst's middle."""
        return self.start.timetuple().tm_yday + (self.end - self.start).days // 2

    def covers(self, day: date) -> bool:
        return self.start <= day <= self.end


@dataclass(frozen=True)
class RecurringEvent:
    centre: int
    bursts: tuple[Burst, ...]

    @property
    def years(self) -> tuple[int, ...]:
        return tuple(sorted({b.year for b in self.bursts}))

    @property
    def count(self) -> int:
        return sum(b.count for b in self.bursts)

    @property
    def key(self) -> str:
        """The stable identity of this event.

        A memory id must not move when the library grows, because dismissals
        and the cooldown are keyed on it - so the key CANNOT be the centre
        day, which shifts by a day or two as years are added.

        The centre is therefore quantised to a fortnight: 26 buckets, named by
        the month and half-month they fall in. Adding a year moves a
        ten-year-old peak by at most a day or two, so the bucket holds unless
        the centre already sat within that distance of a boundary.

        That residual is real and is recorded in `known-limitations.md`
        alongside `place_cluster`'s, which has the same shape. It is not
        pretended away: the alternative - keying on the discovered dates -
        breaks on EVERY new year rather than rarely.
        """
        day = _from_doy(self.centre)
        return f"{day.month:02d}-{'a' if day.day <= 15 else 'b'}"


def _from_doy(doy: int) -> date:
    """A day-of-year as a date in a non-leap reference year."""
    return date.fromordinal(date(2001, 1, 1).toordinal() + (doy - 1) % 365)


def _local_day(photo: Photo) -> date | None:
    moment = photo.meta.taken_at_local
    return moment.date() if moment else None


def day_counts(photos: Iterable[Photo]) -> Counter[date]:
    """How many of `photos` fall on each local calendar day."""
    by_day: Counter[date] = Counter()
    for photo in photos:
        day = _local_day(photo)
        if day is not None:
            by_day[day] += 1
    return by_day


def bursts(photos: Iterable[Photo]) -> list[Burst]:
    """Every run of unusually dense days, per year."""
    return bursts_from(day_counts(photos))


def bursts_from(by_day: Mapping[date, int]) -> list[Burst]:
    """`bursts`, from a day histogram rather than from the photos.

    This split is what lets a 300,000-photo library find its recurring events
    without materialising itself. Nothing below this line reads a `Photo`: the
    density rule was always a question about a histogram, and `MemoryIndex`
    can answer it off the spine (`image_day_counts`) for the cost of a
    `Counter` copy. See `recipes.builtin.RecurringEvent`.
    """
    by_year: dict[int, dict[date, int]] = {}
    for day, count in by_day.items():
        by_year.setdefault(day.year, {})[day] = count

    out: list[Burst] = []
    for year, days in sorted(by_year.items()):
        if len(days) < 10:
            # Too few active days for a median to describe anything. 2000,
            # 2005, 2006 and 2007 in the reference library.
            continue
        counts = sorted(days.values())
        median = counts[len(counts) // 2]
        floor = max(DENSITY * median, MIN_DAY_PHOTOS)
        dense = sorted(day for day, count in days.items() if count >= floor)
        run: list[date] = []
        for day in dense:
            if run and (day - run[-1]).days <= JOIN_GAP + 1:
                run.append(day)
                continue
            if run:
                out.append(_burst(year, run, days))
            run = [day]
        if run:
            out.append(_burst(year, run, days))
    return out


def _burst(year: int, run: list[date], days: dict[date, int]) -> Burst:
    start, end = run[0], run[-1]
    total = sum(count for day, count in days.items() if start <= day <= end)
    return Burst(year=year, start=start, end=end, count=total)


def _circular(a: int, b: int) -> int:
    """Distance around the year, so late December is near early January.

    365 even in a leap year, deliberately. The only value that differs is a
    31 December burst in a leap year, whose day-of-year is 366: against a
    centre near 1 January this still measures 0, and against a centre at 365
    it measures 1 rather than 0. One day, on one day of one year in four, in a
    measure whose window is +/-12 - and correcting it would mean carrying a
    year into a function about seasons.
    """
    gap = abs(a - b)
    return min(gap, 365 - gap)


def events(
    photos: Iterable[Photo],
    *,
    min_years: int = MIN_YEARS,
    half_window: int = HALF_WINDOW,
) -> list[RecurringEvent]:
    """Recurring events, strongest first.

    Greedy: take the day-of-year window the most distinct years agree on,
    claim everything inside it, repeat. Claiming the WHOLE window rather than
    only the chosen bursts matters - otherwise the next pass rediscovers the
    same festival from the bursts it left behind.
    """
    return events_from(day_counts(photos), min_years=min_years, half_window=half_window)


def events_from(
    by_day: Mapping[date, int],
    *,
    min_years: int = MIN_YEARS,
    half_window: int = HALF_WINDOW,
) -> list[RecurringEvent]:
    """`events`, from a day histogram. See `bursts_from` for why."""
    remaining = bursts_from(by_day)
    found: list[RecurringEvent] = []
    while True:
        best: tuple[tuple[int, int, int], int, list[Burst]] | None = None
        for centre in range(1, 366):
            near = [b for b in remaining if _circular(b.doy, centre) <= half_window]
            # One burst per year, the biggest: a year with four dense
            # weekends in October must not count as four years of evidence.
            per_year: dict[int, Burst] = {}
            for burst in near:
                if burst.year not in per_year or burst.count > per_year[burst.year].count:
                    per_year[burst.year] = burst
            members = list(per_year.values())
            if len(members) < min_years:
                continue
            # Third term: of the windows that tie on years and photos, the one
            # whose members sit closest around it. Without it the first
            # qualifying centre wins, which is always the LOWEST - so a
            # festival on 1 October got a centre eleven days earlier, `describe`
            # called it "Mid September", and a second burst a week after the
            # first fell outside the claimed window and was published as a
            # separate memory. Found by a fixture, not by reading.
            spread = sum(_circular(b.doy, centre) for b in members)
            score = (len(members), sum(b.count for b in members), -spread)
            if best is None or score > best[0]:
                best = (score, centre, members)
        if best is None:
            return found
        _, centre, members = best
        found.append(
            RecurringEvent(centre=centre, bursts=tuple(sorted(members, key=lambda b: b.year)))
        )
        remaining = [b for b in remaining if _circular(b.doy, centre) > half_window]


def photos_in(event: RecurringEvent, photos: Iterable[Photo]) -> list[Photo]:
    """Every photo inside any of the event's bursts."""
    return [
        photo
        for photo in photos
        if (day := _local_day(photo)) is not None
        and any(burst.covers(day) for burst in event.bursts)
    ]


def naming_evidence(
    album_years: Mapping[str, set[int]],
    *,
    min_years: int = 2,
    reserved: frozenset[str] = frozenset(),
) -> str | None:
    """The album name this event is called, or None if nothing names it.

    `album_years` is album name AS WRITTEN -> the years of the event it
    appears in, which is the whole of what this rule ever read from a
    photograph. `MemoryIndex.image_album_years` produces it off the spine;
    walking the event's photographs produces the same mapping and is what this
    used to do inline, at 11,422 hydrations on the reference library. The rule
    below is unchanged, and it must stay here rather than migrate into the
    index: the index reports evidence, this decides what it names.

    **Never guesses.** Inferring "Diwali" from a date in late October is
    exactly the kind of confident wrongness this project exists not to commit:
    the date is evidence of density, not of a festival, and getting a
    religious observance wrong in a title a person is shown is worse than
    saying nothing.

    A name counts only when the same album FAMILY appears in at least two
    distinct years of the event. One big album in one year is a thing that
    happened once, not the name of a recurrence - and the reference library
    proves the point: the October peak has 12 years of evidence and no name,
    because `Mahasaptami, 2013` and `Durga Puja 25` are different families and
    each appears once.

    `reserved` holds names another recipe already owns - in practice the
    people in the library. The late-August peak here is named `Avyan` by this
    rule, which is true (it is a child's album, recurring) and useless as a
    title: `person_years` already publishes a memory called `Avyan`, and two
    differently-shaped memories under one name is worse than one honest
    "Late August, most years". Measured: it is two of the eleven events on
    this library, and they are the only two that were named at all.
    """
    years_by_family: dict[str, set[int]] = {}
    for album, years in album_years.items():
        # `presentable` is not cosmetic here. Google's per-year folders
        # are on EVERY photo, so without this filter the winning name for
        # every event on this library was "Photos from". It is applied to the
        # name as written and BEFORE `family`, because `Photos from 2019` is
        # not presentable while its family `Photos from` is.
        if albums.presentable(album):
            years_by_family.setdefault(albums.family(album), set()).update(years)

    candidates = [
        (len(years), family)
        for family, years in years_by_family.items()
        if len(years) >= min_years and family.casefold() not in reserved
    ]
    if not candidates:
        return None
    # Most years wins; the name itself breaks ties, so the choice does not
    # depend on dictionary order.
    return max(candidates, key=lambda c: (c[0], [-ord(ch) for ch in c[1]]))[1]


_PHASE = ((10, "Early"), (20, "Mid"), (31, "Late"))
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


def describe(event: RecurringEvent, *, years_available: int) -> str:
    """A title made only of what is known: when, and how reliably.

    `years_available` is how many years of the event's span hold any photos at
    all, so "every year" is a claim that can be checked rather than a flourish.
    """
    day = _from_doy(event.centre)
    phase = next(label for limit, label in _PHASE if day.day <= limit)
    when = f"{phase} {_MONTHS[day.month - 1]}"
    if years_available and len(event.years) >= years_available:
        return f"{when}, every year"
    return f"{when}, most years"
