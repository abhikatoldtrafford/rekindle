"""The nine built-in recipes.

All nine in one module on purpose: they are 30-60 lines each, they share the
same four helpers, and nine files that each import the same three things make
the set harder to compare, not easier. A third-party recipe is one file and one
`@register` - see docs/writing-recipes.md.

Every threshold here was measured against the reference library; the numbers
are recorded beside each one, because a threshold chosen for this library must
be visibly a threshold chosen for A library rather than a law of nature.
"""

from __future__ import annotations

from datetime import timedelta

from rekindle.memory import albums, captions, recurring, strata
from rekindle.memory.composition import compose
from rekindle.memory.index import MemoryIndex
from rekindle.memory.recipes.base import (
    AS_GIVEN,
    CHRONOLOGICAL,
    MIN_SHOTS,
    Offer,
    Selection,
    chronological,
    years_of,
)
from rekindle.memory.recipes.registry import register
from rekindle.memory.spec import build_fact_sheet
from rekindle.models import Photo

# Which album names may be shown, and which two names are one album, now
# live in `memory.albums` - `recurring` needs the same rules, and the first
# thing that happened without them was every discovered festival being titled
# "Photos from".
_presentable_album = albums.presentable


def _facts(photos: list[Photo], *, title: str, recipe: str, albums=()) -> object:
    return build_fact_sheet(photos, title=title, recipe=recipe, albums=tuple(albums))


# --------------------------------------------------------------------------


@register
class AlbumStory:
    """One memory per named album. The strongest recipe on this library."""

    name = "album_story"
    title = "Album story"

    # 3, not 8. `Durga Puja 25` has 6 photos and `Puri 25` has 3, and the user
    # named the former as an expected output; a higher floor deletes them.
    min_album = MIN_SHOTS

    def offers(self, index: MemoryIndex) -> list[Offer]:
        out = []
        for album, count in index.album_counts().items():
            if count < self.min_album or not _presentable_album(album):
                continue
            photos = index.by_album(album)
            out.append(
                Offer(
                    recipe=self.name,
                    key=album,
                    title=album,
                    subtitle=captions.subtitle_for(photos),
                    size=count,
                )
            )
        # Largest first, then by name - a total order, so two runs agree.
        return sorted(out, key=lambda o: (-o.size, o.key))

    def select(self, index: MemoryIndex, offer: Offer) -> Selection | None:
        photos = index.by_album(offer.key)
        if len(photos) < MIN_SHOTS:
            return None
        ordered = chronological(photos)
        return Selection(
            photos=ordered,
            facts=_facts(ordered, title=offer.key, recipe=self.name, albums=(offer.key,)),
            ordering=CHRONOLOGICAL,
            # Across the album's OWN span: days for a one-week trip, months
            # for an album covering a child's first two years.
            stratify=strata.BY_SPAN,
            captions={p.file_hash: captions.date_caption(p) for p in ordered},
        )


@register
class OnThisDay:
    """The same calendar date across years - the anniversary trigger."""

    name = "on_this_day"
    title = "On this day"

    # >=3 distinct years and >=8 photos. Measured: 192 of the 333 dates
    # present in the library clear this. At >=2 years it would be 283, which
    # includes a lot of thin two-photo pairings; at >=5 years only 113.
    min_years = 3
    min_photos = 8

    def offers(self, index: MemoryIndex) -> list[Offer]:
        out = []
        for month, day in index.month_days():
            photos = index.by_month_day(month, day)
            years = years_of(photos)
            if len(years) < self.min_years or len(photos) < self.min_photos:
                continue
            out.append(
                Offer(
                    recipe=self.name,
                    key=f"{month:02d}-{day:02d}",
                    title=f"{day} {captions.month_name(month)}",
                    subtitle=f"{len(years)} years, {len(photos)} photos",
                    size=len(photos),
                )
            )
        return sorted(out, key=lambda o: (-o.size, o.key))

    def select(self, index: MemoryIndex, offer: Offer) -> Selection | None:
        month, day = (int(x) for x in offer.key.split("-"))
        photos = index.by_month_day(month, day)
        if len(photos) < MIN_SHOTS:
            return None
        ordered = chronological(photos)
        years = sorted(years_of(ordered))
        return Selection(
            photos=ordered,
            facts=_facts(ordered, title=offer.title, recipe=self.name),
            ordering=CHRONOLOGICAL,
            # The passage of time IS the content here, so every shot is
            # captioned with how long ago it was - relative to the most
            # recent year in the memory, NOT to the wall clock, or the same
            # index would render differently tomorrow.
            # BY YEAR, always. The entire concept of this recipe is the same
            # calendar date ACROSS YEARS; a memory confined to one year is not
            # a weaker version of it, it is a different thing.
            stratify=strata.BY_YEAR,
            # Spanning time is this recipe's premise, not a preference.
            min_strata=2,
            captions={p.file_hash: captions.anniversary_caption(p, years[-1]) for p in ordered},
        )


@register
class OnThisMonth:
    """The same calendar month across years. The fallback for days too thin."""

    name = "on_this_month"
    title = "On this month"

    min_years = 3
    # Measured: every month in this library has >=8 distinct years and between
    # 181 (April) and 3,846 (December) photos, so 30 excludes nothing here. It
    # exists for a smaller library, where a month with 12 photos across 3
    # years is not a memory.
    min_photos = 30

    def offers(self, index: MemoryIndex) -> list[Offer]:
        out = []
        for month in index.months():
            photos = index.by_month(month)
            years = years_of(photos)
            if len(years) < self.min_years or len(photos) < self.min_photos:
                continue
            out.append(
                Offer(
                    recipe=self.name,
                    key=f"{month:02d}",
                    title=f"Every {captions.month_name(month)}",
                    subtitle=f"{len(years)} years, {len(photos)} photos",
                    size=len(photos),
                )
            )
        return sorted(out, key=lambda o: (-o.size, o.key))

    def select(self, index: MemoryIndex, offer: Offer) -> Selection | None:
        photos = index.by_month(int(offer.key))
        if len(photos) < MIN_SHOTS:
            return None
        ordered = chronological(photos)
        return Selection(
            photos=ordered,
            facts=_facts(ordered, title=offer.title, recipe=self.name),
            ordering=CHRONOLOGICAL,
            # By year: this is the same MONTH across years, so the years are
            # what must be represented.
            stratify=strata.BY_YEAR,
            # Spanning time is this recipe's premise, not a preference.
            min_strata=2,
            captions={p.file_hash: captions.year_caption(p) for p in ordered},
        )


@register
class PersonYears:
    """One person, walked through time. "Avyan over the years"."""

    name = "person_years"
    title = "Person over the years"

    min_photos = 8
    min_years = 3

    def offers(self, index: MemoryIndex) -> list[Offer]:
        out = []
        for person, count in index.people_counts().items():
            if count < self.min_photos:
                continue
            photos = index.by_person(person)
            years = years_of(photos)
            if len(years) < self.min_years:
                continue
            out.append(
                Offer(
                    recipe=self.name,
                    key=person,
                    title=captions.person_title(person),
                    subtitle=f"{len(years)} years, {count} photos",
                    size=count,
                )
            )
        return sorted(out, key=lambda o: (-o.size, o.key))

    def select(self, index: MemoryIndex, offer: Offer) -> Selection | None:
        photos = index.by_person(offer.key)
        if len(photos) < MIN_SHOTS:
            return None
        ordered = chronological(photos)
        return Selection(
            photos=ordered,
            facts=_facts(ordered, title=offer.title, recipe=self.name),
            ordering=CHRONOLOGICAL,
            # "over the years" is a promise about the SPAN. One year of a
            # person's life is not it.
            stratify=strata.BY_YEAR,
            # Spanning time is this recipe's premise, not a preference.
            min_strata=2,
            captions={p.file_hash: captions.year_caption(p) for p in ordered},
        )


@register
class PairYears:
    """Two named people, together, through time."""

    name = "pair_years"
    title = "Two people over the years"

    min_photos = 8
    min_years = 3

    def offers(self, index: MemoryIndex) -> list[Offer]:
        out = []
        for (a, b), count in index.pair_counts().items():
            if count < self.min_photos:
                continue
            photos = index.by_pair(a, b)
            years = years_of(photos)
            if len(years) < self.min_years:
                continue
            out.append(
                Offer(
                    recipe=self.name,
                    # The index already keys pairs by their SORTED names, so
                    # "A + B" and "B + A" are one offer and one memory id -
                    # and a dismissal of one dismisses the other.
                    key=f"{a} + {b}",
                    title=captions.pair_title(a, b),
                    subtitle=f"{len(years)} years, {count} photos",
                    size=count,
                )
            )
        return sorted(out, key=lambda o: (-o.size, o.key))

    def select(self, index: MemoryIndex, offer: Offer) -> Selection | None:
        a, _, b = offer.key.partition(" + ")
        photos = index.by_pair(a, b)
        if len(photos) < MIN_SHOTS:
            return None
        ordered = chronological(photos)
        return Selection(
            photos=ordered,
            facts=_facts(ordered, title=offer.title, recipe=self.name),
            ordering=CHRONOLOGICAL,
            stratify=strata.BY_YEAR,
            # Spanning time is this recipe's premise, not a preference.
            min_strata=2,
            captions={p.file_hash: captions.year_caption(p) for p in ordered},
        )


@register
class ThenAndNow:
    """The earliest and the latest photo of a subject, paired.

    Exactly two shots, and `AS_GIVEN` ordering - the juxtaposition IS the
    memory, so the engine must not re-sort or cap it.
    """

    name = "then_and_now"
    title = "Then and now"

    min_photos = 8
    # The two shots must be at least this far apart or the pairing says
    # nothing. A year is the smallest gap at which "then and now" is true.
    min_gap = timedelta(days=365)

    @staticmethod
    def _showable(photos: list[Photo]) -> list[Photo]:
        """Narrow to photos that will survive the composition guardrails.

        Every other recipe hands the engine a generous pool and lets it drop
        what cannot be shown. This one picks exactly TWO photos, so if either
        is then dropped the memory dies - and it died silently: on the
        reference library the earliest photo of `Abhik Maiti` is a 6928x2309
        panorama, which the aspect gate rejects, so `offers()` advertised a
        memory `select()` could not build. Found by running the conformance
        suite against the real index, not by reading.

        `compose` is idempotent - the engine runs it again on the pair and
        drops nothing further - so calling it here costs a pass over one
        subject's photos and nothing else.
        """
        showable, _ = compose(photos)
        return showable

    def offers(self, index: MemoryIndex) -> list[Offer]:
        out = []
        for subject, all_photos in self._subjects(index):
            photos = self._showable(all_photos)
            if len(photos) < self.min_photos:
                continue
            first, last = index.earliest(photos), index.latest(photos)
            if first is None or last is None:
                continue
            if last.meta.taken_at_utc - first.meta.taken_at_utc < self.min_gap:
                continue
            label = subject.split(":", 1)[1]
            out.append(
                Offer(
                    recipe=self.name,
                    key=subject,
                    title=f"{label}: then and now",
                    subtitle=(
                        f"{first.meta.taken_at_local.year} and {last.meta.taken_at_local.year}"
                    ),
                    size=len(photos),
                )
            )
        return sorted(out, key=lambda o: (-o.size, o.key))

    @staticmethod
    def _subjects(index: MemoryIndex):
        """Both people and albums. A person is the obvious subject; an album
        is a trip, and "Kashmir: then and now" is the first and last day."""
        for person in sorted(index.people_counts()):
            yield f"person:{person}", index.by_person(person)
        for album in sorted(index.album_counts()):
            if _presentable_album(album):
                yield f"album:{album}", index.by_album(album)

    def select(self, index: MemoryIndex, offer: Offer) -> Selection | None:
        kind, _, subject = offer.key.partition(":")
        raw = index.by_person(subject) if kind == "person" else index.by_album(subject)
        photos = self._showable(raw)
        first, last = index.earliest(photos), index.latest(photos)
        if first is None or last is None or first.file_hash == last.file_hash:
            return None
        if last.meta.taken_at_utc - first.meta.taken_at_utc < self.min_gap:
            return None
        pair = [first, last]
        return Selection(
            photos=pair,
            facts=_facts(pair, title=offer.title, recipe=self.name),
            ordering=AS_GIVEN,
            # Two IS the form. The engine's default floor of three would
            # reject every then-and-now ever built.
            min_shots=2,
            # NOT stratified, deliberately. This recipe wants the EXTREMES of
            # a span, not an even spread across it - spreading it would
            # replace the two photos that are the whole point with a sample.
            stratify=strata.NONE,
            captions={
                first.file_hash: f"Then - {captions.month_year(first.meta.taken_at_local)}",
                last.file_hash: f"Now - {captions.month_year(last.meta.taken_at_local)}",
            },
        )


@register
class YearInReview:
    """The strongest photos of one year."""

    name = "year_in_review"
    title = "Year in review"

    # Measured: 2008 has 72 photos and 2026 has 1,539, so 30 admits every year
    # from 2008 on and excludes 2000-2007, which have 2-4 photos each.
    min_photos = 30

    def offers(self, index: MemoryIndex) -> list[Offer]:
        out = []
        for year in index.years():
            photos = index.by_year(year)
            if len(photos) < self.min_photos:
                continue
            out.append(
                Offer(
                    recipe=self.name,
                    key=str(year),
                    title=str(year),
                    subtitle=f"{len(photos)} photos",
                    size=len(photos),
                )
            )
        # Newest first: a year in review is most interesting for recent years.
        return sorted(out, key=lambda o: -int(o.key))

    def select(self, index: MemoryIndex, offer: Offer) -> Selection | None:
        photos = index.by_year(int(offer.key))
        if len(photos) < MIN_SHOTS:
            return None
        ordered = chronological(photos)
        return Selection(
            photos=ordered,
            facts=_facts(ordered, title=offer.title, recipe=self.name),
            ordering=CHRONOLOGICAL,
            # Across the MONTHS of its one year. Everything here shares a
            # year, so year-stratification would be a single bucket - and
            # three of these showed one month each before it was fixed.
            stratify=strata.BY_MONTH,
            # Spanning time is this recipe's premise, not a preference.
            min_strata=2,
            captions={p.file_hash: captions.month_year(p.meta.taken_at_local) for p in ordered},
        )


@register
class PlaceCluster:
    """Repeated visits to one GPS cell.

    HONESTLY WEAK ON THIS LIBRARY, and said so up front: only 12% of photos
    have GPS at all, and the 2,318 that do fall into just 19 cells at 0.25
    degrees, four of which hold 75% of them. This recipe will produce a
    handful of memories, not a rich set.

    It NEVER names a place. There is no offline gazetteer in this project, so
    a city name would be an invention; the title is fixed and the coordinates
    live in the fact sheet where they can be checked.
    """

    name = "place_cluster"
    title = "A place over time"

    min_photos = 20
    # Two stays separated by more than a fortnight are two visits, not one.
    visit_gap = timedelta(days=14)
    min_visits = 2

    def offers(self, index: MemoryIndex) -> list[Offer]:
        out = []
        for cell, count in index.gps_cells().items():
            if count < self.min_photos:
                continue
            photos = index.by_gps_cell(cell)
            visits = self._visits(photos)
            if len(visits) < self.min_visits:
                continue
            out.append(
                Offer(
                    recipe=self.name,
                    key=f"{cell[0]:.2f},{cell[1]:.2f}",
                    title=captions.PLACE_TITLE,
                    subtitle=f"{len(visits)} visits, {count} photos",
                    size=count,
                )
            )
        return sorted(out, key=lambda o: (-o.size, o.key))

    @staticmethod
    def _visits(photos: list[Photo]) -> list[list[Photo]]:
        """Split a cell's photos into stays separated by more than the gap."""
        ordered = chronological(photos)
        if not ordered:
            return []
        visits = [[ordered[0]]]
        for previous, photo in zip(ordered, ordered[1:], strict=False):
            if photo.meta.taken_at_utc - previous.meta.taken_at_utc > PlaceCluster.visit_gap:
                visits.append([])
            visits[-1].append(photo)
        return visits

    def select(self, index: MemoryIndex, offer: Offer) -> Selection | None:
        lat, _, lon = offer.key.partition(",")
        photos = index.by_gps_cell((float(lat), float(lon)))
        if len(photos) < MIN_SHOTS:
            return None
        ordered = chronological(photos)
        return Selection(
            photos=ordered,
            facts=_facts(ordered, title=offer.title, recipe=self.name),
            ordering=CHRONOLOGICAL,
            # Adaptive: the memory is about RETURNING to a place, so it must
            # show the separate visits rather than the busiest one.
            stratify=strata.BY_SPAN,
            captions={p.file_hash: captions.month_year(p.meta.taken_at_local) for p in ordered},
        )


@register
class RecurringEvent:
    """A burst of photography that happens every year at about the same time.

    The recipe the user asked for without asking for it: "October and November
    are full of Durga Puja and Kali Puja and none of it produces a memory."
    `on_this_day` cannot, because the Bengali festival calendar is lunar and
    no calendar date ever accumulates. See `memory.recurring` for the
    detection, which uses timestamps only and knows no festivals.

    Measured on the reference library: eleven events, the largest being 16
    years around late December (2,307 photos), 12 years around early October
    (2,297 - Durga Puja) and 11 years around late October (1,134 - Kali Puja).
    The October pair are the memories that did not exist before this recipe.
    """

    name = "recurring_event"
    title = "Every year, about this time"

    def offers(self, index: MemoryIndex) -> list[Offer]:
        photos = index.images()
        reserved = frozenset(p.casefold() for p in index.people_counts())
        out: list[Offer] = []
        seen: set[str] = set()
        for event in recurring.events(photos):
            # Two peaks can quantise into one fortnight bucket. Offers are
            # strongest first, so keeping the first is deterministic - but it
            # IS a dropped memory, so it is not done silently: the weaker
            # event becomes unreachable and the subtitle of the survivor does
            # not mention it. Recorded in known-limitations.md.
            if event.key in seen:
                continue
            seen.add(event.key)
            out.append(
                Offer(
                    recipe=self.name,
                    key=event.key,
                    title=self._title(event, photos, reserved),
                    subtitle=f"{len(event.years)} years, {event.count} photos",
                    size=event.count,
                )
            )
        return out

    def select(self, index: MemoryIndex, offer: Offer) -> Selection | None:
        photos = index.images()
        reserved = frozenset(p.casefold() for p in index.people_counts())
        event = next((e for e in recurring.events(photos) if e.key == offer.key), None)
        if event is None:
            return None
        chosen = chronological(recurring.photos_in(event, photos))
        if len(chosen) < MIN_SHOTS:
            return None
        years = sorted(years_of(chosen))
        return Selection(
            photos=chosen,
            facts=_facts(chosen, title=self._title(event, photos, reserved), recipe=self.name),
            ordering=CHRONOLOGICAL,
            # BY YEAR, hard. The premise is that this happens EVERY year; a
            # memory of one year's festival is `album_story`, not this.
            stratify=strata.BY_YEAR,
            min_strata=2,
            captions={p.file_hash: captions.anniversary_caption(p, years[-1]) for p in chosen},
        )

    @staticmethod
    def _title(event, photos: list[Photo], reserved: frozenset[str]) -> str:
        """From evidence only. NEVER a guessed festival name.

        Inferring "Diwali" from a date in late October is exactly the
        confident wrongness this project exists not to commit: the date is
        evidence of density, not of a festival, and a religious observance
        named wrongly in a title a person is shown is worse than saying
        nothing. If an album family names the event across years, that name is
        used because the user wrote it; otherwise the title says only what is
        known, which is when it happens and how often.
        """
        named = recurring.naming_evidence(event, photos, reserved=reserved)
        if named:
            return named
        span = range(event.years[0], event.years[-1] + 1)
        available = len(
            {
                p.meta.taken_at_local.year
                for p in photos
                if p.meta.taken_at_local and p.meta.taken_at_local.year in span
            }
        )
        return recurring.describe(event, years_available=available)
