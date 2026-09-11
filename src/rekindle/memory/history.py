"""Dismissal and resurfacing state: what the user has already seen or refused.

Dismissal is the ONE sensitivity control this project offers. Not a
confirmation prompt, not a quiet period, not a file the user is expected to
hand-edit: a memory is surfaced, and if it is unwelcome the user dismisses it
and it never comes back.

The store behind it is deliberately more capable than that one gesture. It can
hold a dismissed memory, a person, an album or a date range, because the
architectural rule is that exclusions are enforced at ONE chokepoint and the
chokepoint has to be able to express all of them. A user who wants to add an
entry by hand can (the format is documented below and it is plain SQL), but
that is not the expected path and nothing prompts for it.

Both dismissal and the resurfacing cooldown live in the same two tables in the
index database, for one reason: they gate the same thing - what a user sees -
and two mechanisms in two places are two mechanisms that can disagree.

## The identity problem

A dismissed memory must stay dismissed as the library grows. If a memory were
identified by the set of photos in it, one new photo would make a "new" memory
and the dismissal would silently stop working - the worst possible failure for
a control whose entire promise is "never again".

So a memory id is derived from its DEFINING FACTS - recipe plus subject plus
period - and never from its contents:

    album_story:Kashmir          not  album_story:<hash of 507 photos>
    on_this_day:10-20
    person_years:Avyan
    year_in_review:2016

Adding a thousand photos to Kashmir does not change `album_story:Kashmir`.
`test_history.py` pins this directly.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from rekindle.db import PhotoStore
from rekindle.memory.policy import DateRange, ExclusionPolicy

KIND_MEMORY = "memory"
KIND_PERSON = "person"
KIND_ALBUM = "album"
KIND_DATES = "dates"

# How long a surfaced memory stays out of the rotation.
#
# 90 days: long enough that a weekly `rekindle memory --auto` never repeats
# itself within a season, short enough that an annual anniversary is never
# blocked - `on_this_day:12-25` is 365 days apart by construction, so a
# cooldown anywhere below a year cannot suppress it. A shorter value would let
# `year_in_review:2016` reappear a fortnight later, which reads as a bug.
DEFAULT_COOLDOWN_DAYS = 90

# The largest share of photos two memories in one batch may have in common.
#
# 0.5: below half, the two still show mostly different photos and are worth
# watching separately; at or above half a viewer is being shown the same
# memory twice under two titles. The overlap is measured against the SMALLER
# of the two, so a 24-shot memory fully contained in a 200-shot one counts as
# 100% overlap rather than 12% - containment is the case that actually annoys.
DEFAULT_MAX_OVERLAP = 0.5


def memory_id(recipe: str, key: str) -> str:
    """The stable identity of a memory. See the module docstring."""
    return f"{recipe}:{key}"


@dataclass(frozen=True)
class Dismissal:
    kind: str
    value: str
    reason: str
    created_at: str


@dataclass(frozen=True)
class Surfaced:
    memory_id: str
    surfaced_at: datetime
    title: str


class MemoryState:
    """Reader/writer for the two state tables.

    Takes a `PhotoStore` rather than a path so that state and photos are
    always the same database: a dismissal recorded against one index and read
    from another would be a silent no-op.
    """

    def __init__(self, store: PhotoStore) -> None:
        # Reaching into the store's connection is deliberate. The alternative
        # is a second connection to the same file, which on Windows means two
        # writers, a 5-second lock timeout that nothing configures, and a
        # class of "database is locked" failures that only appear under load.
        self._conn: sqlite3.Connection = store._conn

    # ---- dismissal

    def dismiss(self, kind: str, value: str, *, reason: str = "dismissed", now=None) -> None:
        if kind not in (KIND_MEMORY, KIND_PERSON, KIND_ALBUM, KIND_DATES):
            raise ValueError(f"unknown exclusion kind: {kind!r}")
        moment = (now or datetime.now(UTC)).isoformat()
        self._conn.execute(
            "INSERT INTO memory_exclusions(kind, value, reason, created_at) VALUES (?, ?, ?, ?) "
            # A second dismissal of the same thing keeps the FIRST timestamp:
            # "when did you tell me to stop" is the interesting question, and
            # re-dismissing something should not look like a fresh decision.
            "ON CONFLICT(kind, value) DO NOTHING",
            (kind, value, reason, moment),
        )
        self._conn.commit()

    def undismiss(self, kind: str, value: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM memory_exclusions WHERE kind = ? AND value = ?", (kind, value)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def dismissals(self) -> list[Dismissal]:
        return [
            Dismissal(r["kind"], r["value"], r["reason"], r["created_at"])
            for r in self._conn.execute(
                "SELECT kind, value, reason, created_at FROM memory_exclusions ORDER BY kind, value"
            )
        ]

    def dismissed_memory_ids(self) -> frozenset[str]:
        return frozenset(
            r["value"]
            for r in self._conn.execute(
                "SELECT value FROM memory_exclusions WHERE kind = ?", (KIND_MEMORY,)
            )
        )

    def apply_to(self, policy: ExclusionPolicy) -> ExclusionPolicy:
        """Merge persisted dismissals into a loaded policy.

        This is what makes dismissal go through the SAME chokepoint as a
        hand-written exclusion instead of being a second, parallel filter. A
        dismissed person is indistinguishable, downstream, from one listed in
        exclusions.toml.
        """
        people = set(policy.people)
        albums = set(policy.albums)
        dates = list(policy.dates)
        for row in self.dismissals():
            if row.kind == KIND_PERSON:
                people.add(row.value)
            elif row.kind == KIND_ALBUM:
                albums.add(row.value.casefold())
            elif row.kind == KIND_DATES:
                start, _, end = row.value.partition("/")
                try:
                    dates.append(DateRange(date.fromisoformat(start), date.fromisoformat(end)))
                except ValueError:
                    # A hand-edited row with a malformed range. Skipping it
                    # would silently drop an exclusion, which is the one thing
                    # this subsystem may never do.
                    raise ValueError(
                        f"memory_exclusions has a malformed date range: {row.value!r} "
                        "(expected YYYY-MM-DD/YYYY-MM-DD)"
                    ) from None
        return ExclusionPolicy(
            people=frozenset(people),
            albums=frozenset(albums),
            paths=policy.paths,
            dates=tuple(dates),
            public_safe_allow=policy.public_safe_allow,
            public_safe_only=policy.public_safe_only,
            album_aliases=policy.album_aliases,
        )

    # ---- history

    def record_surfaced(self, memory_id_: str, *, title: str = "", now=None) -> None:
        moment = (now or datetime.now(UTC)).isoformat()
        self._conn.execute(
            "INSERT INTO memory_history(memory_id, surfaced_at, title) VALUES (?, ?, ?) "
            # Latest wins: the cooldown asks "how long since I last showed
            # this", so an older timestamp must never survive a newer one.
            "ON CONFLICT(memory_id) DO UPDATE SET surfaced_at = excluded.surfaced_at, "
            "title = excluded.title",
            (memory_id_, moment, title),
        )
        self._conn.commit()

    def history(self) -> list[Surfaced]:
        return [
            Surfaced(r["memory_id"], datetime.fromisoformat(r["surfaced_at"]), r["title"])
            for r in self._conn.execute(
                "SELECT memory_id, surfaced_at, title FROM memory_history ORDER BY surfaced_at DESC"
            )
        ]

    def in_cooldown(self, memory_id_: str, *, days: int = DEFAULT_COOLDOWN_DAYS, now=None) -> bool:
        row = self._conn.execute(
            "SELECT surfaced_at FROM memory_history WHERE memory_id = ?", (memory_id_,)
        ).fetchone()
        if row is None:
            return False
        moment = now or datetime.now(UTC)
        last = datetime.fromisoformat(row["surfaced_at"])
        # A naive stored timestamp (hand-inserted, or written by an older
        # build) must not raise "can't subtract offset-naive and offset-aware".
        if last.tzinfo is None:
            last = last.replace(tzinfo=UTC)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        return moment - last < timedelta(days=days)

    def cooling(self, *, days: int = DEFAULT_COOLDOWN_DAYS, now=None) -> frozenset[str]:
        """Every memory id still inside its cooldown. One query, not N."""
        moment = now or datetime.now(UTC)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        cutoff = moment - timedelta(days=days)
        out = set()
        for row in self._conn.execute("SELECT memory_id, surfaced_at FROM memory_history"):
            last = datetime.fromisoformat(row["surfaced_at"])
            if last.tzinfo is None:
                last = last.replace(tzinfo=UTC)
            if last > cutoff:
                out.add(row["memory_id"])
        return frozenset(out)


def overlap(a: list[str], b: list[str]) -> float:
    """Share of photos two memories have in common, 0.0 to 1.0.

    Measured against the SMALLER set, not the union. A 24-shot memory entirely
    contained in a 200-shot one is 100% redundant to the viewer even though
    Jaccard would call it 12% - and containment is exactly the case that
    happens here, where `on_this_day:10-20` is a subset of `year_in_review`.
    """
    left, right = set(a), set(b)
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))
