"""Dismissal, the resurfacing cooldown, and memory identity.

The test that matters most in this file is
`test_a_dismissed_memory_stays_dismissed_as_the_library_grows`. Dismissal is
the only sensitivity control this project offers, and identity derived from a
photo set would make it silently stop working the moment a new photo landed.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rekindle.db import PhotoStore
from rekindle.memory.history import (
    KIND_ALBUM,
    KIND_DATES,
    KIND_MEMORY,
    KIND_PERSON,
    MemoryState,
    memory_id,
    overlap,
)
from rekindle.memory.index import MemoryIndex
from rekindle.memory.policy import ExclusionPolicy
from rekindle.models import MediaType, Photo, PhotoMeta

NOW = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)


def _p(h, *, people=(), albums=(), local=datetime(2020, 5, 1, 12, 0)) -> Photo:
    return Photo(
        file_hash=h,
        paths=[Path(f"/lib/{h}.jpg")],
        media_type=MediaType.IMAGE,
        meta=PhotoMeta(
            taken_at_utc=local.replace(tzinfo=UTC),
            taken_at_local=local,
            people=list(people),
            width=4000,
            height=3000,
        ),
        first_seen=NOW,
        last_seen=NOW,
        albums=list(albums),
    )


# --------------------------------------------------------------------------
# identity


def test_memory_id_is_recipe_plus_subject():
    assert memory_id("album_story", "Kashmir") == "album_story:Kashmir"
    assert memory_id("on_this_day", "10-20") == "on_this_day:10-20"


def test_a_dismissed_memory_stays_dismissed_as_the_library_grows(tmp_path):
    """THE test for this subsystem.

    If identity were derived from the photo set, adding one photo to Kashmir
    would produce a "new" memory and the dismissal would silently stop
    working - a control whose whole promise is "never again" failing without
    a word.
    """
    db = tmp_path / "db.sqlite"
    with PhotoStore(db) as store:
        store.upsert_many([_p(f"k{i}", albums=["Kashmir"]) for i in range(5)])
        state = MemoryState(store)
        state.dismiss(KIND_MEMORY, memory_id("album_story", "Kashmir"))
        assert "album_story:Kashmir" in state.dismissed_memory_ids()

    # The library grows: 40 more photos land in the same album, on new dates.
    with PhotoStore(db) as store:
        store.upsert_many(
            [
                _p(
                    f"new{i}",
                    albums=["Kashmir"],
                    local=datetime(2026, 1, 1, 9, 0) + timedelta(days=i),
                )
                for i in range(40)
            ]
        )
        state = MemoryState(store)
        assert "album_story:Kashmir" in state.dismissed_memory_ids()


def test_dismissing_a_memory_twice_keeps_the_original_timestamp(tmp_path):
    """ "When did you tell me to stop" is the interesting question; re-running
    a dismissal should not look like a fresh decision."""
    with PhotoStore(tmp_path / "db.sqlite") as store:
        state = MemoryState(store)
        state.dismiss(KIND_MEMORY, "album_story:Kashmir", now=NOW)
        state.dismiss(KIND_MEMORY, "album_story:Kashmir", now=NOW + timedelta(days=30))
        rows = state.dismissals()
    assert len(rows) == 1
    assert rows[0].created_at == NOW.isoformat()


def test_an_unknown_dismissal_kind_is_rejected(tmp_path):
    with PhotoStore(tmp_path / "db.sqlite") as store, pytest.raises(ValueError):
        MemoryState(store).dismiss("nonsense", "x")


def test_a_dismissal_can_be_undone(tmp_path):
    with PhotoStore(tmp_path / "db.sqlite") as store:
        state = MemoryState(store)
        state.dismiss(KIND_MEMORY, "a:b")
        assert state.undismiss(KIND_MEMORY, "a:b") is True
        assert state.dismissed_memory_ids() == frozenset()
        assert state.undismiss(KIND_MEMORY, "a:b") is False


# --------------------------------------------------------------------------
# dismissal flows through the ONE chokepoint


def test_a_dismissed_person_is_enforced_at_the_chokepoint(tmp_path):
    """Dismissal must not be a second, parallel filter. Downstream, a
    dismissed person is indistinguishable from one in exclusions.toml."""
    with PhotoStore(tmp_path / "db.sqlite") as store:
        store.upsert_many([_p("a", people=["Paramita"]), _p("b", people=["Avyan"])])
        state = MemoryState(store)
        state.dismiss(KIND_PERSON, "Paramita")

        index = MemoryIndex.open(store, state.apply_to(ExclusionPolicy()))

        assert {p.file_hash for p in index.all()} == {"b"}
        assert index.by_person("Paramita") == []


def test_a_dismissed_album_is_enforced_and_casefolded(tmp_path):
    with PhotoStore(tmp_path / "db.sqlite") as store:
        store.upsert_many([_p("a", albums=["Kashmir"]), _p("b", albums=["Mysore"])])
        state = MemoryState(store)
        state.dismiss(KIND_ALBUM, "kashmir")

        index = MemoryIndex.open(store, state.apply_to(ExclusionPolicy()))
        assert {p.file_hash for p in index.all()} == {"b"}


def test_a_dismissed_date_range_is_enforced(tmp_path):
    with PhotoStore(tmp_path / "db.sqlite") as store:
        store.upsert_many(
            [
                _p("in", local=datetime(2019, 5, 1, 12, 0)),
                _p("out", local=datetime(2021, 5, 1, 12, 0)),
            ]
        )
        state = MemoryState(store)
        state.dismiss(KIND_DATES, "2019-01-01/2019-12-31")

        index = MemoryIndex.open(store, state.apply_to(ExclusionPolicy()))
        assert {p.file_hash for p in index.all()} == {"out"}


def test_dismissals_are_merged_WITH_the_file_policy_not_instead_of_it(tmp_path):
    """A dismissal must never quietly replace a hand-written exclusion."""
    with PhotoStore(tmp_path / "db.sqlite") as store:
        state = MemoryState(store)
        state.dismiss(KIND_PERSON, "Dismissed")
        merged = state.apply_to(ExclusionPolicy(people=frozenset({"FromFile"})))
    assert merged.people == frozenset({"FromFile", "Dismissed"})


def test_apply_to_preserves_every_other_policy_field(tmp_path):
    """A field forgotten here is an exclusion silently dropped - the same trap
    as merge_meta's explicit field list."""
    with PhotoStore(tmp_path / "db.sqlite") as store:
        base = ExclusionPolicy(
            paths=(Path("/lib/private"),),
            public_safe_allow=frozenset({"Abhik Maiti"}),
            album_aliases={"ladakh": "Leh Ladakh"},
        ).with_public_safe(True)
        merged = MemoryState(store).apply_to(base)
    assert merged.paths == (Path("/lib/private"),)
    assert merged.public_safe_allow == frozenset({"Abhik Maiti"})
    assert merged.public_safe_only is True
    assert merged.album_aliases == {"ladakh": "Leh Ladakh"}


def test_a_malformed_hand_written_date_range_is_fatal(tmp_path):
    """Skipping it would silently drop an exclusion, which is the one thing
    this subsystem may never do."""
    with PhotoStore(tmp_path / "db.sqlite") as store:
        state = MemoryState(store)
        state.dismiss(KIND_DATES, "not-a-date/also-not")
        with pytest.raises(ValueError, match="malformed date range"):
            state.apply_to(ExclusionPolicy())


# --------------------------------------------------------------------------
# cooldown


def test_a_memory_surfaced_today_is_in_cooldown(tmp_path):
    with PhotoStore(tmp_path / "db.sqlite") as store:
        state = MemoryState(store)
        state.record_surfaced("on_this_day:10-20", now=NOW)
        assert state.in_cooldown("on_this_day:10-20", now=NOW) is True


def test_cooldown_expires(tmp_path):
    with PhotoStore(tmp_path / "db.sqlite") as store:
        state = MemoryState(store)
        state.record_surfaced("a:b", now=NOW)
        assert state.in_cooldown("a:b", days=90, now=NOW + timedelta(days=89)) is True
        assert state.in_cooldown("a:b", days=90, now=NOW + timedelta(days=91)) is False


def test_an_annual_anniversary_is_never_blocked_by_the_default_cooldown(tmp_path):
    """`on_this_day:12-25` recurs 365 days apart by construction, so any
    cooldown below a year is safe. 90 days is well inside that."""
    with PhotoStore(tmp_path / "db.sqlite") as store:
        state = MemoryState(store)
        state.record_surfaced("on_this_day:12-25", now=NOW)
        assert state.in_cooldown("on_this_day:12-25", now=NOW + timedelta(days=365)) is False


def test_a_never_surfaced_memory_is_not_in_cooldown(tmp_path):
    with PhotoStore(tmp_path / "db.sqlite") as store:
        assert MemoryState(store).in_cooldown("never:seen", now=NOW) is False


def test_resurfacing_updates_the_timestamp(tmp_path):
    """The cooldown asks "how long since I LAST showed this", so an older
    timestamp must never survive a newer one."""
    with PhotoStore(tmp_path / "db.sqlite") as store:
        state = MemoryState(store)
        state.record_surfaced("a:b", now=NOW)
        state.record_surfaced("a:b", now=NOW + timedelta(days=100))
        assert state.in_cooldown("a:b", now=NOW + timedelta(days=101)) is True


def test_cooling_returns_every_id_still_in_cooldown(tmp_path):
    with PhotoStore(tmp_path / "db.sqlite") as store:
        state = MemoryState(store)
        state.record_surfaced("recent:1", now=NOW)
        state.record_surfaced("old:1", now=NOW - timedelta(days=200))
        assert state.cooling(now=NOW) == frozenset({"recent:1"})


def test_a_naive_stored_timestamp_does_not_crash(tmp_path):
    """Hand-inserted rows, and rows from an older build, may be naive."""
    with PhotoStore(tmp_path / "db.sqlite") as store:
        store._conn.execute(
            "INSERT INTO memory_history(memory_id, surfaced_at) "
            "VALUES ('a:b', '2026-09-10T12:00:00')"
        )
        store._conn.commit()
        state = MemoryState(store)
        assert state.in_cooldown("a:b", now=NOW) is True
        assert "a:b" in state.cooling(now=NOW)


def test_history_is_newest_first(tmp_path):
    with PhotoStore(tmp_path / "db.sqlite") as store:
        state = MemoryState(store)
        state.record_surfaced("old:1", title="Old", now=NOW - timedelta(days=10))
        state.record_surfaced("new:1", title="New", now=NOW)
        assert [h.memory_id for h in state.history()] == ["new:1", "old:1"]


def test_state_survives_reopening_the_database(tmp_path):
    """Dismissal is permanent. It has to outlive the process."""
    db = tmp_path / "db.sqlite"
    with PhotoStore(db) as store:
        MemoryState(store).dismiss(KIND_MEMORY, "album_story:Kashmir")
    with PhotoStore(db) as store:
        assert MemoryState(store).dismissed_memory_ids() == frozenset({"album_story:Kashmir"})


# --------------------------------------------------------------------------
# overlap


def test_overlap_is_measured_against_the_smaller_set():
    """A 24-shot memory entirely inside a 200-shot one is 100% redundant to a
    viewer, even though Jaccard would call it 12%. Containment is the case
    that actually happens - on_this_day is a subset of year_in_review."""
    small = [f"p{i}" for i in range(24)]
    large = [f"p{i}" for i in range(200)]
    assert overlap(small, large) == 1.0


def test_overlap_of_disjoint_sets_is_zero():
    assert overlap(["a", "b"], ["c", "d"]) == 0.0


def test_overlap_of_half_shared_sets():
    assert overlap(["a", "b", "c", "d"], ["c", "d", "e", "f"]) == 0.5


def test_overlap_with_an_empty_set_is_zero():
    assert overlap([], ["a"]) == 0.0
    assert overlap(["a"], []) == 0.0
