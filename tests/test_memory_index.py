"""The chokepoint.

The central test in this file is `test_no_query_method_can_return_a_blocked_photo`.
It enumerates every public method on MemoryIndex rather than listing the ones
that exist today, so a query added later WITHOUT filtering fails automatically.
That is the difference between a guardrail and a convention.
"""

import inspect
from datetime import UTC, date, datetime
from pathlib import Path

from rekindle.db import PhotoStore
from rekindle.memory.index import MemoryIndex, gps_cell
from rekindle.memory.policy import DateRange, ExclusionPolicy
from rekindle.models import Gps, MediaType, Photo, PhotoMeta

ALLOW = frozenset({"Abhik Maiti"})


def _p(h, **kw) -> Photo:
    local = kw.pop("local", datetime(2020, 5, 1, 12, 0))
    gps = kw.pop("gps", None)
    return Photo(
        file_hash=h,
        paths=kw.pop("paths", [Path(f"/lib/{h}.jpg")]),
        media_type=kw.pop("media_type", MediaType.IMAGE),
        meta=PhotoMeta(
            taken_at_utc=kw.pop("utc", local.replace(tzinfo=UTC) if local else None),
            taken_at_local=local,
            people=kw.pop("people", []),
            archived=kw.pop("archived", False),
            trashed=kw.pop("trashed", False),
            gps=Gps(*gps) if gps else None,
        ),
        first_seen=datetime(2020, 1, 1, tzinfo=UTC),
        last_seen=datetime(2020, 1, 1, tzinfo=UTC),
        albums=kw.pop("albums", []),
    )


def _store(tmp_path, photos) -> PhotoStore:
    store = PhotoStore(tmp_path / "db.sqlite")
    store.upsert_many(photos)
    return store


# The blocked photo is deliberately given EVERY attribute that could make it
# turn up in a query: a year, a month-day, a person, a pair, an album and a
# GPS fix. A query method that forgot to filter will therefore return it,
# whichever index it reads.
BLOCKED = "blocked"
OK = "ok"


def _mixed(tmp_path, **policy_kw):
    photos = [
        _p(
            BLOCKED,
            archived=True,
            people=["Paramita", "Abhik Maiti"],
            albums=["Kashmir"],
            gps=(22.5, 87.25),
            local=datetime(2020, 5, 1, 12, 0),
        ),
        _p(
            OK,
            people=["Paramita", "Abhik Maiti"],
            albums=["Kashmir"],
            gps=(22.5, 87.25),
            local=datetime(2020, 5, 1, 12, 0),
        ),
    ]
    store = _store(tmp_path, photos)
    return store, MemoryIndex.open(store, ExclusionPolicy(**policy_kw))


def test_no_query_method_can_return_a_blocked_photo(tmp_path):
    """Enumerates EVERY public method, so a future unfiltered query fails here
    without anyone remembering to extend this test.

    The blocked photo shares a year, month-day, person, pair, album and GPS
    cell with the allowed one, so any index that was not filtered returns it.
    """
    store, index = _mixed(tmp_path)
    try:
        # Arguments chosen to hit the blocked photo in every index.
        args = {
            "by_year": (2020,),
            "by_month": (5,),
            "by_month_day": (5, 1),
            "by_person": ("Paramita",),
            "by_pair": ("Abhik Maiti", "Paramita"),
            "by_album": ("Kashmir",),
            "by_gps_cell": (gps_cell(22.5, 87.25),),
            "get": (BLOCKED,),
            "resolve_many": ([BLOCKED, OK],),
            "by_date": (2020, 5, 1),
            "resolve_path": (index.all()[0],),
            "is_public_safe": (index.all()[0],),
            "years_present": (index.all(),),
            "earliest": (index.all(),),
            "latest": (index.all(),),
        }

        checked = []
        for name, member in inspect.getmembers(MemoryIndex):
            if name.startswith("_") or name == "open":
                continue
            if isinstance(member, property):
                _touch(getattr(index, name), checked, name)
                continue
            if not callable(member):
                continue
            signature = inspect.signature(member)
            # -1 for `self`.
            needed = len(signature.parameters) - 1
            call_args = args.get(name, ())
            if needed and not call_args:
                raise AssertionError(
                    f"MemoryIndex.{name} takes arguments but this test does not "
                    "know how to call it. Add it to `args` with values that "
                    "would reach the blocked photo."
                )
            _touch(getattr(index, name)(*call_args), checked, name)

        assert "by_year" in checked, "the enumeration itself must have run"
        # The three methods the semantic layer comes in through are swept like
        # every other one. Named explicitly as well as counted, because a
        # rename that dropped one from the sweep would still leave the count
        # satisfied by the methods that remain.
        for name in ("resolve_many", "by_date", "dates"):
            assert name in checked, f"{name} was not swept"
        assert len(checked) >= 21, f"only {len(checked)} methods checked: {checked}"
    finally:
        store.close()


def _touch(result, checked, name):
    """Fail if a blocked photo appears anywhere in `result`."""
    checked.append(name)
    items = result if isinstance(result, (list, tuple, set)) else [result]
    for item in items:
        if isinstance(item, Photo):
            assert item.file_hash != BLOCKED, f"MemoryIndex.{name} leaked a blocked photo"
        if isinstance(item, str):
            assert item != BLOCKED, f"MemoryIndex.{name} leaked a blocked photo"


def test_the_blocked_photo_really_is_reachable_without_the_guardrail(tmp_path):
    """Guards the test above against being vacuous.

    If the fixture's blocked photo did not actually share indexes with the
    allowed one, every assertion up there would pass while proving nothing -
    the "test that cannot fail" shape this project keeps hitting.
    """
    store, _ = _mixed(tmp_path)
    try:
        unguarded = MemoryIndex(list(store.iter_photos()), ExclusionPolicy(), _empty_report())
        assert {p.file_hash for p in unguarded.by_year(2020)} == {BLOCKED, OK}
        assert {p.file_hash for p in unguarded.by_album("Kashmir")} == {BLOCKED, OK}
        assert {p.file_hash for p in unguarded.by_person("Paramita")} == {BLOCKED, OK}
        assert {p.file_hash for p in unguarded.by_gps_cell(gps_cell(22.5, 87.25))} == {BLOCKED, OK}
    finally:
        store.close()


def _empty_report():
    from rekindle.memory.index import ExclusionReport

    return ExclusionReport()


def test_person_exclusion_is_honoured_by_every_index(tmp_path):
    store, index = _mixed(tmp_path, people=frozenset({"Paramita"}))
    try:
        assert index.count() == 0
        assert index.by_album("Kashmir") == []
        assert index.people_counts() == {}
    finally:
        store.close()


def test_date_exclusion_is_honoured(tmp_path):
    store, index = _mixed(tmp_path, dates=(DateRange(date(2020, 5, 1), date(2020, 5, 1)),))
    try:
        assert index.count() == 0
    finally:
        store.close()


def test_public_safe_mode_removes_non_qualifying_photos_everywhere(tmp_path):
    photos = [
        _p("solo", people=["Abhik Maiti"], albums=["Kashmir"]),
        _p("pair", people=["Abhik Maiti", "Paramita"], albums=["Kashmir"]),
        _p("untagged", people=[], albums=["Kashmir"]),
    ]
    store = _store(tmp_path, photos)
    try:
        policy = ExclusionPolicy(public_safe_allow=ALLOW).with_public_safe(True)
        index = MemoryIndex.open(store, policy)
        assert {p.file_hash for p in index.all()} == {"solo"}
        assert {p.file_hash for p in index.by_album("Kashmir")} == {"solo"}
    finally:
        store.close()


def test_an_untagged_photo_is_excluded_by_public_safe_mode(tmp_path):
    """43.7% of this library. The count is the point: public-safe mode is not
    a light filter, and widening it to admit untagged photos is what the whole
    rule exists to prevent."""
    store = _store(tmp_path, [_p("untagged", people=[])])
    try:
        policy = ExclusionPolicy(public_safe_allow=ALLOW).with_public_safe(True)
        assert MemoryIndex.open(store, policy).count() == 0
    finally:
        store.close()


# --------------------------------------------------------------------------
# the report


def test_the_report_counts_by_first_reason_and_adds_up(tmp_path):
    photos = [
        _p("a", archived=True),
        _p("b", archived=True),
        _p("c", people=["Unwelcome"]),
        _p("ok"),
    ]
    store = _store(tmp_path, photos)
    try:
        index = MemoryIndex.open(store, ExclusionPolicy(people=frozenset({"Unwelcome"})))
        report = index.report
        assert report.total == 4
        assert report.allowed == 1
        assert report.excluded == 3
        assert report.by_reason == {"archived": 2, "excluded_person": 1}
        assert report.accounted, "every excluded photo must be counted under exactly one reason"
    finally:
        store.close()


def test_the_report_never_names_a_photo(tmp_path):
    """A report that named the excluded photos would defeat the exclusion."""
    store = _store(tmp_path, [_p("secret", archived=True)])
    try:
        report = MemoryIndex.open(store, ExclusionPolicy()).report
        assert "secret" not in repr(report)
    finally:
        store.close()


# --------------------------------------------------------------------------
# queries


def test_an_empty_store_answers_every_query_emptily(tmp_path):
    store = _store(tmp_path, [])
    try:
        index = MemoryIndex.open(store)
        assert index.count() == 0
        assert index.all() == []
        assert index.by_year(2020) == []
        assert index.years() == []
        assert index.people_counts() == {}
        assert index.earliest([]) is None
        assert index.latest([]) is None
    finally:
        store.close()


def test_pairs_are_order_independent(tmp_path):
    store = _store(tmp_path, [_p("a", people=["Paramita", "Abhik Maiti"])])
    try:
        index = MemoryIndex.open(store)
        assert len(index.by_pair("Abhik Maiti", "Paramita")) == 1
        assert len(index.by_pair("Paramita", "Abhik Maiti")) == 1
        # ...and the pair appears ONCE in the counts, not twice.
        assert list(index.pair_counts()) == [("Abhik Maiti", "Paramita")]
    finally:
        store.close()


def test_album_aliases_merge_two_spellings_when_configured(tmp_path):
    """`Leh Ladakh` and `ladakh` are one trip in this library. Nothing merges
    them automatically - that would be asserting a fact no metadata contains -
    but the user can say so explicitly."""
    photos = [_p("a", albums=["Leh Ladakh"]), _p("b", albums=["ladakh"])]
    store = _store(tmp_path, photos)
    try:
        plain = MemoryIndex.open(store)
        assert len(plain.by_album("Leh Ladakh")) == 1
        assert len(plain.by_album("ladakh")) == 1

        merged = MemoryIndex.open(store, ExclusionPolicy(album_aliases={"ladakh": "Leh Ladakh"}))
        assert len(merged.by_album("Leh Ladakh")) == 2
    finally:
        store.close()


def test_the_same_album_with_two_different_years_becomes_one(tmp_path):
    """`Christmas 2025` and `Christmas 15` are one recurring event that
    `album_story` otherwise publishes as two unrelated memories, one of eight
    photos and one of eleven. Measured on the reference library, this is the
    only pair the rule merges."""
    photos = [_p("a", albums=["Christmas 2025"]), _p("b", albums=["Christmas 15"])]
    store = _store(tmp_path, photos)
    try:
        index = MemoryIndex.open(store)
        assert len(index.by_album("Christmas")) == 2
        assert index.by_album("Christmas 2025") == []
        assert index.album_merges == {
            "Christmas 15": "Christmas",
            "Christmas 2025": "Christmas",
        }
    finally:
        store.close()


def test_an_album_with_no_partner_keeps_the_name_the_user_wrote(tmp_path):
    """The measured reason the rule only merges where it merges. Stripping the
    suffix everywhere renames seven albums on the reference library -
    `Durga Puja 25` to `Durga Puja`, `Puri 25` to `Puri` - and merges none of
    them. Each rename changes a memory id, so every dismissal of one stops
    applying, and it buys nothing."""
    photos = [_p("a", albums=["Durga Puja 25"]), _p("b", albums=["Puri 25"])]
    store = _store(tmp_path, photos)
    try:
        index = MemoryIndex.open(store)
        assert len(index.by_album("Durga Puja 25")) == 1
        assert index.by_album("Durga Puja") == []
        assert index.album_merges == {}
    finally:
        store.close()


def test_related_but_distinct_festivals_are_not_merged(tmp_path):
    """`Diwali 25` and `Diwali Kali Puja 22` are different years of
    related-but-distinct festivals. A shared-prefix or shared-word rule merges
    them; a suffix rule does not, and conservative beats clever."""
    photos = [_p("a", albums=["Diwali 25"]), _p("b", albums=["Diwali Kali Puja 22"])]
    store = _store(tmp_path, photos)
    try:
        index = MemoryIndex.open(store)
        assert index.album_merges == {}
        assert len(index.by_album("Diwali 25")) == 1
        assert len(index.by_album("Diwali Kali Puja 22")) == 1
    finally:
        store.close()


def test_googles_year_folders_are_never_merged_with_each_other(tmp_path):
    """`Photos from 2019` and `Photos from 2020` share a family under any
    suffix rule, and merging them would produce one album called `Photos from`
    holding the entire library."""
    photos = [_p("a", albums=["Photos from 2019"]), _p("b", albums=["Photos from 2020"])]
    store = _store(tmp_path, photos)
    try:
        index = MemoryIndex.open(store)
        assert index.album_merges == {}
        assert index.by_album("Photos from") == []
    finally:
        store.close()


def test_an_explicit_alias_beats_the_automatic_rule(tmp_path):
    """The user always wins. `album_aliases` is the documented override and it
    must be able to undo, not just extend, what the automatic rule did."""
    photos = [_p("a", albums=["Christmas 2025"]), _p("b", albums=["Christmas 15"])]
    store = _store(tmp_path, photos)
    try:
        index = MemoryIndex.open(
            store, ExclusionPolicy(album_aliases={"Christmas 15": "Christmas 15"})
        )
        assert len(index.by_album("Christmas 15")) == 1
        assert len(index.by_album("Christmas")) == 1
    finally:
        store.close()


def test_earliest_and_latest_break_ties_on_file_hash(tmp_path):
    """1,430 timestamps in this library are shared by more than one photo, so
    a bare date key is not a total order."""
    same = datetime(2020, 5, 1, 12, 0)
    store = _store(tmp_path, [_p("zzz", local=same), _p("aaa", local=same)])
    try:
        index = MemoryIndex.open(store)
        assert index.earliest(index.all()).file_hash == "aaa"
        assert index.latest(index.all()).file_hash == "zzz"
    finally:
        store.close()


def test_gps_cells_floor_and_handle_negative_coordinates():
    """`round` would put the boundary in the middle of a cell, and `int()`
    truncates towards zero - folding the hemispheres together at 0."""
    assert gps_cell(22.5, 87.25) == (22.5, 87.25)
    assert gps_cell(22.6, 87.3) == (22.5, 87.25)
    assert gps_cell(-0.1, -0.1) == (-0.25, -0.25)
    assert gps_cell(0.1, 0.1) == (0.0, 0.0)


def test_resolve_path_returns_the_copy_that_exists(tmp_path):
    real = tmp_path / "there.jpg"
    real.write_bytes(b"x")
    store = _store(tmp_path, [_p("a", paths=[tmp_path / "gone.jpg", real])])
    try:
        index = MemoryIndex.open(store)
        assert index.resolve_path(index.get("a")) == real
    finally:
        store.close()


def test_resolve_path_returns_none_when_nothing_exists(tmp_path):
    store = _store(tmp_path, [_p("a", paths=[tmp_path / "gone.jpg"])])
    try:
        index = MemoryIndex.open(store)
        assert index.resolve_path(index.get("a")) is None
    finally:
        store.close()


def test_a_query_result_is_a_copy_not_the_internal_store(tmp_path):
    """A recipe holding the result of `all()` must not be able to inject a
    photo that never passed the policy."""
    store = _store(tmp_path, [_p("a")])
    try:
        index = MemoryIndex.open(store)
        index.all().append(_p("smuggled", archived=True))
        index.by_year(2020).append(_p("smuggled", archived=True))
        assert index.count() == 1
        assert {p.file_hash for p in index.all()} == {"a"}
    finally:
        store.close()


def test_the_internal_photo_collection_is_immutable(tmp_path):
    """Defence in depth, and deliberately a SEPARATE test from the one above.

    Every query already returns a copy, so making `_photos` a tuple changes no
    observable behaviour today - the test above passes either way. It is kept
    because the copying is a property of thirteen query methods that each have
    to remember it, while this is a property of the one attribute they all
    read. Asserting it directly is what stops the tuple being quietly relaxed
    to a list by someone who notices no test fails.
    """
    store = _store(tmp_path, [_p("a")])
    try:
        index = MemoryIndex.open(store)
        assert isinstance(index._photos, tuple)
    finally:
        store.close()


# ---------------------------------------------------------------- resolve_many


def test_resolve_many_preserves_order_and_drops_an_unknown_hash(tmp_path):
    store, index = _mixed(tmp_path)
    try:
        photos = index.resolve_many([OK, "never-indexed", OK])
        assert [p.file_hash for p in photos] == [OK, OK]
    finally:
        store.close()


def test_resolve_many_drops_a_blocked_photo_without_leaving_a_none(tmp_path):
    """The guardrail is inherited, not re-implemented.

    A caller doing `[index.get(h) for h in hits]` would get a `None` in the
    list and would have to remember to filter it. This returns photos only,
    so forgetting is not possible.
    """
    store, index = _mixed(tmp_path)
    try:
        photos = index.resolve_many([BLOCKED, OK])
        assert all(p is not None for p in photos)
        assert [p.file_hash for p in photos] == [OK]
    finally:
        store.close()


# -------------------------------------------------------------------- by_date


def test_by_date_buckets_on_the_local_day_not_the_utc_instant(tmp_path):
    """A photo taken at 00:30 local belongs to the day the user lived."""
    late = _p(
        "late",
        local=datetime(2019, 10, 5, 0, 30),
        utc=datetime(2019, 10, 4, 19, 0, tzinfo=UTC),
    )
    same_day = _p("same", local=datetime(2019, 10, 5, 14, 0))
    other = _p("other", local=datetime(2019, 10, 4, 14, 0))
    store = _store(tmp_path, [late, same_day, other])
    try:
        index = MemoryIndex.open(store)
        assert {p.file_hash for p in index.by_date(2019, 10, 5)} == {"late", "same"}
        assert {p.file_hash for p in index.by_date(2019, 10, 4)} == {"other"}
    finally:
        store.close()


def test_by_date_is_not_by_month_day(tmp_path):
    """The anniversary index folds years together; this one must not."""
    store = _store(
        tmp_path,
        [
            _p("a", local=datetime(2019, 10, 5, 12, 0)),
            _p("b", local=datetime(2020, 10, 5, 12, 0)),
        ],
    )
    try:
        index = MemoryIndex.open(store)
        assert len(index.by_month_day(10, 5)) == 2
        assert [p.file_hash for p in index.by_date(2019, 10, 5)] == ["a"]
    finally:
        store.close()


def test_dates_lists_every_local_day_in_order(tmp_path):
    store = _store(
        tmp_path,
        [
            _p("b", local=datetime(2020, 10, 5, 12, 0)),
            _p("a", local=datetime(2019, 10, 5, 12, 0)),
            _p("c", local=datetime(2020, 10, 5, 18, 0)),
        ],
    )
    try:
        index = MemoryIndex.open(store)
        assert index.dates() == [(2019, 10, 5), (2020, 10, 5)]
    finally:
        store.close()
