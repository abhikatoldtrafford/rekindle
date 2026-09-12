"""The chokepoint.

The central test in this file is `test_no_query_method_can_return_a_blocked_photo`.
It enumerates every public method on MemoryIndex rather than listing the ones
that exist today, so a query added later WITHOUT filtering fails automatically.
That is the difference between a guardrail and a convention.
"""

import inspect
from array import array
from collections.abc import Iterator
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
            # The aggregates. They return integers, so `_touch` cannot see a
            # leak in them - `test_the_year_aggregates_honour_an_exclusion`
            # is what actually holds them. They are here so the sweep still
            # refuses to pass over a method it does not know how to call.
            "month_years": (5,),
            "month_day_years": (5, 1),
            "person_years": ("Paramita",),
            "pair_years": ("Abhik Maiti", "Paramita"),
            "album_years": ("Kashmir",),
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
        for name in ("resolve_many", "by_date", "dates", "iter_all", "iter_images"):
            assert name in checked, f"{name} was not swept"
        assert len(checked) >= 32, f"only {len(checked)} methods checked: {checked}"
    finally:
        store.close()


def _touch(result, checked, name):
    """Fail if a blocked photo appears anywhere in `result`.

    An ITERATOR is drained, not skipped. `iter_all` and `iter_images` return
    generators, and a sweep that treated a generator as one opaque object
    would have run every assertion below against the generator itself and
    passed without ever producing a photo - the vacuous-test shape this file
    exists to prevent.
    """
    checked.append(name)
    if isinstance(result, Iterator):
        result = list(result)
        assert result, f"MemoryIndex.{name} yielded nothing, so nothing was checked"
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


def test_the_index_holds_no_photo_collection_at_all(tmp_path):
    """Defence in depth, and deliberately a SEPARATE test from the one above.

    This used to assert that `_photos` was a tuple rather than a list: every
    query already returned a copy, so it changed no observable behaviour, and
    it was kept because the copying was a property thirteen query methods each
    had to remember while immutability was a property of the one attribute
    they all read.

    There is now no such attribute. The spine is `array("q")` of SQLite
    rowids, which cannot hold a `Photo` even in principle, and the LRU cache
    is keyed on rowid, so a photo appended to a returned list reaches nothing.
    The assertion is therefore that no attribute of the index is a container
    OF PHOTOS - which is both the immutability property and the laziness
    property, and fails the moment someone reintroduces a materialised list.
    """
    store = _store(tmp_path, [_p("a")])
    try:
        index = MemoryIndex.open(store)
        index.all()  # fill the cache, so the cache itself is examined too
        assert isinstance(index._ids, array)
        holding = {
            name: value
            for name, value in vars(index).items()
            if isinstance(value, (list, tuple, set, frozenset))
            and any(isinstance(item, Photo) for item in value)
        }
        assert not holding, f"the index is materialising photos in {sorted(holding)}"
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


# --------------------------------------------------------------------------
# laziness
#
# "A lazy index that quietly falls back to materialising everything looks
# identical to a working one until you measure memory." So these tests measure
# it. Nothing below passes on a materialising index.


def _bulk(n, *, year=2020):
    """`n` photos with people and an album, so the derived indexes are real."""
    return [
        _p(
            f"{i:064x}",
            local=datetime(year, 1 + i % 12, 1 + i % 28, 12, 0),
            people=[f"Person {i % 7}", f"Person {(i + 3) % 7}"],
            albums=[f"Trip {i % 20}"],
        )
        for i in range(n)
    ]


def test_open_hydrates_nothing_at_all(tmp_path):
    """The whole claim, at its narrowest: opening the index reads every row,
    applies the policy to every row, and keeps NO photo.

    This is the test that fails the moment someone reintroduces a materialised
    list, whatever else they leave in place.
    """
    store = _store(tmp_path, _bulk(50))
    index = MemoryIndex.open(store)
    try:
        assert index.count() == 50
        assert len(index._cache) == 0, "opening the index materialised photos"
        assert len(index.by_year(2020)) == 50
    finally:
        index.close()
        store.close()


def test_the_cache_is_bounded_and_queries_are_still_complete(tmp_path):
    """A cache smaller than the library must change what is HELD, never what
    is ANSWERED."""
    store = _store(tmp_path, _bulk(60))
    index = MemoryIndex.open(store)
    try:
        index._cache_max = 10
        # 60 at once is larger than the cache, so nothing is admitted...
        assert len(index.all()) == 60
        assert len(index._cache) == 0
        # ...but a query that fits is cached, and bounded.
        for month in range(1, 13):
            assert index.by_month(month)
        assert len(index._cache) <= 10
        # and every answer is still whole.
        assert len(index.all()) == 60
        assert sum(len(index.by_month(m)) for m in index.months()) == 60
    finally:
        index.close()
        store.close()


def test_a_scan_larger_than_the_cache_does_not_evict_the_working_set(tmp_path):
    """`all()` over a library bigger than the cache must not flush it.

    Without the guard, a single `index.all()` - which `rekindle memories`
    printed a warning from - would evict every recipe's working set on the way
    past, and the next query would re-read the database for all of it.
    """
    store = _store(tmp_path, _bulk(60))
    index = MemoryIndex.open(store)
    try:
        index._cache_max = 20
        kept = index.by_month(1)
        assert kept and len(index._cache) == len(kept)
        before = set(index._cache)
        index.all()
        assert set(index._cache) == before, "a big scan flushed the cache"
    finally:
        index.close()
        store.close()


def test_the_index_costs_far_less_than_the_photos_it_indexes(tmp_path):
    """The measurement, not an assertion about it.

    Measured on a synthetic 300,000-row library the photos are 681 MB and the
    indexes over them 32.5 MB, which is the whole reason this class stopped
    holding photos. Here the same comparison is made in miniature and as a
    RATIO, so it means the same thing on any platform and at any Python
    version: opening the index must cost a small fraction of what holding the
    library costs.

    A materialising index scores 1.0 and fails.
    """
    import gc
    import tracemalloc

    store = _store(tmp_path, _bulk(2000))
    index = None
    try:
        gc.collect()
        tracemalloc.start()
        materialised = list(store.iter_photos())
        photos_bytes = tracemalloc.get_traced_memory()[0]
        del materialised
        gc.collect()
        base = tracemalloc.get_traced_memory()[0]
        index = MemoryIndex.open(store)
        index_bytes = tracemalloc.get_traced_memory()[0] - base
        tracemalloc.stop()
        assert index.count() == 2000
        assert index_bytes < photos_bytes / 4, (
            f"the index costs {index_bytes / 1e6:.2f} MB against "
            f"{photos_bytes / 1e6:.2f} MB for the photos - it is materialising them"
        )
    finally:
        if index is not None:
            index.close()
        store.close()


def test_iter_all_yields_the_library_without_holding_it(tmp_path):
    store = _store(tmp_path, _bulk(60))
    index = MemoryIndex.open(store)
    try:
        index._cache_max = 5
        seen = [p.file_hash for p in index.iter_all()]
        assert len(seen) == 60
        assert seen == [p.file_hash for p in index.all()]
        assert len(index._cache) == 0, "streaming admitted photos to the cache"
    finally:
        index.close()
        store.close()


def test_iter_images_yields_only_images(tmp_path):
    photos = _bulk(4)
    photos[1].media_type = MediaType.VIDEO
    store = _store(tmp_path, photos)
    index = MemoryIndex.open(store)
    try:
        assert {p.file_hash for p in index.iter_images()} == {
            p.file_hash for p in photos if p.media_type is MediaType.IMAGE
        }
    finally:
        index.close()
        store.close()


def test_image_day_counts_counts_images_only_and_matches_by_date(tmp_path):
    """The aggregate `recurring_event` reads instead of the whole library. It
    must agree with the index it was derived from, or the recipe finds bursts
    the photos cannot fill."""
    photos = [
        _p("a", local=datetime(2020, 5, 1, 9, 0)),
        _p("b", local=datetime(2020, 5, 1, 10, 0)),
        _p("v", local=datetime(2020, 5, 1, 11, 0), media_type=MediaType.VIDEO),
        _p("c", local=datetime(2020, 5, 2, 9, 0)),
    ]
    store = _store(tmp_path, photos)
    index = MemoryIndex.open(store)
    try:
        counts = index.image_day_counts()
        assert counts == {(2020, 5, 1): 2, (2020, 5, 2): 1}
        for ymd, n in counts.items():
            images = [p for p in index.by_date(*ymd) if p.media_type is MediaType.IMAGE]
            assert len(images) == n
    finally:
        index.close()
        store.close()


def test_a_hydrated_photo_is_the_one_the_scan_admitted(tmp_path):
    """The spine holds rowids, so a fetch that read the wrong row - or a
    policy that disagreed with itself between the scan and the fetch - shows
    up as a query returning fewer photos than the count says."""
    store = _store(tmp_path, [*_bulk(40), _p("hidden", archived=True)])
    index = MemoryIndex.open(store)
    try:
        assert index.count() == 40
        assert len(index.all()) == 40
        assert index.report.total == 41
        assert index.report.by_reason == {"archived": 1}
        assert index.get("hidden") is None
    finally:
        index.close()
        store.close()


def test_closing_the_index_releases_the_file_and_reopening_is_transparent(tmp_path):
    """Windows will not delete an open file, so a caller must be able to let
    go of it - and letting go must not end the index."""
    store = _store(tmp_path, _bulk(5))
    index = MemoryIndex.open(store)
    try:
        assert len(index.all()) == 5
        index.close()
        index.close()  # idempotent
        assert len(index.all()) == 5, "the index did not reopen"
    finally:
        index.close()
        store.close()


def test_a_merged_album_comes_back_in_the_order_the_library_is_scanned_in(tmp_path):
    """Two spellings of one album, interleaved, must come back interleaved.

    The rowid indexes are built per RAW album name and only merged afterwards,
    so the naive merge concatenates - every `Christmas 15` before every
    `Christmas 2025` - and `album_story` then publishes a memory in a
    different order than the materialised index did. Sorting the merged rowids
    restores scan order, and scan order is rowid order only because
    `_StoreSource.scan` says ORDER BY rowid.

    Interleaved on purpose: with the two spellings in separate runs, a
    concatenating merge produces the right answer by accident and this test
    cannot fail.
    """
    photos = [
        _p("a", albums=["Christmas 2025"], local=datetime(2025, 12, 25, 9, 0)),
        _p("b", albums=["Christmas 15"], local=datetime(2015, 12, 25, 9, 0)),
        _p("c", albums=["Christmas 2025"], local=datetime(2025, 12, 25, 10, 0)),
        _p("d", albums=["Christmas 15"], local=datetime(2015, 12, 25, 10, 0)),
    ]
    store = _store(tmp_path, photos)
    index = MemoryIndex.open(store)
    try:
        merged = [p.file_hash for p in index.by_album("Christmas")]
        scanned = [p.file_hash for p in index.all()]
        assert merged == [h for h in scanned if h in set(merged)]
        assert merged == ["a", "b", "c", "d"]
    finally:
        index.close()
        store.close()


def test_streaming_never_admits_a_photo_to_the_cache(tmp_path):
    """Separate from the bounded-cache test, and deliberately so.

    There the library is bigger than the cache, so the scan-resistance guard
    would keep the cache empty even if `iter_all` asked for admission. Here
    the cache is bigger than the library, so the ONLY thing keeping it empty
    is `iter_all` declining to cache - and the test fails if that is removed.
    """
    store = _store(tmp_path, _bulk(60))
    index = MemoryIndex.open(store)
    try:
        index._cache_max = 10_000
        assert len(list(index.iter_all())) == 60
        assert len(index._cache) == 0, "streaming admitted photos to the cache"
        assert len(list(index.iter_images())) == 60
        assert len(index._cache) == 0
    finally:
        index.close()
        store.close()


def test_every_spine_aggregate_agrees_with_the_photos_it_summarises(tmp_path):
    """The offers phase now answers "how many?" and "which years?" off the
    spine instead of loading the slice. That is only safe if the two answers
    are the same answer, so this checks EVERY slice of a real index rather
    than one of each.

    It is the test that would have caught an off-by-one in `_years_in`'s
    bisect, which is the only interesting thing that could go wrong: the years
    array is aligned with `_ids` by position, and a rowid that resolved to its
    neighbour's position would give a plausible wrong year.
    """
    photos = []
    for i in range(120):
        photos.append(
            _p(
                f"{i:064x}",
                # Several years, several months, several days, so no index
                # collapses to a single bucket and the bisect has work to do.
                local=datetime(2015 + i % 8, 1 + i % 12, 1 + i % 27, 9, 0),
                people=[f"Person {i % 5}", f"Person {(i + 2) % 5}"],
                albums=[f"Trip {i % 9}"],
            )
        )
    store = _store(tmp_path, photos)
    index = MemoryIndex.open(store)
    try:
        checked = 0
        for year, count in index.year_counts().items():
            assert count == len(index.by_year(year))
            checked += 1
        for month in index.months():
            assert index.month_counts()[month] == len(index.by_month(month))
            assert index.month_years(month) == index.years_present(index.by_month(month))
            checked += 1
        for month, day in index.month_days():
            assert index.month_day_counts()[(month, day)] == len(index.by_month_day(month, day))
            assert index.month_day_years(month, day) == index.years_present(
                index.by_month_day(month, day)
            )
            checked += 1
        for person in index.people_counts():
            assert index.person_years(person) == index.years_present(index.by_person(person))
            checked += 1
        for a, b in index.pair_counts():
            assert index.pair_years(a, b) == index.years_present(index.by_pair(a, b))
            checked += 1
        for album in index.album_counts():
            assert index.album_years(album) == index.years_present(index.by_album(album))
            checked += 1
        assert checked > 60, f"only {checked} slices compared; the fixture is too thin"
    finally:
        index.close()
        store.close()


def test_the_spine_aggregates_cost_no_photographs(tmp_path):
    """The whole point. Counting and year-listing every slice of the library
    must not hydrate a single photograph - that is the difference between an
    offers pass that reads the library and one that does not."""
    store = _store(tmp_path, _bulk(80))
    index = MemoryIndex.open(store)
    try:
        index.year_counts()
        index.month_counts()
        index.month_day_counts()
        index.image_day_counts()
        for month in index.months():
            index.month_years(month)
        for month, day in index.month_days():
            index.month_day_years(month, day)
        for person in index.people_counts():
            index.person_years(person)
        for a, b in index.pair_counts():
            index.pair_years(a, b)
        for album in index.album_counts():
            index.album_years(album)
        assert len(index._cache) == 0, "an aggregate hydrated a photograph"
    finally:
        index.close()
        store.close()


def test_a_years_answer_is_empty_for_a_slice_that_does_not_exist(tmp_path):
    store = _store(tmp_path, _bulk(10))
    index = MemoryIndex.open(store)
    try:
        assert index.month_years(99) == set()
        assert index.month_day_years(99, 99) == set()
        assert index.person_years("Nobody") == set()
        assert index.pair_years("Nobody", "Nobody Else") == set()
        assert index.album_years("No Such Album") == set()
        assert index.year_counts()[1999] == 0
    finally:
        index.close()
        store.close()


def test_the_year_aggregates_honour_an_exclusion(tmp_path):
    """The aggregates return integers, so the leak sweep cannot see through
    them. This is what does.

    The blocked photograph is in a year, month, day, album, person and pair
    that the allowed one is NOT in, so every aggregate would report a year
    that exists only because of a photograph the user asked never to see. A
    year is not a photograph, but "which years does Kashmir span?" answered
    with a year that only an archived photograph reaches is the exclusion
    leaking anyway - and it would put a wrong subtitle on a real memory.
    """
    photos = [
        _p(
            "hidden",
            archived=True,
            local=datetime(2011, 3, 7, 12, 0),
            people=["Paramita", "Abhik Maiti"],
            albums=["Kashmir"],
        ),
        _p(
            "shown",
            local=datetime(2020, 5, 1, 12, 0),
            people=["Paramita", "Abhik Maiti"],
            albums=["Kashmir"],
        ),
    ]
    store = _store(tmp_path, photos)
    index = MemoryIndex.open(store)
    try:
        assert index.album_years("Kashmir") == {2020}
        assert index.person_years("Paramita") == {2020}
        assert index.pair_years("Abhik Maiti", "Paramita") == {2020}
        assert index.month_years(5) == {2020}
        assert index.month_years(3) == set()
        assert index.month_day_years(5, 1) == {2020}
        assert index.month_day_years(3, 7) == set()
        assert index.year_counts() == {2020: 1}
        assert index.month_counts() == {5: 1}
        assert index.month_day_counts() == {(5, 1): 1}
    finally:
        index.close()
        store.close()


def test_the_hidden_photo_of_that_fixture_really_would_show_up(tmp_path):
    """Guards the test above against being vacuous, the way
    `test_the_blocked_photo_really_is_reachable_without_the_guardrail` guards
    the leak sweep. Built without the policy, every one of those aggregates
    reports 2011."""
    photos = [
        _p(
            "hidden",
            archived=True,
            local=datetime(2011, 3, 7, 12, 0),
            people=["Paramita", "Abhik Maiti"],
            albums=["Kashmir"],
        ),
        _p(
            "shown",
            local=datetime(2020, 5, 1, 12, 0),
            people=["Paramita", "Abhik Maiti"],
            albums=["Kashmir"],
        ),
    ]
    store = _store(tmp_path, photos)
    try:
        unguarded = MemoryIndex(list(store.iter_photos()), ExclusionPolicy(), _empty_report())
        assert unguarded.album_years("Kashmir") == {2011, 2020}
        assert unguarded.person_years("Paramita") == {2011, 2020}
        assert unguarded.pair_years("Abhik Maiti", "Paramita") == {2011, 2020}
        assert unguarded.month_years(3) == {2011}
    finally:
        store.close()


def test_a_dateless_photo_does_not_misalign_the_year_array(tmp_path):
    """`_years` is aligned with `_ids` BY POSITION, so every row must append
    to both or every year after the gap is read off its neighbour.

    `deny_reason` rejects a dateless photo, so `MemoryIndex.open` can never
    produce one - but the direct constructor does not apply the policy, and
    that is deliberate and tested elsewhere. So the alignment line is
    reachable, and without it `album_years` quietly returns the wrong years
    for everything after the gap rather than failing.
    """
    photos = [
        _p("dateless", local=None),
        _p("a", local=datetime(2011, 3, 7, 12, 0), albums=["Kashmir"]),
        _p("b", local=datetime(2020, 5, 1, 12, 0), albums=["Kashmir"]),
    ]
    index = MemoryIndex(photos, ExclusionPolicy(), _empty_report())
    assert index.album_years("Kashmir") == {2011, 2020}
    assert index.month_years(3) == {2011}
    assert index.month_years(5) == {2020}
    assert index.year_counts() == {2011: 1, 2020: 1}
