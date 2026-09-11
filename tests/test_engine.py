"""The pipeline.

The tests that matter here assert the pipeline guarantees against EVERY
REGISTERED RECIPE, via the registry, rather than against a hand-picked list -
so a recipe added later cannot opt out of dedup, the cap, or the guardrails.
"""

import collections
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from rekindle.db import PhotoStore
from rekindle.memory import engine
from rekindle.memory.index import MemoryIndex
from rekindle.memory.policy import ExclusionPolicy
from rekindle.memory.recipes import REGISTRY, Offer, Selection, registered
from rekindle.memory.recipes.base import AS_GIVEN, CHRONOLOGICAL
from rekindle.memory.spec import build_fact_sheet
from rekindle.models import Gps, MediaType, Photo, PhotoMeta

ALLOW = frozenset({"Abhik Maiti"})


def _p(h, *, local=None, people=(), albums=(), gps=None, phash=None, sharp=5.0, size=(4000, 3000)):
    local = local or datetime(2020, 5, 1, 12, 0)
    width, height = size
    return Photo(
        file_hash=h,
        paths=[Path(f"/lib/{h}.jpg")],
        media_type=MediaType.IMAGE,
        meta=PhotoMeta(
            taken_at_utc=local.replace(tzinfo=UTC),
            taken_at_local=local,
            people=list(people),
            gps=Gps(*gps) if gps else None,
            width=width,
            height=height,
            sharpness=sharp,
            brightness=110.0,
            phash=phash if phash is not None else int(f"{abs(hash(h)) % (1 << 60)}"),
        ),
        first_seen=datetime(2020, 1, 1, tzinfo=UTC),
        last_seen=datetime(2020, 1, 1, tzinfo=UTC),
        albums=list(albums),
    )


def _rich_library():
    """A library every recipe can find something in."""
    photos = []
    for year in (2018, 2020, 2022, 2024):
        for i in range(12):
            photos.append(
                _p(
                    f"k{year}{i:02d}",
                    local=datetime(year, 10, 20, 9, i * 3),
                    people=["Abhik Maiti", "Paramita"],
                    albums=["Kashmir"],
                    gps=(22.5, 87.25),
                )
            )
        for i in range(12):
            photos.append(
                _p(
                    f"m{year}{i:02d}",
                    local=datetime(year, 10, 5, 9, i * 3),
                    people=["Abhik Maiti"],
                    albums=["Mysore"],
                )
            )
    return photos


def _index(tmp_path, photos, policy=None):
    store = PhotoStore(tmp_path / "db.sqlite")
    store.upsert_many(photos)
    return store, MemoryIndex.open(store, policy)


# --------------------------------------------------------------------------
# pipeline guarantees, for every recipe


@pytest.mark.parametrize("recipe", registered(), ids=lambda r: r.name)
def test_no_recipe_can_exceed_the_cap(recipe, tmp_path):
    """Asserted through the registry so a new recipe cannot opt out."""
    store, index = _index(tmp_path, _rich_library())
    try:
        for offer in recipe.offers(index):
            spec = engine.build(index, offer, max_shots=6)
            if spec is not None:
                assert len(spec.shots) <= 6, f"{recipe.name} returned {len(spec.shots)} shots"
    finally:
        store.close()


@pytest.mark.parametrize("recipe", registered(), ids=lambda r: r.name)
def test_no_recipe_can_surface_a_blocked_photo(recipe, tmp_path):
    """The guardrail is structural - a recipe has no way to reach the store -
    but this asserts the end-to-end consequence rather than the mechanism."""
    photos = _rich_library()
    blocked = _p(
        "BLOCKED",
        local=datetime(2020, 10, 20, 9, 0),
        people=["Abhik Maiti", "Paramita"],
        albums=["Kashmir"],
        gps=(22.5, 87.25),
    )
    blocked.meta.archived = True
    store, index = _index(tmp_path, [*photos, blocked])
    try:
        for offer in recipe.offers(index):
            spec = engine.build(index, offer)
            if spec is not None:
                assert all(s.file_hash != "BLOCKED" for s in spec.shots)
    finally:
        store.close()


@pytest.mark.parametrize("recipe", registered(), ids=lambda r: r.name)
def test_no_recipe_can_emit_a_duplicate_shot(recipe, tmp_path):
    store, index = _index(tmp_path, _rich_library())
    try:
        for offer in recipe.offers(index):
            spec = engine.build(index, offer)
            if spec is not None:
                hashes = [s.file_hash for s in spec.shots]
                assert len(hashes) == len(set(hashes))
    finally:
        store.close()


@pytest.mark.parametrize("recipe", registered(), ids=lambda r: r.name)
def test_every_spec_round_trips_and_is_reproducible(recipe, tmp_path):
    from rekindle.memory.spec import MemorySpec

    store, index = _index(tmp_path, _rich_library())
    try:
        for offer in recipe.offers(index):
            spec = engine.build(index, offer)
            if spec is not None:
                assert MemorySpec.loads(spec.dumps()) == spec
                assert engine.build(index, offer).dumps() == spec.dumps()
    finally:
        store.close()


# --------------------------------------------------------------------------
# dedup and the cap interact in the right order


def test_dedup_runs_BEFORE_the_cap(tmp_path):
    """Otherwise the cap is filled with 24 slots of which six are
    near-duplicates, instead of 24 distinct photos."""
    photos = []
    # 10 near-identical frames 2 seconds apart, then 10 distinct ones.
    for i in range(10):
        photos.append(_p(f"dup{i}", local=datetime(2020, 5, 1, 9, 0, i * 2), albums=["A"], phash=0))
    for i in range(10):
        photos.append(
            _p(
                f"uniq{i}",
                local=datetime(2020, 5, 2, 9, 0) + timedelta(minutes=i * 10),
                albums=["A"],
                phash=(i + 1) * 0x1111111111111111 % (1 << 63),
            )
        )
    store, index = _index(tmp_path, photos)
    try:
        offer = REGISTRY["album_story"].offers(index)[0]
        spec = engine.build(index, offer, max_shots=11)
        # The 10 duplicates collapse to 1, so 11 slots hold 1 + 10 distinct.
        assert len(spec.shots) == 11
        assert sum(1 for s in spec.shots if s.file_hash.startswith("dup")) == 1
    finally:
        store.close()


def test_a_memory_below_the_floor_is_skipped_and_counted(tmp_path):
    photos = [_p(f"a{i}", local=datetime(2020, 5, i + 1), albums=["Tiny"]) for i in range(3)]
    store, index = _index(tmp_path, photos)
    try:
        report = engine.BuildReport(offered=1)
        offer = REGISTRY["album_story"].offers(index)[0]
        assert engine.build(index, offer, min_shots=10, report=report) is None
        assert report.skipped == {engine.SKIP_TOO_FEW: 1}
    finally:
        store.close()


def test_chronological_order_is_restored_after_the_cap(tmp_path):
    """Ranking picks the best 24 of 500; the memory still plays in order."""
    photos = [
        _p(
            f"a{i:02d}",
            local=datetime(2020, 5, 1) + timedelta(days=i),
            albums=["A"],
            sharp=float(i),
        )
        for i in range(30)
    ]
    store, index = _index(tmp_path, photos)
    try:
        offer = REGISTRY["album_story"].offers(index)[0]
        spec = engine.build(index, offer, max_shots=10)
        stamps = [s.taken_at_local for s in spec.shots]
        assert stamps == sorted(stamps)
    finally:
        store.close()


def test_as_given_ordering_is_not_re_sorted(tmp_path):
    """then_and_now juxtaposes deliberately; re-sorting would destroy it."""
    photos = [_p(f"a{i}", local=datetime(2015 + i, 5, 1), people=["Avyan"]) for i in range(9)]
    store, index = _index(tmp_path, photos)
    try:
        offer = next(o for o in REGISTRY["then_and_now"].offers(index) if o.key == "person:Avyan")
        spec = engine.build(index, offer)
        assert len(spec.shots) == 2
        assert spec.shots[0].caption.startswith("Then")
        assert spec.shots[1].caption.startswith("Now")
    finally:
        store.close()


# --------------------------------------------------------------------------
# ranking


def test_ranking_is_deterministic_under_input_shuffling(tmp_path):
    import random

    photos = [
        _p(f"a{i:02d}", local=datetime(2020, 5, 1) + timedelta(days=i), albums=["A"])
        for i in range(30)
    ]
    store, index = _index(tmp_path, photos)
    try:
        offer = REGISTRY["album_story"].offers(index)[0]
        baseline = engine.build(index, offer, max_shots=10).dumps()
    finally:
        store.close()

    shuffled = photos[:]
    random.Random(3).shuffle(shuffled)
    store, index = _index(tmp_path / "b", shuffled)
    try:
        offer = REGISTRY["album_story"].offers(index)[0]
        assert engine.build(index, offer, max_shots=10).dumps() == baseline
    finally:
        store.close()


def test_the_score_is_a_documented_sum():
    plain = _p("plain")
    assert engine.score(plain, 0.0) == 0.0
    tagged = _p("t", people=["Amy"])
    assert engine.score(tagged, 0.0) == 2.0
    located = _p("g", gps=(1.0, 2.0))
    assert engine.score(located, 0.0) == 1.0
    assert engine.score(plain, 0.75) == 0.75


def test_a_photo_with_people_outranks_one_without_WITHIN_A_BUCKET(tmp_path):
    """People are what makes a memory a memory.

    Both groups share a single day, so they land in ONE stratum and ranking
    alone decides. An earlier version of this test put them in different
    months and asserted that the tagged ones swept every slot - which is
    exactly the clustering the stratification fix removed, so it started
    failing when the fix landed. Ranking now decides WITHIN a period; the
    spread across periods is decided first, and `test_spread_beats_score...`
    below pins that half.
    """
    photos = [_p(f"a{i}", local=datetime(2020, 5, 1, 9, i), albums=["A"]) for i in range(5)]
    photos += [
        _p(f"p{i}", local=datetime(2020, 5, 1, 14, i), albums=["A"], people=["Amy"])
        for i in range(5)
    ]
    store, index = _index(tmp_path, photos)
    try:
        offer = REGISTRY["album_story"].offers(index)[0]
        spec = engine.build(index, offer, max_shots=5)
        assert all(s.file_hash.startswith("p") for s in spec.shots)
    finally:
        store.close()


def test_spread_beats_score_across_buckets(tmp_path):
    """The deliberate behaviour change. A weak period still gets a slot ahead
    of a second slot for the strongest period - otherwise the highest-scoring
    run sweeps the memory, which is what it used to do."""
    photos = [
        _p(f"weak{i}", local=datetime(2020, 5, i + 1), albums=["A"], sharp=1.9) for i in range(2)
    ]
    photos += [
        _p(f"strong{i}", local=datetime(2021, 6, i + 1), albums=["A"], people=["Amy"], sharp=9.0)
        for i in range(20)
    ]
    store, index = _index(tmp_path, photos)
    try:
        offer = REGISTRY["album_story"].offers(index)[0]
        spec = engine.build(index, offer, max_shots=6)
        years = collections.Counter(s.taken_at_local[:4] for s in spec.shots)
        assert years["2020"] >= 1, "the weak period was swept away entirely"
        assert years["2021"] > years["2020"], "the rich period lost its larger share"
    finally:
        store.close()


# --------------------------------------------------------------------------
# public-safe lift


def test_a_memory_is_public_safe_only_when_every_shot_is(tmp_path):
    solo = [
        _p(f"s{i}", local=datetime(2020, 5, i + 1), albums=["A"], people=["Abhik Maiti"])
        for i in range(4)
    ]
    store, index = _index(tmp_path, solo, ExclusionPolicy(public_safe_allow=ALLOW))
    try:
        offer = REGISTRY["album_story"].offers(index)[0]
        assert engine.build(index, offer).public_safe is True
    finally:
        store.close()

    mixed = [
        *solo,
        _p("pair", local=datetime(2020, 6, 1), albums=["A"], people=["Abhik Maiti", "X"]),
    ]
    store, index = _index(tmp_path / "b", mixed, ExclusionPolicy(public_safe_allow=ALLOW))
    try:
        offer = REGISTRY["album_story"].offers(index)[0]
        assert engine.build(index, offer).public_safe is False
    finally:
        store.close()


def test_an_untagged_shot_makes_a_memory_unpublishable(tmp_path):
    """43.7% of this library is untagged, so this is the common case - and the
    reason public-safe memories are rare."""
    photos = [
        _p(f"s{i}", local=datetime(2020, 5, i + 1), albums=["A"], people=["Abhik Maiti"])
        for i in range(4)
    ]
    photos.append(_p("untagged", local=datetime(2020, 6, 1), albums=["A"]))
    store, index = _index(tmp_path, photos, ExclusionPolicy(public_safe_allow=ALLOW))
    try:
        offer = REGISTRY["album_story"].offers(index)[0]
        assert engine.build(index, offer).public_safe is False
    finally:
        store.close()


# --------------------------------------------------------------------------
# build_all: dismissal, cooldown, overlap


def test_build_all_refuses_a_dismissed_memory(tmp_path):
    store, index = _index(tmp_path, _rich_library())
    try:
        offers = engine.all_offers(index)
        dismissed = frozenset({offers[0].memory_id})
        specs, report = engine.build_all(index, offers, dismissed=dismissed)
        assert all(s.recipe + ":" + s.key != offers[0].memory_id for s in specs)
        assert report.skipped[engine.SKIP_DISMISSED] == 1
    finally:
        store.close()


def test_build_all_refuses_a_cooling_memory(tmp_path):
    store, index = _index(tmp_path, _rich_library())
    try:
        offers = engine.all_offers(index)
        specs, report = engine.build_all(index, offers, cooling=frozenset({offers[0].memory_id}))
        assert report.skipped[engine.SKIP_COOLDOWN] == 1
    finally:
        store.close()


def test_build_all_refuses_a_redundant_memory(tmp_path):
    """Two albums holding the SAME photos - the `Leh Ladakh` / `ladakh` shape.
    They stay separate offers (no auto-merge) but only one is built."""
    photos = [
        _p(f"a{i}", local=datetime(2020, 5, i + 1), albums=["Trip", "trip copy"]) for i in range(8)
    ]
    store, index = _index(tmp_path, photos)
    try:
        offers = REGISTRY["album_story"].offers(index)
        assert len(offers) == 2
        specs, report = engine.build_all(index, offers)
        assert len(specs) == 1
        assert report.skipped[engine.SKIP_OVERLAP] == 1
        assert report.accounted
    finally:
        store.close()


def test_two_genuinely_different_memories_both_survive(tmp_path):
    photos = [_p(f"a{i}", local=datetime(2020, 5, i + 1), albums=["A"]) for i in range(8)]
    photos += [_p(f"b{i}", local=datetime(2021, 5, i + 1), albums=["B"]) for i in range(8)]
    store, index = _index(tmp_path, photos)
    try:
        specs, _ = engine.build_all(index, REGISTRY["album_story"].offers(index))
        assert {s.key for s in specs} == {"A", "B"}
    finally:
        store.close()


def test_build_all_accounting_holds(tmp_path):
    store, index = _index(tmp_path, _rich_library())
    try:
        offers = engine.all_offers(index)
        _, report = engine.build_all(index, offers)
        assert report.accounted, (report.offered, report.built, report.skipped)
    finally:
        store.close()


def test_build_all_accounting_holds_under_a_limit(tmp_path):
    """A limit stops the loop, so the un-reached offers are not "skipped" -
    the offered count is corrected rather than a bogus reason invented."""
    store, index = _index(tmp_path, _rich_library())
    try:
        offers = engine.all_offers(index)
        specs, report = engine.build_all(index, offers, limit=2)
        assert len(specs) == 2
        assert report.accounted
    finally:
        store.close()


def test_build_all_is_deterministic(tmp_path):
    store, index = _index(tmp_path, _rich_library())
    try:
        offers = engine.all_offers(index)
        first = [s.dumps() for s in engine.build_all(index, offers)[0]]
        second = [s.dumps() for s in engine.build_all(index, offers)[0]]
        assert first == second
    finally:
        store.close()


# --------------------------------------------------------------------------
# today


def test_offers_for_today_finds_the_anniversary(tmp_path):
    store, index = _index(tmp_path, _rich_library())
    try:
        offers = engine.offers_for_today(index, date(2026, 10, 20))
        assert [o.key for o in offers] == ["10-20"]
        assert offers[0].recipe == "on_this_day"
    finally:
        store.close()


def test_offers_for_today_falls_back_to_the_month(tmp_path):
    """The reason on_this_month exists as a separate recipe."""
    store, index = _index(tmp_path, _rich_library())
    try:
        offers = engine.offers_for_today(index, date(2026, 10, 7))
        assert [o.recipe for o in offers] == ["on_this_month"]
    finally:
        store.close()


def test_offers_for_today_returns_nothing_on_a_bare_date(tmp_path):
    store, index = _index(tmp_path, _rich_library())
    try:
        assert engine.offers_for_today(index, date(2026, 3, 3)) == []
    finally:
        store.close()


def test_offers_for_today_accepts_a_datetime(tmp_path):
    store, index = _index(tmp_path, _rich_library())
    try:
        assert engine.offers_for_today(index, datetime(2026, 10, 20, 9, 0)) != []
    finally:
        store.close()


# --------------------------------------------------------------------------
# a recipe cannot smuggle anything past the pipeline


def test_a_rogue_recipe_still_cannot_exceed_the_cap_or_keep_duplicates(tmp_path):
    """A recipe that ignores every convention - returns 200 photos, duplicates
    them, and asks for no particular order - still produces a legal memory,
    because the engine owns all of that."""

    class Rogue:
        name = "rogue"
        title = "Rogue"

        def offers(self, index):
            return [Offer(recipe="rogue", key="all", title="Everything")]

        def select(self, index, offer):
            photos = index.all() * 3
            return Selection(
                photos=photos,
                facts=build_fact_sheet(photos, title="Everything", recipe="rogue"),
                ordering=CHRONOLOGICAL,
            )

    REGISTRY["rogue"] = Rogue()
    try:
        store, index = _index(tmp_path, _rich_library())
        try:
            offer = REGISTRY["rogue"].offers(index)[0]
            spec = engine.build(index, offer, max_shots=5)
            assert len(spec.shots) == 5
            assert len({s.file_hash for s in spec.shots}) == 5
        finally:
            store.close()
    finally:
        del REGISTRY["rogue"]


def test_the_rogue_recipe_test_is_not_vacuous(tmp_path):
    """Guards the test above: the rogue really does return more than the cap
    and really does duplicate."""

    store, index = _index(tmp_path, _rich_library())
    try:
        assert len(index.all()) > 5
    finally:
        store.close()


def test_as_given_ordering_survives_the_cap(tmp_path):
    """A recipe asking for AS_GIVEN keeps its own sequence among survivors."""

    class Reversed:
        name = "reversed"
        title = "Reversed"

        def offers(self, index):
            return [Offer(recipe="reversed", key="all", title="Backwards")]

        def select(self, index, offer):
            photos = sorted(index.all(), key=lambda p: p.meta.taken_at_utc, reverse=True)
            return Selection(
                photos=photos,
                facts=build_fact_sheet(photos, title="Backwards", recipe="reversed"),
                ordering=AS_GIVEN,
            )

    REGISTRY["reversed"] = Reversed()
    try:
        store, index = _index(tmp_path, _rich_library())
        try:
            spec = engine.build(index, REGISTRY["reversed"].offers(index)[0], max_shots=6)
            stamps = [s.taken_at_local for s in spec.shots]
            assert stamps == sorted(stamps, reverse=True)
        finally:
            store.close()
    finally:
        del REGISTRY["reversed"]


# --------------------------------------------------------------------------
# gaps found by mutation: each of these covers a line that could be deleted
# with the suite green until it was added.


def test_the_re_order_is_not_a_no_op(tmp_path):
    """The earlier ordering test used photos whose sharpness rose with their
    date, so ranked order and chronological order were already the same and
    deleting the re-sort changed nothing.

    Here sharpness FALLS as the date rises, so ranking returns the memory
    backwards and only the re-sort can put it right.
    """
    photos = [
        _p(
            f"a{i:02d}",
            local=datetime(2020, 5, 1) + timedelta(days=i),
            albums=["A"],
            sharp=float(30 - i),
        )
        for i in range(30)
    ]
    store, index = _index(tmp_path, photos)
    try:
        offer = REGISTRY["album_story"].offers(index)[0]
        spec = engine.build(index, offer, max_shots=10)
        stamps = [s.taken_at_local for s in spec.shots]
        assert stamps == sorted(stamps)
        # ...and the chosen ten really are the SHARPEST, i.e. the earliest,
        # which is what makes ranked order differ from chronological order.
        assert spec.shots[0].file_hash == "a00"
    finally:
        store.close()


def test_composition_guardrails_are_applied_by_the_engine(tmp_path):
    """A photo the composition rules reject must not reach a memory even
    though the recipe happily returned it. Deleting the compose() call left
    the suite green until this existed."""
    good = [_p(f"g{i}", local=datetime(2020, 5, i + 1), albums=["A"]) for i in range(5)]
    tiny = _p("TINY", local=datetime(2020, 5, 6), albums=["A"], size=(320, 240))
    pano = _p("PANO", local=datetime(2020, 5, 7), albums=["A"], size=(8874, 943))
    store, index = _index(tmp_path, [*good, tiny, pano])
    try:
        offer = REGISTRY["album_story"].offers(index)[0]
        spec = engine.build(index, offer)
        hashes = {s.file_hash for s in spec.shots}
        assert "TINY" not in hashes
        assert "PANO" not in hashes
        assert len(hashes) == 5
    finally:
        store.close()


def test_composition_drops_are_reported_by_the_engine(tmp_path):
    """Counted with a reason and surfaced - never silently dropped."""
    photos = [_p(f"g{i}", local=datetime(2020, 5, i + 1), albums=["A"]) for i in range(5)]
    photos.append(_p("TINY", local=datetime(2020, 5, 6), albums=["A"], size=(320, 240)))
    store, index = _index(tmp_path, photos)
    try:
        report = engine.BuildReport(offered=1)
        engine.build(index, REGISTRY["album_story"].offers(index)[0], report=report)
        assert report.composition.dropped == {"too_small": 1}
        assert report.composition.examples["too_small"] == "TINY.jpg"
    finally:
        store.close()


def test_the_ranking_tiebreak_decides_between_identical_photos(tmp_path):
    """Two photos with the same score AND the same timestamp. Without the
    file_hash tiebreak the winner depends on the order SQLite hands rows
    back, and the byte-for-byte promise silently rests on the database's
    physical layout."""
    same = datetime(2020, 5, 1, 12, 0)
    photos = [
        _p("zzz", local=same, albums=["A"], sharp=5.0),
        _p("aaa", local=same, albums=["A"], sharp=5.0),
        _p("mmm", local=same, albums=["A"], sharp=5.0),
    ]
    store, index = _index(tmp_path, photos)
    try:
        offer = REGISTRY["album_story"].offers(index)[0]
        spec = engine.build(index, offer, max_shots=2, min_shots=2)
        assert [s.file_hash for s in spec.shots] == ["aaa", "mmm"]
    finally:
        store.close()


def test_ranked_breaks_a_total_tie_on_file_hash(tmp_path):
    """Tested directly on `_ranked` rather than through `build`.

    Every shipped recipe sorts with `chronological()` first, which already
    applies a file_hash tiebreak, so by the time `_ranked` sees the photos the
    tie is usually pre-broken and removing its own tiebreak changes nothing
    observable. That makes it defence in depth for a recipe using AS_GIVEN
    with an arbitrary order - and defence in depth still needs a test, or it
    is a line anyone can delete with the suite green.
    """
    same = datetime(2020, 5, 1, 12, 0)
    photos = [_p("zzz", local=same), _p("aaa", local=same), _p("mmm", local=same)]
    assert [p.file_hash for p in engine._ranked(photos)] == ["aaa", "mmm", "zzz"]


def test_every_score_term_is_reachable_and_weighted_as_documented():
    """The score is a fixed, documented sum so that the reason a photo did or
    did not appear can be reconstructed by hand. Each term is asserted
    separately - a coverage report showed three of them had no test at all,
    which means three lines anyone could delete with the suite green."""
    favourite = _p("f")
    favourite.meta.favorite = True
    assert engine.score(favourite, 0.0) == 3.0

    described = _p("d")
    described.meta.description = "a caption"
    assert engine.score(described, 0.0) == 2.0

    exact = _p("x")
    exact.sidecar_match = "exact"
    assert engine.score(exact, 0.0) == 1.0

    # ...and they add up rather than overriding one another.
    everything = _p("e", people=["Amy"], gps=(1.0, 2.0))
    everything.meta.favorite = True
    everything.meta.description = "c"
    everything.sidecar_match = "exact"
    assert engine.score(everything, 0.5) == 3.0 + 2.0 + 2.0 + 1.0 + 1.0 + 0.5


def test_an_unmeasured_photo_ranks_mid_pack_not_last():
    """Penalising an unfingerprinted photo would make an unfingerprinted
    library rank by metadata alone, which is a different product."""
    measured = [_p(f"m{i}", sharp=float(i)) for i in range(10)]
    unmeasured = _p("u", sharp=None)
    ranked = engine._ranked([*measured, unmeasured])
    position = [p.file_hash for p in ranked].index("u")
    assert 0 < position < len(ranked) - 1


def test_a_registry_with_no_sharpness_at_all_still_ranks():
    """Every photo unfingerprinted: the percentile has no sample to draw on
    and must not divide by zero."""
    photos = [_p(f"a{i}", sharp=None) for i in range(3)]
    assert len(engine._ranked(photos)) == 3


# --------------------------------------------------------------------------
# stratified selection
#
# The defect these pin: selection took top-N by quality score with no temporal
# constraint, so shots clustered wherever the strongest-scoring run happened
# to sit. Measured on the real library BEFORE the fix: 16 of 37 rendered
# memories were confined to a single year and 10 to a single month -
# `on_this_day:12-22` showed 24 shots all from 2019 while 2020 and 2022 had
# photos available, and three `year_in_review` memories showed one month each.


def _dominant_year_library(album="A"):
    """The real `on_this_day:12-22` shape: one year with far more photos than
    the others, plus two thin years that must still be represented.

    The dominant year is also given GPS, which is worth +1 in the score - on
    the real library that is exactly what let one year sweep every slot.
    """
    photos = []
    for i in range(60):
        photos.append(
            _p(
                f"big{i:02d}",
                local=datetime(2019, 12, 22, 9, 0) + timedelta(minutes=i * 7),
                albums=[album],
                people=["Amy"],
                gps=(22.5, 87.25),
                sharp=9.0,
            )
        )
    for i in range(3):
        photos.append(
            _p(f"mid{i}", local=datetime(2020, 12, 22, 10, i * 11), albums=[album], sharp=2.0)
        )
    for i in range(2):
        photos.append(
            _p(f"thin{i}", local=datetime(2022, 12, 22, 11, i * 13), albums=[album], sharp=1.9)
        )
    return photos


def test_on_this_day_is_not_confined_to_one_year(tmp_path):
    """An "on this day" showing a single year defeats the entire concept -
    the point of the recipe is the SAME DATE ACROSS YEARS."""
    store, index = _index(tmp_path, _dominant_year_library())
    try:
        offer = REGISTRY["on_this_day"].offers(index)[0]
        spec = engine.build(index, offer, max_shots=12)
        years = {s.taken_at_local[:4] for s in spec.shots}
        assert years == {"2019", "2020", "2022"}, f"only {years} represented"
    finally:
        store.close()


def test_every_year_with_photos_gets_at_least_one_slot(tmp_path):
    """The floor: a year that has photos on that date must not be silently
    absent while another contributes every shot."""
    store, index = _index(tmp_path, _dominant_year_library())
    try:
        offer = REGISTRY["on_this_day"].offers(index)[0]
        spec = engine.build(index, offer, max_shots=12)
        by_year = collections.Counter(s.taken_at_local[:4] for s in spec.shots)
        # Explicit per-year assertions. `min(by_year.values()) >= 1` looks
        # like the same check and is VACUOUS: an absent year is simply not a
        # key, so the minimum over the years that ARE present is always >= 1
        # and the test passes on the very code it is meant to catch.
        for year in ("2019", "2020", "2022"):
            assert by_year[year] >= 1, f"{year} got no slot at all: {dict(by_year)}"
        # ...and the rich year still gets the largest share.
        assert by_year.most_common(1)[0][0] == "2019"
    finally:
        store.close()


def test_a_richer_bucket_still_gets_more_slots(tmp_path):
    """Proportional-with-a-floor, not an equal split: a bucket with two weak
    photos must not get the same weight as one with sixty good ones."""
    store, index = _index(tmp_path, _dominant_year_library())
    try:
        offer = REGISTRY["on_this_day"].offers(index)[0]
        spec = engine.build(index, offer, max_shots=12)
        by_year = collections.Counter(s.taken_at_local[:4] for s in spec.shots)
        # The thin buckets must be PRESENT before "more than" means anything -
        # otherwise `by_year["2020"]` is 0 by absence and the comparison holds
        # against the unstratified code this test exists to reject.
        assert by_year["2020"] >= 1 and by_year["2022"] >= 1
        assert by_year["2019"] > by_year["2020"]
        assert by_year["2019"] >= 6
    finally:
        store.close()


def test_year_in_review_spreads_across_MONTHS(tmp_path):
    """A "year in review" showing one month is not a year in review. Three of
    them did exactly that on the real library."""
    photos = []
    for month in (1, 4, 7, 11):
        count = 40 if month == 7 else 4
        for i in range(count):
            photos.append(
                _p(
                    f"m{month:02d}{i:02d}",
                    local=datetime(2021, month, 1, 9, 0) + timedelta(hours=i * 5),
                    people=["Amy"] if month == 7 else [],
                    sharp=9.0 if month == 7 else 2.0,
                )
            )
    store, index = _index(tmp_path, photos)
    try:
        offer = REGISTRY["year_in_review"].offers(index)[0]
        spec = engine.build(index, offer, max_shots=12)
        months = {s.taken_at_local[5:7] for s in spec.shots}
        assert months == {"01", "04", "07", "11"}, f"only {months} represented"
    finally:
        store.close()


def test_person_years_spreads_across_the_persons_span(tmp_path):
    photos = []
    for year in (2015, 2018, 2022, 2026):
        count = 30 if year == 2018 else 3
        for i in range(count):
            photos.append(
                _p(
                    f"p{year}{i:02d}",
                    local=datetime(year, 6, 1, 9, 0) + timedelta(days=i),
                    people=["Avyan"],
                    sharp=9.0 if year == 2018 else 2.0,
                )
            )
    store, index = _index(tmp_path, photos)
    try:
        offer = next(o for o in REGISTRY["person_years"].offers(index) if o.key == "Avyan")
        spec = engine.build(index, offer, max_shots=12)
        years = {s.taken_at_local[:4] for s in spec.shots}
        assert years == {"2015", "2018", "2022", "2026"}, f"only {years} represented"
    finally:
        store.close()


def test_an_album_spanning_years_is_represented_across_them(tmp_path):
    """The `Avyan` album spans that child's life and the memory showed one
    August."""
    photos = []
    for year, month in [(2024, 3), (2024, 9), (2025, 2), (2025, 8)]:
        count = 40 if (year, month) == (2025, 8) else 4
        for i in range(count):
            photos.append(
                _p(
                    f"a{year}{month:02d}{i:02d}",
                    local=datetime(year, month, 1, 9, 0) + timedelta(hours=i * 3),
                    albums=["Avyan"],
                    people=["Avyan"] if (year, month) == (2025, 8) else [],
                    sharp=9.0 if (year, month) == (2025, 8) else 2.0,
                )
            )
    store, index = _index(tmp_path, photos)
    try:
        offer = REGISTRY["album_story"].offers(index)[0]
        spec = engine.build(index, offer, max_shots=12)
        periods = {s.taken_at_local[:7] for s in spec.shots}
        assert len(periods) >= 4, f"only {periods} represented"
    finally:
        store.close()


def test_then_and_now_is_NOT_stratified(tmp_path):
    """The opposite case: it wants the extremes, not the spread. Stratifying
    it would defeat the recipe."""
    photos = [_p(f"a{i}", local=datetime(2015 + i, 5, 1), people=["Avyan"]) for i in range(9)]
    store, index = _index(tmp_path, photos)
    try:
        offer = next(o for o in REGISTRY["then_and_now"].offers(index) if o.key == "person:Avyan")
        spec = engine.build(index, offer)
        assert len(spec.shots) == 2
        assert spec.shots[0].taken_at_local[:4] == "2015"
        assert spec.shots[1].taken_at_local[:4] == "2023"
    finally:
        store.close()
