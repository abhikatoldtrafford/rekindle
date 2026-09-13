"""The eight recipes.

Fixture shapes come from the reference library: `Photos from YYYY` auto
albums, `Untitled(1)`, a title beginning with a comma, and the two Kashmir /
ladakh spellings that are deliberately NOT merged.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rekindle.db import PhotoStore
from rekindle.memory.index import MemoryIndex
from rekindle.memory.recipes import REGISTRY, registered
from rekindle.memory.recipes.base import AS_GIVEN
from rekindle.models import Gps, MediaType, Photo, PhotoMeta


def _p(h, *, local, people=(), albums=(), gps=None) -> Photo:
    return Photo(
        file_hash=h,
        paths=[Path(f"/lib/{h}.jpg")],
        media_type=MediaType.IMAGE,
        meta=PhotoMeta(
            taken_at_utc=local.replace(tzinfo=UTC),
            taken_at_local=local,
            people=list(people),
            gps=Gps(*gps) if gps else None,
            width=4000,
            height=3000,
            sharpness=5.0,
            brightness=110.0,
            phash=int(h.encode().hex()[:8], 16) if h else 0,
        ),
        first_seen=datetime(2020, 1, 1, tzinfo=UTC),
        last_seen=datetime(2020, 1, 1, tzinfo=UTC),
        albums=list(albums),
    )


def _index(tmp_path, photos, policy=None) -> tuple[PhotoStore, MemoryIndex]:
    store = PhotoStore(tmp_path / "db.sqlite")
    store.upsert_many(photos)
    return store, MemoryIndex.open(store, policy)


def _empty_index(tmp_path):
    return _index(tmp_path, [])


def _across_years(prefix, *, month=10, day=20, years=(2020, 2021, 2022), per_year=4, **kw):
    """Photos on one calendar date across several years - the anniversary
    shape this library is full of (12-25 spans 13 years, 503 photos)."""
    out = []
    for year in years:
        for i in range(per_year):
            out.append(
                _p(
                    f"{prefix}{year}{i}",
                    local=datetime(year, month, day, 9 + i, 0),
                    **kw,
                )
            )
    return out


# --------------------------------------------------------------------------
# properties every recipe must have


@pytest.mark.parametrize("recipe", registered(), ids=lambda r: r.name)
def test_an_empty_index_yields_no_offers(recipe, tmp_path):
    store, index = _empty_index(tmp_path)
    try:
        assert recipe.offers(index) == []
    finally:
        store.close()


@pytest.mark.parametrize("recipe", registered(), ids=lambda r: r.name)
def test_offers_are_deterministic(recipe, tmp_path):
    photos = _across_years("a", people=["Amy", "Bob"], albums=["Kashmir"], gps=(22.5, 87.25))
    photos += _across_years("b", month=5, day=1, people=["Amy"], albums=["Mysore"])
    store, index = _index(tmp_path, photos)
    try:
        first = [(o.recipe, o.key) for o in recipe.offers(index)]
        second = [(o.recipe, o.key) for o in recipe.offers(index)]
        assert first == second
    finally:
        store.close()


@pytest.mark.parametrize("recipe", registered(), ids=lambda r: r.name)
def test_select_on_a_stale_offer_returns_none_rather_than_raising(recipe, tmp_path):
    """An offer built against a larger library, replayed against a smaller
    one. `rekindle memory --recipe X --key Y` with a stale key is exactly
    this, and it must not traceback."""
    from rekindle.memory.recipes.base import Offer

    store, index = _empty_index(tmp_path)
    try:
        # A key shaped for each recipe, naming a subject that does not exist.
        keys = {
            "album_story": "NoSuchAlbum",
            "on_this_day": "10-20",
            "on_this_month": "10",
            "person_years": "Nobody",
            "pair_years": "Nobody + Nobody Else",
            "then_and_now": "person:Nobody",
            "year_in_review": "1999",
            "place_cluster": "22.50,87.25",
            "recurring_event": "10-a",
        }
        offer = Offer(recipe=recipe.name, key=keys[recipe.name], title="t")
        assert recipe.select(index, offer) is None
    finally:
        store.close()


@pytest.mark.parametrize("recipe", registered(), ids=lambda r: r.name)
def test_every_offer_has_a_nonempty_title_and_a_stable_id(recipe, tmp_path):
    photos = _across_years("a", people=["Amy", "Bob"], albums=["Kashmir"], gps=(22.5, 87.25))
    store, index = _index(tmp_path, photos)
    try:
        for offer in recipe.offers(index):
            assert offer.title.strip(), f"{recipe.name} produced an empty title"
            assert offer.memory_id == f"{recipe.name}:{offer.key}"
    finally:
        store.close()


def test_all_nine_recipes_are_registered():
    assert set(REGISTRY) == {
        "album_story",
        "on_this_day",
        "on_this_month",
        "person_years",
        "pair_years",
        "then_and_now",
        "year_in_review",
        "place_cluster",
        "recurring_event",
    }


# --------------------------------------------------------------------------
# album_story


def test_auto_year_albums_are_never_offered(tmp_path):
    """Every photo is in a `Photos from YYYY` album, so they carry no
    information - and 23 of the 66 albums here are one."""
    photos = [
        _p(f"a{i}", local=datetime(2020, 5, i + 1), albums=["Photos from 2020"]) for i in range(5)
    ]
    store, index = _index(tmp_path, photos)
    try:
        assert REGISTRY["album_story"].offers(index) == []
    finally:
        store.close()


@pytest.mark.parametrize("name", ["Untitled", "Untitled(1)", "Untitled(3)", ", Abhirup, sudipta"])
def test_unpresentable_album_titles_are_never_offered(tmp_path, name):
    """Real metadata, but not a title anyone can read - and inventing a better
    one would be inventing a fact."""
    photos = [_p(f"a{i}", local=datetime(2020, 5, i + 1), albums=[name]) for i in range(5)]
    store, index = _index(tmp_path, photos)
    try:
        assert REGISTRY["album_story"].offers(index) == []
    finally:
        store.close()


def test_the_two_kashmir_spellings_stay_separate(tmp_path):
    """Deliberate: merging them requires asserting a fact no metadata
    contains. `Diwali 25` and `Diwali Kali Puja 22` are DIFFERENT years under
    an equally similar pair of names."""
    photos = [_p(f"a{i}", local=datetime(2018, 8, i + 1), albums=["Leh Ladakh"]) for i in range(4)]
    photos += [_p(f"b{i}", local=datetime(2018, 9, i + 1), albums=["ladakh"]) for i in range(4)]
    store, index = _index(tmp_path, photos)
    try:
        keys = {o.key for o in REGISTRY["album_story"].offers(index)}
        assert keys == {"Leh Ladakh", "ladakh"}
    finally:
        store.close()


def test_a_small_but_real_album_is_still_offered(tmp_path):
    """`Durga Puja 25` has 6 photos and `Puri 25` has 3. The user named the
    former as an expected output, so the floor is 3, not 8."""
    photos = [_p(f"a{i}", local=datetime(2025, 10, i + 1), albums=["Puri 25"]) for i in range(3)]
    store, index = _index(tmp_path, photos)
    try:
        assert [o.key for o in REGISTRY["album_story"].offers(index)] == ["Puri 25"]
    finally:
        store.close()


def test_an_album_below_the_floor_is_not_offered(tmp_path):
    photos = [_p(f"a{i}", local=datetime(2025, 10, i + 1), albums=["Tiny"]) for i in range(2)]
    store, index = _index(tmp_path, photos)
    try:
        assert REGISTRY["album_story"].offers(index) == []
    finally:
        store.close()


# --------------------------------------------------------------------------
# on_this_day / on_this_month


def test_on_this_day_needs_three_years_and_eight_photos(tmp_path):
    two_years = _across_years("a", years=(2020, 2021), per_year=5)
    store, index = _index(tmp_path, two_years)
    try:
        assert REGISTRY["on_this_day"].offers(index) == []
    finally:
        store.close()

    three_years = _across_years("b", years=(2020, 2021, 2022), per_year=3)
    store, index = _index(tmp_path / "two", three_years)
    try:
        assert [o.key for o in REGISTRY["on_this_day"].offers(index)] == ["10-20"]
    finally:
        store.close()


def test_on_this_day_handles_a_leap_day(tmp_path):
    photos = _across_years("l", month=2, day=29, years=(2016, 2020, 2024), per_year=3)
    store, index = _index(tmp_path, photos)
    try:
        assert [o.key for o in REGISTRY["on_this_day"].offers(index)] == ["02-29"]
    finally:
        store.close()


def test_on_this_day_captions_are_relative_to_the_memory_not_the_clock(tmp_path):
    """Otherwise the same index renders differently tomorrow, and the
    byte-for-byte promise is gone."""
    photos = _across_years("a", years=(2020, 2021, 2022), per_year=3)
    store, index = _index(tmp_path, photos)
    try:
        recipe = REGISTRY["on_this_day"]
        offer = recipe.offers(index)[0]
        selection = recipe.select(index, offer)
        texts = set(selection.captions.values())
        assert "2 years ago today" in texts
        assert "today" in texts
    finally:
        store.close()


def test_on_this_month_needs_thirty_photos(tmp_path):
    photos = _across_years("a", years=(2020, 2021, 2022), per_year=4)
    store, index = _index(tmp_path, photos)
    try:
        assert REGISTRY["on_this_month"].offers(index) == []
    finally:
        store.close()

    many = []
    for year in (2020, 2021, 2022):
        many += [_p(f"m{year}{i}", local=datetime(year, 10, (i % 28) + 1, 9)) for i in range(12)]
    store, index = _index(tmp_path / "b", many)
    try:
        assert [o.key for o in REGISTRY["on_this_month"].offers(index)] == ["10"]
    finally:
        store.close()


# --------------------------------------------------------------------------
# person_years / pair_years


def test_person_years_needs_eight_photos_across_three_years(tmp_path):
    photos = _across_years("a", years=(2020, 2021, 2022), per_year=3, people=["Avyan"])
    store, index = _index(tmp_path, photos)
    try:
        offers = REGISTRY["person_years"].offers(index)
        assert [o.key for o in offers] == ["Avyan"]
        assert offers[0].title == "Avyan over the years"
    finally:
        store.close()


def test_a_person_in_one_year_only_is_not_offered(tmp_path):
    photos = [_p(f"a{i}", local=datetime(2020, 5, i + 1), people=["Solo"]) for i in range(10)]
    store, index = _index(tmp_path, photos)
    try:
        assert REGISTRY["person_years"].offers(index) == []
    finally:
        store.close()


def test_a_pair_is_offered_exactly_once_whatever_the_tag_order(tmp_path):
    """Otherwise "A and B" and "B and A" are two memories of the same thing -
    and dismissing one would leave the other."""
    photos = _across_years("a", years=(2020, 2021, 2022), per_year=3, people=["Bob", "Amy"])
    photos += _across_years(
        "b", month=5, years=(2020, 2021, 2022), per_year=1, people=["Amy", "Bob"]
    )
    store, index = _index(tmp_path, photos)
    try:
        offers = REGISTRY["pair_years"].offers(index)
        assert [o.key for o in offers] == ["Amy + Bob"]
        assert offers[0].title == "Amy and Bob"
    finally:
        store.close()


def test_a_pair_key_round_trips_through_select(tmp_path):
    photos = _across_years("a", years=(2020, 2021, 2022), per_year=3, people=["Bob", "Amy"])
    store, index = _index(tmp_path, photos)
    try:
        recipe = REGISTRY["pair_years"]
        offer = recipe.offers(index)[0]
        assert recipe.select(index, offer) is not None
    finally:
        store.close()


# --------------------------------------------------------------------------
# then_and_now


def test_then_and_now_is_exactly_two_shots_in_the_given_order(tmp_path):
    photos = [_p(f"a{i}", local=datetime(2015 + i, 5, 1), people=["Avyan"]) for i in range(9)]
    store, index = _index(tmp_path, photos)
    try:
        recipe = REGISTRY["then_and_now"]
        offer = next(o for o in recipe.offers(index) if o.key == "person:Avyan")
        selection = recipe.select(index, offer)
        assert len(selection.photos) == 2
        assert selection.ordering == AS_GIVEN
        assert selection.photos[0].meta.taken_at_local.year == 2015
        assert selection.photos[1].meta.taken_at_local.year == 2023
    finally:
        store.close()


def test_then_and_now_refuses_a_subject_confined_to_one_year(tmp_path):
    """A "then and now" spanning three weeks says nothing."""
    photos = [_p(f"a{i}", local=datetime(2020, 5, i + 1), people=["Avyan"]) for i in range(10)]
    store, index = _index(tmp_path, photos)
    try:
        assert [o for o in REGISTRY["then_and_now"].offers(index) if o.key == "person:Avyan"] == []
    finally:
        store.close()


def test_then_and_now_offers_albums_as_well_as_people(tmp_path):
    photos = [_p(f"a{i}", local=datetime(2015 + i, 5, 1), albums=["Kashmir"]) for i in range(9)]
    store, index = _index(tmp_path, photos)
    try:
        keys = {o.key for o in REGISTRY["then_and_now"].offers(index)}
        assert "album:Kashmir" in keys
    finally:
        store.close()


# --------------------------------------------------------------------------
# year_in_review


def test_year_in_review_needs_thirty_photos(tmp_path):
    """2008 has 72 photos and 2000 has 2; the floor admits the former and
    excludes the latter."""
    photos = [_p(f"a{i}", local=datetime(2020, 5, 1) + timedelta(days=i)) for i in range(30)]
    photos += [_p("thin", local=datetime(2003, 5, 1))]
    store, index = _index(tmp_path, photos)
    try:
        assert [o.key for o in REGISTRY["year_in_review"].offers(index)] == ["2020"]
    finally:
        store.close()


def test_year_in_review_is_offered_newest_first(tmp_path):
    photos = []
    for year in (2018, 2020, 2022):
        photos += [
            _p(f"a{year}{i}", local=datetime(year, 5, 1) + timedelta(days=i)) for i in range(30)
        ]
    store, index = _index(tmp_path, photos)
    try:
        assert [o.key for o in REGISTRY["year_in_review"].offers(index)] == ["2022", "2020", "2018"]
    finally:
        store.close()


# --------------------------------------------------------------------------
# place_cluster


def test_place_cluster_needs_twenty_photos_and_two_visits(tmp_path):
    first = [_p(f"a{i}", local=datetime(2020, 5, 1, 9, i), gps=(22.5, 87.25)) for i in range(10)]
    second = [_p(f"b{i}", local=datetime(2021, 5, 1, 9, i), gps=(22.5, 87.26)) for i in range(10)]
    store, index = _index(tmp_path, first + second)
    try:
        offers = REGISTRY["place_cluster"].offers(index)
        assert [o.key for o in offers] == ["22.50,87.25"]
        assert offers[0].subtitle.startswith("2 visits")
    finally:
        store.close()


def test_a_single_visit_is_not_a_place_memory(tmp_path):
    photos = [_p(f"a{i}", local=datetime(2020, 5, 1, 9, i), gps=(22.5, 87.25)) for i in range(25)]
    store, index = _index(tmp_path, photos)
    try:
        assert REGISTRY["place_cluster"].offers(index) == []
    finally:
        store.close()


def test_a_place_memory_NEVER_names_a_place(tmp_path):
    """No offline gazetteer exists in this project, so a city name would be
    an invention."""
    photos = [_p(f"a{i}", local=datetime(2020, 5, 1, 9, i), gps=(22.5, 87.25)) for i in range(10)]
    photos += [_p(f"b{i}", local=datetime(2021, 5, 1, 9, i), gps=(22.5, 87.26)) for i in range(10)]
    store, index = _index(tmp_path, photos)
    try:
        offer = REGISTRY["place_cluster"].offers(index)[0]
        assert offer.title == "A place you kept coming back to"
        # The coordinates are in the KEY, which is not shown as a title.
        assert "22.5" not in offer.title
    finally:
        store.close()


def test_then_and_now_select_re_checks_the_year_gap(tmp_path):
    """`offers()` already filters on the gap, but `select()` is reachable
    directly - `rekindle memory --recipe then_and_now --key person:X` builds
    an Offer by hand and never calls offers(). Deleting the guard in select()
    left the suite green until this test existed.
    """
    from rekindle.memory.recipes.base import Offer

    photos = [_p(f"a{i}", local=datetime(2020, 5, i + 1), people=["Avyan"]) for i in range(10)]
    store, index = _index(tmp_path, photos)
    try:
        offer = Offer(recipe="then_and_now", key="person:Avyan", title="hand-made")
        assert REGISTRY["then_and_now"].select(index, offer) is None
    finally:
        store.close()


def test_then_and_now_select_refuses_a_single_photo_subject(tmp_path):
    from rekindle.memory.recipes.base import Offer

    store, index = _index(tmp_path, [_p("only", local=datetime(2020, 5, 1), people=["Solo"])])
    try:
        offer = Offer(recipe="then_and_now", key="person:Solo", title="hand-made")
        assert REGISTRY["then_and_now"].select(index, offer) is None
    finally:
        store.close()


def test_registering_two_recipes_with_one_name_is_refused():
    """Recipe names are half of every memory id, so a collision would make two
    different memories share a dismissal."""
    from rekindle.memory.recipes.registry import register

    with pytest.raises(ValueError, match="duplicate recipe name"):

        @register
        class Clash:
            name = "album_story"
            title = "Clash"

            def offers(self, index):
                return []

            def select(self, index, offer):
                return None


def test_get_returns_none_for_an_unknown_recipe():
    from rekindle.memory.recipes.registry import get

    assert get("no_such_recipe") is None
    assert get("album_story") is not None


def test_then_and_now_skips_a_subject_whose_ends_cannot_be_SHOWN(tmp_path):
    """The bug the conformance suite found on the real library.

    This recipe picks exactly two photos, so if composition then drops either
    one the memory dies - and it died silently: `offers()` advertised a memory
    `select()` could not build. On the reference index the earliest photo of
    `Abhik Maiti` is a 6928x2309 panorama, which the aspect gate rejects.

    The fix is that the recipe narrows to showable photos BEFORE choosing its
    ends, so its offer and its selection agree.
    """

    photos = [_p(f"a{i}", local=datetime(2015 + i, 5, 1), people=["Avyan"]) for i in range(9)]
    # The earliest photo is a real panorama shape, which composition rejects.
    photos[0].meta.width, photos[0].meta.height = 6928, 2309

    store, index = _index(tmp_path, photos)
    try:
        recipe = REGISTRY["then_and_now"]
        offer = next(o for o in recipe.offers(index) if o.key == "person:Avyan")
        selection = recipe.select(index, offer)

        assert selection is not None, "offered a memory it cannot build"
        assert len(selection.photos) == 2
        # The panorama is not one of them; the SECOND-earliest is "then".
        assert selection.photos[0].file_hash == "a1"
        assert "a0" not in {p.file_hash for p in selection.photos}
        # ...and the offer's subtitle reports the year it will actually show.
        assert offer.subtitle.startswith("2016")
    finally:
        store.close()


def test_then_and_now_offers_nothing_when_too_few_photos_are_showable(tmp_path):
    """Eight photos, but seven are thumbnails. The offer must not appear."""
    photos = [_p(f"a{i}", local=datetime(2015 + i, 5, 1), people=["Avyan"]) for i in range(9)]
    for photo in photos[1:8]:
        photo.meta.width, photo.meta.height = 100, 80

    store, index = _index(tmp_path, photos)
    try:
        keys = {o.key for o in REGISTRY["then_and_now"].offers(index)}
        assert "person:Avyan" not in keys
    finally:
        store.close()


def test_then_and_now_DECLARES_no_stratification(tmp_path):
    """Pinned at the declaration, not through `build()`, and deliberately.

    This recipe hands the engine exactly two photos, which the year-gap rule
    guarantees are in different years - so ANY stratification of them is a
    no-op today and a mutation flipping the declaration cannot be caught
    end-to-end. The declaration still matters: it is the contract that says
    this memory wants the EXTREMES of a span rather than a sample across it,
    and it is what would break if the recipe were ever changed to hand over a
    generous pool and rely on AS_GIVEN ordering.
    """
    from rekindle.memory import strata

    photos = [_p(f"a{i}", local=datetime(2015 + i, 5, 1), people=["Avyan"]) for i in range(9)]
    store, index = _index(tmp_path, photos)
    try:
        recipe = REGISTRY["then_and_now"]
        offer = next(o for o in recipe.offers(index) if o.key == "person:Avyan")
        selection = recipe.select(index, offer)
        assert selection.stratify is strata.NONE
    finally:
        store.close()


def test_every_other_recipe_declares_a_dimension(tmp_path):
    """The mirror: a recipe that forgets to declare one falls back to the
    BY_SPAN default, which is right for an album and wrong for `on_this_day`.
    Asserted through the registry so a new recipe cannot quietly skip it."""
    from rekindle.memory import strata

    expected = {
        "album_story": strata.BY_SPAN,
        "on_this_day": strata.BY_YEAR,
        "on_this_month": strata.BY_YEAR,
        "person_years": strata.BY_YEAR,
        "pair_years": strata.BY_YEAR,
        "year_in_review": strata.BY_MONTH,
        "place_cluster": strata.BY_SPAN,
        "then_and_now": strata.NONE,
        "recurring_event": strata.BY_YEAR,
    }
    assert set(expected) == set(REGISTRY), "a recipe was added without a declared dimension"

    photos = _across_years(
        "a",
        years=(2020, 2021, 2022),
        per_year=12,
        people=["Amy", "Bob"],
        albums=["Kashmir"],
        gps=(22.5, 87.25),
    )
    photos += [
        _p(f"y{i}", local=datetime(2021, (i % 12) + 1, 1, 9), albums=["Kashmir"]) for i in range(40)
    ]
    store, index = _index(tmp_path, photos)
    try:
        for name, dimension in expected.items():
            recipe = REGISTRY[name]
            offers = recipe.offers(index)
            if not offers:
                continue
            selection = recipe.select(index, offers[0])
            if selection is None:
                continue
            assert selection.stratify == dimension, f"{name} declares {selection.stratify}"
    finally:
        store.close()


# --------------------------------------------------------- the registry loads
#
# The reported failure: a script looped over `REGISTRY`, got an empty dict
# because nothing had imported `recipes.builtin`, ran zero iterations and
# **exited 0** with an empty output directory. An empty registry is never a
# legitimate state - the nine built-ins are not optional - so every read of it
# has to be able to fill it.


def _forget_the_builtins():
    """Put the process back in the state a fresh interpreter is in.

    Both halves are needed: clearing the dict alone leaves `builtin` in
    `sys.modules`, so `import_module` returns the cached module without
    re-running a single `@register` and the registry stays empty. That is
    itself a way this feature could silently not work, so the helper is
    written to expose it rather than to avoid it.
    """
    import sys

    from rekindle.memory.recipes import registry as reg

    dict.clear(reg.REGISTRY)
    sys.modules.pop(reg.BUILTIN_MODULE, None)


@pytest.fixture
def forgotten():
    _forget_the_builtins()
    yield
    _forget_the_builtins()
    from rekindle.memory.recipes import registry as reg

    assert len(reg.REGISTRY) == 9  # restore for every test that follows


#: Every way a caller can look at the registry, and what a WORKING one
#: answers. One accessor per case: `bool()` falls through to `__len__`, and
#: `len()` loading the registry would hide a broken `__contains__` from any
#: test that checked both. Three mutants survived exactly that way before this
#: was split up.
_READS = [
    pytest.param(lambda r: len(list(r.REGISTRY)), 9, id="iterate"),
    pytest.param(lambda r: len(r.REGISTRY), 9, id="len"),
    pytest.param(lambda r: bool(r.REGISTRY), True, id="bool"),
    pytest.param(lambda r: len(r.REGISTRY.keys()), 9, id="keys"),
    pytest.param(lambda r: len(r.REGISTRY.values()), 9, id="values"),
    pytest.param(lambda r: len(r.REGISTRY.items()), 9, id="items"),
    pytest.param(lambda r: "album_story" in r.REGISTRY, True, id="contains"),
    pytest.param(lambda r: r.REGISTRY["album_story"].name, "album_story", id="getitem"),
    pytest.param(lambda r: r.REGISTRY.get("album_story").name, "album_story", id="dict-get"),
    pytest.param(lambda r: r.get("album_story").name, "album_story", id="get()"),
    pytest.param(lambda r: len(r.registered()), 9, id="registered()"),
    pytest.param(lambda r: len(r.names()), 9, id="names()"),
]


@pytest.mark.parametrize(("read", "expected"), _READS)
def test_every_way_of_reading_the_registry_loads_it(forgotten, read, expected):
    """Not just `registered()`. A caller reaching for the dict is doing a
    reasonable thing and must not be handed nothing.

    `forgotten` puts the process back to "the built-ins were never imported",
    and this is the FIRST read afterwards - so each accessor is tested as the
    one that has to do the loading, not as a passenger on another one.
    """
    from rekindle.memory.recipes import registry as reg

    assert read(reg) == expected


def test_a_script_that_loops_over_the_registry_cannot_silently_do_nothing(forgotten):
    """The exact reported shape: enumerate, do a thing per recipe, exit 0."""
    from rekindle.memory.recipes import registry as reg

    done = [name for name in reg.REGISTRY]
    assert done, "zero iterations and no error is the bug this guards"
    assert len(done) == 9


def test_names_agrees_with_registered_and_keeps_registration_order():
    from rekindle.memory.recipes import names, registered

    assert names() == [r.name for r in registered()]
    assert names()[0] == "album_story"


def test_a_third_party_recipe_cannot_get_ahead_of_the_builtins(forgotten):
    """Registration order decides `rekindle memories` order and breaks `--auto`
    ties, so it must not depend on which module a caller imported first."""
    from rekindle.memory.recipes import registry as reg
    from rekindle.memory.recipes.base import Offer

    @reg.register
    class _Late:
        name = "zzz_third_party"
        title = "Third party"

        def offers(self, index) -> list[Offer]:
            return []

        def select(self, index, offer):
            return None

    try:
        assert reg.names()[0] == "album_story"
        assert reg.names()[-1] == "zzz_third_party"
        assert len(reg.names()) == 10
    finally:
        dict.pop(reg.REGISTRY, "zzz_third_party", None)


def test_registering_the_same_name_twice_is_still_refused():
    from rekindle.memory.recipes import registry as reg

    with pytest.raises(ValueError, match="duplicate recipe name"):

        @reg.register
        class _Dupe:
            name = "album_story"
            title = "Nope"

            def offers(self, index):
                return []

            def select(self, index, offer):
                return None


# --------------------------------------------------------------------------
# offers() must not read the library
#
# The index answers a count and a set of distinct years off its spine, and
# five recipes were changed to ask it rather than to load a slice and measure
# it. Measured on a synthetic 300,000-photo library that took `all_offers`
# from 185.9 s to 46.5 s, and `year_in_review.offers` alone from 38.7 s to
# 0.010 s - it had been hydrating 298,000 photographs to produce 24 integers.
#
# Nothing about the OUTPUT changed, which is why a determinism check cannot
# see this and why it needs its own test: a recipe that quietly went back to
# `by_person(...)` in `offers()` would still produce the right offers.


def _dense_years():
    """Enough photographs per year for every count-and-years recipe to offer.

    `year_in_review` wants 30 a year and `_across_years` can only place 15 in
    one day (it walks the hour), so the density comes from several dates
    rather than from one big group. Thin fixtures are how a "does it hydrate?"
    test becomes a test of nothing, which is why the assertions below check
    that offers were actually produced.
    """
    out = _across_years("a", people=["Amy", "Bob"], albums=["Kashmir"], gps=(22.5, 87.25))
    out += _across_years("b", month=5, day=1, per_year=12, people=["Amy"], albums=["Mysore"])
    out += _across_years("c", month=12, day=25, per_year=12, people=["Amy", "Bob"])
    out += _across_years("d", month=6, day=11, per_year=12, people=["Bob"])
    out += _across_years("e", month=9, day=3, per_year=12, people=["Amy", "Bob"])
    out += _drifting_festival()
    return out


def _drifting_festival():
    """Four Julys that drift, over a quiet baseline, for `recurring_event`.

    Two things the anniversary shape above cannot give it. A burst is a day
    holding four times the median ACTIVE day of its own year, so a fixture
    made only of dense days has no dense day in it at all - every one of the
    dates above sits at the median. And an event must repeat in MIN_YEARS
    distinct years, which is four; `_across_years` spans three.

    The album is on the burst days only, in every year, so the evidence
    `naming_evidence` reads is really there - a fixture that offered a
    recurring event but named nothing would let the album route break
    silently.
    """
    out = []
    n = 0
    for year, day in ((2018, 12), (2019, 4), (2020, 8), (2021, 15)):
        for week in range(52):
            when = datetime(year, 1, 1) + timedelta(weeks=week)
            for _ in range(2):
                n += 1
                out.append(_p(f"quiet{n:05d}", local=when + timedelta(hours=n % 12)))
        for offset in range(3):
            when = datetime(year, 7, day) + timedelta(days=offset)
            for i in range(14):
                n += 1
                out.append(
                    _p(
                        f"burst{n:05d}",
                        local=when + timedelta(minutes=i * 7),
                        albums=["Rathayatra"],
                    )
                )
    return out


#: Recipes whose `offers()` answers entirely from the index spine.
#:
#: `album_story` joined them when the month span moved onto the spine: it
#: needs a count and two dates for its subtitle, and used to get the dates by
#: hydrating every photograph of every album - on a 300k library, the whole
#: library. Measured on the reference index, 0.33s of a 2.60s offers phase,
#: now 0.004s.
#:
#: `recurring_event` joined them when the album evidence for its title moved
#: onto the spine: it needed a local date and an album list per photograph and
#: hydrated every photograph of every burst to read them - 11,422 on the
#: reference index, 0.54s of a 1.18s offers phase, now 0.02s of 0.38s.
#:
#: Two are deliberately absent and are expected to stay that way until the
#: data they need is on the spine too. See `STILL_HYDRATES`.
SPINE_ONLY_OFFERS = (
    "album_story",
    "on_this_day",
    "on_this_month",
    "person_years",
    "pair_years",
    "recurring_event",
    "year_in_review",
)

#: The negative control for the test below, and a record of what is left.
#:
#: `then_and_now` picks exactly TWO photographs, so it has to know which would
#: survive the composition guardrails - width, height, sharpness, brightness
#: and media type per photograph - or it advertises a memory `select()` cannot
#: build. `place_cluster` splits a cell into visits by timestamp gap. Neither
#: is a count, a year or an album, which is all the spine carries.
STILL_HYDRATES = "then_and_now"


@pytest.mark.parametrize("name", SPINE_ONLY_OFFERS)
def test_offers_does_not_hydrate_a_single_photograph(name, tmp_path):
    store, index = _index(tmp_path, _dense_years())
    try:
        offers = REGISTRY[name].offers(index)
        assert offers, f"{name} offered nothing, so the assertion below is vacuous"
        assert len(index._cache) == 0, (
            f"{name}.offers() hydrated {len(index._cache)} photographs; "
            "counts and years are both on the spine"
        )
    finally:
        index.close()
        store.close()


def test_a_recipe_that_reads_the_slice_really_would_show_up(tmp_path):
    """Guards the test above against being vacuous.

    A "does not hydrate" assertion proves nothing unless a recipe that DOES
    hydrate fails it on the same fixture. `album_story` used to be this
    control and no longer hydrates, which is how this test caught the change -
    exactly what a control is for. `then_and_now` needs the composition
    scalars per photograph and cannot answer from the spine.
    """
    photos = _across_years("a", people=["Amy", "Bob"], albums=["Kashmir"])
    store, index = _index(tmp_path, photos)
    try:
        assert REGISTRY[STILL_HYDRATES].offers(index)
        assert len(index._cache) > 0, "the fixture never hydrates, so the test proves nothing"
    finally:
        index.close()
        store.close()


def test_the_offer_count_is_the_real_number_of_photographs(tmp_path):
    """`size` orders the offers and the subtitle is shown to a person, and
    both now come from a spine counter rather than from `len(photos)`. An
    off-by-one there is invisible to every other test in this file."""
    store, index = _index(tmp_path, _dense_years())
    try:
        checked = 0
        for offer in REGISTRY["year_in_review"].offers(index):
            checked += 1
            real = len(index.by_year(int(offer.key)))
            assert offer.size == real, f"{offer.key}: size {offer.size}, really {real}"
            assert offer.subtitle == f"{real} photos"
        for offer in REGISTRY["on_this_day"].offers(index):
            checked += 1
            month, day = (int(x) for x in offer.key.split("-"))
            real = len(index.by_month_day(month, day))
            years = len(index.years_present(index.by_month_day(month, day)))
            assert offer.size == real
            assert offer.subtitle == f"{years} years, {real} photos"
        assert checked > 4, f"only {checked} offers compared; the fixture is too thin"
    finally:
        index.close()
        store.close()
