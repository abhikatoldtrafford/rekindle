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
