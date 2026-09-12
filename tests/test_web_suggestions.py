"""What the empty state offers to type, and what it must never offer.

`api.suggestions` is the only place on this page that VOLUNTEERS a name. Every
other surface answers a question somebody asked; this one puts people, albums
and date windows in front of them unprompted, which makes it the surface where
a guardrail leak would be most visible and least excusable. So the first test
here is the negative one: a person on the exclusion list, with enough
photographs and enough years to qualify on every other count, is not offered.

The thresholds are lowered in these tests rather than the fixture being grown
to sixty photographs across three years. The rule being checked is "a
suggestion comes from the index and honours the policy", and that rule does
not depend on where the bar is - but a three-minute fixture would mean this
file is not run often enough to catch anything.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from rekindle.db import PhotoStore
from rekindle.memory import albums as albums_mod
from rekindle.memory import scenery as scenery_mod
from rekindle.web import api
from tests.fixtures.web import (
    EXCLUDED_PERSON,
    exclude_person,
    jpeg,
    make_library,
    make_workshop,
    photo,
)


@pytest.fixture(autouse=True)
def low_thresholds(monkeypatch):
    """The fixture library is ten photographs in one month of one year."""
    monkeypatch.setattr(api, "MIN_SUGGESTION_PHOTOS", 1)
    monkeypatch.setattr(api, "MIN_SUGGESTION_YEARS", 1)


@pytest.fixture
def shop(tmp_path):
    data_dir, _ = make_library(tmp_path)
    return make_workshop(tmp_path, data_dir), data_dir


def texts(shop, **kwargs):
    return [row["text"] for row in api.suggestions(shop, **kwargs)["suggestions"]]


# ----------------------------------------------------------- the guardrail


def test_a_person_with_photographs_is_offered_as_a_prompt(shop):
    """The positive control for the test below.

    Without this, excluding somebody and finding them absent would prove
    nothing: it would pass just as well if `suggestions` never named a person
    at all, or returned an empty list, or raised and was swallowed.
    """
    workshop, _data_dir = shop
    assert any("abhik maiti" in t for t in texts(workshop, limit=40))


def test_an_excluded_person_is_never_suggested(shop):
    """THE test on this file.

    `EXCLUDED_PERSON` is in the fixture library with photographs of their own,
    and the thresholds above are low enough that they would otherwise qualify.
    Exclusion goes through `MemoryState`, so what is being checked is that the
    suggestion list is built from `MemoryIndex` and not from the photo store -
    an index-bypassing count here would put a name somebody has asked never to
    see on the first screen of the application.
    """
    workshop, data_dir = shop
    before = texts(workshop, limit=40)
    assert any(EXCLUDED_PERSON.lower() in t for t in before), (
        "the excluded person must be suggestible BEFORE the exclusion, or this test cannot fail"
    )

    exclude_person(data_dir)
    workshop.library.load()
    after = texts(workshop, limit=40)
    assert not any(EXCLUDED_PERSON.lower() in t for t in after)
    assert after, "excluding one person must not empty the whole list"


# --------------------------------------------------------- counted, not made


def test_a_suggestion_carries_the_count_it_was_chosen_on(shop):
    workshop, _data_dir = shop
    rows = api.suggestions(workshop, limit=40)["suggestions"]
    person = next(r for r in rows if r["kind"] == "person")
    index = workshop.library.require_index()
    name = person["text"].removesuffix(" over the years")
    real = len(index.by_person(next(n for n in index.people_counts() if n.lower() == name)))
    assert f"{real:,} photographs" in person["note"]


def test_the_ranking_key_does_not_reach_the_page(shop):
    workshop, _data_dir = shop
    for row in api.suggestions(workshop, limit=40)["suggestions"]:
        assert "weight" not in row
        assert set(row) == {"text", "note", "kind"}


def test_kinds_are_interleaved_rather_than_concatenated(shop):
    """Eight rows of one kind teach that a prompt is one kind of thing."""
    workshop, _data_dir = shop
    kinds = [r["kind"] for r in api.suggestions(workshop, limit=8)["suggestions"]]
    assert len(kinds) >= 2
    assert kinds[0] != kinds[1], kinds


# ------------------------------------------------------------------ albums


def album_library(tmp_path: Path) -> Path:
    """A library holding one real album and one Takeout auto-album."""
    data_dir = tmp_path / "data"
    root = tmp_path / "lib"
    rows = []
    for i in range(4):
        rows.append(
            photo(
                f"real{i}",
                jpeg(root / f"real{i}.jpg", colour=(40 + i * 30, 90, 60)),
                local=datetime(2019, 6, 1, 9) + timedelta(days=i),
                albums=("Kashmir",),
            )
        )
        rows.append(
            photo(
                f"auto{i}",
                jpeg(root / f"auto{i}.jpg", colour=(60, 40 + i * 30, 90)),
                local=datetime(2019, 7, 1, 9) + timedelta(days=i),
                albums=("Photos from 2019",),
            )
        )
    with PhotoStore(data_dir / "rekindle.sqlite") as store:
        store.upsert_many(rows)
    return data_dir


def test_an_album_the_recipe_would_not_offer_is_not_suggested(tmp_path):
    """ "Photos from 2019" is Google's filing, not a memory.

    `album_story` already refuses it through `albums.presentable`; raw
    `album_counts()` does not, and the first version of this list suggested
    two of them. Sharing the predicate is what keeps the page from teaching
    that a prompt is a folder name.
    """
    data_dir = album_library(tmp_path)
    workshop = make_workshop(tmp_path, data_dir)
    assert not albums_mod.presentable("Photos from 2019"), (
        "this fixture album is supposed to be the unpresentable one"
    )
    rows = [r for r in api.suggestions(workshop, limit=40)["suggestions"] if r["kind"] == "album"]
    assert [r["text"] for r in rows] == ["kashmir"]


# --------------------------------------------------------------- festivals


def autumn_library(tmp_path: Path) -> Path:
    """Photographs in September, October and November across three years, so
    every festival window in the corpus has evidence behind it."""
    data_dir = tmp_path / "data"
    root = tmp_path / "lib"
    rows = []
    for year in (2018, 2019, 2020):
        for month in (9, 10, 11):
            for day in (5, 15):
                key = f"a{year}{month:02d}{day}"
                rows.append(
                    photo(
                        key,
                        jpeg(root / f"{key}.jpg", colour=(month * 15, day * 10, 60)),
                        local=datetime(year, month, day, 9),
                    )
                )
    with PhotoStore(data_dir / "rekindle.sqlite") as store:
        store.upsert_many(rows)
    return data_dir


def test_only_one_festival_is_suggested_per_month_window(tmp_path):
    """Kali Puja and Jagaddhatri Puja are both October-November.

    Ranked on photographs in the window they score identically, because the
    window is the only evidence there is - so offering both puts two rows in
    front of somebody that carry one fact between them, and pushed Durga Puja
    (September-October, a different window) off the list entirely.
    """
    from rekindle.memory import festivals as festivals_mod

    data_dir = autumn_library(tmp_path)
    workshop = make_workshop(tmp_path, data_dir)
    # `_suggest_festivals` directly, not the interleaved list. `_interleave`
    # takes two of each kind, and two rows drawn from a list that happens to
    # begin with two different windows pass whether the deduplication is
    # there or not - which is exactly how the first version of this test
    # survived having that line deleted.
    rows = api._suggest_festivals(workshop.library.require_index())
    assert len(rows) >= 2, "the fixture has three autumn years; festivals should qualify"

    windows = []
    for row in rows:
        name = row["text"].removesuffix(" over the years")
        festival = festivals_mod.match(name)
        assert festival is not None, f"{name!r} is not a name the festival matcher knows"
        windows.append(festival.window)
    assert len(set(windows)) == len(windows), f"two festivals share a window: {windows}"


def test_a_festival_note_describes_the_window_and_not_the_festival(tmp_path):
    """A photograph in October is not a photograph of Durga Puja, and the
    suggestion must not imply that it is."""
    data_dir = autumn_library(tmp_path)
    workshop = make_workshop(tmp_path, data_dir)
    rows = [
        r for r in api.suggestions(workshop, limit=40)["suggestions"] if r["kind"] == "festival"
    ]
    assert rows
    for row in rows:
        assert "years ·" in row["note"]
        # It names months, never a count of festival photographs.
        assert "photograph" not in row["note"]


# ------------------------------------------------------------------ scenes


def test_a_scene_suggestion_is_a_phrase_the_matcher_accepts(shop, monkeypatch):
    """`names[0]` is the corpus head-word - "mountain", "sea", "flower" - and
    nobody types those. Whatever phrase is chosen instead has to be one
    `scenery.match` still resolves, or the suggestion is a prompt that cannot
    find what it names."""
    workshop, _data_dir = shop
    monkeypatch.setattr(workshop.library, "semantic_available", lambda: True)
    rows = [r for r in api.suggestions(workshop, limit=40)["suggestions"] if r["kind"] == "scene"]
    assert rows, "the scenery corpus is not empty"
    # Every concept, not the two that survive interleaving.
    for concept in scenery_mod.all_concepts():
        phrase = api._scene_phrase(concept)
        found = scenery_mod.match(phrase)
        assert found is not None, f"{phrase!r} resolves to no concept at all"
        assert found.key == concept.key, (
            f"{phrase!r} was generated for {concept.key} but resolves to {found.key}"
        )
    assert {r["text"] for r in rows} <= {api._scene_phrase(c) for c in scenery_mod.all_concepts()}


def test_scenes_are_not_offered_when_the_search_that_runs_them_is_missing(shop, monkeypatch):
    """Suggesting "mountains" to somebody whose only possible answer is
    "install the semantic extra" is a worse empty state than one row fewer."""
    workshop, _data_dir = shop
    monkeypatch.setattr(workshop.library, "semantic_available", lambda: False)
    kinds = {r["kind"] for r in api.suggestions(workshop, limit=40)["suggestions"]}
    assert "scene" not in kinds
