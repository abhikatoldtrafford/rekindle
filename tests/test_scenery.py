"""Scenery memories: the corpus, the vote gate, and what they must not become.

Everything here runs with no model, no embeddings and no photographs.
Retrieval is injected as a callable returning `(file_hash, score)` pairs -
which is the real interface `prompt.consensus` reads - so the tests exercise
the actual consensus code rather than a stand-in for it.
"""

from __future__ import annotations

import textwrap
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rekindle.db import PhotoStore
from rekindle.memory import engine, scenery
from rekindle.memory.festivals import CorpusError
from rekindle.memory.index import MemoryIndex
from rekindle.memory.recipes import REGISTRY, Offer
from rekindle.models import MediaType, Photo, PhotoMeta

T0 = datetime(2020, 1, 1, tzinfo=UTC)


def _photo(h: str, when: datetime, *, people=(), albums=(), gps=None) -> Photo:
    return Photo(
        file_hash=h,
        paths=[Path(f"{h}.jpg")],
        media_type=MediaType.IMAGE,
        meta=PhotoMeta(
            taken_at_utc=when.replace(tzinfo=UTC),
            taken_at_local=when,
            people=list(people),
            gps=gps,
            width=1600,
            height=1200,
            sharpness=0.6,
            brightness=110.0,
            phash=abs(hash(h)) % (1 << 60),
        ),
        first_seen=T0,
        last_seen=T0,
        albums=list(albums),
    )


def _index(photos, tmp_path) -> tuple[PhotoStore, MemoryIndex]:
    store = PhotoStore(tmp_path / "i.sqlite")
    store.upsert_many(photos)
    return store, MemoryIndex.open(store)


def _retriever(per_tag: dict[str, list[str]]):
    """Each tag's hits, best first, as (hash, descending score) pairs."""

    def retrieve(text: str, k: int):
        hits = per_tag.get(text, [])[:k]
        return [(h, 0.9 - i * 0.01) for i, h in enumerate(hits)]

    return retrieve


def _concept(**over) -> scenery.Concept:
    base = {
        "key": "sea",
        "title": "The sea",
        "names": ("sea",),
        "tags": ("waves", "sand", "horizon", "boats"),
    }
    base.update(over)
    return scenery.Concept(**base)


# --------------------------------------------------------------------------
# the corpus


def test_the_shipped_corpus_loads_and_every_entry_is_usable():
    concepts = scenery.all_concepts()
    assert concepts
    for concept in concepts:
        assert concept.title and concept.tags
        assert len(concept.tags) >= 2, f"{concept.key}: one description cannot be a consensus"


def test_no_shipped_title_is_a_place_name():
    """`captions.PLACE_TITLE` exists because this project has no gazetteer.

    A scenery title is copied verbatim onto a memory, so a capitalised word
    inside one is a place or a person that nothing substantiates.
    """
    for concept in scenery.all_concepts():
        for word in concept.title.split()[1:]:
            assert not word[:1].isupper(), (
                f"{concept.title!r} contains {word!r}. A scenery title may not "
                "name a place - there is no gazetteer in this project."
            )


def test_a_concept_with_no_tags_is_refused(tmp_path):
    path = tmp_path / "s.toml"
    path.write_text(
        textwrap.dedent(
            """
            [[concept]]
            key = "x"
            title = "X"
            """
        ),
        encoding="utf-8",
    )
    scenery.load.cache_clear()
    with pytest.raises(CorpusError, match="has no `tags`"):
        scenery.load(path)


def test_a_concept_with_no_title_is_refused(tmp_path):
    """Deriving a title from the key would be inventing one."""
    path = tmp_path / "s.toml"
    path.write_text(
        textwrap.dedent(
            """
            [[concept]]
            key = "x"
            tags = ["a", "b"]
            """
        ),
        encoding="utf-8",
    )
    scenery.load.cache_clear()
    with pytest.raises(CorpusError, match="has no `title`"):
        scenery.load(path)


def test_a_duplicate_key_is_refused(tmp_path):
    """A key is half a memory id, so a duplicate makes one memory shadow
    another and a dismissal of one silently dismiss both."""
    path = tmp_path / "s.toml"
    path.write_text(
        textwrap.dedent(
            """
            [[concept]]
            key = "x"
            title = "X"
            tags = ["a"]
            [[concept]]
            key = "x"
            title = "Y"
            tags = ["b"]
            """
        ),
        encoding="utf-8",
    )
    scenery.load.cache_clear()
    with pytest.raises(CorpusError, match="share the key"):
        scenery.load(path)


def test_a_month_outside_the_year_is_refused(tmp_path):
    path = tmp_path / "s.toml"
    path.write_text('[[concept]]\nkey="x"\ntitle="X"\ntags=["a"]\nmonths=[13]\n', encoding="utf-8")
    scenery.load.cache_clear()
    with pytest.raises(CorpusError, match="month outside"):
        scenery.load(path)


def test_match_prefers_the_longest_name():
    assert scenery.match("show me the sea").key == "sea"
    assert scenery.match("street food over the years").key == "food"


def test_match_is_on_whole_words():
    """`sea` must not match inside `season`, or a user asking for one concept
    silently gets another."""
    assert scenery.match("a season of change") is None


def test_an_unknown_concept_name_lists_the_known_ones():
    with pytest.raises(KeyError, match="Known:"):
        scenery.concept_keys(["volcanoes"])


@pytest.fixture(autouse=True)
def _restore_corpus_cache():
    yield
    scenery.load.cache_clear()


# --------------------------------------------------------------------------
# selection


def test_a_photo_only_one_description_reached_is_left_out(tmp_path):
    """The vote gate. One description's private idea of a concept is not
    evidence, and the stratifier fills a sparse year best-first by QUALITY,
    not by consensus rank - so a one-vote photo would otherwise get a slot."""
    photos = [_photo("a", datetime(2019, 5, 1)), _photo("b", datetime(2020, 5, 1))]
    store, index = _index(photos, tmp_path)
    retrieve = _retriever({"waves": ["a", "b"], "sand": ["a"], "horizon": ["a"], "boats": ["a"]})
    build = scenery.build_selection(index, _concept(), retrieve)
    store.close()
    assert [p.file_hash for p in build.selection.photos] == ["a"]
    assert build.reached == 1
    assert build.votes == {4: 1}


def test_min_votes_of_one_lets_the_tail_back_in(tmp_path):
    photos = [_photo("a", datetime(2019, 5, 1)), _photo("b", datetime(2020, 5, 1))]
    store, index = _index(photos, tmp_path)
    retrieve = _retriever({"waves": ["a", "b"], "sand": ["a"]})
    build = scenery.build_selection(index, _concept(), retrieve, min_votes=1)
    store.close()
    assert {p.file_hash for p in build.selection.photos} == {"a", "b"}


def test_a_single_description_concept_still_works(tmp_path):
    """`min_votes` is 2, and a one-tag concept can never reach 2. The floor
    drops to 1 rather than the concept silently returning nothing."""
    photos = [_photo("a", datetime(2019, 5, 1))]
    store, index = _index(photos, tmp_path)
    build = scenery.build_selection(index, _concept(tags=("waves",)), _retriever({"waves": ["a"]}))
    store.close()
    assert [p.file_hash for p in build.selection.photos] == ["a"]


def test_the_corpus_month_window_narrows_the_pool(tmp_path):
    photos = [_photo("jul", datetime(2019, 7, 1)), _photo("jan", datetime(2019, 1, 1))]
    store, index = _index(photos, tmp_path)
    retrieve = _retriever({"waves": ["jul", "jan"], "sand": ["jul", "jan"]})
    build = scenery.build_selection(
        index, _concept(tags=("waves", "sand"), months=(6, 7, 8, 9)), retrieve
    )
    store.close()
    assert [p.file_hash for p in build.selection.photos] == ["jul"]


def test_an_excluded_photo_cannot_reach_a_scenery_memory(tmp_path):
    """The index is the chokepoint, and `resolve_many` runs BEFORE the vote
    gate and before the year count - so an excluded person's photographs
    cannot influence either.

    NOTE ON MUTATION: no single-line change to `build_selection` makes this
    fail, and that was checked rather than assumed. `MemoryIndex` holds only
    photographs the policy allows, so `index.get` and `index.resolve_many` are
    equally safe and swapping one for the other changes nothing. The guarantee
    is structural. This test guards the BOUNDARY: it fails the day someone
    reaches past the index to a `PhotoStore` query for speed.
    """
    from rekindle.memory.policy import ExclusionPolicy

    photos = [
        _photo("a", datetime(2019, 5, 1), people=["Blocked"]),
        _photo("b", datetime(2020, 5, 1)),
    ]
    store = PhotoStore(tmp_path / "i.sqlite")
    store.upsert_many(photos)
    index = MemoryIndex.open(store, ExclusionPolicy(people=frozenset({"Blocked"})))
    retrieve = _retriever({"waves": ["a", "b"], "sand": ["a", "b"]})
    build = scenery.build_selection(index, _concept(tags=("waves", "sand")), retrieve)
    store.close()
    assert [p.file_hash for p in build.selection.photos] == ["b"]
    assert build.reached == 1


def test_nothing_retrieved_is_a_build_with_no_selection(tmp_path):
    store, index = _index([_photo("a", datetime(2019, 5, 1))], tmp_path)
    build = scenery.build_selection(index, _concept(), _retriever({}))
    store.close()
    assert build.selection is None
    assert build.reached == 0


def test_the_selection_is_chronological(tmp_path):
    photos = [_photo(h, datetime(y, 5, 1)) for h, y in (("c", 2022), ("a", 2019), ("b", 2020))]
    store, index = _index(photos, tmp_path)
    retrieve = _retriever({"waves": ["c", "a", "b"], "sand": ["c", "a", "b"]})
    build = scenery.build_selection(index, _concept(tags=("waves", "sand")), retrieve)
    store.close()
    assert [p.file_hash for p in build.selection.photos] == ["a", "b", "c"]


def test_a_scenery_memory_must_span_more_than_one_year(tmp_path):
    """One afternoon at the beach is a different and lesser memory, and
    nothing in the title would tell the viewer which they were looking at."""
    photos = [_photo(f"h{i}", datetime(2019, 5, i + 1)) for i in range(6)]
    store, index = _index(photos, tmp_path)
    hits = [p.file_hash for p in photos]
    retrieve = _retriever({"waves": hits, "sand": hits})
    build = scenery.build_selection(index, _concept(tags=("waves", "sand")), retrieve)
    report = engine.BuildReport(offered=1)
    spec = engine.build(
        index,
        Offer(recipe=scenery.RECIPE, key="sea", title="The sea"),
        selection=build.selection,
        report=report,
    )
    store.close()
    assert spec is None
    assert report.skipped == {engine.SKIP_TOO_NARROW: 1}


def test_a_two_year_scenery_memory_is_built(tmp_path):
    photos = [_photo(f"h{i}", datetime(2019 + i % 2, 5, i + 1)) for i in range(6)]
    store, index = _index(photos, tmp_path)
    hits = [p.file_hash for p in photos]
    retrieve = _retriever({"waves": hits, "sand": hits})
    build = scenery.build_selection(index, _concept(tags=("waves", "sand")), retrieve)
    spec = engine.build(
        index,
        Offer(recipe=scenery.RECIPE, key="sea", title="The sea"),
        selection=build.selection,
    )
    store.close()
    assert spec is not None
    assert spec.title == "The sea"
    assert spec.recipe == "scenery"


# --------------------------------------------------------------------------
# what a scenery memory must never become


def test_scenery_is_not_a_registered_recipe():
    """A registered recipe would need a retriever inside `select()`, which the
    `Recipe` protocol has no way to supply and which CI - no models, no
    embeddings - could not provide. It would also put scenery memories into
    `rekindle memories` and `--auto`, where nobody had looked at them."""
    assert scenery.RECIPE not in REGISTRY


def test_scenery_does_not_expand_seed_days(tmp_path):
    """The one thing it must NOT borrow from prompt memories.

    `prompt.build_selection` takes a confirmed day WHOLE, because a festival
    memory of only the top-ranked frames is a wall of idols with no family in
    it. A scenery memory's subject is the scene, so the lunch and the hotel
    room photographed on the same day must stay out.
    """
    beach = _photo("beach", datetime(2019, 5, 1, 9))
    lunch = _photo("lunch", datetime(2019, 5, 1, 13))
    later = _photo("beach2", datetime(2021, 5, 1, 9))
    store, index = _index([beach, lunch, later], tmp_path)
    retrieve = _retriever({"waves": ["beach", "beach2"], "sand": ["beach", "beach2"]})
    build = scenery.build_selection(index, _concept(tags=("waves", "sand")), retrieve)
    store.close()
    chosen = {p.file_hash for p in build.selection.photos}
    assert "lunch" not in chosen
    assert chosen == {"beach", "beach2"}


# --------------------------------------------------------------------------
# the reason this exists at all


def test_orphan_is_the_population_no_other_recipe_can_be_about():
    from rekindle.models import Gps

    when = datetime(2019, 5, 1)
    assert scenery.orphan(_photo("a", when))
    assert not scenery.orphan(_photo("b", when, people=["Someone"]))
    assert not scenery.orphan(_photo("c", when, albums=["Kashmir"]))
    assert not scenery.orphan(_photo("d", when, gps=Gps(lat=1.0, lon=2.0)))


def test_googles_own_year_albums_do_not_make_a_photo_reachable():
    """Every photograph in a Takeout export is in `Photos from YYYY`, so an
    orphan test that counted any album would report zero orphans - which is
    exactly what the first run of this measurement did."""
    assert scenery.orphan(_photo("a", datetime(2019, 5, 1), albums=["Photos from 2019"]))
