"""Parsing, consensus and seed-and-expand - all with retrieval injected.

Every test here runs with no torch, no embedding store and no network. The
retriever is a plain function returning `(file_hash, score)` pairs, which is
what makes the consensus arithmetic testable at all: the interesting
behaviours are about RANKS, and a fake retriever is the only way to pin an
exact rank.
"""

import sys
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rekindle.db import PhotoStore
from rekindle.memory import prompt, strata
from rekindle.memory.index import MemoryIndex
from rekindle.memory.policy import ExclusionPolicy
from rekindle.models import MediaType, Photo, PhotoMeta


def _p(h, *, local, people=(), albums=(), archived=False):
    return Photo(
        file_hash=h,
        paths=[Path(f"/lib/{h}.jpg")],
        media_type=MediaType.IMAGE,
        meta=PhotoMeta(
            taken_at_utc=local.replace(tzinfo=UTC),
            taken_at_local=local,
            people=list(people),
            archived=archived,
            width=4000,
            height=3000,
            sharpness=5.0,
            brightness=110.0,
            phash=abs(hash(h)) % (1 << 60),
        ),
        first_seen=datetime(2020, 1, 1, tzinfo=UTC),
        last_seen=datetime(2020, 1, 1, tzinfo=UTC),
        albums=list(albums),
    )


def _index(tmp_path, photos, policy=None):
    store = PhotoStore(tmp_path / "db.sqlite")
    store.upsert_many(photos)
    return store, MemoryIndex.open(store, policy)


def _retriever(per_tag):
    """`{tag: [hash, ...]}` -> a Retriever. Scores descend from 1.0."""

    def retrieve(tag, k):
        hits = per_tag.get(tag, [])[:k]
        return [(h, 1.0 - i * 0.001) for i, h in enumerate(hits)]

    return retrieve


# ------------------------------------------------------------- normalisation


def test_normalisation_is_nfkc_casefold_and_whitespace_collapse():
    assert prompt.normalise("  Durga   Puja  ") == "durga puja"
    assert prompt.normalise("DURGA PUJA") == "durga puja"
    decomposed = unicodedata.normalize("NFD", "café")
    assert decomposed != "café"
    assert prompt.normalise(decomposed) == prompt.normalise("CAFÉ") == "café"


def test_normalisation_preserves_punctuation():
    """A comma changes meaning; folding it away would merge two prompts."""
    assert prompt.normalise("Christmas, 2019") == "christmas, 2019"


# -------------------------------------------------------------------- parse


def test_a_trailing_shape_phrase_sets_the_shape_and_leaves_the_subject(tmp_path):
    store, index = _index(tmp_path, [_p("a", local=datetime(2019, 10, 4, 12, 0))])
    try:
        query = prompt.parse("durga puja over the years", index)
        assert query.shape == prompt.SHAPE_YEARS
        assert query.subject == "durga puja"
        assert query.text == "durga puja over the years"
    finally:
        store.close()


def test_in_is_an_ordinary_word_not_a_place_cue(tmp_path):
    """Measured: the full phrase hand-grades 9/10 while `christmas
    decorations` grades 1/10 and returns Diwali. There is no gazetteer in this
    project, so treating "midnapur" as a place would be inventing one."""
    store, index = _index(tmp_path, [_p("a", local=datetime(2019, 12, 25, 12, 0))])
    try:
        query = prompt.parse("christmas in midnapur", index)
        assert query.shape == prompt.SHAPE_SPAN
        assert query.subject == "christmas in midnapur"
    finally:
        store.close()


def test_a_year_the_library_has_becomes_a_filter_and_leaves_the_subject(tmp_path):
    store, index = _index(tmp_path, [_p("a", local=datetime(2019, 10, 4, 12, 0))])
    try:
        assert prompt.parse("durga puja 2019", index).years == (2019,)
        assert prompt.parse("durga puja 2019", index).subject == "durga puja"
    finally:
        store.close()


def test_a_year_the_library_does_not_have_stays_in_the_subject(tmp_path):
    store, index = _index(tmp_path, [_p("a", local=datetime(2019, 10, 4, 12, 0))])
    try:
        query = prompt.parse("durga puja 1066", index)
        assert query.years == ()
        assert "1066" in query.subject
    finally:
        store.close()


def test_with_a_known_person_becomes_a_filter(tmp_path):
    store, index = _index(
        tmp_path, [_p("a", local=datetime(2019, 10, 4, 12, 0), people=["Avyan Maiti"])]
    )
    try:
        query = prompt.parse("diwali with Avyan Maiti", index)
        assert query.people == ("Avyan Maiti",)
        assert query.subject == "diwali"
    finally:
        store.close()


def test_with_an_unknown_person_is_not_a_filter(tmp_path):
    """`people_counts()` is a closed, verifiable set. A name that is not in it
    is not a person, and the words stay where the user typed them."""
    store, index = _index(
        tmp_path, [_p("a", local=datetime(2019, 10, 4, 12, 0), people=["Avyan Maiti"])]
    )
    try:
        query = prompt.parse("diwali with Gandalf", index)
        assert query.people == ()
        assert query.subject == "diwali with gandalf"
    finally:
        store.close()


def test_a_month_name_comes_from_the_literal_table_not_strftime(tmp_path):
    store, index = _index(tmp_path, [_p("a", local=datetime(2019, 12, 25, 12, 0))])
    try:
        assert prompt.parse("christmas in december", index).months == (12,)
    finally:
        store.close()


def test_prompt_imports_without_torch_numpy_or_transformers():
    """`rekindle --help` must never load CLIP, so this module may not reach
    the semantic layer even transitively."""
    blocked = {"torch": None, "transformers": None, "numpy": None}
    saved = {k: sys.modules.get(k) for k in blocked}
    for name in blocked:
        sys.modules[name] = None  # type: ignore[assignment]
    try:
        import importlib

        importlib.reload(importlib.import_module("rekindle.memory.prompt"))
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


# ---------------------------------------------------------------- memory key


def test_the_memory_key_is_the_normalised_prompt(tmp_path):
    store, index = _index(tmp_path, [_p("a", local=datetime(2019, 10, 4, 12, 0))])
    try:
        query = prompt.parse("Durga  Puja Over The Years", index)
        assert prompt.memory_key(query) == "prompt:durga puja over the years"
    finally:
        store.close()


def test_two_spellings_of_one_prompt_give_one_id(tmp_path):
    store, index = _index(tmp_path, [_p("a", local=datetime(2019, 10, 4, 12, 0))])
    try:
        keys = {
            prompt.memory_key(prompt.parse(text, index))
            for text in ("durga puja", "DURGA  PUJA", " Durga Puja ")
        }
        assert len(keys) == 1
    finally:
        store.close()


def test_the_id_does_not_change_when_a_matching_album_appears(tmp_path):
    """THE dismissal-breaking bug. An id derived from the parse, the matched
    albums or the chosen photos silently mints a new memory the first time the
    library changes, and every dismissal of the old one stops applying."""
    base = [_p(f"b{i}", local=datetime(2019, 10, 4, 12, i)) for i in range(4)]
    store_a, index_a = _index(tmp_path / "a", base)
    store_b, index_b = _index(
        tmp_path / "b",
        base + [_p("extra", local=datetime(2025, 9, 30, 12, 0), albums=["Durga Puja 25"])],
    )
    try:
        query_a = prompt.parse("durga puja", index_a)
        query_b = prompt.parse("durga puja", index_b)
        assert prompt.album_matches(index_a, "durga puja") == []
        assert prompt.album_matches(index_b, "durga puja") == ["Durga Puja 25"]
        assert prompt.memory_key(query_a) == prompt.memory_key(query_b)
    finally:
        store_a.close()
        store_b.close()


# ---------------------------------------------------------------- consensus


def test_a_photo_two_tags_agree_on_outranks_one_that_one_tag_loves():
    """The whole design in one assertion.

    `solo` is rank 1 for its tag and invisible to the other. `both` is rank
    99 for one tag and rank 100 for the other - as deep as the search goes.
    Under plain reciprocal-rank fusion `solo` wins (1/61 = 0.0164 against
    1/159 + 1/160 = 0.0125). Under consensus it loses, because an integer vote
    can never be outweighed by the tiebreak, and THAT is the property that
    separates two festivals with a shared visual vocabulary.
    """
    tag_one = ["solo"] + [f"f{i}" for i in range(97)] + ["both"]
    tag_two = [f"g{i}" for i in range(99)] + ["both"]
    assert tag_one.index("both") == 98 and tag_two.index("both") == 99
    fusion_solo = 1 / (prompt.RRF_C + 1)
    fusion_both = 1 / (prompt.RRF_C + 99) + 1 / (prompt.RRF_C + 100)
    assert fusion_solo > fusion_both, "the fixture no longer pins the interesting case"

    result = prompt.consensus(["one", "two"], _retriever({"one": tag_one, "two": tag_two}))
    assert result.votes["both"] == 2
    assert result.votes["solo"] == 1
    assert result.order.index("both") < result.order.index("solo")


def test_the_tiebreak_orders_within_a_vote_tier():
    result = prompt.consensus(
        ["one", "two"],
        _retriever({"one": ["high", "low"], "two": ["high", "low"]}),
    )
    assert result.votes["high"] == result.votes["low"] == 2
    assert result.order[0] == "high"


def test_equal_scores_are_ordered_by_file_hash_whatever_the_retriever_did():
    """`top_k` breaks ties by store insertion order, so a rebuilt store can
    swap two equal-scoring photos - and a swap across the MIN_SEEDS boundary
    flips a whole capture day in or out of the memory."""
    forwards = prompt.rank_hits([("bbb", 0.5), ("aaa", 0.5)])
    backwards = prompt.rank_hits([("aaa", 0.5), ("bbb", 0.5)])
    assert forwards == backwards == ["aaa", "bbb"]


def test_the_day_quorum_is_half_the_tags_rounded_up():
    assert prompt.day_quorum(1) == 1
    assert prompt.day_quorum(2) == 1
    assert prompt.day_quorum(3) == 2
    assert prompt.day_quorum(4) == 2
    assert prompt.day_quorum(6) == 3


# ---------------------------------------------------------- build_selection


def _festival_library():
    """Two days that look alike to one tag and differ on the others."""
    photos = []
    for year in (2019, 2020, 2021):
        for i in range(6):
            photos.append(_p(f"real{year}{i}", local=datetime(year, 10, 4, 12, i), people=["A B"]))
        for i in range(6):
            photos.append(_p(f"other{year}{i}", local=datetime(year, 3, 1, 12, i), people=["A B"]))
    return photos


def _seed_tags(hashes_per_tag):
    return _retriever(hashes_per_tag)


def test_a_day_with_two_hits_is_a_seed_day_and_a_day_with_one_is_not(tmp_path):
    photos = [
        _p("x1", local=datetime(2019, 10, 4, 12, 0)),
        _p("x2", local=datetime(2019, 10, 4, 12, 5)),
        _p("y1", local=datetime(2020, 5, 1, 12, 0)),
    ]
    store, index = _index(tmp_path, photos)
    try:
        query = prompt.parse("anything", index)
        build = prompt.build_selection(
            index,
            query,
            ["t"],
            _seed_tags({"t": ["x1", "x2", "y1"]}),
        )
        assert [s.iso for s in build.seed_days] == ["2019-10-04"]
    finally:
        store.close()


def test_a_seed_day_expands_to_every_photo_of_that_day(tmp_path):
    """The reason the memory has people in it. Measured: direct top-K gives 4
    of 24 shots with face tags, seed-and-expand gives 22 of 24."""
    photos = [_p(f"d{i}", local=datetime(2019, 10, 4, 12, i)) for i in range(9)]
    store, index = _index(tmp_path, photos)
    try:
        query = prompt.parse("anything", index)
        build = prompt.build_selection(index, query, ["t"], _seed_tags({"t": ["d0", "d1"]}))
        assert build.pool == 9
        assert len(build.selection.photos) == 9
    finally:
        store.close()


def test_a_blocked_hit_cannot_create_a_seed_day(tmp_path):
    """Resolution happens BEFORE day counting. Counting first would let an
    excluded person's photos vote a day into the memory."""
    photos = [
        _p("b1", local=datetime(2019, 10, 4, 12, 0), archived=True),
        _p("b2", local=datetime(2019, 10, 4, 12, 5), archived=True),
        _p("keep", local=datetime(2020, 5, 1, 12, 0)),
    ]
    store, index = _index(tmp_path, photos)
    try:
        query = prompt.parse("anything", index)
        build = prompt.build_selection(index, query, ["t"], _seed_tags({"t": ["b1", "b2", "keep"]}))
        assert build.seed_days == ()
        assert build.selection is None
    finally:
        store.close()


def test_a_day_below_the_tag_quorum_is_not_a_seed_day(tmp_path):
    """One tag's private idea of the concept must not claim a whole day - and
    through the year stratifier, a whole year-bucket of the memory."""
    photos = [
        _p("one1", local=datetime(2019, 10, 4, 12, 0)),
        _p("one2", local=datetime(2019, 10, 4, 12, 5)),
        _p("agree1", local=datetime(2020, 10, 4, 12, 0)),
        _p("agree2", local=datetime(2020, 10, 4, 12, 5)),
    ]
    store, index = _index(tmp_path, photos)
    try:
        query = prompt.parse("anything", index)
        build = prompt.build_selection(
            index,
            query,
            ["a", "b", "c"],
            _seed_tags(
                {
                    "a": ["one1", "one2", "agree1"],
                    "b": ["agree1", "agree2"],
                    "c": ["agree2"],
                }
            ),
        )
        assert [s.iso for s in build.seed_days] == ["2020-10-04"]
    finally:
        store.close()


def test_album_union_matches_all_tokens_not_any(tmp_path):
    """Any-token matches `Diwali Kali Puja 22` on the shared word "puja" and
    drags Kali Puja photos into a Durga Puja memory. That is precisely the
    confusion this module exists to prevent."""
    photos = [
        _p("a", local=datetime(2025, 9, 30, 12, 0), albums=["Durga Puja 25"]),
        _p("b", local=datetime(2022, 10, 24, 12, 0), albums=["Diwali Kali Puja 22"]),
    ]
    store, index = _index(tmp_path, photos)
    try:
        assert prompt.album_matches(index, "durga puja") == ["Durga Puja 25"]
    finally:
        store.close()


def test_a_matching_album_enters_the_pool_even_with_no_seed_day(tmp_path):
    photos = [
        _p("x1", local=datetime(2019, 10, 4, 12, 0)),
        _p("x2", local=datetime(2019, 10, 4, 12, 5)),
        _p("alb", local=datetime(2025, 9, 30, 12, 0), albums=["Durga Puja 25"]),
    ]
    store, index = _index(tmp_path, photos)
    try:
        query = prompt.parse("durga puja", index)
        build = prompt.build_selection(index, query, ["t"], _seed_tags({"t": ["x1", "x2"]}))
        assert build.albums == ("Durga Puja 25",)
        assert "alb" in {p.file_hash for p in build.selection.photos}
    finally:
        store.close()


def test_over_the_years_asks_for_year_strata_and_a_floor_of_three(tmp_path):
    """BY_YEAR, not BY_SPAN. `resolve_level` turns BY_SPAN into whatever
    granularity fits the slots - on a real pool it resolves to MONTH - and
    `min_strata` then counts months, so a promise to span years would be
    satisfied by three months of one year."""
    store, index = _index(tmp_path, _festival_library())
    try:
        query = prompt.parse("festival over the years", index)
        build = prompt.build_selection(
            index, query, ["t"], _seed_tags({"t": ["real20190", "real20191"]})
        )
        assert build.selection.stratify is strata.BY_YEAR
        assert build.selection.min_strata == 3

        span = prompt.parse("festival", index)
        build_span = prompt.build_selection(
            index, span, ["t"], _seed_tags({"t": ["real20190", "real20191"]})
        )
        assert build_span.selection.stratify is strata.BY_SPAN
        assert build_span.selection.min_strata == 1
    finally:
        store.close()


def test_the_pool_is_chronological_and_holds_each_photo_once(tmp_path):
    photos = [_p(f"d{i}", local=datetime(2019, 10, 4, 12, i)) for i in range(6)]
    photos += [_p(f"e{i}", local=datetime(2018, 10, 4, 12, i)) for i in range(6)]
    store, index = _index(tmp_path, photos)
    try:
        query = prompt.parse("anything", index)
        build = prompt.build_selection(
            index,
            query,
            ["t"],
            _seed_tags({"t": ["d0", "d1", "e0", "e1", "d0"]}),
        )
        chosen = build.selection.photos
        assert len({p.file_hash for p in chosen}) == len(chosen)
        assert chosen == sorted(chosen, key=lambda p: (p.meta.taken_at_utc, p.file_hash))
    finally:
        store.close()


def test_an_empty_pool_yields_no_selection(tmp_path):
    store, index = _index(tmp_path, [_p("a", local=datetime(2019, 10, 4, 12, 0))])
    try:
        query = prompt.parse("anything", index)
        build = prompt.build_selection(index, query, ["t"], _seed_tags({"t": []}))
        assert build.selection is None
        assert build.pool == 0
    finally:
        store.close()


def test_a_month_window_narrows_the_seeds(tmp_path):
    photos = [_p(f"o{i}", local=datetime(2019, 10, 4, 12, i)) for i in range(4)]
    photos += [_p(f"m{i}", local=datetime(2019, 5, 4, 12, i)) for i in range(4)]
    store, index = _index(tmp_path, photos)
    try:
        query = prompt.parse("anything", index)
        hits = {"t": ["m0", "m1", "o0", "o1"]}
        wide = prompt.build_selection(index, query, ["t"], _seed_tags(hits))
        narrow = prompt.build_selection(index, query, ["t"], _seed_tags(hits), months=(10,))
        assert {s.iso for s in wide.seed_days} == {"2019-05-04", "2019-10-04"}
        assert {s.iso for s in narrow.seed_days} == {"2019-10-04"}
    finally:
        store.close()


def test_a_person_narrowing_is_applied_to_the_seeds(tmp_path):
    photos = [
        _p("with1", local=datetime(2019, 10, 4, 12, 0), people=["Avyan Maiti"]),
        _p("with2", local=datetime(2019, 10, 4, 12, 5), people=["Avyan Maiti"]),
        _p("no1", local=datetime(2020, 10, 4, 12, 0)),
        _p("no2", local=datetime(2020, 10, 4, 12, 5)),
    ]
    store, index = _index(tmp_path, photos)
    try:
        query = prompt.parse("festival with Avyan Maiti", index)
        build = prompt.build_selection(
            index, query, ["t"], _seed_tags({"t": ["no1", "no2", "with1", "with2"]})
        )
        assert [s.iso for s in build.seed_days] == ["2019-10-04"]
    finally:
        store.close()


def test_a_word_that_narrowed_nothing_is_reported(tmp_path):
    """`christmas in midnapur` builds a Christmas memory from the whole
    library, and the honest line is that "midnapur" narrowed nothing."""
    store, index = _index(tmp_path, [_p("a", local=datetime(2019, 12, 25, 12, 0))])
    try:
        query = prompt.parse("christmas in midnapur", index)
        build = prompt.build_selection(index, query, ["t"], _seed_tags({"t": []}))
        assert "midnapur" in build.unmatched
    finally:
        store.close()


# ------------------------------------------------------- the guardrail seam


def test_an_excluded_person_removes_hits_without_any_filtering_here(tmp_path):
    photos = [
        _p("p1", local=datetime(2019, 10, 4, 12, 0), people=["Paramita"]),
        _p("p2", local=datetime(2019, 10, 4, 12, 5), people=["Paramita"]),
    ]
    store, index = _index(tmp_path, photos, ExclusionPolicy(people=frozenset({"Paramita"})))
    try:
        query = prompt.parse("anything", index)
        build = prompt.build_selection(index, query, ["t"], _seed_tags({"t": ["p1", "p2"]}))
        assert build.selection is None
    finally:
        store.close()


@pytest.mark.parametrize("text", ["", "   "])
def test_an_empty_prompt_normalises_to_nothing(text):
    assert prompt.normalise(text) == ""
