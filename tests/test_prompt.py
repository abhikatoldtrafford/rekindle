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
from rekindle.memory import engine, prompt, strata
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


def test_a_prompt_fact_sheet_marks_its_title_unsubstantiated(tmp_path):
    photos = [_p(f"d{i}", local=datetime(2019, 12, 25, 12, i)) for i in range(4)]
    store, index = _index(tmp_path, photos)
    try:
        query = prompt.parse("Christmas in Midnapur", index)
        build = prompt.build_selection(index, query, ["t"], _seed_tags({"t": ["d0", "d1"]}))
        facts = build.selection.facts
        assert facts.title_substantiated is False
        # And the title is the NORMALISED prompt, so it is also lowercase -
        # two independent reasons a place name in a prompt cannot reach a
        # caption. See tests/test_llm.py for which one is load-bearing.
        assert facts.title == "christmas in midnapur"
    finally:
        store.close()


# ------------------------------------------------------------------- albums
#
# Every test below was written against a measurement on the reference library,
# and the measurements are in `docs/decision-log-prompt-memories.md`.


def _album_library():
    """A library shaped like the reference one: one album that can carry a
    memory, one that cannot, and one trip filed under three names."""
    photos = [_p(f"k{i}", local=datetime(2015, 5, 15, 6, i), albums=["Kashmir"]) for i in range(30)]
    photos += [
        _p(f"k1_{i}", local=datetime(2015, 5, 16, 6, i), albums=["Kashmir, day 1 and 2"])
        for i in range(4)
    ]
    photos += [
        _p(f"k3_{i}", local=datetime(2015, 5, 17, 6, i), albums=["Kashmir day 3"]) for i in range(3)
    ]
    photos += [
        _p(f"puri{i}", local=datetime(2025, 9, 6, 18, i), albums=["Puri 25"]) for i in range(3)
    ]
    photos += [_p(f"other{i}", local=datetime(2019, 10, 4, 12, i)) for i in range(10)]
    return photos


def test_a_framing_word_no_longer_hides_an_album(tmp_path):
    """`memories of puri` is the prompt the owner actually typed, and under
    bare all-token matching it reached NOTHING: `Puri 25` does not contain the
    words "memories" or "of". Measured on the reference library, 37 of 37
    named albums self-match from their bare title and 0 of 37 from
    `memories of <title>`."""
    store, index = _index(tmp_path, _album_library())
    try:
        assert prompt.album_matches(index, "puri") == ["Puri 25"]
        assert prompt.album_matches(index, "memories of puri") == ["Puri 25"]
        assert prompt.album_matches(index, "show me all our photos of kashmir") == [
            "Kashmir",
            "Kashmir day 3",
            "Kashmir, day 1 and 2",
        ]
    finally:
        store.close()


def test_googles_own_per_year_albums_can_never_be_matched(tmp_path):
    """`Photos from YYYY` is on every photograph. Measured on the reference
    library BEFORE this rule: the one-word prompt `photos` matched all 23 of
    them and unioned 17,004 photographs - 88% of the library - into the
    candidate pool."""
    photos = [
        _p(f"auto{i}", local=datetime(2019, 3, 4, 12, i), albums=[f"Photos from 201{i}"])
        for i in range(5)
    ]
    photos.append(_p("named", local=datetime(2019, 3, 4, 13, 0), albums=["Untitled(1)"]))
    store, index = _index(tmp_path, photos)
    try:
        assert prompt.album_matches(index, "photos") == []
        assert prompt.album_matches(index, "photos from") == []
        assert prompt.album_matches(index, "untitled") == []
    finally:
        store.close()


def test_a_prompt_of_nothing_but_framing_words_matches_no_album(tmp_path):
    store, index = _index(tmp_path, _album_library())
    try:
        assert prompt.album_matches(index, "show me some of my photos") == []
    finally:
        store.close()


def test_the_lead_floor_is_the_length_of_a_memory():
    """LEAD_MIN is not a free parameter: it is how many shots a memory holds.
    An album that can fill one on its own needs nothing inferred to complete
    it. Asserted rather than imported, because `engine` imports `prompt`."""
    assert prompt.LEAD_MIN == engine.DEFAULT_MAX_SHOTS


def test_an_album_that_can_fill_a_memory_becomes_the_memory(tmp_path):
    """The whole point of the feature. `Kashmir` holds 510 photographs in the
    reference library; nothing a model infers beats that."""
    store, index = _index(tmp_path, _album_library())
    try:
        query = prompt.parse("memories of kashmir", index)
        match = prompt.match_albums(index, query)
        assert match.led is True
        assert match.names == ("Kashmir", "Kashmir day 3", "Kashmir, day 1 and 2")
        assert match.total == 37

        def explode(tag, k):
            raise AssertionError("an album-led build must not search")

        build = prompt.build_selection(index, query, ["a tag"], explode, album_match=match)
        assert build.tags == ()
        assert build.seed_days == ()
        assert build.pool == 37
        chosen = {p.file_hash for p in build.selection.photos}
        assert not any(h.startswith("other") for h in chosen)
    finally:
        store.close()


def test_a_small_album_is_a_seed_and_the_search_still_runs(tmp_path):
    """`Puri 25` holds three photographs. Three is a memory nobody wants and
    the tags are still needed, so a small album is added to the pool rather
    than becoming it - which is exactly what this module did before."""
    store, index = _index(tmp_path, _album_library())
    try:
        query = prompt.parse("memories of puri", index)
        match = prompt.match_albums(index, query)
        assert (match.names, match.led) == (("Puri 25",), False)
        build = prompt.build_selection(
            index,
            query,
            ["t"],
            _seed_tags({"t": ["other0", "other1"]}),
            album_match=match,
        )
        assert build.tags == ("t",)
        assert [s.iso for s in build.seed_days] == ["2019-10-04"]
        chosen = {p.file_hash for p in build.selection.photos}
        assert {"puri0", "puri1", "puri2"} <= chosen
        assert "other0" in chosen
    finally:
        store.close()


def test_one_trip_under_several_album_names_is_one_cluster(tmp_path):
    """`Kashmir` / `Kashmir, day 1 and 2` / `Kashmir day 3` are one trip in
    three albums, and `album_story` can only ever tell one of them. Routing a
    prompt to that recipe would have to pick one; the union does not."""
    store, index = _index(tmp_path, _album_library())
    try:
        query = prompt.parse("kashmir", index)
        build = prompt.build_selection(index, query, [], _seed_tags({}))
        assert build.albums == ("Kashmir", "Kashmir day 3", "Kashmir, day 1 and 2")
        assert build.pool == 37
    finally:
        store.close()


def test_two_albums_holding_the_same_photo_count_it_once(tmp_path):
    """`Leh Ladakh` and `ladakh` are one trip filed twice and share 195 of
    their photographs on the reference library. Summing the two album sizes
    says 514 where the memory is built from 319, and the CLI printed both
    numbers two lines apart until this was fixed."""
    shared = [
        _p(f"both{i}", local=datetime(2018, 9, 1, 7, i), albums=["Leh Ladakh", "ladakh"])
        for i in range(5)
    ]
    only = [
        _p(f"leh{i}", local=datetime(2018, 9, 2, 7, i), albums=["Leh Ladakh"]) for i in range(2)
    ]
    store, index = _index(tmp_path, shared + only)
    try:
        query = prompt.parse("ladakh", index)
        match = prompt.match_albums(index, query)
        assert match.sizes == {"Leh Ladakh": 7, "ladakh": 5}
        assert sum(match.sizes.values()) == 12
        assert match.total == 7
        build = prompt.build_selection(index, query, [], _seed_tags({}), album_match=match)
        assert build.pool == match.total
    finally:
        store.close()


def test_a_prompt_that_an_album_answers_keeps_its_own_memory_id(tmp_path):
    """An album-led memory is still `prompt:<text>`. Building it under
    `album_story:Kashmir` instead would mean the id changed the day the album
    crossed the floor, and every dismissal of it silently stopped applying."""
    store, index = _index(tmp_path, _album_library())
    try:
        query = prompt.parse("kashmir", index)
        assert prompt.match_albums(index, query).led is True
        assert prompt.memory_key(query) == "prompt:kashmir"
    finally:
        store.close()


def test_an_album_is_narrowed_by_the_year_the_user_typed(tmp_path):
    """Before this, a matched album went into the pool whole and
    `kashmir 2019` still returned the 2015 trip."""
    photos = _album_library()
    photos += [
        _p(f"later{i}", local=datetime(2019, 5, 15, 6, i), albums=["Kashmir"]) for i in range(4)
    ]
    store, index = _index(tmp_path, photos)
    try:
        query = prompt.parse("kashmir 2019", index)
        assert query.years == (2019,)
        match = prompt.match_albums(index, query)
        assert match.sizes == {"Kashmir": 4}
        assert match.led is False
        build = prompt.build_selection(index, query, [], _seed_tags({}), album_match=match)
        assert build.pool == 4
        assert {p.file_hash for p in build.selection.photos} == {f"later{i}" for i in range(4)}
    finally:
        store.close()


def test_a_year_the_album_does_not_have_stops_it_leading(tmp_path):
    """The decision is made on the photographs that survive the user's own
    narrowings, not on the album's raw size - otherwise `kashmir 1999` leads
    with an album that contributes nothing and the memory is empty."""
    store, index = _index(tmp_path, _album_library())
    try:
        query = prompt.parse("kashmir 2025", index)
        match = prompt.match_albums(index, query)
        assert match.names == ()
        assert match.led is False
        assert match.source == prompt.ALBUM_NONE
    finally:
        store.close()


def test_the_festival_month_window_never_deletes_an_album_photo(tmp_path):
    """A corpus month window is rekindle's guess about when a festival falls;
    an album is the user's own labelling. The guess does not get to delete the
    label."""
    photos = [_p("out", local=datetime(2025, 3, 30, 12, 0), albums=["Durga Puja 25"])]
    photos += [_p(f"in{i}", local=datetime(2025, 10, 4, 12, i)) for i in range(3)]
    store, index = _index(tmp_path, photos)
    try:
        query = prompt.parse("durga puja", index)
        build = prompt.build_selection(index, query, [], _seed_tags({}), months=(9, 10))
        assert "out" in {p.file_hash for p in build.selection.photos}
    finally:
        store.close()


def test_a_shortlisted_album_the_library_does_not_have_is_never_searched(tmp_path):
    """The one hard rule of the model path: it may point at an album, never
    invent one."""
    store, index = _index(tmp_path, _album_library())
    try:
        query = prompt.parse("ladhak", index)
        match = prompt.match_albums(index, query, shortlist=["Leh Ladakh", "Kashmir"])
        assert match.names == ("Kashmir",)
        assert match.invented == ("Leh Ladakh",)
        assert match.source == prompt.ALBUM_MODEL
    finally:
        store.close()


def test_the_shortlist_cannot_override_what_the_tokens_matched(tmp_path):
    """Token matching is exact and stays authoritative: a model that answered
    `Diwali Kali Puja 22` to `durga puja` must not be able to reach the pool.
    That confusion is the reason this module exists."""
    photos = [
        _p("d", local=datetime(2025, 9, 30, 12, 0), albums=["Durga Puja 25"]),
        _p("k", local=datetime(2022, 10, 24, 12, 0), albums=["Diwali Kali Puja 22"]),
    ]
    store, index = _index(tmp_path, photos)
    try:
        query = prompt.parse("durga puja", index)
        match = prompt.match_albums(index, query, shortlist=["Diwali Kali Puja 22"])
        assert match.names == ("Durga Puja 25",)
        assert match.source == prompt.ALBUM_TOKENS
    finally:
        store.close()


def test_an_album_led_build_is_byte_identical_across_runs(tmp_path):
    store, index = _index(tmp_path, _album_library())
    try:
        query = prompt.parse("kashmir", index)

        def hashes():
            build = prompt.build_selection(index, query, [], _seed_tags({}))
            return [p.file_hash for p in build.selection.photos]

        assert hashes() == hashes()
    finally:
        store.close()
