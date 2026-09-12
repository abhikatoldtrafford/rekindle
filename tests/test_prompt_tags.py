"""The festival corpus, the tag cache and the optional model layer.

The model is never called. The transport is injected, and - this is the part
that matters - the tests assert what the layer DOES with a reply, never that a
mock was called. A test that mocks the model and asserts the mock was called
proves nothing at all.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rekindle.db import PhotoStore
from rekindle.memory import festivals, prompt, tags
from rekindle.memory.index import ExclusionReport, MemoryIndex
from rekindle.memory.llm import LLMUnavailable
from rekindle.memory.policy import ExclusionPolicy
from rekindle.models import MediaType, Photo, PhotoMeta


def _p(h, *, local, people=(), albums=()):
    return Photo(
        file_hash=h,
        paths=[Path(f"/lib/{h}.jpg")],
        media_type=MediaType.IMAGE,
        meta=PhotoMeta(
            taken_at_utc=local.replace(tzinfo=UTC),
            taken_at_local=local,
            people=list(people),
        ),
        first_seen=datetime(2020, 1, 1, tzinfo=UTC),
        last_seen=datetime(2020, 1, 1, tzinfo=UTC),
        albums=list(albums),
    )


@pytest.fixture
def index(tmp_path):
    store = PhotoStore(tmp_path / "db.sqlite")
    store.upsert_many(
        [
            _p("a", local=datetime(2019, 10, 4, 12, 0), people=["Avyan Maiti"], albums=["Puri 25"]),
            _p("b", local=datetime(2025, 9, 30, 12, 0), albums=["Durga Puja 25"]),
        ]
    )
    try:
        yield MemoryIndex.open(store)
    finally:
        store.close()


def _reply(text):
    """A transport that returns one fixed Responses-API payload."""

    def transport(payload, api_key):
        return {"output_text": text}

    return transport


# ------------------------------------------------------------------- corpus


def test_the_shipped_corpus_loads_and_every_entry_has_tags():
    loaded = festivals.all_festivals()
    assert loaded
    for festival in loaded:
        assert festival.names
        assert festival.tags, f"{festival.key} has no visual tags"
        assert all(1 <= m <= 12 for m in festival.months)


def test_no_corpus_tag_is_the_festivals_own_name():
    """The name is what does not work. If a tag were just "durga puja" the
    corpus would be an elaborate way of doing the thing it exists to avoid.

    A tag may CONTAIN the name where the name is genuinely part of an
    object's English name - "a decorated christmas tree" is a description of a
    visible object, not an event name - so the rule is that a tag is never the
    bare name and is always a real description around it.
    """
    for festival in festivals.all_festivals():
        for tag in festival.tags:
            folded = prompt.normalise(tag)
            assert len(folded.split()) >= 4, f"{tag!r} is too short to be a description"
            for name in festival.names:
                assert prompt.normalise(name) != folded


def test_the_corpus_matches_the_users_own_spellings():
    assert festivals.match("durga puja").key == "durga_puja"
    assert festivals.match("kalipuja diwali celebration").key == "kali_puja"
    assert festivals.match("christmas in midnapur").key == "christmas"
    assert festivals.match("a beach holiday") is None


def test_a_corpus_name_matches_whole_words_only():
    """ "pujo" must not match inside "pujor"."""
    assert festivals.match("pujo") is not None
    assert festivals.match("pujorbari") is None


def test_the_longest_matching_name_wins():
    """`kalipuja diwali` and `diwali` are both in the same entry; a two-word
    festival must never be shadowed by a one-word one inside it."""
    assert festivals.match("kalipuja diwali").key == "kali_puja"


def test_a_corpus_entry_with_no_tags_is_refused(tmp_path):
    path = tmp_path / "f.toml"
    path.write_text(
        'version = 1\n[[festival]]\nkey = "x"\nnames = ["x"]\nmonths = [1]\ntags = []\n',
        encoding="utf-8",
    )
    festivals.load.cache_clear()
    with pytest.raises(festivals.CorpusError, match="no `tags`"):
        festivals.load(path)
    festivals.load.cache_clear()


def test_a_corpus_entry_with_an_impossible_month_is_refused(tmp_path):
    path = tmp_path / "f.toml"
    path.write_text(
        'version = 1\n[[festival]]\nkey = "x"\nnames = ["x"]\nmonths = [13]\ntags = ["a thing"]\n',
        encoding="utf-8",
    )
    festivals.load.cache_clear()
    with pytest.raises(festivals.CorpusError, match="month outside"):
        festivals.load(path)
    festivals.load.cache_clear()


# -------------------------------------------------------------------- cache


def test_the_shipped_starter_cache_is_readable_and_normalised():
    cached = tags.read_caches()
    assert cached
    for key, entry in cached.items():
        assert key == prompt.normalise(key), f"{key!r} is not in normalised form"
        assert len(entry) >= tags.MIN_TAGS


def test_the_starter_cache_carries_no_personal_data():
    """It is committed to a public repo. A key is a generic English phrase and
    a value describes what a kind of photograph looks like; neither may carry
    a year, a path or anything that looks like a filesystem."""
    from rekindle.memory.spec import contains_path_like

    for key, entry in tags.read_caches().items():
        assert not contains_path_like(key)
        assert not tags._YEAR.search(key)
        for tag in entry:
            assert not contains_path_like(tag), f"{tag!r} looks like a path"
            assert not tags._YEAR.search(tag), f"{tag!r} names a year"


def test_a_user_cache_entry_wins_over_the_starter(tmp_path, index):
    tags.write_cache_entry(tmp_path, "a beach", ["something else entirely"])
    merged = tags.read_caches(tmp_path)
    assert merged["a beach"] == ["something else entirely"]


def test_a_cache_entry_round_trips_under_the_normalised_key(tmp_path):
    path = tags.write_cache_entry(tmp_path, "  A  Beach Holiday ", ["waves on sand"])
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert "a beach holiday" in raw["entries"]


def test_a_corrupt_cache_is_treated_as_empty_not_fatal(tmp_path):
    (tmp_path / tags.USER_CACHE_NAME).write_text("{not json", encoding="utf-8")
    assert tags.read_caches(tmp_path) == tags.read_caches()


# ------------------------------------------------------------------ resolve


def test_the_corpus_answers_before_the_cache_and_brings_a_window(index):
    query = prompt.parse("durga puja over the years", index)
    resolution = tags.resolve(query)
    assert resolution.source == tags.SOURCE_CORPUS
    assert resolution.festival == "durga_puja"
    assert resolution.months == (9, 10)


def test_the_cache_answers_when_the_corpus_does_not(index):
    query = prompt.parse("a beach", index)
    resolution = tags.resolve(query)
    assert resolution.source == tags.SOURCE_CACHE
    assert resolution.months == ()


def test_with_no_source_at_all_the_prompt_itself_is_the_tag(index):
    query = prompt.parse("something nobody has described", index)
    resolution = tags.resolve(query)
    assert resolution.source == tags.SOURCE_PROMPT
    assert resolution.tags == ("something nobody has described",)


def test_the_model_is_used_only_when_corpus_and_cache_both_miss(index):
    generator = tags.TagGenerator("k", transport=_reply("a red thing; a blue thing"))
    corpus_hit = tags.resolve(prompt.parse("durga puja", index), generator=generator)
    assert corpus_hit.source == tags.SOURCE_CORPUS
    model_hit = tags.resolve(prompt.parse("a xylophone", index), generator=generator)
    assert model_hit.source == tags.SOURCE_MODEL
    assert model_hit.tags == ("a red thing", "a blue thing")


# ---------------------------------------------------- validating the reply


def test_a_tag_naming_a_year_is_dropped():
    kept, rejected = tags.parse_tags("a red thing; a photo from 1998; a blue thing")
    assert kept == ["a red thing", "a blue thing"]
    assert tags.REJECT_NAMES_A_YEAR in rejected


def test_a_tag_naming_someone_in_the_library_is_dropped():
    kept, rejected = tags.parse_tags(
        "a red thing; Avyan holding a lamp; a blue thing", people=["Avyan Maiti"]
    )
    assert kept == ["a red thing", "a blue thing"]
    assert tags.REJECT_NAMES_A_PERSON in rejected


def test_an_over_long_tag_is_dropped():
    long = "x" * (tags.MAX_TAG_CHARS + 1)
    kept, rejected = tags.parse_tags(f"a red thing; {long}; a blue thing")
    assert kept == ["a red thing", "a blue thing"]
    assert tags.REJECT_TOO_LONG in rejected


def test_a_reply_with_too_few_usable_tags_yields_none():
    kept, rejected = tags.parse_tags("just the one")
    assert kept == []
    assert tags.REJECT_TOO_FEW in rejected


def test_a_model_reply_that_survives_nothing_falls_back_to_the_prompt(index):
    generator = tags.TagGenerator("k", transport=_reply("in 1998 there was a thing"))
    resolution = tags.resolve(prompt.parse("a xylophone", index), generator=generator)
    assert resolution.source == tags.SOURCE_PROMPT
    assert resolution.tags == ("a xylophone",)


def test_the_tag_request_never_contains_the_library(index):
    """The generator is told the prompt and nothing else. Asserted on the real
    payload, not on a mock's call count."""
    seen = {}

    def transport(payload, api_key):
        seen["payload"] = payload
        return {"output_text": "a red thing; a blue thing"}

    tags.TagGenerator("k", transport=transport).tags_for("a xylophone")
    body = json.dumps(seen["payload"])
    assert "a xylophone" in body
    for forbidden in ("Avyan", "Puri 25", "Durga Puja 25", "2019"):
        assert forbidden not in body


# ---------------------------------------------------- the plausibility judge


def test_the_judge_accepts_on_ok(index):
    generator = tags.TagGenerator("k", transport=_reply("OK"))
    assert generator.plausible("durga puja", tags.vocabulary_of(index)) is None


def test_the_judge_refuses_with_a_reason(index):
    generator = tags.TagGenerator("k", transport=_reply("REFUSE: that is not language"))
    assert generator.plausible("qwertyuiop", tags.vocabulary_of(index)) == "that is not language"


def test_an_unparseable_judgement_is_never_a_refusal(index):
    """A reply this code does not understand must not become a refusal: a
    refusal the user cannot inspect is worse than no gate at all."""
    generator = tags.TagGenerator("k", transport=_reply("I think maybe possibly"))
    assert generator.plausible("durga puja", tags.vocabulary_of(index)) is None


def test_the_judge_sees_vocabulary_and_never_photo_content(index):
    seen = {}

    def transport(payload, api_key):
        seen["payload"] = payload
        return {"output_text": "OK"}

    tags.TagGenerator("k", transport=transport).plausible("durga puja", tags.vocabulary_of(index))
    body = json.dumps(seen["payload"])
    assert "Puri 25" in body, "the judge is meant to see album titles"
    assert "Avyan" not in body, "the judge must never be given a person's name"
    assert "/lib/" not in body and "\\lib\\" not in body


def test_the_vocabulary_carries_counts_not_names(index):
    vocabulary = tags.vocabulary_of(index)
    assert vocabulary.people_count == 1
    assert "Avyan Maiti" not in json.dumps(vocabulary.to_json())


def test_no_key_is_a_clear_message_not_a_traceback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMUnavailable, match="OPENAI_API_KEY"):
        tags.generator_from_env()


def test_the_generator_never_repr_s_its_key():
    assert "secret" not in repr(tags.TagGenerator("secret"))


# ------------------------------------------------- shortlisting the albums
#
# The model is never called here either. What is asserted is what the layer
# DOES with a reply.


def _album_index(tmp_path):
    store = PhotoStore(tmp_path / "albums.sqlite")
    store.upsert_many(
        [_p(f"k{i}", local=datetime(2015, 5, 15, 6, i), albums=["Leh Ladakh"]) for i in range(4)]
        + [_p("p", local=datetime(2025, 9, 6, 18, 0), albums=["Puri 25"], people=["Avyan Maiti"])]
    )
    return store, MemoryIndex.open(store)


def test_a_shortlist_reply_is_looked_up_and_never_taken_on_trust():
    titles = ["Leh Ladakh", "Puri 25"]
    names, rejected = tags.parse_album_reply("Leh Ladakh\nKashmir Trip", titles)
    assert names == ["Leh Ladakh"]
    assert rejected == [tags.REJECT_NOT_AN_ALBUM]


def test_a_shortlist_reply_is_matched_on_the_normalised_title():
    """A model that lower-cases a title has still named a real album; a model
    that made one up has not. The library's own spelling comes back."""
    names, rejected = tags.parse_album_reply("leh  ladakh", ["Leh Ladakh"])
    assert names == ["Leh Ladakh"]
    assert rejected == []


def test_none_is_an_answer_and_not_an_album():
    assert tags.parse_album_reply("NONE", ["Leh Ladakh"]) == ([], [])


def test_the_album_request_carries_the_titles_and_nothing_else(tmp_path):
    """The titles are the user's own words and the plausibility judge is
    already given them. Nothing else about the library may go with them."""
    seen = {}

    def transport(payload, api_key):
        seen["payload"] = payload
        return {"output_text": "Puri 25"}

    store, index = _album_index(tmp_path)
    try:
        titles = sorted(index.album_counts())
        tags.TagGenerator("k", transport=transport).albums_for("puri", titles)
        body = json.dumps(seen["payload"])
        assert "Puri 25" in body
        for forbidden in ("Avyan", "2015-05-15", "/lib/", "2025-09-06"):
            assert forbidden not in body
    finally:
        store.close()


def test_no_key_means_token_matching_and_never_an_error(tmp_path):
    """The whole layer is optional. With no generator and no cache the answer
    is empty, which leaves the caller exactly where it was."""
    query = prompt.parse("ladhak", MemoryIndex([], ExclusionPolicy(), ExclusionReport()))
    assert tags.shortlist_albums(query, ["Leh Ladakh"], data_dir=tmp_path).names == ()


def test_a_shortlist_is_cached_so_the_same_prompt_answers_the_same_way(tmp_path):
    calls = []

    def transport(payload, api_key):
        calls.append(1)
        return {"output_text": "Leh Ladakh"}

    store, index = _album_index(tmp_path)
    try:
        query = prompt.parse("ladhak", index)
        titles = sorted(index.album_counts())
        generator = tags.TagGenerator("k", transport=transport)
        first = tags.shortlist_albums(query, titles, data_dir=tmp_path, generator=generator)
        second = tags.shortlist_albums(query, titles, data_dir=tmp_path, generator=generator)
        assert first.names == second.names == ("Leh Ladakh",)
        assert (first.source, second.source) == (tags.SOURCE_MODEL, tags.SOURCE_CACHE)
        assert len(calls) == 1
    finally:
        store.close()


def test_an_empty_shortlist_is_cached_too(tmp_path):
    """Otherwise every prompt that names no album pays for a call, every run."""
    calls = []

    def transport(payload, api_key):
        calls.append(1)
        return {"output_text": "NONE"}

    store, index = _album_index(tmp_path)
    try:
        query = prompt.parse("scuba diving", index)
        titles = sorted(index.album_counts())
        generator = tags.TagGenerator("k", transport=transport)
        tags.shortlist_albums(query, titles, data_dir=tmp_path, generator=generator)
        again = tags.shortlist_albums(query, titles, data_dir=tmp_path, generator=generator)
        assert again.names == ()
        assert again.source == tags.SOURCE_CACHE
        assert len(calls) == 1
    finally:
        store.close()


def test_a_cached_shortlist_is_dropped_when_the_albums_change(tmp_path):
    """A cached answer to "which of THESE albums" is only an answer while the
    album list is the same one. Without the fingerprint, an album added today
    could never be shortlisted for a prompt asked yesterday."""
    calls = []

    def transport(payload, api_key):
        calls.append(1)
        return {"output_text": "Leh Ladakh"}

    store, index = _album_index(tmp_path)
    try:
        query = prompt.parse("ladhak", index)
        generator = tags.TagGenerator("k", transport=transport)
        tags.shortlist_albums(query, ["Leh Ladakh"], data_dir=tmp_path, generator=generator)
        after = tags.shortlist_albums(
            query, ["Leh Ladakh", "ladakh"], data_dir=tmp_path, generator=generator
        )
        assert after.source == tags.SOURCE_MODEL
        assert len(calls) == 2
    finally:
        store.close()


def test_the_shortlist_cache_holds_no_more_than_the_prompt_and_the_titles(tmp_path):
    store, index = _album_index(tmp_path)
    try:
        query = prompt.parse("ladhak", index)
        generator = tags.TagGenerator("k", transport=_reply("Leh Ladakh"))
        tags.shortlist_albums(query, ["Leh Ladakh"], data_dir=tmp_path, generator=generator)
        written = (tmp_path / tags.ALBUM_CACHE_NAME).read_text(encoding="utf-8")
        assert "ladhak" in written and "Leh Ladakh" in written
        assert "Avyan" not in written
    finally:
        store.close()
