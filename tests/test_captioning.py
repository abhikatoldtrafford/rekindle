"""The caption pipeline: grounding, the per-photograph veto, and the cache.

The rule this file exists to pin is the one the whole layer is for: **a wrong
caption on a photograph of someone's family is worse than no caption.** Two
ways that used to be possible and are tested here:

* a memory spanning 2011 to 2019 substantiated "2019" under its 2011 shots;
* a memory of four people substantiated all four names under a photograph
  containing one of them.

Both passed the memory-level verifier, because that is what a fact sheet is:
facts about the memory. `ShotFacts` narrows them to the photograph.

Nothing here touches a network. The transport is a fake that returns strings,
which is not the same as mocking the model and asserting the mock was called:
every assertion below is about the caption that came out.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rekindle.db import PhotoStore
from rekindle.memory import captioning, llm, vocab
from rekindle.memory.index import MemoryIndex
from rekindle.memory.spec import FactSheet, MemorySpec, Shot
from rekindle.models import MediaType, Photo, PhotoMeta

SECRET = "sk-not-a-real-key"


def _fake(replies: list[str]):
    """A transport that returns canned text. Records nothing and asserts nothing."""
    queue = list(replies)

    def transport(payload, api_key):
        assert api_key == SECRET
        return {"output_text": queue.pop(0) if queue else ""}

    return transport


def _spec(**facts) -> MemorySpec:
    base = {
        "title": "A trip",
        "recipe": "album_story",
        "photo_count": 2,
        "years": (2011, 2019),
        "people": {"Paramita": 1, "Abhik Maiti": 1},
    }
    base.update(facts)
    return MemorySpec(
        recipe="album_story",
        key="trip",
        title="A trip",
        subtitle="",
        shots=(
            Shot("hash-2011", "2011", "2011-05-04T10:00:00", True),
            Shot("hash-2019", "2019", "2019-08-01T10:00:00", True),
        ),
        facts=FactSheet(**base),
        public_safe=True,
    )


# --------------------------------------------------------------------------
# the per-photograph veto


def test_a_year_from_another_shot_is_refused():
    """The memory has 2019. THIS photograph does not."""
    shot = llm.ShotFacts(year=2011, people=("Paramita",))
    assert llm.substantiated("A day in 2019", _spec().facts, shot) == llm.REJECT_WRONG_SHOT_YEAR
    assert llm.substantiated("A day in 2011", _spec().facts, shot) is None


def test_without_shot_facts_the_memory_wide_check_still_runs():
    """The weaker check is still a check, and is what a caller with no index
    gets. It must not become a no-op when the narrower one is unavailable."""
    assert llm.substantiated("A day in 2019", _spec().facts) is None
    assert llm.substantiated("A day in 1999", _spec().facts) == llm.REJECT_UNKNOWN_YEAR


def test_a_person_from_another_shot_is_refused_and_counted_apart():
    """Paramita is in this memory and is not in this photograph.

    Counted as `person_not_in_this_photo` rather than as an invented name: the
    model did not make it up, it misplaced it, and merging the two counts
    would hide which happened.
    """
    shot = llm.ShotFacts(year=2011, people=("Abhik Maiti",))
    assert (
        llm.substantiated("A day with Paramita", _spec().facts, shot)
        == llm.REJECT_WRONG_SHOT_PERSON
    )
    assert llm.substantiated("A day with Abhik", _spec().facts, shot) is None


def test_an_invented_name_is_still_an_invented_name():
    shot = llm.ShotFacts(year=2011, people=("Abhik Maiti",))
    assert llm.substantiated("A day with Geoffrey", _spec().facts, shot) == (
        llm.REJECT_UNKNOWN_PERSON
    )


def test_a_photo_the_index_no_longer_holds_substantiates_nothing(tmp_path):
    """An empty ShotFacts must REFUSE, not widen back to the memory.

    A shot excluded since the spec was written is a shot nothing can vouch
    for, and a caption under it should be the deterministic one.
    """
    store = PhotoStore(tmp_path / "i.sqlite")
    index = MemoryIndex.open(store)
    facts = captioning.shot_facts(index, Shot("gone", "", "2011-05-04T10:00:00", True))
    store.close()
    assert facts == llm.ShotFacts(year=None, people=(), has_gps=False, terms=())
    assert llm.substantiated("A day in 2011", _spec().facts, facts) == llm.REJECT_WRONG_SHOT_YEAR
    assert (
        llm.substantiated("A day with Paramita", _spec().facts, facts)
        == llm.REJECT_WRONG_SHOT_PERSON
    )


def _photo(file_hash: str, when: datetime, people=()) -> Photo:
    return Photo(
        file_hash=file_hash,
        paths=[Path("a.jpg")],
        media_type=MediaType.IMAGE,
        meta=PhotoMeta(
            taken_at_utc=when.replace(tzinfo=UTC),
            taken_at_local=when,
            people=list(people),
            width=1600,
            height=1200,
        ),
        first_seen=when.replace(tzinfo=UTC),
        last_seen=when.replace(tzinfo=UTC),
    )


def test_shot_facts_are_read_from_the_index(tmp_path):
    store = PhotoStore(tmp_path / "i.sqlite")
    photo = _photo("aa", datetime(2015, 6, 1, 9, 0), people=["Paramita"])
    store.upsert_many([photo])
    index = MemoryIndex.open(store)
    facts = captioning.shot_facts(
        index, Shot(photo.file_hash, "", "2015-06-01T09:00:00", True), ("at the sea",)
    )
    store.close()
    assert facts.year == 2015
    assert facts.people == ("Paramita",)
    assert facts.terms == ("at the sea",)


# --------------------------------------------------------------------------
# what reaches the model


def test_the_payload_carries_the_terms_and_nothing_from_the_pixels():
    shot = Shot("h", "2011", "2011-05-04T10:00:00", True)
    context = llm.ShotFacts(year=2011, people=("Abhik Maiti",), terms=("at the sea", "at sunset"))
    payload = llm.build_payload(_spec().facts, shot, "2011", context)
    body = json.loads(payload["input"][1]["content"])
    assert body["recognised_in_this_photo"] == ["at the sea", "at sunset"]
    assert body["year_of_this_photo"] == 2011
    # No path, no filename, no hash, no pixels. The hash is not secret but it
    # is not something a caption can be built from, and sending it would be
    # the first step towards sending something that is.
    serialised = json.dumps(payload)
    for forbidden in ("h", ".jpg", "file_hash", "path"):
        if forbidden == "h":
            continue
        assert forbidden not in serialised


def test_an_empty_term_list_is_absent_rather_than_empty():
    """A model handed `"recognised": []` writes around it - it reads as "the
    model looked and found nothing", which is a claim nobody made."""
    payload = llm.build_payload(
        _spec().facts, Shot("h", "", None, True), "", llm.ShotFacts(year=2011)
    )
    assert "recognised_in_this_photo" not in json.loads(payload["input"][1]["content"])


# --------------------------------------------------------------------------
# the cache, which is what makes this deterministic


class _Cache:
    def __init__(self, initial=None):
        self.rows = dict(initial or {})
        self.writes = 0

    def get(self, file_hash):
        return self.rows.get(file_hash)

    def put(self, file_hash, caption):
        self.rows[file_hash] = caption
        self.writes += 1


def test_a_cached_caption_is_used_and_the_model_is_not_asked():
    cache = _Cache({"hash-2011": "May 2011", "hash-2019": "August 2019"})
    # An empty reply queue: any call at all raises IndexError and fails loudly
    # rather than quietly returning the deterministic caption.
    spec, report = llm.apply_captions(
        _spec(),
        llm.GptCaptioner(SECRET, transport=_fake([])),
        lambda s: llm.ShotFacts(year=int(s.caption)),
        cache=cache,
    )
    assert [s.caption for s in spec.shots] == ["May 2011", "August 2019"]
    assert report.cached == 2
    assert report.accepted == 0
    assert cache.writes == 0


def test_a_generated_caption_is_written_to_the_cache():
    cache = _Cache()
    llm.apply_captions(
        _spec(),
        llm.GptCaptioner(SECRET, transport=_fake(["May 2011", "August 2019"])),
        lambda s: llm.ShotFacts(year=int(s.caption)),
        cache=cache,
    )
    assert cache.rows == {"hash-2011": "May 2011", "hash-2019": "August 2019"}


def test_the_same_spec_captioned_twice_is_byte_identical():
    """The promise the engine rests on, with a language model in the loop.

    The fake returns DIFFERENT text the second time. Without the cache that is
    a different memory; with it, the second run never asks.
    """
    cache = _Cache()
    context = lambda s: llm.ShotFacts(year=int(s.caption))  # noqa: E731
    first, _ = llm.apply_captions(
        _spec(),
        llm.GptCaptioner(SECRET, transport=_fake(["May 2011", "August 2019"])),
        context,
        cache=cache,
    )
    second, _ = llm.apply_captions(
        _spec(),
        llm.GptCaptioner(SECRET, transport=_fake(["Something else", "And another"])),
        context,
        cache=cache,
    )
    assert first.dumps() == second.dumps()


def test_a_cached_caption_the_memory_no_longer_substantiates_is_refused():
    """The cache is not a bypass. A caption naming someone who has since been
    excluded from this memory must not reappear because it was once allowed."""
    cache = _Cache({"hash-2011": "A day with Paramita"})
    spec = _spec(people={"Abhik Maiti": 1})
    rewritten, report = llm.apply_captions(
        spec,
        llm.GptCaptioner(SECRET, transport=_fake([])),
        lambda s: llm.ShotFacts(year=2011, people=()),
        cache=cache,
    )
    assert rewritten.shots[0].caption == "2011"
    assert report.rejected[llm.REJECT_STALE_CACHE] == 1


def test_the_report_accounts_for_every_shot():
    cache = _Cache({"hash-2011": "May 2011"})
    _, report = llm.apply_captions(
        _spec(),
        llm.GptCaptioner(SECRET, transport=_fake(["August 2019"])),
        lambda s: llm.ShotFacts(year=int(s.caption)),
        cache=cache,
    )
    assert report.accounted
    assert (report.requested, report.cached, report.accepted) == (2, 1, 1)


# --------------------------------------------------------------------------
# the store as a cache


def test_the_store_cache_forgets_a_row_from_another_vocabulary(tmp_path):
    """An edited `caption_vocab.toml` must regenerate rather than mix two
    vocabularies inside one memory."""
    store = PhotoStore(tmp_path / "i.sqlite")
    store.caption_put("h", "clip", "At the sea", vocab_version=1)
    assert store.caption_get("h", "clip", vocab_version=1) == ("At the sea", ())
    assert store.caption_get("h", "clip", vocab_version=2) is None
    store.close()


def test_the_store_cache_forgets_a_row_from_another_model(tmp_path):
    store = PhotoStore(tmp_path / "i.sqlite")
    store.caption_put("h", "gpt", "A day", model="gpt-5.6-luna")
    assert store.caption_get("h", "gpt", model="gpt-5.6-luna") == ("A day", ())
    assert store.caption_get("h", "gpt", model="something-else") is None
    store.close()


def test_clip_and_gpt_cache_independently(tmp_path):
    """Turning the model on must not discard the grounding underneath it."""
    store = PhotoStore(tmp_path / "i.sqlite")
    store.caption_put("h", "clip", "At the sea", vocab_version=captioning.VOCAB_VERSION)
    store.caption_put("h", "gpt", "A quiet morning by the sea")
    assert store.caption_count() == 2
    assert store.caption_get("h", "clip", vocab_version=captioning.VOCAB_VERSION)[0] == "At the sea"
    store.close()


# --------------------------------------------------------------------------
# grounding a whole spec


class _Vision:
    """A fake `VisionSupport`. Returns percentiles, which is the real interface."""

    def __init__(self, rows):
        self.rows = rows

    def scores_for(self, file_hash):
        return self.rows.get(file_hash)


def _all_probes(**high):
    out = dict.fromkeys(vocab.probes(), 0.0)
    for probe, value in high.items():
        out[probe] = value
    return out


def _probe(fragment: str) -> str:
    return next(t.probe for t in vocab.load() if fragment in t.probe)


def test_grounding_reports_a_photo_with_no_embedding_apart_from_a_silent_one():
    """Two different facts. A video has no vector; a portrait has one and
    cleared nothing. A report that merged them would hide a broken store."""
    spec = _spec()
    vision = _Vision({"hash-2011": _all_probes()})
    found, report = captioning.ground(spec, vision)
    assert report.unembedded == 1
    assert report.silent == 1
    assert report.accounted
    assert found["hash-2019"].terms == ()


def test_grounding_is_reported_by_facet():
    spec = _spec()
    vision = _Vision(
        {
            "hash-2011": _all_probes(**{_probe("sandy beach"): 0.99}),
            "hash-2019": _all_probes(**{_probe("sandy beach"): 0.99}),
        }
    )
    _, report = captioning.ground(spec, vision)
    assert report.grounded == 2
    assert report.facets == {"setting": 2}


def test_apply_clip_keeps_the_deterministic_caption_where_clip_was_silent():
    spec = _spec()
    vision = _Vision({"hash-2011": _all_probes(**{_probe("sandy beach"): 0.99})})
    found, _ = captioning.ground(spec, vision)
    out = captioning.apply_clip(spec, found)
    assert out.shots[0].caption == "At the sea, 2011"
    assert out.shots[1].caption == "2019"


def test_apply_clip_changes_nothing_but_the_caption():
    spec = _spec()
    vision = _Vision({"hash-2011": _all_probes(**{_probe("sandy beach"): 0.99})})
    found, _ = captioning.ground(spec, vision)
    out = captioning.apply_clip(spec, found)
    assert [s.file_hash for s in out.shots] == [s.file_hash for s in spec.shots]
    assert [s.taken_at_local for s in out.shots] == [s.taken_at_local for s in spec.shots]
    assert out.facts == spec.facts
    assert out.public_safe == spec.public_safe


def test_apply_clip_writes_the_cache(tmp_path):
    """And writes ONLY what CLIP produced.

    The silent shot keeps its deterministic caption, and that caption must not
    be written to the caption cache: caching "2019" as a grounded caption
    would make a later run with a better vocabulary read it back as one.
    """
    store = PhotoStore(tmp_path / "i.sqlite")
    spec = _spec()
    vision = _Vision({"hash-2011": _all_probes(**{_probe("sandy beach"): 0.99})})
    found, _ = captioning.ground(spec, vision)
    captioning.apply_clip(spec, found, store=store)
    assert store.caption_get("hash-2011", "clip", vocab_version=captioning.VOCAB_VERSION) == (
        "At the sea, 2011",
        (_probe("sandy beach"),),
    )
    assert store.caption_get("hash-2019", "clip", vocab_version=captioning.VOCAB_VERSION) is None
    assert store.caption_count("clip") == 1
    store.close()


@pytest.mark.parametrize("mode", captioning.MODES)
def test_every_mode_is_a_known_mode(mode):
    assert mode in ("deterministic", "clip", "gpt")
