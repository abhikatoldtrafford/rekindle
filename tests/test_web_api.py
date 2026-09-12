"""The API layer: the prompt stream, dismissal, and parity with the CLI.

The prompt path is exercised with an INJECTED retriever rather than the real
one, for the same reason `tests/test_prompt.py` does it: `prompt.consensus`
takes retrieval as a callable, so the whole path can be driven with no torch,
no onnxruntime, no embedding store and no model download. The retriever here
is not a mock that records calls - it returns real hashes from the fixture
library, and the assertions are about which photos end up in the memory.
"""

from __future__ import annotations

import pytest

from rekindle.memory.history import MemoryState, memory_id
from rekindle.web import api
from rekindle.web.draft import ORIGIN_PROMPT
from tests.fixtures.web import ALBUM, make_library, make_workshop

PROMPT = "kashmir"


def fake_retriever(hashes):
    """`retrieve(text, k)` over a fixed ranking. Deterministic, offline."""

    def retrieve(_text: str, k: int):
        return [(h, 1.0 - i / 100) for i, h in enumerate(hashes[:k])]

    return retrieve


@pytest.fixture
def workshop(tmp_path):
    data_dir, _ = make_library(tmp_path)
    return make_workshop(tmp_path, data_dir), data_dir


# ------------------------------------------------------------- prompt stream


def test_a_prompt_memory_streams_tags_then_days_then_shots(workshop, monkeypatch):
    shop, _data_dir = workshop
    # The burst day plus three ordinary days: enough to pass MIN_SEEDS (2) and
    # the tag quorum with a single tag.
    monkeypatch.setattr(
        shop.library,
        "retriever",
        lambda: fake_retriever(["ok00", "ok01", "ok02", "burst_keep", "burst_drop"]),
    )
    events = list(api.build_events(shop, prompt=PROMPT))
    names = [e["event"] for e in events]
    assert names[-1] == "ready", events[-1]
    assert "tags" in names and "searched" in names and "found" in names

    found = next(e["data"] for e in events if e["event"] == "found")
    assert found["seed_days"], "the fake ranking must produce at least one seed day"
    session = events[-1]["data"]
    assert session["origin"]["kind"] == ORIGIN_PROMPT
    assert session["origin"]["command"] == ["rekindle", "memory", PROMPT]
    # A prompt memory's title is a query, not a fact.
    assert session["facts"]["title_substantiated"] is False


EXCLUDED_DAY = "2020-05-27"
EXCLUDED_HASHES = ("excluded", "excluded1")


def test_an_exclusion_holds_all_the_way_through_a_prompt_memory(tmp_path, monkeypatch):
    """The retriever may return anything; `MemoryIndex` is the filter.

    Paired with the test below, which shows the same two photos DO reach the
    memory when nothing excludes them - so this is not passing because the
    fixture happens to be thin.
    """
    from tests.fixtures.web import exclude_person

    data_dir, _ = make_library(tmp_path)
    exclude_person(data_dir)
    shop = make_workshop(tmp_path, data_dir)
    monkeypatch.setattr(shop.library, "retriever", lambda: fake_retriever([*EXCLUDED_HASHES]))

    events = list(api.build_events(shop, prompt="scuba diving underwater"))
    found = next(e["data"] for e in events if e["event"] == "found")
    assert found["seed_days"] == [], "an excluded photo must not vote a capture day in"
    assert events[-1]["event"] == "error"


def test_without_the_exclusion_those_same_photos_build_a_memory(tmp_path, monkeypatch):
    data_dir, _ = make_library(tmp_path)
    shop = make_workshop(tmp_path, data_dir)
    monkeypatch.setattr(shop.library, "retriever", lambda: fake_retriever([*EXCLUDED_HASHES]))

    events = list(api.build_events(shop, prompt="scuba diving underwater"))
    found = next(e["data"] for e in events if e["event"] == "found")
    assert [day["day"] for day in found["seed_days"]] == [EXCLUDED_DAY]
    assert events[-1]["event"] == "ready"
    session = events[-1]["data"]
    assert set(EXCLUDED_HASHES) <= {c["file_hash"] for c in session["candidates"]}
    # Two photos is below MIN_SHOTS, so the engine builds nothing and the
    # window opens on an empty memory with its rejected pile visible - which
    # is exactly when overruling a guardrail by hand is the right move.
    assert session["order"] == []
    assert "too few photos" in session["note"]


def test_an_archived_photo_never_enters_a_prompt_memory(tmp_path, monkeypatch):
    """Archived is not configurable, so there is no "without it" mirror: the
    check is that it is absent from the candidate pool even when the search
    ranks it first."""
    data_dir, _ = make_library(tmp_path)
    shop = make_workshop(tmp_path, data_dir)
    monkeypatch.setattr(
        shop.library, "retriever", lambda: fake_retriever(["archived", *EXCLUDED_HASHES])
    )
    events = list(api.build_events(shop, prompt="scuba diving underwater"))
    assert events[-1]["event"] == "ready"
    session = events[-1]["data"]
    assert "archived" not in {c["file_hash"] for c in session["candidates"]}
    assert "archived" not in session["order"]


def test_a_prompt_that_finds_nothing_ends_with_an_error_not_an_empty_memory(workshop, monkeypatch):
    shop, _data_dir = workshop
    monkeypatch.setattr(shop.library, "retriever", lambda: fake_retriever([]))
    # A subject that matches no album either - otherwise the album union below
    # would build a memory and there would be nothing to refuse.
    events = list(api.build_events(shop, prompt="scuba diving underwater"))
    assert events[-1]["event"] == "error"
    assert "agreement" in events[-1]["data"]["message"]


def test_a_prompt_naming_your_own_album_builds_from_it_with_no_search_hits(workshop, monkeypatch):
    """`prompt.build_selection` adds a matching album whole, unconditionally.

    Worth pinning here because it is why the test above has to use a word the
    library has never heard of: on this library "kashmir" builds a perfectly
    good memory from the album alone.
    """
    shop, _data_dir = workshop
    monkeypatch.setattr(shop.library, "retriever", lambda: fake_retriever([]))
    events = list(api.build_events(shop, prompt=PROMPT))
    assert events[-1]["event"] == "ready"
    found = next(e["data"] for e in events if e["event"] == "found")
    assert found["seed_days"] == []
    assert found["albums"] == [ALBUM]
    assert events[-1]["data"]["order"]


def test_an_album_big_enough_to_be_the_memory_skips_the_search_entirely(tmp_path, monkeypatch):
    """30 photos in one album is more than a memory holds, so the album IS the
    answer. `retriever` raises here: reaching it at all is the failure, and it
    is also what the claim "this works without the semantic extra" means."""

    def refuse():
        raise AssertionError("an album-led prompt memory must not open the store")

    data_dir, _ = make_library(tmp_path, days=30)
    shop = make_workshop(tmp_path, data_dir)
    monkeypatch.setattr(shop.library, "retriever", refuse)

    events = list(api.build_events(shop, prompt=PROMPT))
    assert events[-1]["event"] == "ready", events[-1]
    names = [e["event"] for e in events]
    assert "searched" not in names, "nothing was searched, so nothing may say it was"

    tags = next(e["data"] for e in events if e["event"] == "tags")
    assert tags["album_led"] is True
    assert tags["tags"] == []
    assert tags["albums"] == [ALBUM]
    assert tags["weak"] is False, "the weak-path warning is about a search that ran"

    found = next(e["data"] for e in events if e["event"] == "found")
    assert found["album_led"] is True
    assert found["seed_days"] == []
    assert found["albums"] == [ALBUM]
    # Still keyed on the prompt, not on the album.
    assert events[-1]["data"]["origin"]["key"] == PROMPT


def test_a_prompt_without_the_extra_says_what_to_install(workshop, monkeypatch):
    shop, _data_dir = workshop
    from rekindle.semantic import availability

    monkeypatch.setattr(
        availability,
        "probe",
        lambda: availability.Availability(
            cpu=False, gpu=False, missing_cpu=("numpy",), missing_gpu=("torch",)
        ),
    )
    events = list(api.build_events(shop, prompt=PROMPT))
    assert events[-1]["event"] == "error"
    assert "uv sync --extra semantic" in events[-1]["data"]["message"]


def test_an_empty_prompt_is_refused(workshop):
    shop, _data_dir = workshop
    events = list(api.build_events(shop, prompt="   "))
    assert events[-1]["event"] == "error"


# ---------------------------------------------------- parity with the CLI


def recipe_session(shop) -> str:
    events = list(api.build_events(shop, recipe="album_story", key=ALBUM))
    assert events[-1]["event"] == "ready", events[-1]
    return events[-1]["data"]["session_id"]


def test_rendering_records_the_memory_as_surfaced(workshop):
    """Parity with `rekindle memory`: what you have seen enters the cooldown."""
    shop, data_dir = workshop
    session_id = recipe_session(shop)
    from rekindle.db import PhotoStore

    with PhotoStore(data_dir / "rekindle.sqlite") as store:
        assert memory_id("album_story", ALBUM) not in MemoryState(store).cooling()

    api.render(shop, session_id, {"no_mp4": True})

    with PhotoStore(data_dir / "rekindle.sqlite") as store:
        assert memory_id("album_story", ALBUM) in MemoryState(store).cooling()


def test_dismissing_from_the_ui_is_the_same_permanent_act_as_the_cli(workshop):
    shop, data_dir = workshop
    session_id = recipe_session(shop)
    assert not any(o["dismissed"] for o in api.offers(shop)["offers"])

    api.dismiss(shop, session_id)

    from rekindle.db import PhotoStore

    with PhotoStore(data_dir / "rekindle.sqlite") as store:
        assert memory_id("album_story", ALBUM) in MemoryState(store).dismissed_memory_ids()
    rows = {o["key"]: o for o in api.offers(shop)["offers"]}
    assert rows[ALBUM]["dismissed"] is True


def test_offers_come_from_the_registry(workshop):
    from rekindle.memory.recipes import registered

    shop, _data_dir = workshop
    payload = api.offers(shop)
    assert payload["recipes"] == [r.name for r in registered()]
    assert "prompt" not in payload["recipes"], (
        "a prompt memory is never an offer - it must not appear in this list"
    )
    assert any(o["recipe"] == "album_story" and o["key"] == ALBUM for o in payload["offers"])


def test_status_reports_what_the_guardrails_withheld(tmp_path):
    data_dir, rows = make_library(tmp_path)
    shop = make_workshop(tmp_path, data_dir)
    payload = api.status(shop)
    assert payload["state"] == "ready"
    assert payload["withheld_by_reason"] == {"archived": 1}
    # The accounting identity: nothing vanishes without being counted.
    assert payload["photos"] + payload["withheld"] == len(rows)


def test_a_missing_session_is_a_404_not_a_crash(workshop):
    shop, _data_dir = workshop
    with pytest.raises(api.ApiError) as caught:
        shop.get("no-such-session")
    assert caught.value.status == 404


def test_music_is_chosen_by_name_from_the_music_folder_only(workshop, tmp_path):
    shop, _data_dir = workshop
    shop.music_dir.mkdir(parents=True, exist_ok=True)
    (shop.music_dir / "bed.mp3").write_bytes(b"not really an mp3")
    session_id = recipe_session(shop)

    state = api.edit(shop, session_id, {"op": "pace", "music": "bed.mp3"})
    assert state["pace"]["music"] == "bed.mp3"

    for hostile in ("../../data/rekindle.sqlite", str(tmp_path / "elsewhere.mp3")):
        with pytest.raises(api.ApiError):
            api.edit(shop, session_id, {"op": "pace", "music": hostile})


def test_excluding_one_person_does_not_hide_a_different_one_with_a_similar_name(tmp_path):
    """Measured on the real library and then pinned here.

    Excluding `Paramita` left 500 hits for a search on "paramita" - which looks
    exactly like a guardrail that failed. It is not: `policy.deny_reason` tests
    `p in self.people`, an EXACT name match, and the remaining hits were
    `Paramita Dadabhai` and `Paramita Dadu`, two different people nobody
    excluded. Zero of the 500 carried the excluded tag.

    Metadata search matches substrings, so the two behaviours will keep meeting.
    What makes it honest is that every hit says WHICH field and WHICH value
    matched, so the page shows "person Paramita Dadabhai" rather than an
    unexplained result.
    """
    from datetime import datetime

    from rekindle.db import PhotoStore
    from tests.fixtures.web import EXCLUDED_PERSON, exclude_person, jpeg, open_library, photo

    data_dir, _ = make_library(tmp_path)
    namesake = f"{EXCLUDED_PERSON} Junior"
    with PhotoStore(data_dir / "rekindle.sqlite") as store:
        store.upsert_many(
            [
                photo(
                    "namesake",
                    jpeg(tmp_path / "lib" / "namesake.jpg", colour=(10, 10, 90)),
                    local=datetime(2020, 5, 28, 9, 0),
                    people=(namesake,),
                )
            ]
        )

    before = open_library(data_dir)
    hits = before.search_metadata(EXCLUDED_PERSON, limit=50)
    assert {h.file_hash for h in hits} >= {"excluded", "excluded1", "namesake"}

    exclude_person(data_dir, EXCLUDED_PERSON)
    after = open_library(data_dir)

    remaining = after.search_metadata(EXCLUDED_PERSON, limit=50)
    found = {h.file_hash for h in remaining}
    assert "namesake" in found, "a different person must not be hidden by someone else's name"
    assert not ({"excluded", "excluded1"} & found), "the excluded person must be gone"
    # And the page can explain the difference, because the hit names the value
    # that matched rather than just saying "person".
    assert next(h.why for h in remaining if h.file_hash == "namesake") == f"person {namesake}"


def test_an_unknown_edit_is_refused(workshop):
    shop, _data_dir = workshop
    session_id = recipe_session(shop)
    with pytest.raises(api.ApiError, match="Unknown edit"):
        api.edit(shop, session_id, {"op": "delete-everything"})
