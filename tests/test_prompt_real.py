"""Opt-in regression against the reference library and a real CLIP store.

Skipped everywhere except a machine that has both. CI has no photos, no GPU
and no embeddings, and must stay green - but the numbers in
`docs/decision-log-prompt-memories.md` are the whole justification for this
feature, and a number nobody re-measures is a number that quietly goes wrong.
Six figures in this project have now been carried forward without
re-measurement and every one of them was wrong.

The assertions are RANGES, not the exact values. Re-embedding the library is
legitimately a new store and a new answer; a test that pinned exact hashes
would fail for a correct reason and teach everyone to ignore it.

Point it at another library with REKINDLE_REAL_DATA_DIR.
"""

from __future__ import annotations

import collections
import datetime as dt
import os
from pathlib import Path

import pytest

from rekindle.db import PhotoStore
from rekindle.memory import engine, festivals, prompt, tags
from rekindle.memory.index import MemoryIndex
from rekindle.memory.policy import load_policy
from rekindle.memory.recipes import Offer

DATA = Path(os.environ.get("REKINDLE_REAL_DATA_DIR", r"D:\google_photos\data"))
DB = DATA / "rekindle.sqlite"
STORE = DATA / "semantic" / "clip-vit-l14"

pytestmark = pytest.mark.skipif(
    not (DB.is_file() and (STORE / "manifest.sqlite").is_file()),
    reason="needs the reference index and a clip-vit-l14 embedding store",
)


# --------------------------------------------------------------------------
# Ground truth for the two festivals that look alike to CLIP.
#
# Vijaya Dashami and Kali Puja dates from human recall, then corroborated two
# ways before being used here: every Dashami below sits at the end of, or one
# day after, that year's dense September-October capture cluster in this
# library, and four festival-years are pinned independently by the user's own
# album titles (`Durga Puja 25` 2025-09-28..30, `Mahasaptami, 2013`
# 2013-10-10, `Diwali Kali Puja 22` 2022-10-22..25, `Diwali 25` 2025-10-19/20).
#
# This table is a MEASUREMENT AID and lives in the test, not in the shipped
# corpus. rekindle asserts no festival date anywhere.

DASHAMI = {
    2010: "2010-10-17", 2011: "2011-10-06", 2012: "2012-10-24", 2013: "2013-10-14",
    2014: "2014-10-03", 2015: "2015-10-22", 2016: "2016-10-11", 2017: "2017-09-30",
    2018: "2018-10-19", 2019: "2019-10-08", 2020: "2020-10-26", 2021: "2021-10-15",
    2022: "2022-10-05", 2023: "2023-10-24", 2024: "2024-10-12", 2025: "2025-10-02",
}  # fmt: skip
KALI = {
    2010: "2010-11-05", 2011: "2011-10-26", 2012: "2012-11-13", 2013: "2013-11-02",
    2014: "2014-10-23", 2015: "2015-11-11", 2016: "2016-10-30", 2017: "2017-10-19",
    2018: "2018-11-07", 2019: "2019-10-27", 2020: "2020-11-14", 2021: "2021-11-04",
    2022: "2022-10-24", 2023: "2023-11-12", 2024: "2024-10-31", 2025: "2025-10-20",
}  # fmt: skip


def festival_of(day: dt.date) -> str | None:
    end = DASHAMI.get(day.year)
    if end and dt.date.fromisoformat(end) - dt.timedelta(days=6) <= day <= dt.date.fromisoformat(
        end
    ) + dt.timedelta(days=1):
        return "durga"
    kali = KALI.get(day.year)
    if kali and dt.date.fromisoformat(kali) - dt.timedelta(days=1) <= day <= dt.date.fromisoformat(
        kali
    ) + dt.timedelta(days=2):
        return "kali"
    return None


# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def index():
    store = PhotoStore(DB)
    try:
        yield MemoryIndex.open(store, load_policy(DATA / "exclusions.toml"))
    finally:
        store.close()


@pytest.fixture(scope="module")
def retrieve():
    """The real CLIP encoder on the real store, cached for the module."""
    pytest.importorskip("torch")
    from rekindle.semantic.encoder import load_encoder
    from rekindle.semantic.registry import embed_model
    from rekindle.semantic.search import SemanticSearch, embed_query
    from rekindle.semantic.setup import cache_dir_for
    from rekindle.semantic.store import EmbeddingStore

    spec = embed_model(None)
    store = EmbeddingStore(
        STORE, dim=spec.dim, model_key=spec.key, model_revision=spec.pin("torch").revision
    )
    search = SemanticSearch(store, None)
    encoder = load_encoder(spec.key, device="auto", cache_dir=cache_dir_for(DATA)).encoder
    cache: dict[tuple[str, int], list[tuple[str, float]]] = {}

    def _retrieve(text: str, k: int) -> list[tuple[str, float]]:
        if (text, k) not in cache:
            hits = search.search_vector(embed_query(encoder, text), k=k)
            cache[(text, k)] = [(h.file_hash, h.score) for h in hits]
        return cache[(text, k)]

    return _retrieve


def _build(index, retrieve, text, **kw):
    query = prompt.parse(text, index)
    resolution = tags.resolve(query, data_dir=None)
    matched = festivals.match(query.subject)
    build = prompt.build_selection(
        index,
        query,
        resolution.tags,
        retrieve,
        months=resolution.months,
        festival=resolution.festival,
        known_words=matched.names if matched else (),
        **kw,
    )
    if build.selection is None:
        return build, None
    spec = engine.build(
        index,
        Offer(recipe=prompt.RECIPE, key=query.text, title=query.text),
        selection=build.selection,
        report=engine.BuildReport(offered=1),
    )
    return build, spec


def _days(spec, index):
    return [index.get(s.file_hash).meta.taken_at_local.date() for s in spec.shots]


def _tally(spec, index):
    return collections.Counter(festival_of(d) for d in _days(spec, index))


def _faces(spec, index):
    return sum(1 for s in spec.shots if any(index.get(s.file_hash).meta.people))


# ------------------------------------------------------ acceptance criteria


def test_durga_puja_over_the_years(index, retrieve):
    build, spec = _build(index, retrieve, "durga puja over the years")
    assert spec is not None
    assert len(spec.shots) == 24
    years = {s.taken_at_local[:4] for s in spec.shots}
    assert len(years) >= 8, sorted(years)
    months = collections.Counter(int(s.taken_at_local[5:7]) for s in spec.shots)
    assert months[9] + months[10] >= 20, dict(months)
    assert _faces(spec, index) >= 15
    # The user's own album is what gets 2025 in at all: those six photos are
    # portraits, not idols, and rank about 800 on any content query.
    assert "Durga Puja 25" in build.albums
    assert "2025" in years
    # Measured 24/24 the day this was written.
    assert _tally(spec, index)["durga"] >= 20


def test_kalipuja_does_not_bleed_into_durga_puja(index, retrieve):
    """THE regression. Sending these words straight to CLIP put 9 of 24 shots
    on a Durga Puja day and 4 on a genuine Kali Puja or Diwali day. With the
    tags it measured 24 and 0."""
    build, spec = _build(index, retrieve, "kalipuja diwali celebration")
    assert spec is not None
    assert len(spec.shots) == 24
    tally = _tally(spec, index)
    assert tally["durga"] <= 2, f"festival bleed is back: {dict(tally)}"
    assert tally["kali"] >= 18, dict(tally)
    assert len({s.taken_at_local[:4] for s in spec.shots}) >= 4


def test_christmas_in_midnapur_says_the_place_narrowed_nothing(index, retrieve):
    """The place half cannot work and the CLI must not pretend it ran: this
    library has 8 photos in the Medinipur cell in the week of Christmas, from
    one morning of 2019, and burst dedup takes that to about two."""
    build, spec = _build(index, retrieve, "christmas in midnapur")
    assert spec is not None
    assert "midnapur" in build.unmatched
    months = collections.Counter(int(s.taken_at_local[5:7]) for s in spec.shots)
    assert months[12] >= 18, dict(months)
    gps = sum(1 for s in spec.shots if index.get(s.file_hash).meta.gps is not None)
    assert gps <= 4, "this library has almost no GPS in December; check the claim"


def test_a_concept_this_library_does_not_hold_still_builds(index, retrieve):
    """Deliberately asserts the KNOWN HOLE.

    There is no refusal gate. `scuba diving underwater` builds a 24-shot
    memory from a library with no scuba photographs in it, and that is the
    documented behaviour, not a bug to be silently fixed with a threshold -
    eight statistics have now been tested as refusal signals and all eight
    failed. A test that writes the hole down is worth more than a hole nobody
    recorded. If this ever starts failing, something introduced a threshold;
    go and read `prompt.tag_agreement` before celebrating.
    """
    tags_for_scuba = [
        "a diver in a wetsuit and mask with an air tank underwater",
        "a stream of silver bubbles rising through blue water",
        "brightly coloured coral and fish below the surface",
    ]
    query = prompt.parse("scuba diving underwater", index)
    build = prompt.build_selection(index, query, tags_for_scuba, retrieve)
    spec = engine.build(
        index,
        Offer(recipe=prompt.RECIPE, key=query.text, title=query.text),
        selection=build.selection,
        report=engine.BuildReport(offered=1),
    )
    assert spec is not None and len(spec.shots) == 24


# --------------------------------------------------------- the design itself


def test_the_tags_are_what_separate_the_two_festivals(index, retrieve):
    """Runs the naive path beside the tag path and asserts the tags win.

    Without this, every "tag consensus reduced the bleed" claim in the docs
    rests on a number measured once by hand.
    """
    query = prompt.parse("kalipuja diwali celebration", index)
    naive = prompt.build_selection(index, query, [query.subject], retrieve)
    naive_spec = engine.build(
        index,
        Offer(recipe=prompt.RECIPE, key=query.text, title=query.text),
        selection=naive.selection,
        report=engine.BuildReport(offered=1),
    )
    naive_bleed = _tally(naive_spec, index)["durga"]

    _, tagged_spec = _build(index, retrieve, "kalipuja diwali celebration")
    tagged_bleed = _tally(tagged_spec, index)["durga"]

    assert naive_bleed >= 5, f"the naive path is meant to be bad; it scored {naive_bleed}"
    assert tagged_bleed < naive_bleed, f"naive {naive_bleed}, tagged {tagged_bleed}"


def test_seed_and_expand_is_what_puts_people_in_the_memory(index, retrieve):
    """Direct top-K gives a wall of idols: measured 2-4 of 24 shots with face
    tags. Expanding a confirmed seed day to its whole capture session gives
    20 or more."""
    query = prompt.parse("durga puja over the years", index)
    resolution = tags.resolve(query, data_dir=None)
    consensus = prompt.consensus(resolution.tags, retrieve)
    direct = index.resolve_many(consensus.order)[:150]
    direct_faces = sum(1 for p in direct[:24] if any(p.meta.people))

    _, spec = _build(index, retrieve, "durga puja over the years")
    assert _faces(spec, index) > direct_faces
    assert _faces(spec, index) >= 15


def test_every_corpus_festival_finds_something_in_this_library(index, retrieve):
    """A corpus entry whose tags find nothing here is not necessarily wrong -
    this library may simply not hold that festival - but an entry that finds
    nothing for EVERY festival would mean the tags are broken, and nothing
    else in the suite would notice."""
    found = 0
    for festival in festivals.all_festivals():
        query = prompt.parse(festival.names[0], index)
        build = prompt.build_selection(
            index, query, festival.tags, retrieve, months=festival.months
        )
        if build.seed_days:
            found += 1
    assert found >= 4, f"only {found} corpus festivals found any day in this library"
