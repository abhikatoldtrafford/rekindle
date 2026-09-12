"""Opt-in re-measurement of scenery memories against the reference library.

Skipped everywhere except a machine that has the index, an embedding store and
the semantic extra.

The number this file exists to defend is **6,419 photographs - 33.2% of the
library - with no face tag, no album anyone named and no GPS**. That number is
the entire justification for the feature and it is quoted in
`memory/scenery.py`, in `corpus/scenery.toml` and in the README. The first
time it was measured it came out as ZERO, because every photograph in a
Takeout export is in a `Photos from YYYY` folder and the measurement counted
that as a named album.

Point it at another library with REKINDLE_REAL_DATA_DIR.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rekindle.db import PhotoStore
from rekindle.memory import engine, scenery
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


@pytest.fixture(scope="module")
def index():
    store = PhotoStore(DB)
    try:
        yield MemoryIndex.open(store, load_policy(DATA))
    finally:
        store.close()


@pytest.fixture(scope="module")
def retrieve():
    pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from rekindle.semantic.encoder import load_encoder
    from rekindle.semantic.registry import embed_model
    from rekindle.semantic.search import SemanticSearch, embed_query
    from rekindle.semantic.setup import cache_dir_for
    from rekindle.semantic.store import EmbeddingStore, store_root

    spec = embed_model(None)
    root = store_root(DATA, spec.key)
    store = EmbeddingStore(
        root,
        dim=spec.dim,
        model_key=spec.key,
        model_revision=spec.pin("torch").revision if spec.torch else "",
    )
    search = SemanticSearch(store, None)
    encoder = load_encoder(spec.key, device="auto", cache_dir=cache_dir_for(DATA)).encoder

    def go(text: str, k: int):
        return [
            (h.file_hash, h.score) for h in search.search_vector(embed_query(encoder, text), k=k)
        ]

    return go


def test_about_a_third_of_the_library_is_unreachable_by_any_other_recipe(index):
    """6,419 of 19,318 rows, 33.2%, measured 2026-09-12. The brief said 6,415.

    A RANGE, because the exclusion policy and the library both move. What
    would be a real failure is this collapsing towards zero, which is what a
    measurement that counted Google's own `Photos from YYYY` folders as named
    albums produced.
    """
    rows = index.all()
    orphans = sum(1 for p in rows if scenery.orphan(p))
    share = orphans / len(rows)
    assert 0.25 <= share <= 0.40, f"{orphans}/{len(rows)} = {share:.1%}"


def test_the_orphans_are_mostly_still_images_and_therefore_reachable(index):
    """Videos are never embedded, so a scenery memory cannot reach them.
    Measured: 5,962 of the 6,419 orphans are images."""
    images = {p.file_hash for p in index.images()}
    orphans = [p for p in index.all() if scenery.orphan(p)]
    reachable = sum(1 for p in orphans if p.file_hash in images)
    assert reachable / len(orphans) >= 0.85


@pytest.mark.parametrize("key", [c.key for c in scenery.all_concepts()])
def test_every_shipped_concept_finds_something(index, retrieve, key):
    """A corpus entry that retrieves nothing on the library it was written
    against is a corpus entry with a typo in it."""
    concept = scenery.get(key)
    build = scenery.build_selection(index, concept, retrieve)
    assert build.pool > 20, f"{key}: only {build.pool} photos matched any description"
    assert build.reached >= 10, f"{key}: only {build.reached} reached the vote gate"


@pytest.mark.parametrize("key", ["sea", "mountains", "flowers", "temples", "food"])
def test_the_strong_concepts_build_a_full_memory(index, retrieve, key):
    """The five entries graded at 22 of 24 or better. `night` and `rain` are
    deliberately excluded: they are recorded in the corpus as the weak ones,
    and pinning them here would pin a result nobody is defending."""
    concept = scenery.get(key)
    build = scenery.build_selection(index, concept, retrieve)
    spec = engine.build(
        index,
        Offer(recipe=scenery.RECIPE, key=key, title=concept.title),
        selection=build.selection,
    )
    assert spec is not None
    assert len(spec.shots) >= 12, f"{key}: {len(spec.shots)} shots"
    years = {s.taken_at_local[:4] for s in spec.shots}
    assert len(years) >= scenery.MIN_YEARS


def test_a_scenery_memory_actually_reaches_the_photos_it_exists_for(index, retrieve):
    """The claim the feature rests on, measured on the memory that makes it.

    `scenery:sea` is the strongest entry in the corpus. If it were made
    entirely of photographs that already have a face tag or an album, the
    feature would be finding what other recipes already find.
    """
    concept = scenery.get("sea")
    build = scenery.build_selection(index, concept, retrieve)
    spec = engine.build(
        index,
        Offer(recipe=scenery.RECIPE, key="sea", title=concept.title),
        selection=build.selection,
    )
    orphans = sum(1 for s in spec.shots if scenery.orphan(index.get(s.file_hash)))
    assert orphans >= 4, f"only {orphans} of {len(spec.shots)} were unreachable otherwise"


def test_the_same_concept_builds_the_same_memory_twice(index, retrieve):
    """Byte-for-byte, which is what the whole engine promises and what a
    retrieval path with a floating-point ranking in it could quietly break."""
    concept = scenery.get("mountains")
    specs = []
    for _ in range(2):
        build = scenery.build_selection(index, concept, retrieve)
        specs.append(
            engine.build(
                index,
                Offer(recipe=scenery.RECIPE, key="mountains", title=concept.title),
                selection=build.selection,
            ).dumps()
        )
    assert specs[0] == specs[1]
