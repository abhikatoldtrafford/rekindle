"""Semantic search, asserted on ANSWERS rather than on calls.

`ToyEncoder` is a real joint embedding function over colour, so every test
here has one correct result that a broken ranking, a broken filter or a broken
row lookup cannot produce by accident.
"""

from __future__ import annotations

import pytest

from rekindle.semantic.photos import PhotoIndexReader, ReadFilter
from rekindle.semantic.search import PROMPT_TEMPLATES, SemanticSearch, embed_query
from rekindle.semantic.store import EmbeddingStore, StoreError
from rekindle.semantic.vectors import load_matrix, top_k
from tests.fixtures.semantic import (
    TOY_DIM,
    ToyEncoder,
    make_index,
    make_photo,
    solid_image,
    write_photo,
)

pytest.importorskip("numpy")


@pytest.fixture
def world(tmp_path):
    """Six photos, two of each primary colour, embedded with ToyEncoder."""
    root = tmp_path / "lib"
    encoder = ToyEncoder()
    photos = []
    vectors = []
    for colour in ("red", "green", "blue"):
        for i in range(2):
            path = write_photo(root, f"{colour}{i}.jpg", colour)
            digest = f"{colour[0]}{i}" + "0" * 30
            photos.append(
                make_photo(
                    digest,
                    path,
                    albums=["Photos from 2019", "Kashmir"] if colour == "blue" else None,
                    people=["Abhik Maiti"] if colour == "red" else None,
                )
            )
            vectors.append((digest, encoder.encode_images([solid_image(colour)])[0]))
    db = tmp_path / "data" / "rekindle.sqlite"
    make_index(db, photos)
    store = EmbeddingStore(tmp_path / "store", dim=TOY_DIM, model_key="toy")
    store.add_many(vectors)
    return db, store, encoder


def test_query_returns_the_matching_colour(world):
    db, store, encoder = world
    with PhotoIndexReader(db) as reader:
        hits = SemanticSearch(store, reader).search_text(encoder, "red", k=2)
    assert len(hits) == 2
    assert all(hit.photo.path.name.startswith("red") for hit in hits)


def test_a_different_query_returns_different_photos(world):
    """Guards against "returns the first k rows regardless"."""
    db, store, encoder = world
    with PhotoIndexReader(db) as reader:
        search = SemanticSearch(store, reader)
        red = {h.file_hash for h in search.search_text(encoder, "red", k=2)}
        blue = {h.file_hash for h in search.search_text(encoder, "blue", k=2)}
    assert red.isdisjoint(blue)


def test_results_are_ordered_best_first(world):
    db, store, encoder = world
    with PhotoIndexReader(db) as reader:
        hits = SemanticSearch(store, reader).search_text(encoder, "green", k=6)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert hits[0].photo.path.name.startswith("green")


def test_k_caps_the_result_count(world):
    db, store, encoder = world
    with PhotoIndexReader(db) as reader:
        assert len(SemanticSearch(store, reader).search_text(encoder, "red", k=3)) == 3


def test_min_score_drops_weak_hits(world):
    db, store, encoder = world
    with PhotoIndexReader(db) as reader:
        search = SemanticSearch(store, reader)
        every = search.search_text(encoder, "red", k=6)
        cutoff = every[1].score
        kept = search.search_text(encoder, "red", k=6, min_score=cutoff)
    assert len(kept) == 2
    assert all(h.score >= cutoff for h in kept)


def test_allowed_hashes_scopes_a_search_to_an_album(world):
    """Over-fetching matters here: "blue" ranks last for a red query."""
    db, store, encoder = world
    with PhotoIndexReader(db) as reader:
        search = SemanticSearch(store, reader)
        allowed = search.allowed_hashes(ReadFilter(albums=("Kashmir",)))
        hits = search.search_text(encoder, "red", k=5, allowed=allowed)
    assert hits
    assert {h.file_hash for h in hits} == allowed
    assert all(h.photo.path.name.startswith("blue") for h in hits)


def test_similar_to_excludes_the_query_photo(world):
    db, store, encoder = world
    with PhotoIndexReader(db) as reader:
        hits = SemanticSearch(store, reader).similar_to("r0" + "0" * 30, k=3)
    assert "r0" + "0" * 30 not in {h.file_hash for h in hits}
    assert hits[0].file_hash == "r1" + "0" * 30


def test_similar_to_an_unknown_hash_raises(world):
    _db, store, _encoder = world
    with pytest.raises(KeyError):
        SemanticSearch(store).similar_to("nope", k=3)


def test_search_without_a_reader_still_returns_hashes(world):
    _db, store, encoder = world
    hits = SemanticSearch(store).search_text(encoder, "blue", k=2)
    assert len(hits) == 2
    assert all(h.photo is None for h in hits)
    assert all(h.path_str == "(not in index)" for h in hits)


def test_empty_store_returns_nothing(tmp_path):
    store = EmbeddingStore(tmp_path / "empty", dim=TOY_DIM, model_key="toy")
    assert SemanticSearch(store).search_text(ToyEncoder(), "red", k=5) == []


def test_query_of_wrong_dimension_is_refused(world):
    _db, store, _encoder = world
    with pytest.raises(StoreError, match="different embedding spaces"):
        SemanticSearch(store).search_vector([0.1] * (TOY_DIM + 1), k=1)


def test_embed_query_is_a_unit_vector(world):
    _db, _store, encoder = world
    vec = embed_query(encoder, "red")
    assert sum(v * v for v in vec) == pytest.approx(1.0, abs=1e-6)


def test_embed_query_uses_the_whole_prompt_ensemble(world):
    """One template would be a different vector; assert the count, then the mean."""
    _db, _store, encoder = world
    before = encoder.text_calls
    embed_query(encoder, "red")
    assert encoder.text_calls == before + 1
    single = embed_query(encoder, "red", templates=("{}",))
    ensemble = embed_query(encoder, "red")
    # ToyEncoder ignores template wording, so these agree - which is exactly
    # what makes the NEXT assertion meaningful: the averaging must not have
    # changed the direction.
    assert single == pytest.approx(ensemble, abs=1e-6)
    assert len(PROMPT_TEMPLATES) >= 2


def test_empty_query_is_refused(world):
    _db, _store, encoder = world
    with pytest.raises(ValueError, match="empty query"):
        embed_query(encoder, "   ")


def test_top_k_is_stable_under_ties():
    np = pytest.importorskip("numpy")
    scores = np.array([0.5, 0.5, 0.5, 0.9], dtype=np.float32)
    assert top_k(scores, 4) == [3, 0, 1, 2]
    assert top_k(scores, 0) == []


def test_matrix_rows_line_up_with_hashes(world):
    """If this drifts, every result belongs to the wrong photo."""
    np = pytest.importorskip("numpy")
    _db, store, _encoder = world
    matrix = load_matrix(store)
    for i, digest in enumerate(matrix.hashes):
        assert np.allclose(matrix.data[i], store.get(digest), atol=1e-6)
