"""Scene clustering. The colour library has ONE correct k=3 partition.

That is the point of building the fixture from solid colours: a clustering
that is subtly wrong - a broken centroid update, a seeding bug, a mis-indexed
assignment - cannot produce the right answer by luck.
"""

from __future__ import annotations

import pytest

from rekindle.semantic.cluster import (
    K_MAX,
    K_MIN,
    SCENE_VOCABULARY,
    label_clusters,
    spherical_kmeans,
    suggest_k,
)
from rekindle.semantic.store import EmbeddingStore
from rekindle.semantic.vectors import Matrix, load_matrix
from tests.fixtures.semantic import TOY_DIM, ToyEncoder, solid_image

pytest.importorskip("numpy")


def build(tmp_path, colours, per_colour=4):
    encoder = ToyEncoder()
    store = EmbeddingStore(tmp_path / "store", dim=TOY_DIM, model_key="toy")
    truth = {}
    rows = []
    for colour in colours:
        vec = encoder.encode_images([solid_image(colour)])[0]
        for i in range(per_colour):
            digest = f"{colour}{i}".ljust(32, "0")
            rows.append((digest, vec))
            truth[digest] = colour
    store.add_many(rows)
    return store, truth, encoder


def test_three_colours_land_in_three_clusters(tmp_path):
    store, truth, _ = build(tmp_path, ("red", "green", "blue"))
    result = spherical_kmeans(load_matrix(store), 3, seed=7)
    assert len(result.clusters) == 3
    for cluster in result.clusters:
        colours = {truth[h] for h in cluster.members}
        assert len(colours) == 1, f"cluster {cluster.cluster_id} mixes {colours}"
    assert sum(len(c) for c in result.clusters) == 12


def test_clustering_is_deterministic(tmp_path):
    """A user writes down "cluster 12"; it must still be cluster 12 tomorrow."""
    store, _truth, _ = build(tmp_path, ("red", "green", "blue", "yellow"))
    matrix = load_matrix(store)
    first = spherical_kmeans(matrix, 4, seed=3)
    second = spherical_kmeans(matrix, 4, seed=3)
    assert [(c.cluster_id, c.members) for c in first.clusters] == [
        (c.cluster_id, c.members) for c in second.clusters
    ]


def test_a_different_seed_may_differ_but_stays_correct(tmp_path):
    store, truth, _ = build(tmp_path, ("red", "green", "blue"))
    for seed in (0, 1, 2, 99):
        result = spherical_kmeans(load_matrix(store), 3, seed=seed)
        for cluster in result.clusters:
            assert len({truth[h] for h in cluster.members}) == 1


def test_clusters_are_ordered_largest_first(tmp_path):
    encoder = ToyEncoder()
    store = EmbeddingStore(tmp_path / "store", dim=TOY_DIM, model_key="toy")
    rows = []
    for colour, count in (("red", 6), ("green", 3), ("blue", 1)):
        vec = encoder.encode_images([solid_image(colour)])[0]
        rows.extend((f"{colour}{i}".ljust(32, "0"), vec) for i in range(count))
    store.add_many(rows)
    result = spherical_kmeans(load_matrix(store), 3, seed=1)
    assert [len(c) for c in result.clusters] == [6, 3, 1]


def test_cohesion_is_high_for_identical_members(tmp_path):
    store, _truth, _ = build(tmp_path, ("red", "blue"))
    result = spherical_kmeans(load_matrix(store), 2, seed=1)
    for cluster in result.clusters:
        assert cluster.cohesion > 0.999
        assert cluster.separation < cluster.cohesion


def test_every_photo_is_assigned_when_no_min_cosine(tmp_path):
    store, truth, _ = build(tmp_path, ("red", "green", "blue"))
    result = spherical_kmeans(load_matrix(store), 3, seed=5)
    assigned = {h for c in result.clusters for h in c.members}
    assert assigned == set(truth)
    assert result.outliers == ()


def test_min_cosine_holds_back_what_matches_nothing(tmp_path):
    store, _truth, _ = build(tmp_path, ("red", "green", "blue"))
    result = spherical_kmeans(load_matrix(store), 3, seed=5, min_cosine=0.999999)
    strict = spherical_kmeans(load_matrix(store), 1, seed=5, min_cosine=0.99)
    assert result.outliers == ()  # identical members always clear the bar
    assert strict.outliers, "k=1 over three colours must leave outliers"


def test_k_larger_than_n_is_clamped(tmp_path):
    store, _truth, _ = build(tmp_path, ("red",), per_colour=2)
    result = spherical_kmeans(load_matrix(store), 50, seed=1)
    assert result.k == 2
    assert sum(len(c) for c in result.clusters) == 2


def test_empty_matrix_clusters_to_nothing():
    result = spherical_kmeans(Matrix((), _empty()), 3, seed=1)
    assert result.clusters == ()
    assert result.n == 0


def _empty():
    import numpy as np

    return np.zeros((0, TOY_DIM), dtype=np.float32)


def test_of_reports_membership(tmp_path):
    store, truth, _ = build(tmp_path, ("red", "blue"))
    result = spherical_kmeans(load_matrix(store), 2, seed=1)
    some = next(iter(truth))
    assert result.of(some) is not None
    assert result.of("not-a-hash") is None


def test_labels_name_clusters_from_the_vocabulary(tmp_path):
    """ToyEncoder understands colour words, so the label has a right answer."""
    store, truth, encoder = build(tmp_path, ("red", "blue"))
    result = spherical_kmeans(load_matrix(store), 2, seed=1)
    vocabulary = ("a red thing", "a blue thing")
    vectors = encoder.encode_texts(vocabulary)
    labelled = label_clusters(result, vocabulary, vectors)
    for cluster in labelled.clusters:
        colour = truth[cluster.members[0]]
        assert colour in cluster.label
        assert cluster.label_score > 0.5


def test_min_label_score_suppresses_a_weak_label(tmp_path):
    store, _truth, encoder = build(tmp_path, ("red",))
    result = spherical_kmeans(load_matrix(store), 1, seed=1)
    vectors = encoder.encode_texts(("a blue thing",))
    labelled = label_clusters(result, ("a blue thing",), vectors, min_score=0.99)
    assert labelled.clusters[0].label == ""
    assert labelled.clusters[0].label_score < 0.99


def test_labelling_preserves_membership(tmp_path):
    store, _truth, encoder = build(tmp_path, ("red", "green"))
    result = spherical_kmeans(load_matrix(store), 2, seed=1)
    labelled = label_clusters(result, ("a red thing",), encoder.encode_texts(("a red thing",)))
    assert [c.members for c in labelled.clusters] == [c.members for c in result.clusters]


def test_labelling_with_no_vocabulary_is_a_no_op(tmp_path):
    store, _truth, _ = build(tmp_path, ("red",))
    result = spherical_kmeans(load_matrix(store), 1, seed=1)
    assert label_clusters(result, (), []) is result


def test_suggest_k_is_bounded():
    assert suggest_k(0) == 0
    assert suggest_k(2) == 2
    assert suggest_k(8) == 2
    # 18,201 is the live-image count on the reference library: sqrt(n/2) = 95.4.
    assert suggest_k(18201) == 95
    assert suggest_k(10**6) == K_MAX
    assert K_MIN <= suggest_k(5000) <= K_MAX


def test_scene_vocabulary_has_no_duplicates():
    assert len(SCENE_VOCABULARY) == len(set(SCENE_VOCABULARY))
