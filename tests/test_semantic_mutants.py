"""Tests added because a mutation SURVIVED.

Forty-nine deliberate defects were injected into `rekindle.semantic`, one at a
time, and the suite was run against each. Forty-one turned it red. The eight
below did not — meaning eight lines could be deleted or inverted with the
suite still green, which is the exact defect class this project's decision log
says cost it the most.

The project's standing ruling is "untested guards get tests, not arguments",
so each of these is the test that now kills its mutant. The mutation that
survived is named in each docstring; re-injecting it must fail the test.
"""

from __future__ import annotations

import sqlite3

import pytest

from rekindle.semantic.availability import SemanticUnavailable
from rekindle.semantic.cluster import spherical_kmeans
from rekindle.semantic.encoder import EncodeError, Preprocess, _load_preprocess, _normalise
from rekindle.semantic.photos import PhotoIndexReader, ReadFilter
from rekindle.semantic.store import EmbeddingStore, StoreError
from rekindle.semantic.vectors import Matrix, load_matrix, top_k
from tests.fixtures.semantic import make_index, make_photo, write_photo

np = pytest.importorskip("numpy")


# 1 ------------------------------------------------------ photos.without_people
def test_without_people_also_rejects_a_takeout_only_tag(tmp_path):
    """MUTANT: `people = '[]'` without `AND takeout_people = '[]'`.

    Nothing in the reference index has one field populated and the other empty
    - `enrich` writes both - so the old tests could not tell the two clauses
    apart. That is precisely why the narrower clause survived: it is correct on
    today's data and wrong for any future source that writes only one field,
    and the failure mode is a person-tagged photo being treated as untagged by
    the FACE GATE.
    """
    root = tmp_path / "lib"
    photos = [
        make_photo("a" * 32, write_photo(root, "a.jpg", "red"), people=[], takeout_people=[]),
        make_photo(
            "b" * 32,
            write_photo(root, "b.jpg", "green"),
            people=[],
            takeout_people=["Abhik Maiti"],
        ),
    ]
    db = tmp_path / "data" / "rekindle.sqlite"
    make_index(db, photos)
    with PhotoIndexReader(db) as reader:
        untagged = {p.file_hash for p in reader.iter_photos(ReadFilter(without_people=True))}
    assert untagged == {"a" * 32}


# 2 ------------------------------------------------- vectors row contiguity
def test_a_gap_in_the_row_numbering_is_refused(tmp_path):
    """MUTANT: the `row != i` check deleted.

    `load_matrix` reads the whole file as one block and zips it against
    `ordered_hashes()`. If row 1 is missing, every hash from there on names the
    WRONG vector, and nothing downstream can notice. Reproduced by deleting a
    manifest row - the shape an interrupted prune or a hand-edited manifest
    would leave.
    """
    store = EmbeddingStore(tmp_path / "s", dim=4, model_key="toy")
    store.add_many([(f"h{i}", [float(i)] * 4) for i in range(4)])
    store.close()
    conn = sqlite3.connect(tmp_path / "s" / "manifest.sqlite")
    conn.execute("DELETE FROM vectors WHERE row = 1")
    conn.commit()
    conn.close()
    store = EmbeddingStore(tmp_path / "s", dim=4, model_key="toy")
    with pytest.raises(StoreError, match="not contiguous"):
        load_matrix(store)


# 3 ---------------------------------------------------------- vectors.top_k
def test_top_k_sorts_whatever_order_the_partition_hands_it(monkeypatch):
    """MUTANT: `return [int(i) for i in part]`, dropping the argsort.

    THIS MUTANT COULD NOT BE KILLED BY REAL DATA, and the reason is worth
    writing down. `numpy.argpartition` documents the order WITHIN the returned
    partition as undefined, but numpy 2.5.3's introselect happens to leave a
    small top-k already descending - checked against random arrays from n=6 to
    n=5000 and k=2..20, and against hand-built adversarial ones. So on this
    numpy the argsort is a no-op, the mutation is invisible, and a test built
    on real scores would be pinning an implementation detail of numpy rather
    than the behaviour of `top_k`.

    What `top_k` actually promises is "best first", for whatever order the
    partition produces. So the partition is made to produce a bad one: numpy's
    documented freedom is exercised deliberately, and the sort is what has to
    fix it. That is a stub for an input whose order is explicitly unspecified -
    not a mock of the code under test.
    """
    real = np.argpartition

    def shuffled(arr, kth, **kwargs):
        out = real(arr, kth, **kwargs)
        k = kth + 1
        return np.concatenate([out[:k][::-1], out[k:]])

    monkeypatch.setattr(np, "argpartition", shuffled)
    scores = np.array([0.1, 0.9, 0.3, 0.7, 0.5, 0.8, 0.2, 0.6], dtype=np.float32)
    picks = top_k(scores, 4)
    assert [float(scores[i]) for i in picks] == pytest.approx([0.9, 0.8, 0.7, 0.6])
    assert picks == [1, 5, 3, 7]


# 4 ------------------------------------------------------- cluster seeding
def _hard_matrix(n: int = 120, dim: int = 16, seed: int = 5) -> Matrix:
    """Vectors with no clean cluster structure, so the SEED decides the answer.

    Solid-colour fixtures have one correct partition that every seeding finds,
    which is why "ignore the seed" survived against them.
    """
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n, dim)).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    return Matrix(tuple(f"h{i:03d}" for i in range(n)), np.ascontiguousarray(data))


def test_the_seed_is_actually_used():
    """MUTANT: `default_rng(seed)` -> `default_rng()`.

    Two runs at one seed must agree; two runs at different seeds are ALLOWED to
    agree, but over eight seed pairs on unstructured data they will not all
    agree unless the seed is being ignored.
    """
    matrix = _hard_matrix()
    a = [c.members for c in spherical_kmeans(matrix, 8, seed=1).clusters]
    b = [c.members for c in spherical_kmeans(matrix, 8, seed=1).clusters]
    assert a == b, "the same seed must give the same partition"

    partitions = {
        tuple(c.members for c in spherical_kmeans(matrix, 8, seed=s).clusters) for s in range(8)
    }
    assert len(partitions) > 1, (
        "eight different seeds produced one partition on unstructured data; "
        "the seed is not reaching the RNG"
    )


# 5 --------------------------------------------------------- cluster min_cosine
def test_an_outlier_is_removed_from_its_cluster_not_only_listed():
    """MUTANT: the `mask & (best_sim >= min_cosine)` line deleted.

    `outliers` is computed separately at the end, so deleting the mask left the
    outlier list correct while the photo STAYED in a cluster - reported as held
    back and used anyway, the worst of both. Asserted here as a partition
    property: members and outliers must be disjoint and must together cover n.
    """
    matrix = _hard_matrix(n=60, dim=8, seed=11)
    result = spherical_kmeans(matrix, 4, seed=3, min_cosine=0.6)
    members = {h for c in result.clusters for h in c.members}
    outliers = set(result.outliers)
    assert outliers, "this matrix must produce outliers or the test is vacuous"
    assert members.isdisjoint(outliers)
    assert members | outliers == set(matrix.hashes)


# 6 ------------------------------------------------------- encoder._normalise
def test_encoder_output_is_normalised_to_unit_length():
    """MUTANT: `out.append(list(row))`, dropping the division.

    Only TorchEncoder and OnnxEncoder call `_normalise`, and neither runs in
    CI, so nothing exercised it. Cosine similarity over unnormalised vectors is
    not cosine similarity: a bright photo with a large-magnitude embedding
    outranks a better match, for every query.
    """
    got = _normalise([[3.0, 4.0], [1.0, 0.0], [-2.0, -2.0]])
    assert got[0] == pytest.approx([0.6, 0.8])
    assert got[1] == pytest.approx([1.0, 0.0])
    for row in got:
        assert sum(v * v for v in row) == pytest.approx(1.0)


def test_a_zero_vector_is_an_error_not_a_nan():
    assert pytest.raises(EncodeError, _normalise, [[0.0, 0.0]])


# 7 --------------------------------------------- encoder.load_encoder guard
def test_loading_an_encoder_with_no_extras_names_the_install_command(tmp_path):
    """MUTANT: `require(feature="Embedding")` -> `pass`.

    Without the guard the ONNX branch still fails - on huggingface_hub being
    absent - so an "exit code 3" assertion could not tell the two apart. What
    the user actually needs is the extra and the command, so that is what is
    asserted.
    """
    import importlib.util

    from rekindle.semantic.encoder import load_encoder

    if importlib.util.find_spec("onnxruntime") is not None:
        pytest.skip("onnxruntime is installed; this asserts the no-extras path")
    with pytest.raises(SemanticUnavailable) as exc:
        load_encoder("clip-vit-l14", runtime="onnx", cache_dir=tmp_path)
    message = str(exc.value)
    assert "uv sync --extra" in message
    # "Embedding" comes ONLY from `require(feature="Embedding")`. Without that
    # assertion the test passed with the guard deleted, because the next thing
    # to fail - huggingface_hub being absent - raises a message that also says
    # "uv sync --extra semantic". Naming the FEATURE is what distinguishes
    # "you need the extra to embed" from "an internal import happened to fail".
    assert "Embedding" in message
    assert "missing:" in message


# 8 ------------------------------------------------- preprocess config reading
def test_preprocessor_config_values_are_read_not_hardcoded(tmp_path):
    """MUTANT: `shortest = 224`, ignoring the config.

    The real CLIP config happens to say 224, which is also the fallback, so
    reading it and ignoring it were indistinguishable. SigLIP's is 384 and a
    future export could be anything; silently preprocessing at the wrong
    resolution degrades every embedding without erroring.
    """
    import json

    path = tmp_path / "preprocessor_config.json"
    path.write_text(
        json.dumps(
            {
                "size": {"shortest_edge": 384},
                "crop_size": {"height": 336, "width": 336},
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "rescale_factor": 1 / 127.5,
                "resample": 2,
            }
        ),
        encoding="utf-8",
    )
    pre = _load_preprocess(path)
    assert pre.size == 384
    assert pre.crop == 336
    assert pre.mean == pytest.approx((0.5, 0.5, 0.5))
    assert pre.rescale == pytest.approx(1 / 127.5)

    from PIL import Image

    from rekindle.semantic.encoder import preprocess_image

    assert preprocess_image(Image.new("RGB", (800, 600)), pre).shape == (3, 336, 336)


def test_preprocess_is_a_plain_record():
    """Frozen and comparable, so a config change is visible in a diff."""
    a = Preprocess(224, 224, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 1 / 255, 3)
    b = Preprocess(224, 224, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 1 / 255, 3)
    assert a == b
