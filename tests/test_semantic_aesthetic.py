"""The aesthetic head, the .pth reader it needs, and the spread rule.

The real LAION checkpoint is 3.7 MB of downloaded weights that CI will never
have, so these tests build a head whose answer is known by construction (a
chain of affine layers with hand-chosen coefficients) and check the arithmetic
and the selection policy against it.
"""

from __future__ import annotations

import pickle
import zipfile

import pytest

from rekindle.semantic.aesthetic import (
    SCORE_MAX,
    AestheticError,
    AestheticHead,
    Scored,
    pick_best,
    score_store,
)
from rekindle.semantic.store import EmbeddingStore
from rekindle.semantic.torchfile import TorchFileError, load_state_dict

np = pytest.importorskip("numpy")


def linear_head(in_dim=4, gain=1.0, bias=0.0) -> AestheticHead:
    """A one-layer head: score = gain * sum(x) + bias, on a unit vector."""
    return AestheticHead([np.full((1, in_dim), gain, dtype=np.float32)], [np.array([bias])])


def test_score_is_the_arithmetic_it_claims():
    head = linear_head(gain=2.0, bias=1.0)
    # [1,0,0,0] is already unit-norm, so score = 2*1 + 1 = 3.
    assert head.score([[1.0, 0.0, 0.0, 0.0]])[0] == pytest.approx(3.0)


def test_input_is_renormalised_before_scoring():
    """The published head expects a unit vector; a caller may not supply one."""
    head = linear_head(gain=1.0)
    a = head.score([[1.0, 0.0, 0.0, 0.0]])[0]
    b = head.score([[5.0, 0.0, 0.0, 0.0]])[0]
    assert a == pytest.approx(b)


def test_scores_are_clamped_to_the_published_range():
    assert linear_head(gain=1000.0).score([[1.0, 0, 0, 0]])[0] == SCORE_MAX
    assert linear_head(gain=-1000.0).score([[1.0, 0, 0, 0]])[0] == 0.0


def test_a_multi_layer_chain_composes():
    """Two affine layers: y = 2x + 1, then z = 3y - 2. For x=1: z = 7."""
    head = AestheticHead(
        [np.full((1, 4), 2.0, dtype=np.float32), np.array([[3.0]], dtype=np.float32)],
        [np.array([1.0]), np.array([-2.0])],
    )
    assert head.score([[1.0, 0, 0, 0]])[0] == pytest.approx(7.0)


def test_a_batch_scores_in_order():
    head = linear_head(gain=1.0)
    got = head.score([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0.5, 0.5, 0.5, 0.5]])
    assert len(got) == 3
    assert got[2] > got[0]  # sum of four halves beats one


def test_wrong_dimension_is_refused_not_broadcast():
    """Silently scoring a 1152-dim SigLIP vector with a 768-wide head would
    produce numbers, and they would be meaningless."""
    with pytest.raises(AestheticError, match="768|4-dim|takes"):
        linear_head(in_dim=4).score([[1.0, 0.0]])


def test_score_store_refuses_a_mismatched_embedding_space(tmp_path):
    from rekindle.semantic.registry import aesthetic_model

    store = EmbeddingStore(tmp_path / "s", dim=4, model_key="siglip-so400m")
    store.add_many([("aa", [1.0, 0, 0, 0])])
    head = linear_head()
    object.__setattr__(head, "spec", aesthetic_model())
    with pytest.raises(AestheticError, match="siglip-so400m"):
        score_store(store, head)


def test_score_store_returns_one_score_per_vector(tmp_path):
    store = EmbeddingStore(tmp_path / "s", dim=4, model_key="clip-vit-l14")
    store.add_many([("aa", [1.0, 0, 0, 0]), ("bb", [0, 1.0, 0, 0])])
    got = score_store(store, linear_head())
    assert {s.file_hash for s in got} == {"aa", "bb"}


def test_score_store_on_an_empty_store(tmp_path):
    store = EmbeddingStore(tmp_path / "s", dim=4, model_key="clip-vit-l14")
    assert score_store(store, linear_head()) == []


# ------------------------------------------------------------------ pick_best


def test_pick_best_without_vectors_is_plain_top_k():
    scored = [Scored(f"h{i}", float(i)) for i in range(5)]
    assert [s.file_hash for s in pick_best(scored, 2)] == ["h4", "h3"]


def test_pick_best_breaks_ties_stably():
    scored = [Scored("bb", 1.0), Scored("aa", 1.0)]
    assert [s.file_hash for s in pick_best(scored, 2)] == ["aa", "bb"]


def test_pick_best_suppresses_near_duplicates():
    """A 508-photo trip's top 40 is routinely eight scenes without this."""
    same = [1.0, 0.0, 0.0, 0.0]
    other = [0.0, 1.0, 0.0, 0.0]
    scored = [Scored("a", 9.0), Scored("b", 8.9), Scored("c", 8.0)]
    vectors = {"a": same, "b": same, "c": other}
    picked = [s.file_hash for s in pick_best(scored, 2, vectors=vectors)]
    assert picked == ["a", "c"]


def test_pick_best_backfills_rather_than_returning_short():
    """A recipe asked for k photos; duplicates beat coming up empty."""
    same = [1.0, 0.0, 0.0, 0.0]
    scored = [Scored("a", 9.0), Scored("b", 8.9), Scored("c", 8.0)]
    vectors = dict.fromkeys(("a", "b", "c"), same)
    picked = pick_best(scored, 3, vectors=vectors)
    assert len(picked) == 3


def test_pick_best_keeps_a_photo_with_no_vector():
    scored = [Scored("a", 9.0), Scored("b", 8.0)]
    picked = pick_best(scored, 2, vectors={"a": [1.0, 0, 0, 0]})
    assert {s.file_hash for s in picked} == {"a", "b"}


def test_max_similarity_controls_how_aggressive_the_spread_is():
    near = [1.0, 0.05, 0.0, 0.0]
    scored = [Scored("a", 9.0), Scored("b", 8.0)]
    vectors = {"a": [1.0, 0.0, 0.0, 0.0], "b": near}
    assert len(pick_best(scored, 2, vectors=vectors, max_similarity=0.99)) == 2
    loose = pick_best(scored, 1, vectors=vectors, max_similarity=0.5)
    assert [s.file_hash for s in loose] == ["a"]


# ------------------------------------------------------- the torch-file reader


def test_reader_rejects_a_non_zip(tmp_path):
    bad = tmp_path / "legacy.pth"
    bad.write_bytes(b"\x80\x02}q\x00.")
    with pytest.raises(TorchFileError, match="legacy format"):
        load_state_dict(bad)


def test_reader_rejects_a_zip_without_data_pkl(tmp_path):
    path = tmp_path / "x.pth"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/version", "3\n")
    with pytest.raises(TorchFileError, match="no data.pkl"):
        load_state_dict(path)


def test_reader_refuses_a_dangerous_global(tmp_path):
    """A .pth is a pickle. The allow-list is the only thing between a model
    download and arbitrary code execution."""
    path = tmp_path / "evil.pth"
    payload = pickle.dumps(_Evil(), protocol=2)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", payload)
        zf.writestr("archive/version", "3\n")
    with pytest.raises(pickle.UnpicklingError, match="refusing to unpickle"):
        load_state_dict(path)


class _Evil:
    def __reduce__(self):
        import os

        return (os.system, ("echo pwned",))


def test_head_from_a_file_with_no_layers_says_so(tmp_path):
    path = tmp_path / "wrong.pth"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", pickle.dumps({"not_a_layer": 1}, protocol=2))
        zf.writestr("archive/version", "3\n")
    with pytest.raises(AestheticError, match="not the LAION aesthetic head"):
        AestheticHead.from_file(path)


def test_layers_are_ordered_numerically_not_lexically():
    """layers.10 must come after layers.2, or the chain is scrambled."""
    import rekindle.semantic.aesthetic as module

    state = {}
    for index in (0, 2, 10):
        state[f"layers.{index}.weight"] = np.ones((1, 1), dtype=np.float32)
        state[f"layers.{index}.bias"] = np.array([float(index)], dtype=np.float32)
    original = module.load_state_dict
    module.load_state_dict = lambda _path: state
    try:
        head = AestheticHead.from_file(tmp_path_placeholder())
    finally:
        module.load_state_dict = original
    # biases applied in order 0, 2, 10 => x + 0 + 2 + 10 = x + 12
    assert head.score([[1.0]])[0] == pytest.approx(SCORE_MAX)  # clamped from 13
    assert len(head._b) == 3
    assert [float(b[0]) for b in head._b] == [0.0, 2.0, 10.0]


def tmp_path_placeholder():
    from pathlib import Path

    return Path("unused-because-load_state_dict-is-patched.pth")
