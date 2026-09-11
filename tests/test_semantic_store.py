"""The embedding store: round-trip, versioning, and crash recovery.

Stdlib only - these run on a CI machine with none of the optional extras.
"""

from __future__ import annotations

import sqlite3
import struct

import pytest

from rekindle.semantic.store import (
    ITEM_SIZE,
    STORE_SCHEMA_VERSION,
    EmbeddingStore,
    StoreError,
    store_root,
)


def open_store(tmp_path, dim=4, key="toy", revision="r1"):
    return EmbeddingStore(tmp_path / "s", dim=dim, model_key=key, model_revision=revision)


def test_round_trip_preserves_values(tmp_path):
    with open_store(tmp_path) as store:
        store.add_many([("aa", [1.0, 2.0, 3.0, 4.0]), ("bb", [-1.5, 0.0, 0.25, 9.0])])
        assert store.get("aa") == pytest.approx([1.0, 2.0, 3.0, 4.0])
        assert store.get("bb") == pytest.approx([-1.5, 0.0, 0.25, 9.0])
        assert store.get("missing") is None
        assert store.count() == 2


def test_rows_are_dense_and_ordered(tmp_path):
    with open_store(tmp_path) as store:
        store.add_many([(f"h{i}", [float(i)] * 4) for i in range(5)])
        assert [row for _, row in store.iter_rows()] == [0, 1, 2, 3, 4]
        assert store.ordered_hashes() == ["h0", "h1", "h2", "h3", "h4"]


def test_rewriting_a_hash_reuses_its_row(tmp_path):
    """Otherwise a re-embed grows the matrix without bound and orphans rows."""
    with open_store(tmp_path) as store:
        store.add_many([("aa", [1.0] * 4), ("bb", [2.0] * 4)])
        store.add_many([("aa", [7.0] * 4)])
        assert store.count() == 2
        assert store.get("aa") == pytest.approx([7.0] * 4)
        assert store.vectors_path.stat().st_size == 2 * 4 * ITEM_SIZE


def test_bytes_on_disk_are_little_endian_float32(tmp_path):
    """The layout is a promise to `numpy.frombuffer(..., '<f4')` in vectors.py."""
    with open_store(tmp_path) as store:
        store.add_many([("aa", [1.0, 2.0, 3.0, 4.0])])
    raw = (tmp_path / "s" / "vectors.f32").read_bytes()
    assert raw == struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)


def test_missing_reports_only_absent_hashes_in_order(tmp_path):
    with open_store(tmp_path) as store:
        store.add_many([("bb", [1.0] * 4)])
        assert store.missing(["aa", "bb", "cc", "aa"]) == ["aa", "cc"]


def test_wrong_dimension_is_refused(tmp_path):
    with open_store(tmp_path) as store, pytest.raises(StoreError, match="3 dims"):
        store.add_many([("aa", [1.0, 2.0, 3.0])])


def test_reopening_with_a_different_dim_is_refused(tmp_path):
    open_store(tmp_path, dim=4).close()
    with pytest.raises(StoreError, match="cannot share one matrix"):
        open_store(tmp_path, dim=8)


def test_reopening_with_a_different_model_is_refused(tmp_path):
    open_store(tmp_path, key="toy").close()
    with pytest.raises(StoreError, match="own store directory"):
        open_store(tmp_path, key="other")


def test_reopening_at_a_different_revision_is_refused(tmp_path):
    """Two revisions of one model are two embedding spaces wearing one name."""
    open_store(tmp_path, revision="r1").close()
    with pytest.raises(StoreError, match="not comparable"):
        open_store(tmp_path, revision="r2")


def test_creating_without_dim_is_refused(tmp_path):
    with pytest.raises(StoreError, match="needs both"):
        EmbeddingStore(tmp_path / "s")


def test_future_store_schema_is_refused(tmp_path):
    open_store(tmp_path).close()
    conn = sqlite3.connect(tmp_path / "s" / "manifest.sqlite")
    conn.execute(
        "UPDATE meta SET value = ? WHERE key = 'store_schema_version'",
        (str(STORE_SCHEMA_VERSION + 1),),
    )
    conn.commit()
    conn.close()
    with pytest.raises(StoreError, match="embedding-store schema"):
        open_store(tmp_path)


def test_orphan_tail_from_an_interrupted_run_is_truncated(tmp_path):
    """Bytes written, manifest not committed: the next open must reclaim them.

    Without this the next write allocates row N over bytes an earlier crash
    already put there, and the vector it reads back belongs to nobody.
    """
    with open_store(tmp_path) as store:
        store.add_many([("aa", [1.0] * 4)])
    with (tmp_path / "s" / "vectors.f32").open("ab") as fh:
        fh.write(struct.pack("<4f", 9.0, 9.0, 9.0, 9.0))
    with open_store(tmp_path) as store:
        assert store.vectors_path.stat().st_size == 1 * 4 * ITEM_SIZE
        store.add_many([("bb", [5.0] * 4)])
        assert store.get("bb") == pytest.approx([5.0] * 4)


def test_truncated_matrix_is_refused_not_guessed(tmp_path):
    with open_store(tmp_path) as store:
        store.add_many([("aa", [1.0] * 4), ("bb", [2.0] * 4)])
    with (tmp_path / "s" / "vectors.f32").open("r+b") as fh:
        fh.truncate(4 * ITEM_SIZE)  # one row short
    with pytest.raises(StoreError, match="truncated"):
        open_store(tmp_path)


def test_manifest_without_its_matrix_is_refused(tmp_path):
    with open_store(tmp_path) as store:
        store.add_many([("aa", [1.0] * 4)])
    (tmp_path / "s" / "vectors.f32").unlink()
    with pytest.raises(StoreError, match="does not exist"):
        open_store(tmp_path)


def test_store_root_separates_models(tmp_path):
    a = store_root(tmp_path, "clip-vit-l14")
    b = store_root(tmp_path, "siglip-so400m")
    assert a != b
    assert a.parent == b.parent == tmp_path / "semantic"


def test_empty_batch_is_a_no_op(tmp_path):
    with open_store(tmp_path) as store:
        assert store.add_many([]) == 0
        assert store.count() == 0
