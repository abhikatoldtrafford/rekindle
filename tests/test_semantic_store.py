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


# --------------------------------------------------------------- provenance
#
# `decode_key` is an OPAQUE string to this module. These tests therefore use
# obviously fake keys: what is under test is the comparison and the three-way
# split, not what the embedder chooses to put in the string. The end-to-end
# proof that a real orientation verdict moves a real key, and that the vector
# then changes, is in tests/test_semantic_embed.py.


def test_a_vector_records_what_it_was_computed_from(tmp_path):
    with open_store(tmp_path) as store:
        store.add_many([("aa", [1.0] * 4)], decode_keys={"aa": "v1|px224|exif"})
        assert store.decode_key_of("aa") == "v1|px224|exif"
        assert store.decode_key_of("nope") is None


def test_plan_calls_a_vector_STALE_when_its_decode_key_moved(tmp_path):
    """The whole point. Same hash, same model, different decode - redo it."""
    with open_store(tmp_path) as store:
        store.add_many([("aa", [1.0] * 4)], decode_keys={"aa": "v1|px224|exif"})
        plan = store.plan([("aa", "v1|px224|raw")])
        assert plan.stale == ["aa"]
        assert plan.fresh == 0
        assert plan.missing == []
        assert plan.to_embed() == ["aa"]


def test_plan_calls_a_vector_FRESH_when_the_key_is_unchanged(tmp_path):
    with open_store(tmp_path) as store:
        store.add_many([("aa", [1.0] * 4)], decode_keys={"aa": "k"})
        plan = store.plan([("aa", "k")])
        assert plan.fresh == 1
        assert plan.to_embed() == []
        assert plan.to_embed(include_unverified=True) == []


def test_plan_calls_a_pre_provenance_vector_UNVERIFIED_not_fresh(tmp_path):
    """NULL means "nobody recorded it", which is not the same as "it matches".

    Claiming fresh here is how the feature would quietly do nothing on every
    store that already exists.
    """
    with open_store(tmp_path) as store:
        store.add_many([("aa", [1.0] * 4)])  # no decode_keys - the old call
        plan = store.plan([("aa", "k")])
        assert plan.unverified == ["aa"]
        assert plan.fresh == 0
        assert plan.stale == []
        assert plan.to_embed() == []
        assert plan.to_embed(include_unverified=True) == ["aa"]
        assert store.unverified_hashes() == {"aa"}


def test_plan_splits_missing_stale_and_fresh_in_one_pass(tmp_path):
    with open_store(tmp_path) as store:
        store.add_many(
            [("fresh", [1.0] * 4), ("stale", [2.0] * 4), ("old", [3.0] * 4)],
            decode_keys={"fresh": "k", "stale": "other"},
        )
        plan = store.plan(
            [("fresh", "k"), ("stale", "k"), ("old", "k"), ("new", "k"), ("new", "k")]
        )
        assert plan.missing == ["new"]
        assert plan.stale == ["stale"]
        assert plan.unverified == ["old"]
        assert plan.fresh == 1
        assert plan.duplicates == 1
        assert plan.to_embed() == ["new", "stale"]


def test_rewriting_a_vector_without_a_key_CLEARS_the_recorded_one(tmp_path):
    """A caller that will not say what it decoded has made the row unverified.

    Carrying the old key forward would assert a provenance nobody vouched for,
    and the row would then read as fresh against a decode that never ran.
    """
    with open_store(tmp_path) as store:
        store.add_many([("aa", [1.0] * 4)], decode_keys={"aa": "k"})
        store.add_many([("aa", [9.0] * 4)])
        assert store.decode_key_of("aa") is None
        assert store.plan([("aa", "k")]).unverified == ["aa"]


def test_a_store_written_before_the_decode_key_column_still_opens(tmp_path):
    """`CREATE TABLE IF NOT EXISTS` does nothing to a table that exists, so a
    column added to `_SCHEMA` never reaches a store somebody already has.

    Opening one must migrate it, not refuse it and not corrupt it: refusing
    would demand the five-hour rebuild this feature exists to avoid.
    """
    root = tmp_path / "legacy"
    root.mkdir()
    conn = sqlite3.connect(root / "manifest.sqlite")
    conn.executescript(
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
        "CREATE TABLE vectors (file_hash TEXT PRIMARY KEY,"
        " row INTEGER NOT NULL UNIQUE, embedded_at TEXT NOT NULL);"
    )
    conn.executemany(
        "INSERT INTO meta(key, value) VALUES(?, ?)",
        [
            ("store_schema_version", str(STORE_SCHEMA_VERSION)),
            ("dim", "4"),
            ("model_key", "toy"),
            ("model_revision", "r1"),
            ("normalized", "1"),
            ("created_at", "2026-01-01T00:00:00+00:00"),
        ],
    )
    conn.execute("INSERT INTO vectors VALUES('aa', 0, '2026-01-01T00:00:00+00:00')")
    conn.commit()
    conn.close()
    (root / "vectors.f32").write_bytes(struct.pack("<4f", 1.0, 2.0, 3.0, 4.0))

    with EmbeddingStore(root, dim=4, model_key="toy", model_revision="r1") as store:
        assert store.count() == 1
        assert store.get("aa") == pytest.approx([1.0, 2.0, 3.0, 4.0])
        assert store.unverified_hashes() == {"aa"}
        # And the migrated store can record provenance from here on.
        store.add_many([("bb", [5.0] * 4)], decode_keys={"bb": "k"})
        assert store.decode_key_of("bb") == "k"


def test_the_migration_does_not_bump_the_schema_version(tmp_path):
    """Bumping it would make every existing store refuse to open. The vectors
    file is byte-identical and every statement an older reader issues is still
    valid against the wider table, so there is nothing for it to misread."""
    with open_store(tmp_path) as store:
        assert store.get_meta("store_schema_version") == str(STORE_SCHEMA_VERSION)
        store.add_many([("aa", [1.0] * 4)], decode_keys={"aa": "k"})
    # An "older reader": the exact SQL store.py shipped before the column.
    conn = sqlite3.connect(tmp_path / "s" / "manifest.sqlite")
    conn.row_factory = sqlite3.Row
    rows = list(conn.execute("SELECT file_hash, row FROM vectors ORDER BY row"))
    conn.execute(
        "INSERT INTO vectors(file_hash, row, embedded_at) VALUES(?, ?, ?) "
        "ON CONFLICT(file_hash) DO UPDATE SET embedded_at = excluded.embedded_at",
        ("bb", 1, "later"),
    )
    conn.commit()
    conn.close()
    assert [r["file_hash"] for r in rows] == ["aa"]
