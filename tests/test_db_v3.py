"""Schema v3 and v4: the fingerprint columns and the migration ladder.

The v1 -> v3 path is covered in test_db.py (that file already owns the v1
fixture). What is here is everything v3 adds: the v2 rung, the ladder's
refusal to touch an unknown version, round-tripping the three new fields, and
the merge rule that stops a re-index destroying a six-minute pass.
"""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rekindle.db import SCHEMA_VERSION, FingerprintRow, PhotoStore
from rekindle.models import MediaType, Photo, PhotoMeta, TzSource, merge_meta

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

# Every column a v2 database had. Written out rather than generated so that a
# change to _SCHEMA cannot silently change what "v2" means in this test.
_V2_CREATE = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE photos (
    file_hash TEXT PRIMARY KEY, media_type TEXT NOT NULL,
    paths TEXT NOT NULL, albums TEXT NOT NULL, edited_of TEXT,
    first_seen TEXT NOT NULL, last_seen TEXT NOT NULL,
    taken_at_utc TEXT, taken_at_local TEXT, tz_source TEXT NOT NULL,
    gps_lat REAL, gps_lon REAL, gps_alt REAL, people TEXT NOT NULL,
    face_regions TEXT NOT NULL, keywords TEXT NOT NULL,
    description TEXT, favorite INTEGER NOT NULL DEFAULT 0,
    camera_make TEXT, camera_model TEXT, width INTEGER, height INTEGER,
    source TEXT NOT NULL DEFAULT 'folder',
    metadata_conflict INTEGER NOT NULL DEFAULT 0,
    exif_taken_at_utc TEXT, takeout_people TEXT NOT NULL DEFAULT '[]',
    archived INTEGER NOT NULL DEFAULT 0, trashed INTEGER NOT NULL DEFAULT 0,
    sidecar_match TEXT NOT NULL DEFAULT 'none'
);
CREATE TABLE photo_paths (
    file_hash TEXT NOT NULL, path TEXT NOT NULL,
    name_cf TEXT NOT NULL, parent TEXT NOT NULL,
    PRIMARY KEY (file_hash, path)
);
"""


def _write_v2_database(path: Path, *, version: str = "2") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(_V2_CREATE)
    conn.execute("INSERT INTO meta(key, value) VALUES('schema_version', ?)", (version,))
    conn.execute(
        "INSERT INTO photos (file_hash, media_type, paths, albums, first_seen,"
        " last_seen, tz_source, people, face_regions, keywords, description,"
        " archived, sidecar_match, width, height) VALUES"
        " ('v2row', 'image', '[\"/lib/a.jpg\"]', '[\"Kashmir\"]',"
        " '2020-01-01T00:00:00+00:00', '2020-01-01T00:00:00+00:00',"
        " 'exif_offset', '[\"Abhik Maiti\"]', '[]', '[]', 'a caption',"
        " 1, 'exact', 4000, 3000)"
    )
    conn.commit()
    conn.close()


def _photo(h="abc", **kw) -> Photo:
    return Photo(
        file_hash=h,
        paths=kw.pop("paths", [Path("/photos/a.jpg")]),
        media_type=kw.pop("media_type", MediaType.IMAGE),
        meta=kw.pop("meta", PhotoMeta.empty()),
        first_seen=T0,
        last_seen=T0,
        **kw,
    )


def test_v2_database_migrates_to_v5_keeping_every_prior_value(tmp_path):
    """The additive promise: a v2 row keeps everything it had and gains the
    new columns as NULLs. If ALTER TABLE were ever swapped for a table
    rebuild, this is the test that would notice the data loss."""
    db = tmp_path / "db.sqlite"
    _write_v2_database(db)

    with PhotoStore(db) as store:
        assert store.schema_version() == SCHEMA_VERSION == 5
        photo = store.get("v2row")

    assert photo is not None
    # Every v2 field, not just one: a rebuild that dropped a column would
    # otherwise pass on whichever field the test happened to check.
    assert photo.albums == ["Kashmir"]
    assert photo.meta.people == ["Abhik Maiti"]
    assert photo.meta.description == "a caption"
    assert photo.meta.archived is True
    assert photo.meta.tz_source is TzSource.EXIF_OFFSET
    assert (photo.meta.width, photo.meta.height) == (4000, 3000)
    assert photo.sidecar_match == "exact"
    # ...and the new columns exist, empty.
    assert photo.meta.phash is None
    assert photo.meta.sharpness is None
    assert photo.meta.phash_error is None
    assert photo.meta.colour is None


def test_a_current_database_opens_without_migrating(tmp_path):
    db = tmp_path / "db.sqlite"
    with PhotoStore(db) as store:
        store.upsert_many([_photo()])
    with PhotoStore(db) as store:
        assert store.schema_version() == SCHEMA_VERSION
        assert store.count() == 1


def test_a_v3_database_gains_the_colour_column(tmp_path):
    """The rung added for the diversity signal. A v3 index already holds
    hashes; it must gain the histogram without a full re-index."""
    db = tmp_path / "db.sqlite"
    _write_v2_database(db)
    with PhotoStore(db) as store:  # v2 -> v5
        store.set_fingerprints([FingerprintRow("v2row", phash=7, colour="ab" * 64)])
        assert store.get("v2row").meta.colour == "ab" * 64


# --------------------------------------------------------------------------
# v5: the migration that deletes measurements on purpose


def _fingerprinted_v4(db):
    """A v4 database holding one measured row and one row that failed."""
    _write_v2_database(db)
    with PhotoStore(db) as store:
        assert store.schema_version() == 5  # v2 -> v5 on open
        store.set_fingerprints(
            [
                FingerprintRow(
                    "v2row",
                    phash=7,
                    sharpness=9.5,
                    brightness=110.0,
                    colour="ab" * 64,
                    width=4000,
                    height=3000,
                )
            ]
        )
    # Wind it back to v4 with the v4-scale values in place, which is exactly
    # what an index fingerprinted by M2 looks like on disk.
    conn = sqlite3.connect(db)
    conn.execute("UPDATE meta SET value = '4' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()


def test_v5_clears_every_measurement_taken_on_the_old_scale(tmp_path):
    """The old sharpness was a mean gradient in [0, 25] and the new one is a
    ratio in [0, 1]. Every use of the column compares one row against another,
    so a database holding both scales does not fail - it silently ranks every
    stale row above every fresh one. Leaving one of the four behind is enough
    to corrupt a comparison, so all four are asserted."""
    db = tmp_path / "db.sqlite"
    _fingerprinted_v4(db)

    with PhotoStore(db) as store:
        assert store.schema_version() == 5
        meta = store.get("v2row").meta

    assert meta.sharpness is None
    assert meta.phash is None
    assert meta.brightness is None
    assert meta.colour is None


def test_v5_keeps_the_dimensions_it_did_not_invalidate(tmp_path):
    """width/height do not depend on the measure, and the v3 pass repaired
    ~2,475 of them. Clearing them would blind the resolution floor until the
    re-run finished, so they must survive."""
    db = tmp_path / "db.sqlite"
    _fingerprinted_v4(db)
    with PhotoStore(db) as store:
        photo = store.get("v2row")
    assert (photo.meta.width, photo.meta.height) == (4000, 3000)


def test_v5_leaves_a_recorded_failure_alone(tmp_path):
    """`phash_error` is finished work: why a file could not be decoded does
    not change with the measure. Clearing it would put every undecodable file
    back in the queue on every upgrade, which is the thing the resume
    predicate exists to prevent."""
    db = tmp_path / "db.sqlite"
    _write_v2_database(db)
    with PhotoStore(db) as store:
        store.set_fingerprints([FingerprintRow("v2row", error="undecodable")])
    conn = sqlite3.connect(db)
    conn.execute("UPDATE meta SET value = '4' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()

    with PhotoStore(db) as store:
        assert store.get("v2row").meta.phash_error == "undecodable"
        # ...and it is still excluded from the re-run.
        assert [p.file_hash for p in store.iter_unfingerprinted()] == []


def test_v5_puts_the_cleared_rows_back_in_the_fingerprint_queue(tmp_path):
    """Clearing the columns is only half the migration. If the resume
    predicate did not pick the row up again the index would sit permanently
    unfingerprinted, which reads as an empty library rather than an error."""
    db = tmp_path / "db.sqlite"
    _fingerprinted_v4(db)
    with PhotoStore(db) as store:
        assert [p.file_hash for p in store.iter_unfingerprinted()] == ["v2row"]


def test_a_newer_schema_is_refused_rather_than_downgraded(tmp_path):
    db = tmp_path / "db.sqlite"
    _write_v2_database(db, version="9")
    with pytest.raises(RuntimeError, match="Upgrade rekindle"):
        PhotoStore(db)


def test_an_unknown_version_is_left_alone_not_stamped(tmp_path):
    """The ladder walks KNOWN rungs only. A version with no step entry stops
    the loop, so __init__'s explicit error fires instead of the database
    being silently declared current."""
    db = tmp_path / "db.sqlite"
    _write_v2_database(db, version="99")
    with pytest.raises(RuntimeError, match="Upgrade rekindle"):
        PhotoStore(db)

    conn = sqlite3.connect(db)
    try:
        version = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
    finally:
        conn.close()
    assert version == "99", "a refused database must not have been written to"


def test_fingerprint_fields_roundtrip(tmp_path):
    meta = PhotoMeta(phash=0xDEADBEEFCAFEF00D, sharpness=12.5, phash_error=None)
    with PhotoStore(tmp_path / "db.sqlite") as store:
        store.upsert_many([_photo(meta=meta)])
        got = store.get("abc")
    assert got.meta.phash == 0xDEADBEEFCAFEF00D
    assert got.meta.sharpness == pytest.approx(12.5)
    assert got.meta.phash_error is None


def test_a_full_64_bit_hash_survives_sqlite(tmp_path):
    """SQLite INTEGER is signed 64-bit. A dHash with the top bit set is a
    perfectly ordinary value (any image whose first gradient rises), so if
    it were stored as an unsigned Python int without care this is where it
    would overflow or come back negative."""
    meta = PhotoMeta(phash=0xFFFFFFFFFFFFFFFF)
    with PhotoStore(tmp_path / "db.sqlite") as store:
        store.upsert_many([_photo(meta=meta)])
        got = store.get("abc")
    assert got.meta.phash == 0xFFFFFFFFFFFFFFFF


def test_reindex_does_not_destroy_a_stored_fingerprint(tmp_path):
    """The trap that once destroyed enrichment, applied to fingerprints.

    `merge_meta(stored, freshly_scanned)` is what every re-index calls, and a
    folder scan never computes a phash. A field left out of merge_meta's
    explicit field list is DELETED by the next `rekindle index` - so without
    the phash line, `rekindle fingerprint` (six minutes) would be undone by
    `rekindle index` (seconds).
    """
    stored = PhotoMeta(taken_at_utc=T0, tz_source=TzSource.EXIF_OFFSET, phash=123, sharpness=4.5)
    scanned = PhotoMeta(taken_at_utc=T0, tz_source=TzSource.EXIF_OFFSET)

    merged, _ = merge_meta(stored, scanned)

    assert merged.phash == 123
    assert merged.sharpness == pytest.approx(4.5)


def test_reindex_through_the_store_preserves_the_fingerprint(tmp_path):
    """The same guarantee, through the real upsert path rather than a direct
    merge_meta call - because that is how `rekindle index` reaches it."""
    db = tmp_path / "db.sqlite"
    fingerprinted = PhotoMeta(taken_at_utc=T0, tz_source=TzSource.EXIF_OFFSET, phash=999)
    with PhotoStore(db) as store:
        store.upsert_many([_photo(meta=fingerprinted)])
        # A second sighting of the same bytes, as a folder rescan produces it.
        store.upsert_many([_photo(meta=PhotoMeta(taken_at_utc=T0, tz_source=TzSource.EXIF_OFFSET))])
        assert store.get("abc").meta.phash == 999


def test_a_zero_phash_is_not_treated_as_absent(tmp_path):
    """dHash 0 is a real, reachable value: any perfectly flat image (a
    scanned blank page, a black video poster) hashes to it. An `old.phash or
    new.phash` fallback would silently discard it - the exact `x or y`
    scalar-fallback bug already recorded in known-limitations for width and
    height."""
    stored = PhotoMeta(taken_at_utc=T0, tz_source=TzSource.EXIF_OFFSET, phash=0)
    scanned = PhotoMeta(taken_at_utc=T0, tz_source=TzSource.EXIF_OFFSET, phash=77)

    merged, _ = merge_meta(stored, scanned)

    assert merged.phash == 0


def test_iter_unfingerprinted_skips_done_and_failed_rows(tmp_path):
    """Resumability. A row that succeeded and a row that permanently FAILED
    are both finished work; only a row that was never attempted comes back."""
    with PhotoStore(tmp_path / "db.sqlite") as store:
        store.upsert_many(
            [
                _photo("done", meta=PhotoMeta(phash=1, colour="ff" * 64)),
                _photo("failed", meta=PhotoMeta(phash_error="undecodable")),
                _photo("video", meta=PhotoMeta(phash_error="video")),
                _photo("todo"),
            ]
        )
        assert {p.file_hash for p in store.iter_unfingerprinted()} == {"todo"}


def test_a_row_missing_only_a_LATER_measurement_is_re_offered(tmp_path):
    """What lets a new signal reach an index that already has hashes.

    The v4 colour histogram was added after this library was fingerprinted.
    Without this, `rekindle fingerprint` would report "nothing to do" and the
    diversity signal would be permanently half-blind on every existing index.
    """
    with PhotoStore(tmp_path / "db.sqlite") as store:
        store.upsert_many(
            [
                _photo("hashed_no_colour", meta=PhotoMeta(phash=1)),
                _photo("complete", meta=PhotoMeta(phash=2, colour="ab" * 64)),
                _photo("video", meta=PhotoMeta(phash_error="video")),
            ]
        )
        assert {p.file_hash for p in store.iter_unfingerprinted()} == {"hashed_no_colour"}


def test_set_fingerprints_writes_only_the_three_columns(tmp_path):
    """A targeted UPDATE, not a full row rewrite.

    If this were routed through `update_many`, the stale Photo objects the
    fingerprint pass loaded would overwrite anything written since - so the
    check is that a concurrent change to another column SURVIVES the
    fingerprint write.
    """
    db = tmp_path / "db.sqlite"
    with PhotoStore(db) as store:
        store.upsert_many([_photo("h1")])
        # Something else updates the row after the fingerprint pass read it.
        store.update_photo(_photo("h1", meta=PhotoMeta(description="written later")))

        written = store.set_fingerprints([FingerprintRow("h1", phash=42, sharpness=3.5)])

        assert written == 1
        got = store.get("h1")
        assert got.meta.phash == 42
        assert got.meta.sharpness == pytest.approx(3.5)
        assert got.meta.description == "written later"


def test_set_fingerprints_reports_rows_that_do_not_exist(tmp_path):
    """A photo deleted between the read and the write must not be counted as
    written - silently dropping input is the bug the accounting identities in
    this project exist to prevent."""
    with PhotoStore(tmp_path / "db.sqlite") as store:
        store.upsert_many([_photo("h1")])
        assert (
            store.set_fingerprints([FingerprintRow("h1", phash=1), FingerprintRow("gone", phash=2)])
            == 1
        )
