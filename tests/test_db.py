import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rekindle.db import SCHEMA_VERSION, PhotoStore
from rekindle.models import FaceRegion, Gps, MediaType, Photo, PhotoMeta, TzSource, merge_meta

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
T1 = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def _photo(h="abc123", **kw) -> Photo:
    meta = kw.pop("meta", PhotoMeta.empty())
    return Photo(
        file_hash=h,
        paths=kw.pop("paths", [Path("/photos/a.jpg")]),
        media_type=kw.pop("media_type", MediaType.IMAGE),
        meta=meta,
        first_seen=kw.pop("first_seen", T0),
        last_seen=kw.pop("last_seen", T0),
        albums=kw.pop("albums", []),
        edited_of=kw.pop("edited_of", None),
        source=kw.pop("source", "folder"),
        metadata_conflict=kw.pop("metadata_conflict", False),
    )


def test_creates_schema_and_records_version(tmp_path):
    with PhotoStore(tmp_path / "db.sqlite") as s:
        assert s.schema_version() == SCHEMA_VERSION
        assert s.count() == 0


def test_insert_then_read_back_roundtrips_metadata(tmp_path):
    # T1 for taken_at_local so it's distinguishable from taken_at_utc in the
    # assertions below - a swap of the two columns must fail this test.
    meta = PhotoMeta(
        taken_at_utc=T0,
        taken_at_local=T1,
        tz_source=TzSource.EXIF_OFFSET,
        gps=Gps(lat=15.3, lon=74.08, alt=12.5),
        people=["Alice", "Bob"],
        face_regions=[FaceRegion(name="Alice", x=0.5, y=0.4, w=0.2, h=0.25)],
        keywords=["beach"],
        description="sunset",
        favorite=True,
        camera_make="Canon",
        camera_model="EOS R",
        width=4000,
        height=3000,
    )
    with PhotoStore(tmp_path / "db.sqlite") as s:
        inserted, updated = s.upsert_many(
            [
                _photo(
                    meta=meta,
                    albums=["Goa Trip"],
                    # Non-default values: a field that silently fell back to
                    # its default on read-back would still pass a test that
                    # merely wrote the default.
                    source="takeout",
                    metadata_conflict=True,
                )
            ]
        )
        assert (inserted, updated) == (1, 0)
        got = s.get("abc123")

    assert got is not None
    assert got.albums == ["Goa Trip"]
    assert got.source == "takeout"
    assert got.metadata_conflict is True
    assert got.meta.taken_at_utc == T0
    assert got.meta.taken_at_local == T1
    assert got.meta.tz_source is TzSource.EXIF_OFFSET
    assert got.meta.gps == Gps(lat=15.3, lon=74.08, alt=12.5)
    assert got.meta.people == ["Alice", "Bob"]
    assert got.meta.face_regions == [FaceRegion(name="Alice", x=0.5, y=0.4, w=0.2, h=0.25)]
    assert got.meta.keywords == ["beach"]
    assert got.meta.description == "sunset"
    assert got.meta.favorite is True
    assert got.meta.camera_make == "Canon"
    assert got.meta.camera_model == "EOS R"
    assert got.meta.width == 4000
    assert got.meta.height == 3000


def test_upsert_same_hash_updates_last_seen_and_unions_albums(tmp_path):
    with PhotoStore(tmp_path / "db.sqlite") as s:
        s.upsert_many([_photo(albums=["Year 2014"], paths=[Path("/a/x.jpg")])])
        inserted, updated = s.upsert_many(
            [_photo(albums=["Goa Trip"], paths=[Path("/b/x.jpg")], first_seen=T1, last_seen=T1)]
        )
        assert (inserted, updated) == (0, 1)
        got = s.get("abc123")

    assert sorted(got.albums) == ["Goa Trip", "Year 2014"]
    # Compare Paths, not strings: Path("/a/x.jpg") is "\a\x.jpg" on Windows.
    assert sorted(got.paths) == sorted([Path("/a/x.jpg"), Path("/b/x.jpg")])
    assert got.first_seen == T0  # earliest wins
    assert got.last_seen == T1  # latest wins


def test_reindex_does_not_reset_a_resolved_sidecar_match(tmp_path):
    """A folder rescan always writes sidecar_match='none' (it knows nothing
    about Takeout sidecars). If _merge let that clobber a prior 'exact'
    resolution, every `rekindle index` after an enrich would erase it."""
    with PhotoStore(tmp_path / "db.sqlite") as s:
        enriched = _photo()
        enriched.sidecar_match = "exact"
        s.upsert_many([enriched])

        rescanned = _photo()  # sidecar_match defaults to "none"
        assert rescanned.sidecar_match == "none"
        inserted, updated = s.upsert_many([rescanned])
        assert (inserted, updated) == (0, 1)

        got = s.get("abc123")
    assert got.sidecar_match == "exact"


def test_get_returns_none_for_unknown_hash(tmp_path):
    with PhotoStore(tmp_path / "db.sqlite") as s:
        assert s.get("nope") is None


def test_all_hashes_returns_every_stored_hash(tmp_path):
    # Named for what it actually checks. Incremental scan is an explicit
    # scope cut in the M0 plan - nothing here exercises it, and the old name
    # claimed a capability the code does not have.
    with PhotoStore(tmp_path / "db.sqlite") as s:
        s.upsert_many([_photo("h1"), _photo("h2")])
        assert s.all_hashes() == {"h1", "h2"}


def test_store_persists_across_sessions(tmp_path):
    db = tmp_path / "db.sqlite"
    with PhotoStore(db) as s:
        s.upsert_many([_photo()])
    with PhotoStore(db) as s:
        assert s.count() == 1


def test_upsert_many_commits_once_not_per_photo(tmp_path):
    """A regression guard for "one transaction per run, never one commit per
    photo": if a commit were moved inside the loop, the two good photos
    ahead of the failing one would already be durable when the batch blows
    up, and this test would see count() == 2 instead of 0.

    sqlite3 rolls back an uncommitted transaction when its connection is
    closed without calling commit(), so a mid-batch failure that never
    reaches the single end-of-batch commit() leaves nothing persisted -
    provided nothing committed earlier in the loop.
    """
    db = tmp_path / "db.sqlite"
    with pytest.raises(AttributeError), PhotoStore(db) as s:
        # object() has no .file_hash: upsert_many's loop raises on the third
        # item, after two real photos would otherwise have gone in.
        s.upsert_many([_photo("h1"), _photo("h2"), object()])

    with PhotoStore(db) as s:
        assert s.count() == 0


def _write_v1_database(path):
    """An M0-schema database, byte-for-byte as v1 wrote it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(
        """
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
            metadata_conflict INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    conn.execute("INSERT INTO meta(key, value) VALUES('schema_version', '1')")
    conn.execute(
        "INSERT INTO photos (file_hash, media_type, paths, albums, first_seen,"
        " last_seen, tz_source, people, face_regions, keywords) VALUES"
        " ('abc', 'image', ?, ?, '2020-01-01T00:00:00+00:00',"
        " '2020-01-01T00:00:00+00:00', 'exif_naive', ?, '[]', '[]')",
        (
            json.dumps(["/lib/Photos from 2014/IMG_0001.jpg"]),
            json.dumps(["Photos from 2014"]),
            json.dumps(["Ada"]),
        ),
    )
    conn.commit()
    conn.close()


def test_v1_database_migrates_without_losing_rows(tmp_path):
    """A data-loss regression test. M0 users must not be told to re-index."""
    db_path = tmp_path / "data" / "rekindle.sqlite"
    _write_v1_database(db_path)

    with PhotoStore(db_path) as store:
        assert store.schema_version() == SCHEMA_VERSION == 2
        assert store.count() == 1
        photo = store.get("abc")
        assert photo is not None
        assert photo.meta.people == ["Ada"]
        assert photo.albums == ["Photos from 2014"]
        # New columns arrive with their defaults, not with NULLs that crash
        # deserialisation.
        assert photo.sidecar_match == "none"
        assert photo.meta.archived is False
        assert photo.meta.trashed is False
        assert photo.meta.exif_taken_at_utc is None
        assert photo.meta.takeout_people == []


def test_migration_backfills_the_photo_paths_table(tmp_path):
    db_path = tmp_path / "data" / "rekindle.sqlite"
    _write_v1_database(db_path)
    with PhotoStore(db_path) as store:
        hits = store.hashes_for_filename("img_0001.jpg")
    assert hits == {"abc"}


def test_new_meta_fields_survive_a_reindex_merge():
    """merge_meta enumerates fields; one left out is enrichment destroyed."""
    enriched = PhotoMeta(
        taken_at_utc=datetime(2014, 6, 1, tzinfo=UTC),
        tz_source=TzSource.TAKEOUT,
        people=["Ada"],
        takeout_people=["Ada"],
        archived=True,
        trashed=False,
        exif_taken_at_utc=datetime(2010, 1, 1, tzinfo=UTC),
    )
    fresh_from_disk = PhotoMeta(
        taken_at_utc=datetime(2020, 1, 1, tzinfo=UTC),
        tz_source=TzSource.FILE_MTIME,
    )
    merged, _ = merge_meta(enriched, fresh_from_disk)
    assert merged.takeout_people == ["Ada"]
    assert merged.archived is True
    assert merged.exif_taken_at_utc == datetime(2010, 1, 1, tzinfo=UTC)


def test_takeout_is_a_real_tz_source():
    assert TzSource("takeout") is TzSource.TAKEOUT
