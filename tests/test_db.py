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
    # A second photo with THREE paths across different album directories -
    # this library routinely has 2+ paths per photo, and one is not ASCII
    # (a German export directory name and an eszett in the filename), which
    # exercises str.casefold() rather than str.lower() in the backfill.
    conn.execute(
        "INSERT INTO photos (file_hash, media_type, paths, albums, first_seen,"
        " last_seen, tz_source, people, face_regions, keywords) VALUES"
        " ('multi1', 'image', ?, ?, '2021-01-01T00:00:00+00:00',"
        " '2021-01-01T00:00:00+00:00', 'exif_naive', '[]', '[]', '[]')",
        (
            json.dumps(
                [
                    "/lib/Album A/IMG_0002.jpg",
                    "/lib/Album B/IMG_0002.jpg",
                    "/lib/Alben/STRAßE.JPG",
                ]
            ),
            json.dumps(["Album A", "Album B", "Alben"]),
        ),
    )
    conn.commit()
    conn.close()


def test_v1_database_migrates_without_losing_rows(tmp_path):
    """A data-loss regression test. M0 users must not be told to re-index."""
    db_path = tmp_path / "data" / "rekindle.sqlite"
    _write_v1_database(db_path)

    with PhotoStore(db_path) as store:
        # v1 -> v2 -> v3 in ONE open. This is the ladder's whole reason to
        # exist: the single-step `!= 1` guard it replaced would have run the
        # v2 step, left the database at v2, and then had __init__ reject it as
        # unmigratable - turning every M0 database into "delete and re-index".
        assert store.schema_version() == SCHEMA_VERSION == 3
        assert store.count() == 2
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


def test_migration_backfill_writes_one_row_per_hash_and_path(tmp_path):
    """The regression the reviewer's mutation caught: indexing only the
    first path of `json.loads(row["paths"])` still passes every OTHER
    assertion in this suite. `multi1` (from `_write_v1_database`) has three
    paths across three album directories, one with a non-ASCII filename
    (a German eszett), and this library routinely has 2+ paths per photo.
    """
    db_path = tmp_path / "data" / "rekindle.sqlite"
    _write_v1_database(db_path)
    with PhotoStore(db_path):
        pass  # migration runs on open; check the raw table afterwards

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT path, name_cf, parent FROM photo_paths WHERE file_hash = 'multi1' ORDER BY path"
        ).fetchall()
    finally:
        conn.close()

    # Exactly one row per (file_hash, path): three distinct paths in, three
    # distinct rows out - not one, and not deduplicated by filename despite
    # two of the three sharing "IMG_0002.jpg". Compare via Path so the
    # expected separators match this platform (Path stringifies "/" as "\\"
    # on Windows).
    assert len(rows) == 3
    got = {(path, name_cf, parent) for path, name_cf, parent in rows}
    assert got == {
        (
            str(Path("/lib/Album A/IMG_0002.jpg")),
            "img_0002.jpg",
            str(Path("/lib/Album A")),
        ),
        (
            str(Path("/lib/Album B/IMG_0002.jpg")),
            "img_0002.jpg",
            str(Path("/lib/Album B")),
        ),
        (
            str(Path("/lib/Alben/STRAßE.JPG")),
            "strasse.jpg",
            str(Path("/lib/Alben")),
        ),
    }

    with PhotoStore(db_path) as store:
        # Two of the three paths share a filename: hashes_for_filename must
        # still resolve to the one photo, not silently drop it.
        assert store.hashes_for_filename("img_0002.jpg") == {"multi1"}
        # str.casefold(), not str.lower(): "ß" folds to "ss", so a lookup
        # spelled with plain ASCII "ss" must still find the eszett filename.
        assert store.hashes_for_filename("strasse.jpg") == {"multi1"}
        assert store.hashes_for_filename("STRASSE.JPG") == {"multi1"}


def test_new_meta_fields_survive_a_reindex_merge():
    """merge_meta enumerates fields; one left out is enrichment destroyed.

    `trashed=True` here is purely to exercise the merge line - the model's
    own comment notes it is expected to be permanently False for a real
    Takeout export - but `merge_meta` has no way to know that, and a merge
    that dropped this field would be exactly as broken as one that dropped
    any other.
    """
    enriched = PhotoMeta(
        taken_at_utc=datetime(2014, 6, 1, tzinfo=UTC),
        tz_source=TzSource.TAKEOUT,
        people=["Ada"],
        takeout_people=["Ada"],
        archived=True,
        trashed=True,
        exif_taken_at_utc=datetime(2010, 1, 1, tzinfo=UTC),
    )
    fresh_from_disk = PhotoMeta(
        taken_at_utc=datetime(2020, 1, 1, tzinfo=UTC),
        tz_source=TzSource.FILE_MTIME,
    )
    merged, _ = merge_meta(enriched, fresh_from_disk)
    assert merged.takeout_people == ["Ada"]
    assert merged.archived is True
    assert merged.trashed is True
    assert merged.exif_taken_at_utc == datetime(2010, 1, 1, tzinfo=UTC)


def test_takeout_is_a_real_tz_source():
    assert TzSource("takeout") is TzSource.TAKEOUT


def test_pre_v1_database_raises_instead_of_being_silently_stamped(tmp_path):
    """The ladder only walks KNOWN versions, and 0 is not one of them.

    A `>=` guard plus an unconditional bump to SCHEMA_VERSION would stamp any
    older database - including one no migration step knows anything about -
    straight to the current version and declare success, making __init__'s
    friendly "no migration exists" error unreachable. The ladder preserves
    that property by looking each version up in its step table and stopping
    when there is no entry.
    """
    db_path = tmp_path / "data" / "rekindle.sqlite"
    _write_v1_database(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE meta SET value = '0' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="schema v0"):
        PhotoStore(db_path)

    # Must not have been silently stamped to v2 on the way to raising.
    conn = sqlite3.connect(db_path)
    try:
        version = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0]
    finally:
        conn.close()
    assert version == "0"


def test_migration_failure_closes_the_connection_and_propagates_the_real_error(
    tmp_path,
):
    """A `_migrate` failure must not leave the connection open holding a
    write transaction: on Windows that keeps the file locked until GC or
    process exit, turning a legible root cause (corrupt JSON, here) into a
    baffling "database is locked" on the very next attempt.

    Corrupting 'multi1' (not 'abc') matters: the backfill processes rows in
    insertion order, so 'abc' is INSERTed into photo_paths successfully
    first - opening an implicit write transaction on the connection - and
    THEN the loop dies on 'multi1'. That leaves a real uncommitted write
    transaction behind, which is what actually blocks a second connection's
    writes on Windows; corrupting 'abc' instead fails before any write ever
    starts and does not reproduce the lock.
    """
    db_path = tmp_path / "data" / "rekindle.sqlite"
    _write_v1_database(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE photos SET paths = 'not-json' WHERE file_hash = 'multi1'")
    conn.commit()
    conn.close()

    with pytest.raises(json.JSONDecodeError):
        PhotoStore(db_path)

    # The failed attempt must not leave the file locked for a write: an
    # immediate retry (no del()/gc.collect()) must see the SAME real error
    # again, not sqlite3.OperationalError("database is locked").
    with pytest.raises(json.JSONDecodeError):
        PhotoStore(db_path)


def test_hashes_for_filename_casefolds_its_own_argument(tmp_path):
    """The parameter is named `name`, not `name_cf`: callers should not have
    to pre-casefold, and one that forgets must not silently get back an
    empty set instead of the match.
    """
    with PhotoStore(tmp_path / "db.sqlite") as s:
        s.upsert_many([_photo(paths=[Path("/photos/IMG_0001.JPG")])])
        assert s.hashes_for_filename("img_0001.jpg") == {"abc123"}
        assert s.hashes_for_filename("IMG_0001.JPG") == {"abc123"}
        assert s.hashes_for_filename("Img_0001.Jpg") == {"abc123"}


def test_iter_photos_yields_every_row(tmp_path):
    store = PhotoStore(tmp_path / "d" / "r.sqlite")
    now = datetime(2020, 1, 1, tzinfo=UTC)
    store.upsert_many(
        [
            Photo(
                file_hash=h,
                paths=[tmp_path / f"{h}.jpg"],
                media_type=MediaType.IMAGE,
                meta=PhotoMeta(),
                first_seen=now,
                last_seen=now,
            )
            for h in ("aa", "bb", "cc")
        ]
    )
    assert {p.file_hash for p in store.iter_photos()} == {"aa", "bb", "cc"}
    store.close()


def test_update_photo_does_not_merge(tmp_path):
    """merge_meta keeps the EARLIEST real date. Enrichment must be able to
    move a date FORWARD when Google says so, which upsert_many cannot do."""
    store = PhotoStore(tmp_path / "d" / "r.sqlite")
    now = datetime(2020, 1, 1, tzinfo=UTC)
    original = Photo(
        file_hash="aa",
        paths=[tmp_path / "aa.jpg"],
        media_type=MediaType.IMAGE,
        meta=PhotoMeta(
            taken_at_utc=datetime(2010, 1, 1, tzinfo=UTC),
            tz_source=TzSource.EXIF_NAIVE,
            people=["Ada", "Grace"],
        ),
        first_seen=now,
        last_seen=now,
    )
    store.upsert_many([original])

    original.meta.taken_at_utc = datetime(2015, 6, 1, tzinfo=UTC)
    original.meta.tz_source = TzSource.TAKEOUT
    original.meta.people = ["Grace"]
    original.sidecar_match = "exact"
    store.update_photo(original)

    reloaded = store.get("aa")
    assert reloaded.meta.taken_at_utc == datetime(2015, 6, 1, tzinfo=UTC)
    assert reloaded.meta.tz_source is TzSource.TAKEOUT
    assert reloaded.meta.people == ["Grace"]  # not unioned back to two
    assert reloaded.sidecar_match == "exact"
    store.close()


def test_meta_keys_round_trip(tmp_path):
    store = PhotoStore(tmp_path / "d" / "r.sqlite")
    assert store.get_meta("enriched_at") is None
    store.set_meta("enriched_at", "2026-09-11T10:00:00+00:00")
    store.set_meta("enriched_at", "2026-09-12T10:00:00+00:00")
    assert store.get_meta("enriched_at") == "2026-09-12T10:00:00+00:00"
    store.close()


def test_update_many_does_not_merge_and_commits_once(tmp_path):
    """update_many is update_photo batched: writes must land AS GIVEN (no
    merge_meta tiebreak) and all-or-nothing in one transaction, exactly like
    upsert_many's crash-mid-batch guarantee above.
    """
    db = tmp_path / "db.sqlite"
    now = datetime(2020, 1, 1, tzinfo=UTC)
    with PhotoStore(db) as s:
        original = Photo(
            file_hash="aa",
            paths=[tmp_path / "aa.jpg"],
            media_type=MediaType.IMAGE,
            meta=PhotoMeta(
                taken_at_utc=datetime(2010, 1, 1, tzinfo=UTC),
                tz_source=TzSource.EXIF_NAIVE,
            ),
            first_seen=now,
            last_seen=now,
        )
        s.upsert_many([original])

        original.meta.taken_at_utc = datetime(2015, 6, 1, tzinfo=UTC)
        original.meta.tz_source = TzSource.TAKEOUT
        n = s.update_many([original])
        assert n == 1
        reloaded = s.get("aa")
        assert reloaded.meta.taken_at_utc == datetime(2015, 6, 1, tzinfo=UTC)

    with pytest.raises(AttributeError), PhotoStore(db) as s:
        second = s.get("aa")
        second.meta.taken_at_utc = datetime(2016, 1, 1, tzinfo=UTC)
        # object() has no .meta/.file_hash: the loop raises on the second
        # item, after the first would otherwise have gone in.
        s.update_many([second, object()])

    with PhotoStore(db) as s:
        # The half-applied batch above must not have been committed: the
        # date must still read back as the LAST value durably committed
        # before the failing update_many call, not 2016.
        assert s.get("aa").meta.taken_at_utc == datetime(2015, 6, 1, tzinfo=UTC)


def test_update_photo_retires_stale_photo_paths_row_on_rename(tmp_path):
    """`photo_paths` must stay consistent with `photos` on ANY write path,
    not just `upsert_many`/migration. Both `update_photo` and `update_many`
    are correct today only because they route through `_insert`, which does
    the DELETE + re-INSERT upkeep; a future hand-rolled `UPDATE photos SET
    ...` for either writer would silently stop retiring old rows, and
    nothing else in this suite would notice.
    """
    with PhotoStore(tmp_path / "db.sqlite") as s:
        original = _photo("aa", paths=[tmp_path / "old_name.jpg"])
        s.upsert_many([original])
        assert s.hashes_for_filename("old_name.jpg") == {"aa"}

        renamed = _photo("aa", paths=[tmp_path / "new_name.jpg"])
        s.update_photo(renamed)

        assert s.hashes_for_filename("old_name.jpg") == set()
        assert s.hashes_for_filename("new_name.jpg") == {"aa"}

        rows = s._conn.execute("SELECT path FROM photo_paths WHERE file_hash = 'aa'").fetchall()
        # Exactly one row: the stale "old_name.jpg" row must be GONE, not
        # merely shadowed by a second row for "new_name.jpg".
        assert [r["path"] for r in rows] == [str(tmp_path / "new_name.jpg")]


def test_update_many_keeps_photo_paths_consistent_for_multi_path_photos(tmp_path):
    """Same guarantee as above, batched, with multi-path photos: a partial
    delete (e.g. only the first path's row retired) would only show up when
    a photo has more than one path.
    """
    with PhotoStore(tmp_path / "db.sqlite") as s:
        m1 = _photo("m1", paths=[tmp_path / "m1_old_a.jpg", tmp_path / "m1_old_b.jpg"])
        m2 = _photo("m2", paths=[tmp_path / "m2_old.jpg"])
        s.upsert_many([m1, m2])

        m1_renamed = _photo("m1", paths=[tmp_path / "m1_new.jpg"])
        m2_renamed = _photo("m2", paths=[tmp_path / "m2_new_a.jpg", tmp_path / "m2_new_b.jpg"])
        n = s.update_many([m1_renamed, m2_renamed])
        assert n == 2

        assert s.hashes_for_filename("m1_old_a.jpg") == set()
        assert s.hashes_for_filename("m1_old_b.jpg") == set()
        assert s.hashes_for_filename("m1_new.jpg") == {"m1"}
        assert s.hashes_for_filename("m2_old.jpg") == set()
        assert s.hashes_for_filename("m2_new_a.jpg") == {"m2"}
        assert s.hashes_for_filename("m2_new_b.jpg") == {"m2"}

        m1_rows = s._conn.execute("SELECT path FROM photo_paths WHERE file_hash = 'm1'").fetchall()
        assert [r["path"] for r in m1_rows] == [str(tmp_path / "m1_new.jpg")]

        m2_rows = {
            r["path"]
            for r in s._conn.execute(
                "SELECT path FROM photo_paths WHERE file_hash = 'm2'"
            ).fetchall()
        }
        assert m2_rows == {str(tmp_path / "m2_new_a.jpg"), str(tmp_path / "m2_new_b.jpg")}
