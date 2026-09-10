from datetime import UTC, datetime
from pathlib import Path

from rekindle.db import SCHEMA_VERSION, PhotoStore
from rekindle.models import FaceRegion, Gps, MediaType, Photo, PhotoMeta, TzSource

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
    )


def test_creates_schema_and_records_version(tmp_path):
    with PhotoStore(tmp_path / "db.sqlite") as s:
        assert s.schema_version() == SCHEMA_VERSION
        assert s.count() == 0


def test_insert_then_read_back_roundtrips_metadata(tmp_path):
    meta = PhotoMeta(
        taken_at_utc=T0,
        taken_at_local=T0,
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
        inserted, updated = s.upsert_many([_photo(meta=meta, albums=["Goa Trip"])])
        assert (inserted, updated) == (1, 0)
        got = s.get("abc123")

    assert got is not None
    assert got.meta.gps == Gps(lat=15.3, lon=74.08, alt=12.5)
    assert got.meta.people == ["Alice", "Bob"]
    assert got.meta.face_regions[0].name == "Alice"
    assert got.meta.favorite is True
    assert got.meta.tz_source is TzSource.EXIF_OFFSET
    assert got.albums == ["Goa Trip"]
    assert got.meta.taken_at_utc == T0


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


def test_get_returns_none_for_unknown_hash(tmp_path):
    with PhotoStore(tmp_path / "db.sqlite") as s:
        assert s.get("nope") is None


def test_all_hashes_supports_incremental_scan(tmp_path):
    with PhotoStore(tmp_path / "db.sqlite") as s:
        s.upsert_many([_photo("h1"), _photo("h2")])
        assert s.all_hashes() == {"h1", "h2"}


def test_store_persists_across_sessions(tmp_path):
    db = tmp_path / "db.sqlite"
    with PhotoStore(db) as s:
        s.upsert_many([_photo()])
    with PhotoStore(db) as s:
        assert s.count() == 1
