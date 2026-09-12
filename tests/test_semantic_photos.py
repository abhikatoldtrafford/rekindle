"""The read-only index reader: never writes, never pins a schema version."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pytest

from rekindle.db import SCHEMA_VERSION
from rekindle.models import MediaType
from rekindle.semantic.photos import (
    DATE_BUCKET_PREFIX,
    IndexUnavailable,
    PhotoIndexReader,
    ReadFilter,
)
from tests.fixtures.semantic import make_index, make_photo, write_photo


@pytest.fixture
def library(tmp_path):
    root = tmp_path / "lib"
    rows = [
        make_photo("a" * 32, write_photo(root, "a.jpg", "red"), people=["Abhik Maiti"]),
        make_photo("b" * 32, write_photo(root, "b.jpg", "green")),
        make_photo(
            "c" * 32, write_photo(root, "c.jpg", "blue"), albums=["Photos from 2019", "Kashmir"]
        ),
        make_photo("d" * 32, write_photo(root, "d.jpg", "white"), archived=True),
        make_photo("e" * 32, write_photo(root, "e.jpg", "black"), trashed=True),
        make_photo(
            "f" * 32,
            root / "f.mp4",
            media_type=MediaType.VIDEO,
        ),
    ]
    db = tmp_path / "data" / "rekindle.sqlite"
    make_index(db, rows, index_root=root)
    return db, root


def test_missing_index_names_the_command_to_run(tmp_path):
    with pytest.raises(IndexUnavailable, match="rekindle index"):
        PhotoIndexReader(tmp_path / "nope.sqlite")


def test_a_non_database_file_is_rejected_clearly(tmp_path):
    junk = tmp_path / "junk.sqlite"
    junk.write_bytes(b"this is not a database" * 40)
    with pytest.raises(IndexUnavailable):
        PhotoIndexReader(junk)


def test_a_database_without_photos_is_rejected(tmp_path):
    other = tmp_path / "other.sqlite"
    conn = sqlite3.connect(other)
    conn.execute("CREATE TABLE something(x INTEGER)")
    conn.commit()
    conn.close()
    with pytest.raises(IndexUnavailable, match="no `photos` table"):
        PhotoIndexReader(other)


def test_a_photos_table_missing_columns_names_them(tmp_path):
    old = tmp_path / "old.sqlite"
    conn = sqlite3.connect(old)
    conn.execute("CREATE TABLE photos(file_hash TEXT, media_type TEXT, paths TEXT)")
    conn.commit()
    conn.close()
    with pytest.raises(IndexUnavailable, match="missing column"):
        PhotoIndexReader(old)


def test_reader_does_not_write_to_the_index(library):
    """`PhotoStore.__init__` writes on open; this reader must not.

    Compared byte for byte, not by mtime: on Windows mtime granularity can hide
    a small write entirely.
    """
    db, _root = library
    before = db.read_bytes()
    with PhotoIndexReader(db) as reader:
        list(reader.iter_photos())
        reader.count()
    assert db.read_bytes() == before


def test_reader_accepts_a_schema_version_it_does_not_know(library):
    """M2 is bumping SCHEMA_VERSION concurrently; a reader must not care.

    The reference index is already at v3 while this branch's `db.py` says v2 -
    `PhotoStore` refuses it and this reader opens it, which is the whole reason
    this class exists.
    """
    db, _root = library
    conn = sqlite3.connect(db)
    conn.execute("UPDATE meta SET value = ? WHERE key='schema_version'", (str(SCHEMA_VERSION + 7),))
    conn.commit()
    conn.close()
    with PhotoIndexReader(db) as reader:
        assert reader.schema_version() == SCHEMA_VERSION + 7
        assert reader.count() == 3  # unchanged behaviour


def test_default_filter_excludes_archived_trashed_and_video(library):
    db, _root = library
    with PhotoIndexReader(db) as reader:
        hashes = {p.file_hash for p in reader.iter_photos()}
    assert hashes == {"a" * 32, "b" * 32, "c" * 32}


def test_archived_can_be_included_deliberately(library):
    db, _root = library
    with PhotoIndexReader(db) as reader:
        hashes = {p.file_hash for p in reader.iter_photos(ReadFilter(include_archived=True))}
    assert "d" * 32 in hashes


def test_without_people_needs_both_person_fields_empty(library):
    """`people` alone would be enough today and silently wrong tomorrow."""
    db, _root = library
    with PhotoIndexReader(db) as reader:
        hashes = {p.file_hash for p in reader.iter_photos(ReadFilter(without_people=True))}
    assert hashes == {"b" * 32, "c" * 32}


def test_real_albums_drops_the_takeout_year_bucket(library):
    db, _root = library
    with PhotoIndexReader(db) as reader:
        by_hash = {p.file_hash: p for p in reader.iter_photos()}
    assert by_hash["c" * 32].albums == ("Photos from 2019", "Kashmir")
    assert by_hash["c" * 32].real_albums == ("Kashmir",)
    assert by_hash["b" * 32].real_albums == ()
    assert by_hash["b" * 32].albums[0].startswith(DATE_BUCKET_PREFIX)


def test_album_filter_selects_by_membership(library):
    db, _root = library
    with PhotoIndexReader(db) as reader:
        hashes = {p.file_hash for p in reader.iter_photos(ReadFilter(albums=("Kashmir",)))}
    assert hashes == {"c" * 32}


def test_limit_caps_the_scan(library):
    db, _root = library
    with PhotoIndexReader(db) as reader:
        assert len(list(reader.iter_photos(ReadFilter(limit=2)))) == 2


def test_get_many_resolves_a_batch(library):
    db, _root = library
    with PhotoIndexReader(db) as reader:
        found = reader.get_many(["a" * 32, "c" * 32, "zz"])
    assert set(found) == {"a" * 32, "c" * 32}


def test_existing_path_skips_a_deleted_copy(tmp_path):
    """A photo indexed from two folders can lose one of them."""
    root = tmp_path / "lib"
    real = write_photo(root, "keep.jpg", "red")
    photo = make_photo("h" * 32, root / "gone.jpg")
    photo.paths = [root / "gone.jpg", real]
    db = tmp_path / "data" / "rekindle.sqlite"
    make_index(db, [photo])
    with PhotoIndexReader(db) as reader:
        found = next(iter(reader.iter_photos()))
    assert found.path == root / "gone.jpg"
    assert found.existing_path() == real


def test_existing_path_is_none_when_nothing_is_on_disk(tmp_path):
    db = tmp_path / "data" / "rekindle.sqlite"
    make_index(db, [make_photo("h" * 32, tmp_path / "nowhere.jpg")])
    with PhotoIndexReader(db) as reader:
        assert next(iter(reader.iter_photos())).existing_path() is None


def test_index_root_is_reported(library):
    db, root = library
    with PhotoIndexReader(db) as reader:
        assert reader.index_root() == str(root)


def test_photos_come_back_in_a_stable_order(library):
    db, _root = library
    with PhotoIndexReader(db) as reader:
        first = [p.file_hash for p in reader.iter_photos()]
        second = [p.file_hash for p in reader.iter_photos()]
    assert first == second == sorted(first)


def test_taken_at_survives_the_round_trip(tmp_path):
    when = datetime(2014, 3, 9, 7, 30, tzinfo=UTC)
    db = tmp_path / "data" / "rekindle.sqlite"
    make_index(db, [make_photo("h" * 32, tmp_path / "x.jpg", taken_at=when)])
    with PhotoIndexReader(db) as reader:
        photo = next(iter(reader.iter_photos()))
    assert photo.taken_at_utc.startswith("2014-03-09")


def test_count_applies_the_album_filter_like_iter_photos_does(library):
    """`count` built its SQL from `_where`, which never sees `albums` -
    the column is a JSON blob and the match happens in Python. So the two
    disagreed: measured on the reference index, `count` said 18,201 for a
    filter that `iter_photos` answered with 508. A caller sizing a progress
    bar, or deciding whether anything matched at all, is not wrong by a
    little."""
    db, _root = library
    with PhotoIndexReader(db) as reader:
        where = ReadFilter(albums=("Kashmir",))
        assert reader.count(where) == len(list(reader.iter_photos(where)))
        assert reader.count(where) == 1
        assert reader.count(ReadFilter()) > 1, "the unfiltered count must be unaffected"


def test_count_ignores_a_limit(library):
    """A limit answers "give me some", a count answers "how many are there".
    Passing one through would make `count` report the limit back."""
    db, _root = library
    with PhotoIndexReader(db) as reader:
        assert reader.count(ReadFilter(albums=("Kashmir",), limit=1)) == 1
        assert reader.count(ReadFilter(limit=1)) == reader.count(ReadFilter())


def test_a_limit_is_applied_AFTER_the_album_match(tmp_path):
    """The LIMIT went into the SQL, which runs before the Python-side album
    test, so it cut the candidates rather than the results: measured on a real
    library, `limit=200` returned 17 rows.

    Here the wanted album is the last thing in `file_hash` order, so a SQL
    LIMIT of 3 would reach none of it.
    """
    root = tmp_path / "lib"
    rows = [
        make_photo(f"{i:032x}", write_photo(root, f"n{i}.jpg", "red"), albums=["Other"])
        for i in range(10)
    ]
    rows += [
        make_photo(f"f{i:031x}", write_photo(root, f"k{i}.jpg", "blue"), albums=["Kashmir"])
        for i in range(5)
    ]
    db = tmp_path / "data" / "rekindle.sqlite"
    make_index(db, rows, index_root=root)

    with PhotoIndexReader(db) as reader:
        got = list(reader.iter_photos(ReadFilter(albums=("Kashmir",), limit=3)))

    assert len(got) == 3, "the limit must cut the RESULTS, not the candidates"
    assert all("Kashmir" in p.albums for p in got)
