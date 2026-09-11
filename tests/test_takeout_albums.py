import json
from datetime import UTC, datetime
from pathlib import Path

from rekindle.db import PhotoStore
from rekindle.enrich.takeout import (
    EnrichReport,
    SidecarIndex,
    TakeoutEnricher,
    album_renames,
    build_index,
    retitle_albums,
)
from rekindle.models import MediaType, Photo, PhotoMeta
from rekindle.sources.folder import FolderSource
from tests.fixtures.takeout import build_takeout

NOW = datetime(2020, 1, 1, tzinfo=UTC)


def _photo(*albums: str) -> Photo:
    return Photo(
        file_hash="h",
        paths=[Path("x.jpg")],
        media_type=MediaType.IMAGE,
        meta=PhotoMeta(),
        first_seen=NOW,
        last_seen=NOW,
        albums=list(albums),
    )


def test_a_differing_title_replaces_the_sanitised_folder_name(tmp_path):
    root = build_takeout(tmp_path / "Takeout")
    report = EnrichReport()
    index = build_index(root, report)
    renames = album_renames(index, report)
    assert renames["Goa Trip"] == "Goa/ Trip"

    photo = _photo("Goa Trip", "Photos from 2011")
    retitle_albums(photo, renames, report)
    assert photo.albums == ["Goa/ Trip", "Photos from 2011"]
    assert report.albums_retitled == 1


def test_an_empty_or_null_title_leaves_the_folder_name_in_place(tmp_path):
    root = build_takeout(tmp_path / "Takeout")
    report = EnrichReport()
    renames = album_renames(build_index(root, report), report)
    assert "No Name" not in renames
    photo = _photo("No Name")
    retitle_albums(photo, renames, report)
    assert photo.albums == ["No Name"]


def test_a_title_matching_its_own_folder_is_not_a_rename(tmp_path):
    root = build_takeout(tmp_path / "Takeout")
    report = EnrichReport()
    renames = album_renames(build_index(root, report), report)
    assert "Untitled" not in renames


def test_colliding_titles_are_reported_and_not_applied(tmp_path):
    """Untitled(1) has title 'Untitled', which is already another album."""
    root = build_takeout(tmp_path / "Takeout")
    report = EnrichReport()
    renames = album_renames(build_index(root, report), report)
    assert "Untitled(1)" not in renames
    assert ("Untitled(1)", "Untitled") in report.album_title_collisions


def test_an_empty_title_is_not_a_rename_even_if_it_reaches_the_index():
    """`build_index` already excludes empty titles, but `album_renames` must
    not rely on that as its only defence - "never fabricate a value" applies
    to a hypothetically-mispopulated `SidecarIndex` too, not just the one
    `build_index` produces."""
    index = SidecarIndex()
    index.albums[Path("/lib/No Title")] = ""
    report = EnrichReport()
    renames = album_renames(index, report)
    assert renames == {}
    assert report.album_title_collisions == []


def test_retitling_is_idempotent():
    report = EnrichReport()
    renames = {"Goa Trip": "Goa/ Trip"}
    photo = _photo("Goa Trip")
    retitle_albums(photo, renames, report)
    retitle_albums(photo, renames, report)
    assert photo.albums == ["Goa/ Trip"]


def test_collision_resolution_does_not_depend_on_the_platform():
    """Replaces a first-draft test whose scenario was unrepresentable:
    index.albums only ever holds directories that HAD a metadata.json, so
    "ignore directories without one" could not be constructed.

    This one is real. sorted() on Path compares case-folded on Windows and raw
    on POSIX, so which of two albums claiming one title wins would differ
    between a contributor's machine and CI. M0 hit this exact bug.
    """
    # "Zulu" vs "beta" is chosen deliberately: raw ASCII puts 'Z' (0x5A)
    # before 'b' (0x62), casefolding puts "beta" before "zulu". A pair like
    # Alpha/beta orders the same either way and would NOT catch the bug.
    #
    # Verified: unfixed, this FAILS on Linux and macOS and PASSES on Windows,
    # because PureWindowsPath already compares case-insensitively. That is the
    # bug - the same code picking different albums on different machines - so
    # do not conclude the test is inert from a green run on Windows. CI covers
    # all three.
    index = SidecarIndex()
    index.albums[Path("/lib/Zulu")] = "Shared"
    index.albums[Path("/lib/beta")] = "Shared"
    report = EnrichReport()
    renames = album_renames(index, report)
    assert renames == {"beta": "Shared"}
    assert report.album_title_collisions == [("Zulu", "Shared")]


def _album_export(tmp_path):
    """One album whose Google title differs from its sanitised folder name."""
    from tests.fixtures.gen import make_jpeg

    root = tmp_path / "Takeout"
    album = root / "Trip 2019"
    make_jpeg(album / "IMG_A.jpg")
    (album / "metadata.json").write_text(json.dumps({"title": "Our Coast Trip"}), encoding="utf-8")
    (album / "IMG_A.jpg.supplemental-metadata.json").write_text(
        json.dumps({"title": "IMG_A.jpg", "photoTakenTime": {"timestamp": "1400000000"}}),
        encoding="utf-8",
    )
    return root


def test_a_second_enrich_retitles_nothing_and_changes_no_album(tmp_path):
    """The idempotency property `albums_retitled` is supposed to have."""
    root = _album_export(tmp_path)
    photos, _ = FolderSource().scan(root)
    store = PhotoStore(tmp_path / "data" / "rekindle.sqlite")
    store.upsert_many(photos)
    digest = photos[0].file_hash

    first = TakeoutEnricher().enrich(root, store)
    assert first.albums_retitled == 1
    assert store.get(digest).albums == ["Our Coast Trip"]

    second = TakeoutEnricher().enrich(root, store)
    assert second.albums_retitled == 0
    assert store.get(digest).albums == ["Our Coast Trip"]
    store.close()


def test_a_re_index_then_re_enrich_does_not_duplicate_an_album_entry(tmp_path):
    """`index -> enrich -> index -> enrich` must not double the album.

    `PhotoStore._merge` unions the already-retitled title with the raw folder
    name the fresh scan re-derives, so the photo comes back carrying BOTH.
    `retitle_albums` then maps the folder name onto the title that is already
    present, and a plain list comprehension stores `['Our Coast Trip', 'Our
    Coast Trip']` - stable there forever. 0 such rows on the reference export
    today because only one index+enrich has ever run against it; the defect is
    latent until the first re-index, which doctor's own orphan warning tells
    users to perform.

    MUTATION (run, not assumed): change `photo.albums =
    list(dict.fromkeys(updated))` back to `photo.albums = updated` in
    `retitle_albums` and this fails with `['Our Coast Trip', 'Our Coast
    Trip'] != ['Our Coast Trip']`.
    """
    root = _album_export(tmp_path)
    photos, _ = FolderSource().scan(root)
    store = PhotoStore(tmp_path / "data" / "rekindle.sqlite")
    store.upsert_many(photos)
    digest = photos[0].file_hash

    TakeoutEnricher().enrich(root, store)
    assert store.get(digest).albums == ["Our Coast Trip"]

    for _cycle in range(2):
        rescanned, _ = FolderSource().scan(root)
        store.upsert_many(rescanned)
        # The scan re-derives the folder name, so both spellings are stored
        # between the two passes. That part is correct - the union is what
        # keeps an album a photo left in place.
        assert store.get(digest).albums == ["Our Coast Trip", "Trip 2019"]
        TakeoutEnricher().enrich(root, store)
        assert store.get(digest).albums == ["Our Coast Trip"]
    store.close()
