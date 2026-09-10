from datetime import UTC, datetime
from pathlib import Path

from rekindle.enrich.takeout import (
    EnrichReport,
    SidecarIndex,
    album_renames,
    build_index,
    retitle_albums,
)
from rekindle.models import MediaType, Photo, PhotoMeta
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
