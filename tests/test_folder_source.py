from rekindle.models import MediaType, TzSource
from rekindle.sources.folder import FolderSource
from tests.fixtures.gen import build_library, make_jpeg


def _scan(root):
    return FolderSource().scan(root)


def test_indexes_images_and_skips_non_media(tmp_path):
    root = build_library(tmp_path / "lib")
    photos, report = _scan(root)
    assert report.skipped.get("not_media", 0) >= 1  # notes.txt
    names = {p.name for photo in photos for p in photo.paths}
    assert "notes.txt" not in names
    # A real check on media_indexed: the fixture library has exactly 7
    # distinct media items once IMG_0001's cross-folder duplicate is merged
    # (IMG_0001, IMG_0001-edited, IMG_0002, IMG_0003, actually_jpeg.heic,
    # PXL_0001.jpg, PXL_0001.MP). A wrong count here - counting files_seen
    # instead of distinct photos, or missing the dedupe - would fail this
    # even though `media_indexed == len(photos)` trivially holds either way.
    assert report.media_indexed == 7
    assert len(photos) == 7


def test_json_sidecars_are_ignored_but_counted(tmp_path):
    root = build_library(tmp_path / "lib")
    _, report = _scan(root)
    assert report.json_sidecars >= 1


def test_duplicate_across_folders_becomes_one_photo_with_both_albums(tmp_path):
    root = build_library(tmp_path / "lib")
    photos, report = _scan(root)
    dupes = [p for p in photos if len(p.paths) > 1]
    assert len(dupes) == 1
    assert sorted(dupes[0].albums) == ["Goa Trip", "Photos from 2014"]
    assert report.duplicates_merged == 1


def test_edited_variant_links_to_its_original(tmp_path):
    root = build_library(tmp_path / "lib")
    photos, report = _scan(root)
    edited = [p for p in photos if any("-edited" in x.name for x in p.paths)]
    assert len(edited) == 1
    assert edited[0].edited_of is not None
    assert report.edited_linked == 1
    originals = {p.file_hash for p in photos}
    assert edited[0].edited_of in originals


def test_extension_lying_file_is_typed_by_content(tmp_path):
    root = build_library(tmp_path / "lib")
    photos, _ = _scan(root)
    liar = [p for p in photos if any(x.name == "actually_jpeg.heic" for x in p.paths)]
    assert liar and liar[0].media_type is MediaType.IMAGE


def test_xmp_metadata_is_read_into_the_photo(tmp_path):
    root = build_library(tmp_path / "lib")
    photos, report = _scan(root)
    with_people = [p for p in photos if p.meta.people]
    assert len(with_people) == 1
    p = with_people[0]
    assert sorted(p.meta.people) == ["Alice", "Bob"]
    assert p.meta.face_regions[0].name == "Alice"
    assert p.meta.description == "morning on the beach"
    assert report.with_people == 1
    assert report.with_xmp == 1


def test_exif_date_and_gps_are_read(tmp_path):
    root = build_library(tmp_path / "lib")
    photos, report = _scan(root)
    dated = [p for p in photos if p.meta.taken_at_utc]
    assert dated
    geo = [p for p in photos if p.meta.gps]
    assert len(geo) == 1
    assert report.with_gps == 1
    assert any(p.meta.tz_source is TzSource.EXIF_OFFSET for p in photos)


def test_album_is_the_containing_folder_name(tmp_path):
    root = tmp_path / "lib"
    make_jpeg(root / "Holiday" / "x.jpg")
    photos, _ = _scan(root)
    assert photos[0].albums == ["Holiday"]


def test_photo_directly_in_root_has_no_album(tmp_path):
    root = tmp_path / "lib"
    make_jpeg(root / "x.jpg")
    photos, _ = _scan(root)
    assert photos[0].albums == []


def test_corrupt_file_is_indexed_but_reported_as_unreadable(tmp_path):
    """It must not pass as a healthy photo - doctor would report false health."""
    root = tmp_path / "lib"
    root.mkdir(parents=True)
    make_jpeg(root / "good.jpg")
    (root / "truncated.jpg").write_bytes(b"\xff\xd8\xff" + b"\x00" * 20)
    photos, report = _scan(root)
    assert any("good.jpg" in x.name for p in photos for x in p.paths)
    assert report.files_seen == 2
    assert len(report.unreadable) == 1
    bad_path, reason = report.unreadable[0]
    assert bad_path.name == "truncated.jpg"
    assert reason
    assert report.skipped.get("undecodable") == 1


def test_rescan_is_idempotent(tmp_path):
    root = build_library(tmp_path / "lib")
    first, _ = _scan(root)
    second, _ = _scan(root)
    assert {p.file_hash for p in first} == {p.file_hash for p in second}


def test_empty_directory_yields_empty_report(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    photos, report = _scan(root)
    assert photos == []
    assert report.files_seen == 0


def test_trash_folder_is_never_indexed(tmp_path):
    """Deleted photos must not become memories."""
    root = build_library(tmp_path / "lib")
    photos, report = _scan(root)
    assert report.excluded_dirs >= 1
    assert not any(p.name == "deleted.jpg" for photo in photos for p in photo.paths)


def test_orphan_sidecars_are_counted(tmp_path):
    """A sidecar with no photo means archive parts are missing."""
    root = build_library(tmp_path / "lib")
    _, report = _scan(root)
    assert report.orphan_sidecars >= 1


def test_sidecar_target_strips_supplemental_metadata_and_counter():
    from rekindle.sources.folder import _sidecar_target

    assert _sidecar_target("IMG_1234.jpg.supplemental-metadata.json") == "IMG_1234.jpg"
    assert _sidecar_target("DSC_0880.JPG.supplemental-metadata(1).json") == "DSC_0880.JPG"
    assert _sidecar_target("IMG_1234.jpg.supplemental-metad.json") == "IMG_1234.jpg"
    assert _sidecar_target("IMG_1234.jpg.json") == "IMG_1234.jpg"


def test_motion_photo_video_is_indexed_as_video_and_paired(tmp_path):
    root = build_library(tmp_path / "lib")
    photos, report = _scan(root)
    mp = [p for p in photos if any(x.suffix.lower() == ".mp" for x in p.paths)]
    assert len(mp) == 1
    assert mp[0].media_type is MediaType.VIDEO
    assert report.motion_pairs == 1


def test_photo_source_is_set_explicitly_not_only_by_default(tmp_path):
    """RULING 2: source must be set via `source=self.name`, not rely on the
    dataclass default. Every photo returned by FolderSource must carry it."""
    root = build_library(tmp_path / "lib")
    photos, _ = _scan(root)
    assert photos
    assert all(p.source == "folder" for p in photos)
    assert FolderSource().name == "folder"


def test_edit_links_when_filename_is_nfd_and_suffix_has_an_accent(tmp_path):
    """RULING 1: stem_index keys and _split_edited's base must share one
    Unicode normal form, and slicing must use the normalised length.

    "-modifié" (the French edited suffix) contains an accented "é" that is 1
    codepoint in NFC but 2 in NFD ("e" + combining acute). macOS hands back
    NFD filenames from the filesystem. If the base is sliced from the RAW
    (NFD) stem using the NFC suffix's length, one stray combining character
    is left dangling and the computed base never matches the plain-ASCII
    stem key registered for the original - edit-linking silently fails.
    This is exactly the platform+locale combination the NFC handling exists
    for, so it must actually work end to end, not just in the matching step.
    """
    import unicodedata

    root = tmp_path / "lib"
    make_jpeg(root / "IMG_6000.jpg", color=(1, 2, 3))
    nfd_edited_stem = unicodedata.normalize("NFD", "IMG_6000-modifié")
    assert nfd_edited_stem != "IMG_6000-modifié"  # sanity: genuinely decomposed
    make_jpeg(root / f"{nfd_edited_stem}.jpg", color=(4, 5, 6))

    photos, report = _scan(root)
    edited = [p for p in photos if "modifi" in next(iter(p.paths)).stem]
    assert len(edited) == 1
    assert edited[0].edited_of is not None
    assert report.edited_linked == 1
