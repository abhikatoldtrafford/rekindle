from rekindle.models import MediaType, TzSource
from rekindle.sources.folder import FolderSource
from tests.fixtures.gen import build_library, make_jpeg, make_xmp_sidecar


def _scan(root):
    return FolderSource().scan(root)


def test_indexes_images_and_skips_non_media(tmp_path):
    root = build_library(tmp_path / "lib")
    photos, report = _scan(root)
    assert report.skipped.get("not_media", 0) >= 1  # notes.txt
    names = {p.name for photo in photos for p in photo.paths}
    assert "notes.txt" not in names
    # A real check on media_indexed: the fixture library has exactly 9
    # distinct media items once cross-folder duplicates are merged
    # (IMG_0001, IMG_0001-edited, IMG_0002, IMG_0003, actually_jpeg.heic,
    # PXL_0001.MP.jpg, PXL_0001.MP, PXL_0001.MP-edited, IMG_8000). A wrong
    # count here - counting files_seen instead of distinct photos, or missing
    # the dedupe - would fail this even though `media_indexed == len(photos)`
    # trivially holds either way.
    assert report.media_indexed == 9
    assert len(photos) == 9


def test_json_sidecars_are_ignored_but_counted(tmp_path):
    root = build_library(tmp_path / "lib")
    _, report = _scan(root)
    # Exact count (FIX 5): >=1 would also pass if .xmp were miscounted as
    # JSON, or if .aae/.thm leaked into the count (FIX 1). The fixture has
    # exactly 3 real Google JSON sidecars (IMG_0001.jpg.json,
    # IMG_9999...supplemental-metadata.json, DSC_0880...supplemental-
    # metadata(1).json); the .xmp and .aae sidecars must not add to this.
    assert report.json_sidecars == 3


def test_duplicate_across_folders_becomes_one_photo_with_both_albums(tmp_path):
    root = build_library(tmp_path / "lib")
    photos, report = _scan(root)
    dupes = [p for p in photos if len(p.paths) > 1]
    # IMG_0001 (year + album) and IMG_8000 (FIX 4's AAA_First/ZZZ_Second pair).
    assert len(dupes) == 2
    img_0001 = next(p for p in dupes if any(x.name == "IMG_0001.jpg" for x in p.paths))
    assert sorted(img_0001.albums) == ["Goa Trip", "Photos from 2014"]
    assert report.duplicates_merged == 2


def test_edited_variant_links_to_its_original(tmp_path):
    root = build_library(tmp_path / "lib")
    photos, report = _scan(root)
    edited = [p for p in photos if any("-edited" in x.name for x in p.paths)]
    # IMG_0001-edited and PXL_0001.MP-edited (FIX 2's motion-photo edit).
    assert len(edited) == 2
    assert all(p.edited_of is not None for p in edited)
    assert report.edited_linked == 2
    originals = {p.file_hash for p in photos}
    assert all(p.edited_of in originals for p in edited)


def test_extension_lying_file_is_typed_by_content(tmp_path):
    root = build_library(tmp_path / "lib")
    photos, _ = _scan(root)
    liar = [p for p in photos if any(x.name == "actually_jpeg.heic" for x in p.paths)]
    assert liar and liar[0].media_type is MediaType.IMAGE


def test_xmp_metadata_is_read_into_the_photo(tmp_path):
    root = build_library(tmp_path / "lib")
    photos, report = _scan(root)
    with_people = [p for p in photos if p.meta.people]
    # IMG_0002 (Alice, Bob) and IMG_8000 (Carol - FIX 4's duplicate whose
    # sidecar lives beside only the second copy).
    assert len(with_people) == 2
    portrait = next(p for p in with_people if any(x.name == "IMG_0002.jpg" for x in p.paths))
    assert sorted(portrait.meta.people) == ["Alice", "Bob"]
    assert portrait.meta.face_regions[0].name == "Alice"
    assert portrait.meta.description == "morning on the beach"
    dup = next(p for p in with_people if any(x.name == "IMG_8000.jpg" for x in p.paths))
    assert dup.meta.people == ["Carol"]
    assert report.with_people == 2
    assert report.with_xmp == 2


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
    # Exact count (FIX 6): protects FIX 1's change. Only IMG_9999 and
    # DSC_0880's JSON sidecars are genuinely orphaned; the fixture's orphan
    # .aae (IMG_7000.aae) must NOT add to this - it is not a Google JSON
    # sidecar, so _sidecar_target should never even see it.
    assert report.orphan_sidecars == 2


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
    dataclass default `"folder"`. A subclass with a DIFFERENT name proves
    it - `source == "folder"` on a FolderSource-default instance would still
    be true even if `source=self.name` were dropped entirely, since that
    happens to be the dataclass default too. Only a different `name` value
    exposes the difference (FIX 3)."""

    class ProbeSource(FolderSource):
        name = "probe"

    root = build_library(tmp_path / "lib")
    photos, _ = ProbeSource().scan(root)
    assert photos
    assert all(p.source == "probe" for p in photos)
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


def test_aae_sidecar_is_skipped_not_miscounted_as_json_or_orphan(tmp_path):
    """FIX 1: .aae (and .thm) share _SIDECAR_EXTS with .json but are not
    Google JSON sidecars. _sidecar_target only understands the Google
    naming convention, so feeding an .aae through it returns the filename
    unchanged, which never matches any media file and would be counted as a
    permanently orphaned sidecar - inflating doctor's "missing archive
    parts" warning on an iPhone library that never had one."""
    root = tmp_path / "lib"
    make_jpeg(root / "IMG_1.jpg")
    (root / "IMG_1.aae").write_text("apple edit sidecar", encoding="utf-8")
    (root / "orphan.aae").write_text("no matching photo at all", encoding="utf-8")

    photos, report = _scan(root)
    assert len(photos) == 1
    assert report.json_sidecars == 0
    assert report.orphan_sidecars == 0


def test_stem_index_prefers_image_over_video_sharing_a_stem(tmp_path):
    """FIX 2 (guard), direct unit test. With Google's real motion-photo
    naming a video ("PXL_x.MP", stem "PXL_x") and its still ("PXL_x.MP.jpg",
    stem "PXL_x.MP") do NOT share a stem - see FINDING B - so this
    exercises the stem_index guard with a generic same-stem image/video pair
    instead. Without the guard, the video (visited after the image
    alphabetically) would overwrite the image's stem_index entry, and the
    edited variant would silently link to the wrong original."""
    root = tmp_path / "lib"
    make_jpeg(root / "clip.jpg", color=(9, 8, 7))
    (root / "clip.mov").write_bytes(b"\x00\x00\x00\x14ftypqt  \x00\x00\x02\x00" + b"\x00" * 128)
    make_jpeg(root / "clip-edited.jpg", color=(1, 2, 3))

    photos, _ = _scan(root)
    still = next(p for p in photos if any(x.name == "clip.jpg" for x in p.paths))
    video = next(p for p in photos if any(x.name == "clip.mov" for x in p.paths))
    edited = next(p for p in photos if any(x.name == "clip-edited.jpg" for x in p.paths))
    assert video.media_type is MediaType.VIDEO
    assert edited.edited_of == still.file_hash
    assert edited.edited_of != video.file_hash


def test_duplicate_merge_reads_metadata_present_on_only_one_copy(tmp_path):
    """FIX 4: the duplicate branch must still call merge_meta. The fixture's
    two copies are byte-identical but only the SECOND-visited one
    ("ZZZ" sorts after "AAA") carries an XMP sidecar - the exact Takeout
    layout where an album copy has no sidecar of its own. Deleting the
    merge_meta call on the duplicate branch leaves the photo with whichever
    copy was seen FIRST (no sidecar), silently losing this metadata - and
    all 18 pre-fix tests stayed green with that bug in place."""
    root = tmp_path / "lib"
    bytes_ = make_jpeg(root / "AAA" / "dup.jpg", color=(1, 2, 3)).read_bytes()
    second = root / "ZZZ" / "dup.jpg"
    second.parent.mkdir(parents=True, exist_ok=True)
    second.write_bytes(bytes_)
    make_xmp_sidecar(second, people=["Dana"], description="only on the second copy")

    photos, report = _scan(root)
    assert len(photos) == 1
    dup = photos[0]
    assert dup.meta.people == ["Dana"]
    assert report.duplicates_merged == 1
    assert report.with_xmp == 1


def test_duplicate_merge_does_not_double_count_with_xmp_for_one_photo(tmp_path):
    """FIX 4 (guard): when BOTH copies of a duplicate carry their own XMP
    sidecar, `with_xmp` must still count the PHOTO once, not once per path -
    the saw_xmp guard exists for exactly this. Also confirms merge_meta
    unions data from both copies rather than keeping only the first."""
    root = tmp_path / "lib"
    first = make_jpeg(root / "AAA" / "dup.jpg", color=(4, 5, 6))
    second = root / "ZZZ" / "dup.jpg"
    second.parent.mkdir(parents=True, exist_ok=True)
    second.write_bytes(first.read_bytes())
    make_xmp_sidecar(first, description="first copy note")
    make_xmp_sidecar(second, people=["Dana"], description="second copy note")

    photos, report = _scan(root)
    assert len(photos) == 1
    dup = photos[0]
    assert dup.meta.people == ["Dana"]
    assert report.duplicates_merged == 1
    assert report.with_xmp == 1  # one PHOTO has xmp, even though both its paths do


def test_duplicate_first_seen_is_the_earliest_and_last_seen_the_latest(tmp_path):
    """FIX 8: the duplicate branch narrows last_seen to the MAX mtime but
    left first_seen untouched. Give the FIRST-visited copy the NEWER mtime
    and the duplicate (second-visited) copy the OLDER one, so first_seen can
    only end up correct (the older date) if it is explicitly narrowed with
    min() on the duplicate branch."""
    import os
    from datetime import UTC, datetime

    root = tmp_path / "lib"
    older = datetime(2010, 1, 1, tzinfo=UTC)
    newer = datetime(2020, 1, 1, tzinfo=UTC)

    first = make_jpeg(root / "AAA" / "dup.jpg", color=(7, 8, 9))  # visited first
    second = root / "ZZZ" / "dup.jpg"  # visited second -> duplicate branch
    second.parent.mkdir(parents=True, exist_ok=True)
    second.write_bytes(first.read_bytes())
    os.utime(first, (newer.timestamp(), newer.timestamp()))
    os.utime(second, (older.timestamp(), older.timestamp()))

    photos, _ = _scan(root)
    assert len(photos) == 1
    dup = photos[0]
    assert dup.first_seen == older
    assert dup.last_seen == newer


def test_unstatable_long_path_is_reported_not_silently_dropped(tmp_path, monkeypatch):
    """FIX 7: Path.is_file() swallows a stat() failure and returns False -
    exactly as it would for an ordinary directory. Without an explicit
    check, a file whose path is too long for Windows' MAX_PATH is simply
    invisible: neither counted in files_seen nor reported anywhere, which is
    exactly the silent drop the source's constraints forbid.

    is_file()/is_dir() are monkeypatched (scoped to one marker path) to
    simulate a genuine stat() failure without needing to actually create an
    unstatable file on disk, which is unreliable across platforms."""
    from pathlib import Path

    import rekindle.sources.folder as folder_mod

    root = tmp_path / "lib"
    make_jpeg(root / "good.jpg")
    ghost = root / "ghost_too_long.jpg"
    ghost.write_bytes(b"\xff\xd8\xff" + b"\x00" * 10)  # a real, short-path file

    real_is_file = Path.is_file
    real_is_dir = Path.is_dir

    def fake_is_file(self):
        return False if self == ghost else real_is_file(self)

    def fake_is_dir(self):
        return False if self == ghost else real_is_dir(self)

    monkeypatch.setattr(Path, "is_file", fake_is_file)
    monkeypatch.setattr(Path, "is_dir", fake_is_dir)
    monkeypatch.setattr(folder_mod, "is_long_path", lambda p: p == ghost)

    photos, report = _scan(root)
    assert report.files_seen == 2
    assert ghost in report.long_paths
    assert report.skipped.get("long_path_unreadable") == 1
    assert not any(x.name == "ghost_too_long.jpg" for p in photos for x in p.paths)
