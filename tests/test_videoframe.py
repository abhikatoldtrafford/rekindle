"""Videos as still frames: which videos, which frame, and what happens without ffmpeg.

**CI HAS NO FFMPEG, AND MOST OF THIS FILE STILL RUNS.** That is deliberate and
it is the point of the design: extraction is the only step that needs ffmpeg,
and everything downstream reads a cached FILE. So a test can put a JPEG in the
frame cache by hand and exercise the whole pipeline - fingerprint, dedup
inputs, composition, `resolve_path`, the semantic reader - with no video codec
anywhere. Only the handful of tests that actually run ffmpeg are skipped.

That is not a testing trick. It is the same property the feature promises the
user: two machines sharing a data directory build the same memory whether or
not either of them can decode video today.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from rekindle.db import PhotoStore
from rekindle.memory import composition as comp
from rekindle.memory import fingerprint as fp
from rekindle.memory import videoframe as vf
from rekindle.memory.index import MemoryIndex
from rekindle.models import MediaType, Photo, PhotoMeta

T0 = datetime(2020, 5, 1, 12, 0, tzinfo=UTC)

HAVE_FFMPEG = vf.have_ffmpeg()
needs_ffmpeg = pytest.mark.skipif(HAVE_FFMPEG is False, reason="needs ffmpeg and ffprobe on PATH")


def _photo(h: str, path: Path, media: MediaType = MediaType.VIDEO, **kw) -> Photo:
    return Photo(
        file_hash=h,
        paths=[path],
        media_type=media,
        meta=PhotoMeta(
            taken_at_utc=T0,
            taken_at_local=T0.replace(tzinfo=None),
            **kw,
        ),
        first_seen=T0,
        last_seen=T0,
        albums=["Photos from 2020"],
    )


def _detailed_jpeg(path: Path, *, blur: bool = False, size=(320, 240)) -> Path:
    """A frame with real high-frequency detail, so `sharpness` has something to
    measure - a flat colour scores the same however it is decoded."""
    im = Image.new("RGB", size, (30, 40, 60))
    draw = ImageDraw.Draw(im)
    for x in range(0, size[0], 7):
        draw.line([(x, 0), (x, size[1])], fill=(230, 220, 40), width=2)
    for y in range(0, size[1], 9):
        draw.line([(0, y), (size[0], y)], fill=(220, 30, 30), width=2)
    if blur:
        from PIL import ImageFilter

        im = im.filter(ImageFilter.GaussianBlur(6))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, "JPEG", quality=92)
    return path


def _cache_frame(data_dir: Path, file_hash: str, **kw) -> Path:
    """Put a still in the frame cache as if ffmpeg had extracted it."""
    return _detailed_jpeg(vf.frame_path(data_dir, file_hash), **kw)


# --------------------------------------------------------------------------
# which videos: the motion-photo pairing rule
#
# Measured on the reference library: of 1,117 videos, 640 are `<name>.MP`
# beside `<name>.MP.jpg` and 235 are `MVIMG_*.MP4` beside `MVIMG_*.jpg`. 875 of
# 1,117 (78%) already have their still in the library, and extracting a frame
# from those would put two near-identical images into the same memory.


def _dirmap(*items: tuple[Path, MediaType]) -> dict[tuple[Path, str], MediaType]:
    return {(p.parent, p.name.casefold()): t for p, t in items}


def test_a_pixel_motion_photo_half_is_recognised(tmp_path):
    video = tmp_path / "a" / "PXL_20260321_083923528.MP"
    still = tmp_path / "a" / "PXL_20260321_083923528.MP.jpg"
    assert vf.pairs_with_a_still(
        _photo("v", video), _dirmap((video, MediaType.VIDEO), (still, MediaType.IMAGE))
    )


def test_an_MVIMG_motion_photo_half_is_recognised(tmp_path):
    """Google's older naming: the STEM matches, the extension does not."""
    video = tmp_path / "a" / "MVIMG_20190910_125917.MP4"
    still = tmp_path / "a" / "MVIMG_20190910_125917.jpg"
    assert vf.pairs_with_a_still(
        _photo("v", video), _dirmap((video, MediaType.VIDEO), (still, MediaType.IMAGE))
    )


def test_the_pairing_is_case_insensitive(tmp_path):
    """A real export mixes .MP/.mp and .jpg/.JPG freely."""
    video = tmp_path / "a" / "pxl_1.mp"
    still = tmp_path / "a" / "PXL_1.MP.JPG"
    assert vf.pairs_with_a_still(
        _photo("v", video), _dirmap((video, MediaType.VIDEO), (still, MediaType.IMAGE))
    )


def test_a_same_named_still_in_ANOTHER_directory_does_not_pair(tmp_path):
    """Scoped to one directory, for the reason `propagate_to_derivatives`
    documents: a `.MP` in one album must never pair with an unrelated still of
    the same name in another."""
    video = tmp_path / "a" / "PXL_1.MP"
    elsewhere = tmp_path / "b" / "PXL_1.MP.jpg"
    assert not vf.pairs_with_a_still(
        _photo("v", video), _dirmap((video, MediaType.VIDEO), (elsewhere, MediaType.IMAGE))
    )


@pytest.mark.parametrize(
    ("video_name", "sibling_name"),
    [("PXL_1.MP", "PXL_1.MP.jpg"), ("MVIMG_1.MP4", "MVIMG_1.jpg")],
    ids=["MP rule", "MVIMG rule"],
)
def test_a_sibling_that_is_itself_a_VIDEO_does_not_count_as_the_still(
    tmp_path, video_name, sibling_name
):
    """Both rules, because they are two separate lookups and only one of them
    was checking the media type when this was written."""
    video = tmp_path / "a" / video_name
    fake = tmp_path / "a" / sibling_name
    assert not vf.pairs_with_a_still(
        _photo("v", video), _dirmap((video, MediaType.VIDEO), (fake, MediaType.VIDEO))
    )


def test_a_standalone_video_does_not_pair(tmp_path):
    video = tmp_path / "a" / "20131220_180302.mp4"
    assert not vf.pairs_with_a_still(_photo("v", video), _dirmap((video, MediaType.VIDEO)))


def test_index_by_dir_name_keeps_the_FIRST_claim_on_a_name(tmp_path):
    a = tmp_path / "d" / "x.jpg"
    mapping = vf.index_by_dir_name(
        [_photo("i", a, MediaType.IMAGE), _photo("v", a, MediaType.VIDEO)]
    )
    assert mapping[(a.parent, "x.jpg")] is MediaType.IMAGE


# --------------------------------------------------------------------------
# the reason a video has no still, per row


@pytest.mark.parametrize(
    "reason", [vf.ERR_NO_FFMPEG, vf.ERR_PAIRED, vf.ERR_NO_FRAME, vf.ERR_LEGACY]
)
def test_is_video_reason_knows_every_reason_this_module_writes(reason):
    """All four, not just the legacy one. `run_fingerprints` buckets on this,
    so a reason it does not recognise is counted as a decode FAILURE - a bug
    report about the user's library instead of a fact about ffmpeg."""
    assert vf.is_video_reason(reason)


@pytest.mark.parametrize("reason", ["undecodable", "missing", "unreadable", None, "", "vid"])
def test_is_video_reason_rejects_everything_else(reason):
    assert not vf.is_video_reason(reason)


def test_the_reason_a_video_has_no_still_is_STORED_not_just_counted(tmp_path):
    """ "It needs ffmpeg" is the only one a user can act on, and it is
    indistinguishable from "it is a video" unless the row says which."""
    data, _ = _store_with_video(tmp_path)
    with PhotoStore(data / "rekindle.sqlite") as store:
        fp.run_fingerprints(store, data_dir=data, ffmpeg=False)
        assert store.get("v").meta.phash_error == vf.ERR_NO_FFMPEG

    data2, _ = _store_with_video(tmp_path / "paired", paired=True)
    with PhotoStore(data2 / "rekindle.sqlite") as store:
        fp.run_fingerprints(store, data_dir=data2, ffmpeg=True)
        assert store.get("v").meta.phash_error == vf.ERR_PAIRED

    data3 = tmp_path / "broken"
    lib = data3 / "lib"
    lib.mkdir(parents=True)
    (lib / "bad.mp4").write_bytes(b"not a container")
    with PhotoStore(data3 / "rekindle.sqlite") as store:
        store.upsert_many([_photo("b", lib / "bad.mp4")])
        fp.run_fingerprints(store, data_dir=data3, ffmpeg=True)
        assert store.get("b").meta.phash_error == vf.ERR_NO_FRAME


# --------------------------------------------------------------------------
# without ffmpeg: excluded, and SAID SO


def test_without_ffmpeg_nothing_is_extracted_and_the_report_says_why(tmp_path):
    video = tmp_path / "lib" / "clip.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"not really a video")
    frames, report = vf.extract_for([_photo("v", video)], tmp_path / "data", {}, ffmpeg=False)
    assert frames == {}
    assert report.skipped_no_ffmpeg == 1
    assert report.extracted == 0
    assert report.ffmpeg_missing
    assert report.accounted


def test_a_paired_video_is_skipped_the_same_way_WITH_or_WITHOUT_ffmpeg(tmp_path):
    """The pairing check runs before the ffmpeg check on purpose: the count
    then means the same thing on a machine that has ffmpeg and one that does
    not, so two people comparing reports are comparing the same number."""
    video = tmp_path / "lib" / "PXL_1.MP"
    still = tmp_path / "lib" / "PXL_1.MP.jpg"
    mapping = _dirmap((video, MediaType.VIDEO), (still, MediaType.IMAGE))
    for available in (True, False):
        _, report = vf.extract_for(
            [_photo("v", video)], tmp_path / "data", mapping, ffmpeg=available
        )
        assert (report.paired, report.skipped_no_ffmpeg, report.extracted) == (1, 0, 0)
        assert report.accounted


def test_a_cached_frame_is_reused_without_ffmpeg(tmp_path):
    """The whole design in one test. The frame is DATA: once it exists, the
    machine that reads it never needs a video codec."""
    data = tmp_path / "data"
    cached = _cache_frame(data, "v")
    frames, report = vf.extract_for(
        [_photo("v", tmp_path / "lib" / "clip.mp4")], data, {}, ffmpeg=False
    )
    assert frames == {"v": cached}
    assert (report.extracted, report.already_cached, report.skipped_no_ffmpeg) == (1, 1, 0)
    assert report.accounted


def test_an_unreadable_video_is_a_failure_not_an_exception(tmp_path):
    """One bad file among 1,117 must not end the pass."""
    video = tmp_path / "lib" / "broken.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"\x00\x01\x02 definitely not a container")
    _, report = vf.extract_for([_photo("v", video)], tmp_path / "data", {}, ffmpeg=True)
    assert report.failed == 1
    assert report.extracted == 0
    assert report.accounted


def test_a_video_whose_file_is_gone_is_a_failure(tmp_path):
    _, report = vf.extract_for(
        [_photo("v", tmp_path / "nowhere" / "clip.mp4")], tmp_path / "data", {}, ffmpeg=True
    )
    assert report.failed == 1
    assert report.accounted


# --------------------------------------------------------------------------
# the fingerprint pass


def _store_with_video(tmp_path, *, paired: bool = False) -> tuple[Path, Path]:
    data = tmp_path / "data"
    lib = tmp_path / "lib"
    lib.mkdir(parents=True, exist_ok=True)
    video = lib / ("PXL_1.MP" if paired else "clip.mp4")
    video.write_bytes(b"pretend video")
    photos = [_photo("v", video)]
    if paired:
        still = _detailed_jpeg(lib / "PXL_1.MP.jpg")
        photos.append(_photo("s", still, MediaType.IMAGE))
    with PhotoStore(data / "rekindle.sqlite") as store:
        store.upsert_many(photos)
    return data, video


def test_a_video_with_a_cached_still_gets_a_REAL_fingerprint(tmp_path):
    """Not a marker row: the hash, sharpness, brightness, colour and stored
    dimensions are all measurements of the frame the renderer will draw."""
    data, _ = _store_with_video(tmp_path)
    _cache_frame(data, "v", size=(320, 240))
    with PhotoStore(data / "rekindle.sqlite") as store:
        report = fp.run_fingerprints(store, data_dir=data, ffmpeg=False)
        row = store.get("v")
    assert report.videos_hashed == 1
    assert report.hashed == 1
    assert report.skipped_video == 0
    assert report.accounted
    assert row.meta.phash is not None
    assert row.meta.phash_error is None
    assert row.meta.sharpness is not None
    assert row.meta.brightness is not None
    assert row.meta.colour is not None
    assert (row.meta.width, row.meta.height) == (320, 240)


def test_the_fingerprint_is_the_FRAMES_not_something_generic(tmp_path):
    """Two videos with different frames must not get the same hash - otherwise
    every video is a duplicate of every other one and dedup deletes them all."""
    data = tmp_path / "data"
    lib = tmp_path / "lib"
    lib.mkdir(parents=True)
    with PhotoStore(data / "rekindle.sqlite") as store:
        store.upsert_many([_photo("v1", lib / "a.mp4"), _photo("v2", lib / "b.mp4")])
    _cache_frame(data, "v1")
    _cache_frame(data, "v2", blur=True)
    with PhotoStore(data / "rekindle.sqlite") as store:
        fp.run_fingerprints(store, data_dir=data, ffmpeg=False)
        one, two = store.get("v1"), store.get("v2")
    assert one.meta.sharpness > two.meta.sharpness, "the blurred frame must score lower"
    assert one.meta.colour is not None and two.meta.colour is not None


def test_a_video_with_no_still_is_skipped_and_stays_retryable(tmp_path):
    """`iter_unfingerprinted` excludes a row with a recorded reason, and the
    commonest reason here is "ffmpeg was not installed" - which stops being
    true the moment somebody installs it. So the video pass does not use that
    predicate, and a second run picks the video up."""
    data, _ = _store_with_video(tmp_path)
    with PhotoStore(data / "rekindle.sqlite") as store:
        first = fp.run_fingerprints(store, data_dir=data, ffmpeg=False)
    assert (first.skipped_video, first.videos_hashed) == (1, 0)
    assert first.frames.skipped_no_ffmpeg == 1

    _cache_frame(data, "v")  # as if ffmpeg had arrived
    with PhotoStore(data / "rekindle.sqlite") as store:
        second = fp.run_fingerprints(store, data_dir=data, ffmpeg=False)
        row = store.get("v")
    assert second.videos_hashed == 1
    assert row.meta.phash is not None


def test_a_video_already_fingerprinted_is_not_done_again(tmp_path):
    data, _ = _store_with_video(tmp_path)
    _cache_frame(data, "v")
    with PhotoStore(data / "rekindle.sqlite") as store:
        fp.run_fingerprints(store, data_dir=data, ffmpeg=False)
        again = fp.run_fingerprints(store, data_dir=data, ffmpeg=False)
    assert again.considered == 0
    assert again.accounted


def test_a_motion_photo_half_is_never_fingerprinted_even_with_a_frame_cached(tmp_path):
    """78% of the reference library's videos are these. Their still is already
    in the library, and a second near-identical image in the same memory is
    a duplicate this pass would have created."""
    data, _ = _store_with_video(tmp_path, paired=True)
    with PhotoStore(data / "rekindle.sqlite") as store:
        report = fp.run_fingerprints(store, data_dir=data, ffmpeg=True)
        row = store.get("v")
    assert report.frames.paired == 1
    assert report.frames.extracted == 0
    assert row.meta.phash is None
    assert not vf.frame_path(data, "v").exists()


def test_without_a_data_dir_the_old_behaviour_is_exact(tmp_path):
    """Every existing caller passes only a store, and must see what it saw."""
    data, _ = _store_with_video(tmp_path)
    _cache_frame(data, "v")  # present, and deliberately not looked at
    with PhotoStore(data / "rekindle.sqlite") as store:
        report = fp.run_fingerprints(store)
        row = store.get("v")
    assert report.frames is None
    assert report.skipped_video == 1
    assert report.videos_hashed == 0
    assert row.meta.phash is None
    assert row.meta.phash_error == fp.ERR_VIDEO


# --------------------------------------------------------------------------
# downstream: the video is a photograph everywhere or nowhere


def test_resolve_path_hands_out_the_STILL_and_never_the_video(tmp_path):
    """`frames.fit_photo` opens whatever it is given with Pillow. Handing it
    an .mp4 raises, so this substitution is the only thing between the design
    and a crash in the renderer."""
    data, video = _store_with_video(tmp_path)
    still = _cache_frame(data, "v")
    with PhotoStore(data / "rekindle.sqlite") as store:
        fp.run_fingerprints(store, data_dir=data, ffmpeg=False)
        index = MemoryIndex.open(store)
        photo = index.get("v")
        assert index.resolve_path(photo) == still
        assert index.resolve_path(photo) != video


def test_resolve_path_returns_None_for_a_video_with_no_still(tmp_path):
    data, _ = _store_with_video(tmp_path)
    with PhotoStore(data / "rekindle.sqlite") as store:
        index = MemoryIndex.open(store)
        assert index.resolve_path(index.get("v")) is None


def test_composition_admits_a_video_that_has_a_still(tmp_path):
    data, _ = _store_with_video(tmp_path)
    _cache_frame(data, "v", size=(1600, 1200))
    with PhotoStore(data / "rekindle.sqlite") as store:
        fp.run_fingerprints(store, data_dir=data, ffmpeg=False)
        photos = list(store.iter_photos())
    kept, report = comp.compose(photos)
    assert [p.file_hash for p in kept] == ["v"]
    assert comp.DROP_VIDEO not in report.dropped


def test_composition_still_refuses_a_video_without_one(tmp_path):
    data, _ = _store_with_video(tmp_path)
    with PhotoStore(data / "rekindle.sqlite") as store:
        fp.run_fingerprints(store, data_dir=data, ffmpeg=False)
        photos = list(store.iter_photos())
    kept, report = comp.compose(photos)
    assert kept == []
    assert report.dropped[comp.DROP_VIDEO] == 1


def test_the_semantic_reader_offers_a_framed_video_and_points_at_the_frame(tmp_path):
    """The embedder and the face gate both decode `existing_path()`. If that
    returned the .mp4 the encoder would fail and the face gate - which governs
    what may be PUBLISHED - would silently have no verdict for it."""
    from rekindle.semantic.photos import PhotoIndexReader, ReadFilter

    data, video = _store_with_video(tmp_path)
    still = _cache_frame(data, "v")
    with PhotoStore(data / "rekindle.sqlite") as store:
        fp.run_fingerprints(store, data_dir=data, ffmpeg=False)
    with PhotoIndexReader(data / "rekindle.sqlite") as reader:
        rows = list(reader.iter_photos(ReadFilter(images_only=True)))
        assert [r.file_hash for r in rows] == ["v"]
        row = rows[0]
        assert row.still == still
        assert row.decode_paths == (still,)
        assert row.existing_path() == still
        assert row.paths == (video,), "the video's own path is kept, not overwritten"
        assert reader.count(ReadFilter(images_only=True)) == 1


def test_the_semantic_reader_never_claims_a_still_that_is_not_THERE(tmp_path):
    """A deleted frame cache with the fingerprints still in the index. `still`
    must read None, not a path to nothing - a caller that trusts it and opens
    the file gets a traceback instead of a skipped photo."""
    from rekindle.semantic.photos import PhotoIndexReader, ReadFilter

    data, _ = _store_with_video(tmp_path)
    frame = _cache_frame(data, "v")
    with PhotoStore(data / "rekindle.sqlite") as store:
        fp.run_fingerprints(store, data_dir=data, ffmpeg=False)
    frame.unlink()
    with PhotoIndexReader(data / "rekindle.sqlite") as reader:
        rows = list(reader.iter_photos(ReadFilter(images_only=True)))
        assert [r.file_hash for r in rows] == ["v"], "the index still says it has a phash"
        assert rows[0].still is None
        assert rows[0].existing_path() is None


def test_the_semantic_reader_hides_a_video_with_no_frame(tmp_path):
    from rekindle.semantic.photos import PhotoIndexReader, ReadFilter

    data, _ = _store_with_video(tmp_path)
    with PhotoStore(data / "rekindle.sqlite") as store:
        fp.run_fingerprints(store, data_dir=data, ffmpeg=False)
    with PhotoIndexReader(data / "rekindle.sqlite") as reader:
        assert list(reader.iter_photos(ReadFilter(images_only=True))) == []
        assert reader.count(ReadFilter(images_only=True)) == 0


def test_a_framed_video_can_actually_be_rendered(tmp_path):
    """End to end through the real renderer, because "it should work" is how
    the .mp4-to-Pillow crash would have shipped."""
    from rekindle.memory.render.frames import fit_photo

    data, _ = _store_with_video(tmp_path)
    _cache_frame(data, "v", size=(1600, 1200))
    with PhotoStore(data / "rekindle.sqlite") as store:
        fp.run_fingerprints(store, data_dir=data, ffmpeg=False)
        index = MemoryIndex.open(store)
        path = index.resolve_path(index.get("v"))
    frame, _mode = fit_photo(path, (640, 480))
    assert frame.size == (640, 480)


# --------------------------------------------------------------------------
# the screenshot heuristic, and the false positive this feature created


def test_a_video_frame_at_a_SCREEN_SIZE_is_not_a_screenshot(tmp_path):
    """Found by measurement, not reasoning.

    When extraction first ran on the reference library, 87 of the 242
    extracted frames (36%) were classified as screenshots and dropped. None
    was one. The commonest video resolutions ARE the commonest screen sizes -
    70 frames at 1920x1080, 29 at 1080x1920, 8 at 1280x720 - and a video
    carries no camera make or model at all (0 of 242), so the conjunction that
    protects photographs protected nothing.
    """
    frame = _photo("v", tmp_path / "clip.mp4")
    frame.meta.width, frame.meta.height = 1920, 1080
    frame.meta.phash = 1
    assert not comp.is_screenshot(frame)

    same_pixels_but_a_photo = _photo("i", tmp_path / "wallpaper.jpg", MediaType.IMAGE)
    same_pixels_but_a_photo.meta.width, same_pixels_but_a_photo.meta.height = 1920, 1080
    assert comp.is_screenshot(same_pixels_but_a_photo), (
        "the size branch must still fire for a downloaded image, which is what it is for"
    )


@pytest.mark.parametrize("size", [(1920, 1080), (1080, 1920), (1280, 720), (720, 1280)])
def test_the_common_video_resolutions_all_survive(tmp_path, size):
    frame = _photo("v", tmp_path / "clip.mp4")
    frame.meta.width, frame.meta.height = size
    frame.meta.phash = 1
    assert not comp.is_screenshot(frame)


def test_a_SCREEN_RECORDING_is_still_refused(tmp_path):
    """The filename branch is a positive assertion by the operating system,
    and a screen recording is exactly what this gate exists to keep out."""
    frame = _photo("v", tmp_path / "Screenshot_20161008-222024.mp4")
    frame.meta.width, frame.meta.height = 1080, 1920
    frame.meta.phash = 1
    assert comp.is_screenshot(frame)


# --------------------------------------------------------------------------
# the parts that genuinely need ffmpeg


@pytest.fixture
def real_video(tmp_path):
    """A four-second clip whose SECOND half is sharp and whose first is not.

    Built with ffmpeg from generated frames, so the right answer is known: a
    strategy that takes the first frame gets the blur, and one that samples
    later does not.
    """
    frames = tmp_path / "src"
    frames.mkdir()
    for i in range(1, 41):
        _detailed_jpeg(frames / f"f{i:03d}.jpg", blur=i <= 20, size=(320, 240))
    out = tmp_path / "clip.mp4"
    import subprocess

    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-framerate",
            "10",
            "-i",
            str(frames / "f%03d.jpg"),
            "-c:v",
            "mjpeg",
            "-q:v",
            "2",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


@needs_ffmpeg
def test_ffprobe_reads_the_duration(real_video):
    assert 3.5 <= vf.duration_s(real_video) <= 4.5


@needs_ffmpeg
def test_extraction_avoids_the_blurred_opening(tmp_path, real_video):
    """The documented failure: the first frame is a blur or a lens cap.

    The clip's first two seconds are blurred and its last two are not, so the
    first frame is measurably the wrong answer and best-of-five must not pick
    it.
    """
    first = tmp_path / "first.jpg"
    assert vf._grab(real_video, 0.0, first)
    with Image.open(first) as im:
        first_score = fp.sharpness(im.convert("RGB"))

    got = vf.extract_frame(real_video, tmp_path / "out.jpg")
    assert got is not None
    assert got.candidates == 5
    assert got.at_fraction >= 0.5, f"picked {got.at_fraction}, which is inside the blur"
    assert got.sharpness > first_score * 1.5


@needs_ffmpeg
def test_extraction_is_deterministic(tmp_path, real_video):
    """Same library in, same memory out - the project's binding constraint,
    and this is the only step with a subprocess in it."""
    a = vf.extract_frame(real_video, tmp_path / "a.jpg")
    b = vf.extract_frame(real_video, tmp_path / "b.jpg")
    assert a is not None and b is not None
    assert a.at_fraction == b.at_fraction
    assert (tmp_path / "a.jpg").read_bytes() == (tmp_path / "b.jpg").read_bytes()


@needs_ffmpeg
def test_a_frame_lands_in_the_cache_under_the_file_hash(tmp_path, real_video):
    data = tmp_path / "data"
    frames, report = vf.extract_for([_photo("abc123", real_video)], data, {})
    assert report.extracted == 1
    assert report.accounted
    assert frames["abc123"] == data / "frames" / "abc123.jpg"
    assert frames["abc123"].is_file()
    with Image.open(frames["abc123"]) as im:
        assert im.size == (320, 240)


@needs_ffmpeg
def test_no_candidate_file_is_left_behind(tmp_path, real_video):
    """The extractor writes a temporary candidate per sample. Leaving 244 of
    them in the user's data directory would be a silent mess."""
    data = tmp_path / "data"
    vf.extract_for([_photo("abc123", real_video)], data, {})
    assert sorted(p.name for p in (data / "frames").iterdir()) == ["abc123.jpg"]


@needs_ffmpeg
def test_the_whole_pass_runs_with_a_real_video(tmp_path, real_video):
    data = tmp_path / "data"
    shutil.copy2(real_video, tmp_path / "lib_clip.mp4")
    with PhotoStore(data / "rekindle.sqlite") as store:
        store.upsert_many([_photo("v", tmp_path / "lib_clip.mp4")])
        report = fp.run_fingerprints(store, data_dir=data)
        row = store.get("v")
    assert report.videos_hashed == 1
    assert row.meta.phash is not None
    assert (row.meta.width, row.meta.height) == (320, 240)
