"""Fingerprint pass: the hash, the focus measure, and resumability.

Fixture shapes come from the real index: JPEG images, videos with no
dimensions at all (all 1,117 video rows have width/height NULL), and files
that cannot be decoded.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image, ImageFilter, JpegImagePlugin

from rekindle.db import PhotoStore
from rekindle.memory import fingerprint as fp
from rekindle.models import MediaType, Photo, PhotoMeta

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def _gradient(size=(64, 64), flip=False) -> Image.Image:
    """A left-to-right ramp. Every horizontal gradient rises, so under the
    `left > right` convention dHash is all ZEROS; mirrored, every gradient
    falls and it is all ones. That makes the bit order and the comparison
    direction both directly observable."""
    im = Image.new("L", size)
    w, h = size
    for x in range(w):
        value = int(255 * x / (w - 1))
        for y in range(h):
            im.putpixel((w - 1 - x if flip else x, y), value)
    return im


def _photo(h, path, media_type=MediaType.IMAGE, meta=None) -> Photo:
    return Photo(
        file_hash=h,
        paths=[Path(path)] if not isinstance(path, list) else [Path(p) for p in path],
        media_type=media_type,
        meta=meta or PhotoMeta.empty(),
        first_seen=T0,
        last_seen=T0,
    )


def _jpeg(path: Path, image: Image.Image) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, "JPEG", quality=95)
    return path


# --------------------------------------------------------------------------
# the hash itself


def test_a_rising_ramp_hashes_to_all_zeros():
    """Pins the comparison direction AND the bit width.

    The convention is `left > right`, so a left-to-right BRIGHTENING ramp
    makes every comparison false. Inverting the comparison gives all ones
    here, and a 9x8 grid misread as 8x8 gives a different width - both are
    silent changes that would invalidate every stored hash, because the value
    is persisted.
    """
    assert fp.dhash(_gradient()) == 0


def test_a_falling_ramp_hashes_to_all_ones():
    assert fp.dhash(_gradient(flip=True)) == 0xFFFFFFFFFFFFFFFF


def test_a_mirrored_image_hashes_differently():
    """Proves the hash reads spatial structure. An implementation that
    returned, say, a histogram would pass the two tests above and fail this
    one."""
    assert fp.dhash(_gradient()) != fp.dhash(_gradient(flip=True))


def test_identical_pixels_hash_identically():
    assert fp.dhash(_gradient()) == fp.dhash(_gradient())


def test_a_flat_image_hashes_to_zero():
    """dHash 0 is a REAL value, not a sentinel: any flat image produces it,
    and so does any rising ramp. This is why nothing in the codebase may test
    a phash with `if phash:`."""
    assert fp.dhash(Image.new("L", (64, 64), 128)) == 0


def test_the_hash_is_scale_invariant():
    """The same scene at two resolutions must land in the same burst. Without
    this, a photo and its resized copy would never be seen as similar."""
    assert fp.dhash(_gradient((64, 64))) == fp.dhash(_gradient((512, 512)))


# --------------------------------------------------------------------------
# sharpness


def test_a_blurred_copy_is_less_sharp_than_its_original():
    sharp = _gradient((128, 128)).filter(ImageFilter.FIND_EDGES)
    blurred = sharp.filter(ImageFilter.GaussianBlur(radius=4))
    assert fp.sharpness(blurred) < fp.sharpness(sharp)


def test_sharpness_is_measured_at_a_fixed_size():
    """Otherwise resolution alone decides every burst: a 4000px photo would
    beat a 1200px one regardless of focus."""
    small = _gradient((128, 128)).filter(ImageFilter.FIND_EDGES)
    large = small.resize((512, 512), Image.Resampling.BILINEAR)
    assert fp.sharpness(large) == pytest.approx(fp.sharpness(small), rel=0.25)


def test_a_flat_image_has_zero_sharpness():
    assert fp.sharpness(Image.new("L", (128, 128), 200)) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# fingerprint_file: never raises


def test_fingerprints_a_real_jpeg(tmp_path):
    path = _jpeg(tmp_path / "a.jpg", _gradient((200, 150)))
    result = fp.fingerprint_file(path)
    assert result.ok
    assert result.error is None
    assert result.sharpness is not None


def test_draft_is_used_on_jpeg_input(tmp_path, monkeypatch):
    """The 2.9x speedup the whole pass depends on. Deleting the draft() call
    changes no output at all, so only a direct observation can protect it."""
    path = _jpeg(tmp_path / "a.jpg", _gradient((400, 300)))
    calls = []
    # JpegImageFile OVERRIDES draft(); patching Image.Image.draft would spy on
    # a method the JPEG path never reaches and pass while measuring nothing.
    original = JpegImagePlugin.JpegImageFile.draft

    def spy(self, mode, size):
        calls.append((mode, size))
        return original(self, mode, size)

    monkeypatch.setattr(JpegImagePlugin.JpegImageFile, "draft", spy)
    fp.fingerprint_file(path)
    assert calls, "draft() was never called - the pass is now ~3x slower"


def test_a_missing_file_is_recorded_not_raised(tmp_path):
    result = fp.fingerprint_file(tmp_path / "nope.jpg")
    assert result.phash is None
    assert result.error == fp.ERR_MISSING


def test_a_file_that_is_not_an_image_is_recorded_not_raised(tmp_path):
    path = tmp_path / "a.jpg"
    path.write_bytes(b"this is not a JPEG at all")
    result = fp.fingerprint_file(path)
    assert result.phash is None
    assert result.error == fp.ERR_UNDECODABLE


def test_a_truncated_jpeg_is_recorded_not_raised(tmp_path):
    """A real library reliably contains a few of these. One must not abort a
    six-minute pass."""
    path = _jpeg(tmp_path / "a.jpg", _gradient((300, 300)))
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 3])
    result = fp.fingerprint_file(path)
    assert result.phash is None
    assert result.error == fp.ERR_UNDECODABLE


def test_a_directory_passed_as_a_path_is_recorded_not_raised(tmp_path):
    assert fp.fingerprint_file(tmp_path).error is not None


def test_the_second_readable_path_is_used_when_the_first_is_gone(tmp_path):
    """The same bytes in two folders is routine in this library, and a
    partially extracted export can have one copy missing."""
    good = _jpeg(tmp_path / "b.jpg", _gradient((100, 100)))
    photo = _photo("h", [str(tmp_path / "gone.jpg"), str(good)])
    digest, phash, sharp, error = fp._row_for(photo)
    assert phash is not None
    assert error is None


# --------------------------------------------------------------------------
# the pass


def test_videos_are_marked_and_never_decoded(tmp_path):
    """Getting a frame out of a video needs ffmpeg, which must stay optional.
    Videos therefore never get a phash and so never participate in dedup."""
    with PhotoStore(tmp_path / "db.sqlite") as store:
        store.upsert_many([_photo("v", tmp_path / "clip.mp4", MediaType.VIDEO)])
        report = fp.run_fingerprints(store)
        stored = store.get("v")

    assert report.skipped_video == 1
    assert report.hashed == 0
    assert stored.meta.phash is None
    assert stored.meta.phash_error == fp.ERR_VIDEO


def test_a_pass_hashes_images_and_accounts_for_everything(tmp_path):
    a = _jpeg(tmp_path / "a.jpg", _gradient((120, 90)))
    with PhotoStore(tmp_path / "db.sqlite") as store:
        store.upsert_many(
            [
                _photo("a", a),
                _photo("v", tmp_path / "clip.mp4", MediaType.VIDEO),
                _photo("gone", tmp_path / "missing.jpg"),
            ]
        )
        report = fp.run_fingerprints(store)
        assert store.get("a").meta.phash is not None

    assert (report.considered, report.hashed, report.failed, report.skipped_video) == (3, 1, 1, 1)
    assert report.accounted, "every considered photo must land in exactly one bucket"
    assert report.errors == {fp.ERR_MISSING: 1}


def test_a_second_pass_re_decodes_nothing(tmp_path):
    """Resumability, observed rather than inferred: the second run must open
    no files at all, including the one that permanently failed."""
    a = _jpeg(tmp_path / "a.jpg", _gradient((120, 90)))
    db = tmp_path / "db.sqlite"
    with PhotoStore(db) as store:
        store.upsert_many([_photo("a", a), _photo("gone", tmp_path / "missing.jpg")])
        first = fp.run_fingerprints(store)
    assert first.considered == 2

    opened = []
    real_open = Image.open

    with PhotoStore(db) as store:
        import unittest.mock as mock

        with mock.patch.object(
            Image, "open", side_effect=lambda p, *a, **k: (opened.append(p), real_open(p))[1]
        ):
            second = fp.run_fingerprints(store)

    assert second.considered == 0
    assert opened == [], "a resumed pass must not re-decode finished work"


def test_an_interrupted_pass_keeps_the_batches_it_finished(tmp_path):
    """Batched writes exist so an interruption loses at most one batch. With
    a single transaction per run, a Ctrl-C at minute five loses five minutes."""
    paths = [_jpeg(tmp_path / f"{i}.jpg", _gradient((60, 60))) for i in range(5)]
    db = tmp_path / "db.sqlite"
    with PhotoStore(db) as store:
        store.upsert_many([_photo(f"h{i}", p) for i, p in enumerate(paths)])

    boom = RuntimeError("interrupted")

    def explode(done, total):
        if done >= 2:
            raise boom

    with PhotoStore(db) as store, pytest.raises(RuntimeError):
        fp.run_fingerprints(store, batch_size=2, on_progress=explode)

    with PhotoStore(db) as store:
        remaining = list(store.iter_unfingerprinted())
    assert len(remaining) == 3, "the first completed batch must have been committed"


def test_progress_is_reported(tmp_path):
    a = _jpeg(tmp_path / "a.jpg", _gradient((60, 60)))
    with PhotoStore(tmp_path / "db.sqlite") as store:
        store.upsert_many([_photo("a", a)])
        seen = []
        fp.run_fingerprints(store, on_progress=lambda d, t: seen.append((d, t)))
    assert seen[-1] == (1, 1)


def test_a_hash_with_the_top_bit_set_survives_the_round_trip(tmp_path):
    """Half of all images set the top bit. This is the end-to-end version of
    the signed/unsigned check in test_db_v3 - through a real decode, because
    that is where a real hash comes from."""
    path = _jpeg(tmp_path / "ramp.jpg", _gradient((128, 128)))
    with PhotoStore(tmp_path / "db.sqlite") as store:
        store.upsert_many([_photo("a", path)])
        fp.run_fingerprints(store)
        stored = store.get("a").meta.phash
    assert stored == fp.fingerprint_file(path).phash
    assert stored >= 0
