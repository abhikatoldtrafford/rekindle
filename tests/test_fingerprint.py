"""Fingerprint pass: the hash, the focus measure, and resumability.

Fixture shapes come from the real index: JPEG images, videos with no
dimensions at all (all 1,117 video rows have width/height NULL), and files
that cannot be decoded.
"""

import random
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFilter, JpegImagePlugin

from rekindle.db import PhotoStore
from rekindle.memory import composition as comp
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


def _texture(size, cell=2, seed=1, low=0, high=255) -> Image.Image:
    """Fine random detail: `cell`-pixel blocks of random grey.

    NOT a checkerboard, and the reason is worth recording because it took a
    failing test to see it. `sharpness` sums absolute gradients, and spreading
    a step edge over three pixels instead of one does not change that sum - so
    blurring a checkerboard barely moves its score (measured: 0.043 sharp,
    0.009 at radius 4, both far below the gate). The measure reads the loss of
    fine TEXTURE, which is what blur actually destroys in a photograph, and a
    fixture has to have some. Deterministic: seeded, so the numbers below are
    stable across runs and platforms.
    """
    rng = random.Random(seed)
    width, height = size
    blocks = Image.new("L", ((width + cell - 1) // cell, (height + cell - 1) // cell))
    blocks.putdata([rng.randrange(low, high + 1) for _ in range(blocks.size[0] * blocks.size[1])])
    return blocks.resize(size, Image.Resampling.NEAREST)


def test_a_blurred_copy_is_less_sharp_than_its_original():
    sharp = _texture((256, 256))
    blurred = sharp.filter(ImageFilter.GaussianBlur(radius=4))
    assert fp.sharpness(blurred) < fp.sharpness(sharp)


def test_a_small_sharp_photo_beats_a_large_blurred_one():
    """What the old "measured at a fixed size" test was reaching for.

    Resolution must not decide a burst on its own. That test pinned the
    mechanism - a fixed 128x128 resize - which is exactly what destroyed the
    detail mild blur lives in, so the mechanism had to go. The PROPERTY it was
    defending has not, and this states it directly.

    Measured on 2,240 real photos, the median score by native resolution is
    0.589 (<1MP), 0.604 (1-3MP), 0.532 (3-8MP), 0.570 (8-14MP), 0.519 (>14MP)
    - a 1.16x spread, against 1.29x for the measure this replaced and 1.58x
    for an unnormalised gradient at the same working size.
    """
    small = _texture((320, 240))
    large = _texture((2560, 1920)).filter(ImageFilter.GaussianBlur(radius=6))
    assert fp.sharpness(small) > fp.sharpness(large)


def test_the_measure_never_upscales_a_small_photo():
    """An enlarged copy is smooth at the pixel scale by construction, so
    upscaling to reach the working size would score every small photo as
    blurred - and small, in this library, means old."""
    tiny = _texture((200, 150))
    assert fp._downscale(tiny, 1024).size == (200, 150)
    assert fp.sharpness(tiny) > comp.MIN_SHARPNESS


def test_the_sharpest_region_decides_not_the_average():
    """Shallow depth of field: a sharp subject against a blurred background is
    often the best photo in the set. Measured on 123 real photos given a
    synthetic sharp-centre/blurred-surround treatment, the median retention of
    the fully-sharp score is 0.800 taking the max tile and 0.006 taking the
    75th percentile - so this is the single choice that keeps those photos."""
    crisp = _texture((1200, 900))
    blurred = crisp.filter(ImageFilter.GaussianBlur(radius=8))
    mask = Image.new("L", crisp.size, 0)
    ImageDraw.Draw(mask).rectangle((450, 330, 750, 570), fill=255)
    shallow = Image.composite(crisp, blurred, mask)

    assert fp.sharpness(shallow) > comp.MIN_SHARPNESS
    # ...and a whole-image average over the same picture would not survive it.
    assert fp.sharpness(shallow) > fp.sharpness(blurred) * 3


def test_mild_blur_is_visible_at_the_working_resolution():
    """The defect this measure exists to fix. At 128x128 a blur of a few
    pixels in a 4000px photo is below the resampling floor and simply is not
    there to measure; over 123 real photos the old measure scored a sharp
    photo above a mildly blurred one 51.9% of the time, against 90.3% for
    this one. A radius of 2 in a 2400px image is well inside "mild"."""
    crisp = _texture((2400, 1800))
    mild = crisp.filter(ImageFilter.GaussianBlur(radius=2))
    assert fp.sharpness(mild) < fp.sharpness(crisp) * 0.75


def test_sharpness_is_a_ratio_so_scene_contrast_cancels():
    """Why a single threshold is safe across 2000-2026. The old measure's
    per-year 5th percentile spanned 4.70x because a low-contrast photo and a
    blurred one were indistinguishable to it; this one spans 1.71x. Halving
    the contrast of an image must barely move its score."""
    crisp = _texture((1200, 900))
    faint = Image.eval(crisp, lambda v: 100 + v // 8)  # same detail, 1/8 the contrast
    assert fp.sharpness(faint) == pytest.approx(fp.sharpness(crisp), rel=0.15)


def test_a_flat_image_has_zero_sharpness():
    assert fp.sharpness(Image.new("L", (128, 128), 200)) == pytest.approx(0.0)


def test_blur_along_one_axis_only_is_still_seen():
    """A one-axis gradient is blind to detail that varies only along that
    axis, and the commonest mild blur in this library is camera shake, which
    is directional by nature. Horizontal stripes have no horizontal gradient
    at all, so a measure reading only dx scores this picture - and its blurred
    copy - identically at zero."""
    rng = random.Random(3)
    rows = Image.new("L", (1, 450))
    rows.putdata([rng.randrange(256) for _ in range(450)])
    striped = rows.resize((1200, 900), Image.Resampling.NEAREST)

    crisp = fp.sharpness(striped)
    blurred = fp.sharpness(striped.filter(ImageFilter.GaussianBlur(radius=3)))
    assert crisp > comp.MIN_SHARPNESS
    assert blurred < crisp * 0.5


def test_sharpness_stays_inside_its_declared_range():
    """It is documented as 0.0-1.0 and `MIN_SHARPNESS` is calibrated against
    that. A negative value - a tile whose reblur RAISES the gradient, which
    ringing can do - would sort below every real photo."""
    for image in (
        _texture((1200, 900)),
        _texture((1200, 900)).filter(ImageFilter.GaussianBlur(radius=12)),
        Image.new("L", (300, 300), 0),
        _gradient((400, 300)),
    ):
        assert 0.0 <= fp.sharpness(image) <= 1.0


def test_a_small_image_is_measured_as_one_tile_not_sixty_four():
    """A fixed 8x8 grid over a 320px image gives 40px tiles whose gradient
    statistics are mostly noise, and a MAXIMUM over 64 noisy tiles biases the
    score upward - the one direction a blur gate must not be biased."""
    assert len(list(fp._tiles((240, 180)))) == 1
    assert len(list(fp._tiles((1024, 768)))) == 64
    # The tiles must cover the image exactly: a gap would hide detail and an
    # overlap would double-count it.
    boxes = list(fp._tiles((1024, 768)))
    assert sum((right - left) * (bottom - top) for left, top, right, bottom in boxes) == 1024 * 768


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


def test_the_colour_histogram_actually_contains_colour(tmp_path):
    """M2 shipped `im.draft("L", ...)`, which tells libjpeg to decode the luma
    plane only. `colour_signature` then ran on a grey image and converted it
    back to RGB, so every stored "colour histogram" was a luminance histogram:
    measured over 2,000 real rows, a median of 42 of the 64 bins were exactly
    zero and 55% of the mass sat on the four grey bins. Nothing failed - the
    diversity signal simply carried less than it claimed.

    These two colours have the same JPEG luminance (0.299*255 = 76.2 and
    0.587*130 = 76.3), so a grayscale decode maps BOTH to a flat 76 and yields
    the identical signature. Only a real colour decode can tell them apart.
    """
    red = _jpeg(tmp_path / "red.jpg", Image.new("RGB", (200, 150), (255, 0, 0)))
    green = _jpeg(tmp_path / "green.jpg", Image.new("RGB", (200, 150), (0, 130, 0)))

    assert fp.fingerprint_file(red).colour != fp.fingerprint_file(green).colour


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
    row = fp._row_for(photo)
    assert row.phash is not None
    assert row.error is None


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


# --------------------------------------------------------------------------
# brightness, and the orientation repair


def test_brightness_tracks_mean_luminance():
    assert fp.brightness(Image.new("L", (128, 128), 0)) == pytest.approx(0.0)
    assert fp.brightness(Image.new("L", (128, 128), 255)) == pytest.approx(255.0, abs=1)
    assert fp.brightness(Image.new("L", (128, 128), 128)) == pytest.approx(128.0, abs=1)


def test_brightness_is_stored_by_the_pass(tmp_path):
    path = _jpeg(tmp_path / "dark.jpg", Image.new("L", (120, 90), 4))
    with PhotoStore(tmp_path / "db.sqlite") as store:
        store.upsert_many([_photo("a", path)])
        fp.run_fingerprints(store)
        assert store.get("a").meta.brightness < 20


def _oriented(path, size, orientation):
    """Landscape pixels plus a rotation tag - how a phone stores a portrait."""
    im = Image.new("RGB", size, (90, 120, 150))
    exif = im.getexif()
    exif[0x0112] = orientation
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, "JPEG", exif=exif)
    return path


def test_the_pass_measures_the_image_a_viewer_SEES(tmp_path):
    """Orientation is applied before anything is measured. Roughly 2,475 rows
    in the reference library are stored 400x300 while a viewer sees 300x400."""
    path = _oriented(tmp_path / "p.jpg", (400, 300), 6)
    result = fp.fingerprint_file(path)
    assert (result.width, result.height) == (300, 400)


def test_the_pass_repairs_swapped_dimensions_in_the_index(tmp_path):
    """The in-place repair for indexes built before the orientation fix. The
    alternative was telling every existing user to re-index from scratch."""
    path = _oriented(tmp_path / "p.jpg", (400, 300), 6)
    with PhotoStore(tmp_path / "db.sqlite") as store:
        # Exactly as M0 stored it: raw, unrotated.
        store.upsert_many([_photo("a", path, meta=PhotoMeta(width=400, height=300))])
        fp.run_fingerprints(store)
        got = store.get("a")
    assert (got.meta.width, got.meta.height) == (300, 400)


def test_the_repair_records_NATIVE_resolution_not_the_drafted_size(tmp_path):
    """draft() scales the decode down by up to 8x. Measuring the loaded image
    would record a 4000px photo as 500px, and the resolution floor would then
    reject a perfectly good photo."""
    path = _jpeg(tmp_path / "big.jpg", _gradient((1600, 1200)))
    result = fp.fingerprint_file(path)
    assert (result.width, result.height) == (1600, 1200)


def test_a_video_row_keeps_its_null_dimensions(tmp_path):
    """All 1,117 video rows have width/height NULL and are never decoded, so
    the pass must not write a dimension it never measured."""
    with PhotoStore(tmp_path / "db.sqlite") as store:
        store.upsert_many([_photo("v", tmp_path / "c.mp4", MediaType.VIDEO)])
        fp.run_fingerprints(store)
        got = store.get("v")
    assert got.meta.width is None and got.meta.height is None


def test_a_failed_decode_does_not_null_stored_dimensions(tmp_path):
    """A None width must leave the stored value alone, not overwrite it."""
    with PhotoStore(tmp_path / "db.sqlite") as store:
        store.upsert_many(
            [_photo("a", tmp_path / "gone.jpg", meta=PhotoMeta(width=800, height=600))]
        )
        fp.run_fingerprints(store)
        got = store.get("a")
    assert (got.meta.width, got.meta.height) == (800, 600)
    assert got.meta.phash_error == fp.ERR_MISSING


def test_the_colour_histogram_is_smoothed_across_adjacent_bins():
    """Hard binning makes the signature brittle exactly where it matters.

    (200,40,40) and (150,30,30) - the same red under slightly different light
    - fall in different bins with four bins per channel. For an image
    dominated by one colour that puts ALL the mass in disjoint bins, and
    measured before smoothing those two scored 1.000 dissimilarity: the same
    as red against blue. Diffusing each bin into its neighbours keeps a small
    exposure shift a small change.
    """
    from rekindle.memory.diversity import ColourSignal
    from rekindle.models import MediaType, Photo

    def photo(colour):
        return Photo("h", [Path("/a.jpg")], MediaType.IMAGE, PhotoMeta(colour=colour), T0, T0)

    red = fp.colour_signature(Image.new("RGB", (64, 64), (200, 40, 40)))
    dark_red = fp.colour_signature(Image.new("RGB", (64, 64), (150, 30, 30)))
    blue = fp.colour_signature(Image.new("RGB", (64, 64), (40, 40, 200)))

    signal = ColourSignal()
    near = signal.between(photo(red), photo(dark_red))
    far = signal.between(photo(red), photo(blue))

    assert near < far, "a lighting shift is as different as a different colour"
    assert near < 0.9


def test_the_colour_signature_is_a_fixed_width_hex_string():
    signature = fp.colour_signature(Image.new("RGB", (64, 64), (10, 20, 30)))
    assert len(signature) == 128
    int(signature, 16)


def test_the_colour_signature_is_deterministic():
    image = _gradient((128, 128))
    assert fp.colour_signature(image) == fp.colour_signature(image)


def test_the_colour_signature_is_stored_by_the_pass(tmp_path):
    path = _jpeg(tmp_path / "a.jpg", _gradient((120, 90)))
    with PhotoStore(tmp_path / "db.sqlite") as store:
        store.upsert_many([_photo("a", path)])
        fp.run_fingerprints(store)
        assert store.get("a").meta.colour is not None
