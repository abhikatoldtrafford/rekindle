"""Composition guardrails.

Fixture shapes are all real ones from the reference library: a 8874x943 VR
panorama, a 101x24 barcode, a 1220x2712 screenshot with no camera metadata, a
video row with NULL dimensions, and photos whose EXIF was stripped but which
are perfectly ordinary pictures.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from rekindle.memory import composition as comp
from rekindle.memory.composition import (
    Orientation,
    canvas_for,
    classify,
    cohesive_orientation,
    compose,
    fit_within,
    is_screenshot,
)
from rekindle.models import MediaType, Photo, PhotoMeta

T0 = datetime(2020, 5, 1, 12, 0, tzinfo=UTC)


def _p(h="a", *, size=(4000, 3000), name=None, make="Google", model="Pixel", **kw) -> Photo:
    width, height = size if size else (None, None)
    return Photo(
        file_hash=h,
        paths=[Path(f"/lib/{name or h + '.jpg'}")],
        media_type=kw.pop("media_type", MediaType.IMAGE),
        meta=PhotoMeta(
            taken_at_utc=T0,
            taken_at_local=datetime(2020, 5, 1, 12, 0),
            width=width,
            height=height,
            camera_make=make,
            camera_model=model,
            brightness=kw.pop("brightness", 110.0),
            sharpness=kw.pop("sharpness", 9.0),
            phash=kw.pop("phash", 1),
            phash_error=kw.pop("phash_error", None),
        ),
        first_seen=T0,
        last_seen=T0,
    )


# --------------------------------------------------------------------------
# orientation


def test_classification():
    assert classify(_p(size=(4000, 3000))) is Orientation.LANDSCAPE
    assert classify(_p(size=(3000, 4000))) is Orientation.PORTRAIT
    assert classify(_p(size=(2000, 2000))) is Orientation.SQUARE
    assert classify(_p(size=None)) is Orientation.UNKNOWN


def test_the_square_tolerance_matches_the_measured_cliff():
    """178 live images are exactly 1:1 and 210 are within 1.02; only 21 more
    sit between 1.02 and 1.05. 1.05 forgives a crop a pixel off without
    swallowing a real 4:3 (1.33)."""
    assert classify(_p(size=(1024, 1000))) is Orientation.SQUARE  # 1.024
    assert classify(_p(size=(1040, 1000))) is Orientation.SQUARE  # 1.04
    assert classify(_p(size=(1200, 1000))) is Orientation.LANDSCAPE  # 1.20
    assert classify(_p(size=(4000, 3000))) is Orientation.LANDSCAPE  # 1.33


def test_dimensions_are_read_as_stored_not_re_transposed():
    """The index already stores post-rotation dimensions. Re-applying the
    orientation tag here would swap them back and undo the M0 fix."""
    portrait = _p(size=(3000, 4000))
    assert comp.dimensions(portrait) == (3000, 4000)
    assert classify(portrait) is Orientation.PORTRAIT


def test_the_majority_orientation_wins():
    photos = [_p(f"l{i}", size=(4000, 3000)) for i in range(3)]
    photos += [_p("p1", size=(3000, 4000))]
    assert cohesive_orientation(photos) is Orientation.LANDSCAPE

    photos = [_p(f"p{i}", size=(3000, 4000)) for i in range(3)] + [_p("l1", size=(4000, 3000))]
    assert cohesive_orientation(photos) is Orientation.PORTRAIT


def test_an_exact_tie_goes_to_landscape():
    """73% of this library is landscape and a montage is watched in a
    landscape frame, so the tie is not arbitrary - but it IS a choice, and it
    must be deterministic."""
    photos = [_p("l", size=(4000, 3000)), _p("p", size=(3000, 4000))]
    assert cohesive_orientation(photos) is Orientation.LANDSCAPE


def test_square_photos_never_outvote_and_never_get_dropped():
    """Square letterboxes acceptably into either canvas, so it is a compatible
    minority rather than a competing majority."""
    photos = [_p("l", size=(4000, 3000))] + [_p(f"s{i}", size=(2000, 2000)) for i in range(5)]
    assert cohesive_orientation(photos) is Orientation.LANDSCAPE
    kept, report = compose(photos)
    assert len(kept) == 6
    assert comp.DROP_MINORITY_ORIENTATION not in report.dropped


def test_an_all_square_set_is_square():
    assert cohesive_orientation([_p("s", size=(2000, 2000))]) is Orientation.SQUARE


def test_the_minority_orientation_is_dropped_and_counted(tmp_path):
    photos = [_p(f"l{i}", size=(4000, 3000)) for i in range(4)] + [_p("p", size=(3000, 4000))]
    kept, report = compose(photos)
    assert {p.file_hash for p in kept} == {"l0", "l1", "l2", "l3"}
    assert report.dropped[comp.DROP_MINORITY_ORIENTATION] == 1
    assert report.orientation is Orientation.LANDSCAPE
    assert report.accounted


def test_orientation_enforcement_can_be_turned_off():
    photos = [_p("l", size=(4000, 3000)), _p("p", size=(3000, 4000))]
    kept, _ = compose(photos, enforce_orientation=False)
    assert len(kept) == 2


def test_orientation_is_decided_AFTER_the_per_photo_gates():
    """Otherwise a pile of rejected portrait thumbnails outvotes the real
    landscape photos and empties the memory."""
    photos = [_p(f"tiny{i}", size=(200, 300)) for i in range(9)]
    photos += [_p("good", size=(4000, 3000))]
    kept, report = compose(photos)
    assert [p.file_hash for p in kept] == ["good"]
    assert report.orientation is Orientation.LANDSCAPE


# --------------------------------------------------------------------------
# resolution and aspect


def test_a_photo_below_the_resolution_floor_is_dropped():
    kept, report = compose([_p("ok"), _p("thumb", size=(320, 240))])
    assert [p.file_hash for p in kept] == ["ok"]
    assert report.dropped[comp.DROP_TOO_SMALL] == 1


def test_the_floor_is_measured_on_the_SHORT_edge():
    """A photo is not usable just because ONE edge is large.

    600x300 is chosen so the aspect ratio is 2.0 - comfortably inside the
    2.5 limit - which means only the resolution floor can reject it. An
    earlier version used 4000x200, which the EXTREME ASPECT rule rejected
    anyway, so the test passed while measuring nothing: swapping min for max
    here left it green.
    """
    kept, report = compose([_p("strip", size=(600, 300))])
    assert kept == []
    assert report.dropped == {comp.DROP_TOO_SMALL: 1}


def test_a_real_panorama_is_dropped():
    """8874x943 is the widest image in the reference library, at 9.41:1. It
    would render as a 1280x136 band inside a black frame."""
    _, report = compose([_p("pano", size=(8874, 943))])
    assert report.dropped[comp.DROP_EXTREME_ASPECT] == 1


def test_a_very_tall_crop_is_dropped():
    """1886x8485, also real, at 4.5:1 the other way."""
    _, report = compose([_p("tall", size=(1886, 8485))])
    assert report.dropped[comp.DROP_EXTREME_ASPECT] == 1


def test_an_ordinary_16_by_9_photo_survives():
    """1.78:1 is the 90th percentile of this library. The aspect rule must not
    touch it."""
    kept, _ = compose([_p("wide", size=(3840, 2160))])
    assert len(kept) == 1


def test_the_barcode_is_dropped():
    """101x24, genuinely present in the library, twice."""
    _, report = compose([_p("barcode", size=(101, 24), name="barcode.jpeg")])
    assert report.total_dropped == 1


# --------------------------------------------------------------------------
# screenshots


def test_a_screenshot_with_no_camera_metadata_is_detected():
    """1220x2712 with no camera make or model - the exact shape of 57 files in
    the reference library."""
    assert is_screenshot(_p(size=(1220, 2712), make=None, model=None)) is True


def test_a_screenshot_named_by_convention_is_detected():
    shot = _p(size=(1100, 2000), make=None, model=None, name="Screenshot_20161008-222024.png")
    assert is_screenshot(shot) is True


def test_a_PHOTO_at_a_screen_size_is_NOT_a_screenshot():
    """The conjunction is the whole safety margin: 1080x1920 is a perfectly
    ordinary photo size, and camera metadata is what distinguishes them."""
    assert is_screenshot(_p(size=(1080, 1920), make="Google", model="Pixel")) is False


def test_a_photo_with_stripped_exif_is_NOT_a_screenshot():
    """1,801 live images (9.9%) have no camera make or model, mostly from a
    re-save or a messaging app. Dropping all of them was explicitly declined:
    absence of EXIF alone proves nothing."""
    assert is_screenshot(_p(size=(4000, 3000), make=None, model=None)) is False
    kept, _ = compose([_p("stripped", size=(4000, 3000), make=None, model=None)])
    assert len(kept) == 1


def test_screenshots_are_dropped_and_counted():
    photos = [_p("ok"), _p("shot", size=(1220, 2712), make=None, model=None)]
    _, report = compose(photos)
    assert report.dropped[comp.DROP_SCREENSHOT] == 1


# --------------------------------------------------------------------------
# quality


def test_a_near_black_frame_is_dropped():
    _, report = compose([_p("pocket", brightness=4.0)])
    assert report.dropped[comp.DROP_TOO_DARK] == 1


def test_a_blown_out_frame_is_dropped():
    _, report = compose([_p("blown", brightness=250.0)])
    assert report.dropped[comp.DROP_TOO_BRIGHT] == 1


def test_a_badly_out_of_focus_frame_is_dropped():
    _, report = compose([_p("blur", sharpness=0.4)])
    assert report.dropped[comp.DROP_OUT_OF_FOCUS] == 1


def test_the_quality_gates_do_not_touch_an_ordinarily_soft_old_photo():
    """2011's 5th-percentile sharpness is 1.83 and 2008's is 2.61. A threshold
    set at the whole-library p5 (3.23) would delete a fifth of those years;
    1.5 sits below every year's p5 so no year is singled out."""
    kept, _ = compose([_p("soft2011", sharpness=1.9), _p("soft2008", sharpness=2.7)])
    assert len(kept) == 2


def test_a_dim_indoor_photo_is_not_mistaken_for_a_pocket_shot():
    """The 1st percentile of mean luminance in this library is 33. The gate is
    at 20, well below anything real."""
    kept, _ = compose([_p("dim", brightness=35.0)])
    assert len(kept) == 1


def test_an_unfingerprinted_photo_is_KEPT():
    """The quality gates filter on measured evidence; they do not require that
    evidence to exist. Otherwise an index that has never been fingerprinted
    would produce zero memories instead of a warning."""
    kept, _ = compose([_p("new", brightness=None, sharpness=None, phash=None)])
    assert len(kept) == 1


# --------------------------------------------------------------------------
# videos and unreadable files


def test_videos_are_dropped_and_counted():
    """1,117 rows. They have no dimensions, no hash, and rendering one needs
    ffmpeg - including them only when ffmpeg happens to be installed would
    make the MemorySpec depend on the machine."""
    _, report = compose([_p("v", size=None, media_type=MediaType.VIDEO)])
    assert report.dropped[comp.DROP_VIDEO] == 1


def test_an_undecodable_file_is_counted_not_silently_dropped():
    _, report = compose([_p("bad", phash=None, phash_error="undecodable")])
    assert report.dropped[comp.DROP_UNREADABLE] == 1


def test_a_photo_with_no_dimensions_is_counted():
    _, report = compose([_p("nodim", size=None)])
    assert report.dropped[comp.DROP_NO_DIMENSIONS] == 1


# --------------------------------------------------------------------------
# accounting and the canvas


def test_everything_is_accounted_for():
    photos = [
        _p("ok"),
        _p("tiny", size=(100, 100)),
        _p("pano", size=(8874, 943)),
        _p("dark", brightness=2.0),
        _p("v", size=None, media_type=MediaType.VIDEO),
        _p("portrait", size=(3000, 4000)),
    ]
    kept, report = compose(photos)
    assert report.accounted, report.dropped
    assert report.considered == 6
    assert report.kept == len(kept)


def test_every_drop_carries_a_reason_and_an_example():
    _, report = compose([_p("pano", size=(8874, 943), name="PANO_20191005.jpg")])
    assert report.examples[comp.DROP_EXTREME_ASPECT] == "PANO_20191005.jpg"
    assert comp.describe_drops(report) == ["1 extreme aspect (e.g. PANO_20191005.jpg)"]


def test_the_report_shows_filenames_only_never_full_paths():
    _, report = compose([_p("pano", size=(8874, 943))])
    assert "/lib/" not in report.examples[comp.DROP_EXTREME_ASPECT]


def test_the_canvas_is_the_MEDIAN_not_the_minimum():
    """The rule that replaced the minimum, and why.

    Under the minimum, ONE 640x480 photo from 2014 pinned an entire
    "Paramita over the years" memory to 640x480 - rendering two 7008x4672
    photos at a 120th of their pixel count. Measured across 45 memories, 11 of
    them landed on 640x480 for exactly that reason.

    Width and height are still taken independently, so a set mixing 4:3 and
    16:9 gets a canvas both can sit in.
    """
    photos = [_p("a", size=(4000, 3000)), _p("b", size=(1600, 2000)), _p("c", size=(2000, 1200))]
    assert canvas_for(photos) == (2000, 2000)


def test_one_tiny_photo_no_longer_drags_the_whole_canvas_down():
    """The regression this replaced, stated as a test."""
    photos = [_p(f"big{i}", size=(4000, 3000)) for i in range(9)]
    photos.append(_p("tiny2014", size=(640, 480)))
    width, height = canvas_for(photos)
    assert width == 4000 and height == 3000


def test_the_median_is_the_LOWER_of_two_middles():
    """An even-length set takes the smaller middle rather than averaging, so
    the canvas is always a real size at least half the set can meet."""
    photos = [_p("a", size=(1000, 1000)), _p("b", size=(3000, 3000))]
    assert canvas_for(photos) == (1000, 1000)


def test_the_canvas_is_derived_from_the_KEPT_photos():
    """A rejected 320px thumbnail must not shrink the canvas for everyone."""
    photos = [_p("a", size=(4000, 3000)), _p("thumb", size=(320, 240))]
    _, report = compose(photos)
    assert report.canvas == (4000, 3000)


def test_the_canvas_is_none_for_an_empty_set():
    assert canvas_for([]) is None


def test_fit_never_upscales():
    """Upscaling a 640px photo to sit beside a 4000px one produces visible
    mush, which is the whole reason the canvas comes from the set."""
    assert fit_within((640, 480), (1920, 1080)) == (640, 480)


def test_fit_downscales_preserving_aspect():
    assert fit_within((4000, 3000), (800, 800)) == (800, 600)
    assert fit_within((3000, 4000), (800, 800)) == (600, 800)


def test_fit_never_returns_a_zero_dimension():
    assert fit_within((4000, 10), (100, 100)) == (100, 1)


def test_an_empty_input_composes_to_nothing():
    kept, report = compose([])
    assert kept == []
    assert report.accounted
    assert report.canvas is None


def test_reports_merge_additively():
    _, first = compose([_p("pano", size=(8874, 943))])
    _, second = compose([_p("tiny", size=(100, 100))])
    first.merge(second)
    assert first.considered == 2
    assert first.total_dropped == 2
    assert set(first.dropped) == {comp.DROP_EXTREME_ASPECT, comp.DROP_TOO_SMALL}


@pytest.mark.parametrize(
    "size,expected",
    [((4000, 3000), Orientation.LANDSCAPE), ((3000, 4000), Orientation.PORTRAIT)],
)
def test_a_single_photo_defines_its_own_orientation(size, expected):
    kept, report = compose([_p("only", size=size)])
    assert len(kept) == 1
    assert report.orientation is expected


def test_a_face_tagged_photo_at_a_screen_size_is_not_a_screenshot():
    """Measured: the size branch alone flagged 58 photos carrying Google face
    tags - re-compressed photos (many from WhatsApp) that happen to land on a
    common screen size. A photo Google found a person in is a photograph."""
    photo = _p(size=(1920, 1200), make=None, model=None)
    photo.meta.people = ["Abhik Maiti"]
    assert is_screenshot(photo) is False


def test_the_filename_convention_is_trusted_even_with_face_tags():
    """A screenshot of a video call has faces in it. The OS filename is a
    positive assertion; a matching screen size is only circumstantial."""
    photo = _p(size=(1080, 2340), make=None, model=None, name="Screenshot_20210311.png")
    photo.meta.people = ["Abhik Maiti"]
    assert is_screenshot(photo) is True


def test_a_wallpaper_at_a_desktop_size_is_still_caught():
    """222 files at 1920x1200 in the reference library are downloaded
    wallpapers. Untagged, no camera metadata - correctly excluded."""
    assert is_screenshot(_p(size=(1920, 1200), make=None, model=None, name="IT-wp4.jpg")) is True


# --------------------------------------------------------------------------
# placement: downscale, upscale within tolerance, or pad


def test_a_larger_photo_is_downscaled_to_fit():
    target, mode = comp.plan_placement((7008, 4672), (3984, 2988))
    assert mode == comp.FIT_DOWNSCALE
    assert target[0] <= 3984 and target[1] <= 2988


def test_a_slightly_smaller_photo_is_upscaled_within_tolerance():
    """Padding a photo 5% below the canvas would read as an inconsistency
    rather than a deliberate signal, and a 25% linear stretch is
    imperceptible."""
    _, mode = comp.plan_placement((3800, 2850), (3984, 2988))
    assert mode == comp.FIT_UPSCALE


def test_a_far_smaller_photo_is_padded_at_NATIVE_size():
    target, mode = comp.plan_placement((640, 480), (3984, 2988))
    assert mode == comp.FIT_PAD
    assert target == (640, 480), "a padded photo must not be resized at all"


def test_the_tolerance_boundary_is_exact():
    # Exactly 1.25x is still an upscale; a hair beyond it pads.
    _, mode = comp.plan_placement((800, 600), (1000, 750))
    assert mode == comp.FIT_UPSCALE
    _, mode = comp.plan_placement((790, 592), (1000, 750))
    assert mode == comp.FIT_PAD


def test_nothing_is_ever_excluded_for_being_small():
    """The rule that keeps a memory's earliest years.

    On this library small means OLD, so excluding sub-canvas photos would
    quietly delete the early years of exactly the memories - "person over the
    years" - whose whole subject is the span. Two individually reasonable
    rules would have combined to defeat each other.
    """
    photos = [_p(f"big{i}", size=(4000, 3000)) for i in range(9)]
    photos.append(_p("old2014", size=(640, 480)))
    kept, report = compose(photos)
    assert len(kept) == 10
    assert "too_small" not in report.dropped


def test_the_resolution_FLOOR_still_applies_though():
    """Padding is for photos below the canvas, not for thumbnails. The 480px
    floor is a separate gate and still removes genuine junk."""
    _, report = compose([_p("ok"), _p("thumb", size=(320, 240))])
    assert report.dropped[comp.DROP_TOO_SMALL] == 1
