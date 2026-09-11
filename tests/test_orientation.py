"""One contract, asserted at every place that decodes pixels.

M2 fixed the EXIF orientation tag in the INDEX: `meta.exif.read_exif` now
records post-rotation `width`/`height`, and `memory.fingerprint` repairs rows
written before that. Those tests all assert NUMBERS, and numbers are exactly
what a missing transpose can survive: 400x300 tagged 90 degrees is 300x400
whichever direction the rotation went, and orientations 2, 3 and 4 do not
change the size at all.

So the index was right and two shipped decode sites were still wrong.
`semantic.embed._decode` and `semantic.faces._examine` handed the encoder and
the face detector the raw pixels. Measured against the reference library:
2,503 of 18,363 images (13.6%) carry a 90/270-degree tag, and on 200 of them
the missing transpose changed **14.5% of face-gate verdicts**, with 7% showing
the detector no face at all where the upright image has one - a default-deny
publishing gate failing open.

Every test in this file therefore asserts the PIXELS, through
`tests.fixtures.oriented`, whose four-colour marker distinguishes all eight
orientations from each other. A test that can only see width and height is
the test that let this through.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageOps

from rekindle.memory import fingerprint as fp
from rekindle.memory.render import frames as fr
from rekindle.meta.exif import open_upright, read_exif
from rekindle.semantic.faces import DEFAULT_DETECT_THRESHOLD, DEFAULT_GATE_THRESHOLD, _examine
from rekindle.semantic.photos import IndexedPhoto
from tests.fixtures.oriented import (
    ALL_ORIENTATIONS,
    ORIENTATION_TAG,
    UPRIGHT,
    quadrants,
    upright_marker,
    write_oriented,
    write_untagged,
)

# Orientations that exchange the two axes. The other four are the ones a
# dimension assertion is blind to, which is why they are all parametrised.
SWAPS_AXES = (5, 6, 7, 8)
KEEPS_AXES = (1, 2, 3, 4)


# ------------------------------------------------------ the fixture is honest
#
# Checked against Pillow's own implementation, not against ours. A fixture
# that encodes the same misunderstanding as the code under test certifies the
# bug - which is how M0's motion-photo convention got invented.


@pytest.mark.parametrize("orientation", ALL_ORIENTATIONS)
def test_the_fixture_is_upright_under_pillows_own_transpose(tmp_path, orientation):
    path = tmp_path / f"o{orientation}.jpg"
    write_oriented(path, orientation)
    with Image.open(path) as im:
        assert im.getexif()[ORIENTATION_TAG] == orientation
        assert quadrants(ImageOps.exif_transpose(im)) == UPRIGHT


@pytest.mark.parametrize("orientation", [2, 3, 4, 5, 6, 7, 8])
def test_the_fixtures_raw_pixels_are_NOT_upright(tmp_path, orientation):
    """Without this, a fixture that forgot to transform its pixels would pass
    every test below while proving nothing at all."""
    path = tmp_path / f"o{orientation}.jpg"
    write_oriented(path, orientation)
    with Image.open(path) as im:
        assert quadrants(im) != UPRIGHT


def test_the_marker_distinguishes_all_eight_orientations(tmp_path):
    """A two-colour marker cannot tell a mirror from a rotation. Every one of
    the eight raw arrangements must be different, or a test asserting
    `quadrants(...) == UPRIGHT` is weaker than it looks."""
    seen = set()
    for orientation in ALL_ORIENTATIONS:
        path = tmp_path / f"o{orientation}.jpg"
        write_oriented(path, orientation)
        with Image.open(path) as im:
            seen.add(quadrants(im))
    assert len(seen) == len(ALL_ORIENTATIONS)


# --------------------------------------------------------------- open_upright


@pytest.mark.parametrize("orientation", ALL_ORIENTATIONS)
def test_open_upright_returns_the_image_a_viewer_sees(tmp_path, orientation):
    path = tmp_path / f"o{orientation}.jpg"
    expected = write_oriented(path, orientation)
    got = open_upright(path)
    assert quadrants(got) == UPRIGHT
    assert got.size == expected.size


def test_open_upright_leaves_an_untagged_photo_alone(tmp_path):
    """1,852 live images carry no orientation tag, and another 1,107 carry the
    invalid value 0. Neither may be rotated."""
    path = tmp_path / "none.jpg"
    write_untagged(path)
    assert quadrants(open_upright(path)) == UPRIGHT
    assert open_upright(path).size == (400, 300)


def test_open_upright_ignores_an_out_of_range_tag(tmp_path):
    """Orientation 0 is not a legal EXIF value and 1,107 live files have it."""
    marker = upright_marker()
    exif = marker.getexif()
    exif[ORIENTATION_TAG] = 0
    path = tmp_path / "zero.jpg"
    marker.save(path, "JPEG", quality=95, exif=exif)
    assert quadrants(open_upright(path)) == UPRIGHT


@pytest.mark.parametrize("orientation", KEEPS_AXES)
def test_the_axis_keeping_orientations_keep_their_size(tmp_path, orientation):
    """The `!= 1` test that `known-limitations.md` warns about would rotate
    these. Only a pixel assertion can tell - the size is identical either
    way."""
    path = tmp_path / f"o{orientation}.jpg"
    write_oriented(path, orientation)
    assert open_upright(path).size == (400, 300)
    assert quadrants(open_upright(path)) == UPRIGHT


@pytest.mark.parametrize("orientation", SWAPS_AXES)
def test_the_axis_swapping_orientations_exchange_their_size(tmp_path, orientation):
    """The real-world shape: a phone writes a portrait photo as LANDSCAPE
    pixels plus a tag. 2,503 live images are stored this way."""
    path = tmp_path / f"o{orientation}.jpg"
    write_oriented(path, orientation, size=(300, 400))
    with Image.open(path) as raw:
        assert raw.size == (400, 300)
    assert open_upright(path).size == (300, 400)


@pytest.mark.parametrize("orientation", ALL_ORIENTATIONS)
def test_open_upright_agrees_with_the_dimensions_the_index_stores(tmp_path, orientation):
    """One source of truth. `read_exif` writes these numbers into the index,
    `canvas_for` sizes the canvas from them and `fit_photo` draws the pixels;
    if the two disagree the canvas is shaped for the other orientation."""
    path = tmp_path / f"o{orientation}.jpg"
    write_oriented(path, orientation)
    exif = read_exif(path)
    assert (exif.width, exif.height) == open_upright(path).size


def test_the_draft_hint_never_decodes_below_the_size_it_was_asked_for(tmp_path):
    """`draft` measures the pixels ON DISK and knows nothing about the tag, so
    a target given in upright axes has to be transposed back before it is
    handed over. Passing it straight through compares a width against a
    height, and on a rotated photo that can pick a scale whose output is
    SMALLER than the caller asked for - which the caller then upscales, the
    one thing `plan_placement` exists to avoid.

    The numbers are chosen to make the two pairings disagree: upright 400x1600
    is 1600x400 on disk, so the wrong pairing reads 1600//400 and 400//100 and
    picks a 4x reduction, decoding to 100x400 - a quarter of the width asked
    for. The right pairing reads 400//400 and 1600//100 and picks 1x.
    """
    path = tmp_path / "pano.jpg"
    write_oriented(path, 6, size=(400, 1600))  # raw 1600x400
    canvas = (400, 100)
    got = open_upright(path, draft=canvas)
    assert got.width >= canvas[0]
    assert got.height >= canvas[1]


def test_the_draft_hint_is_left_alone_when_the_tag_keeps_the_axes(tmp_path):
    """The mirror image of the test above, and the reason the guard names
    `_SWAPS_AXES` rather than `!= 1`. Orientations 2, 3 and 4 do NOT exchange
    the axes, so transposing their hint is the same mistake pointing the other
    way: here it would pick a 4x reduction where 1x is correct."""
    path = tmp_path / "flat.jpg"
    write_oriented(path, 3, size=(1600, 400))  # 180 degrees: raw is 1600x400 too
    canvas = (100, 400)
    got = open_upright(path, draft=canvas)
    assert got.width >= canvas[0]
    assert got.height >= canvas[1]


def test_the_draft_hint_still_shrinks_the_decode(tmp_path):
    """The guard above must not be bought by disabling `draft` altogether -
    it is the single biggest speed lever in the embed pass."""
    path = tmp_path / "big.jpg"
    write_oriented(path, 6, size=(1600, 1200))  # upright 1200x1600
    assert open_upright(path, draft=(150, 200)).size < (1200, 1600)


def test_the_draft_hint_does_not_change_which_way_up_the_photo_is(tmp_path):
    path = tmp_path / "d.jpg"
    write_oriented(path, 8, size=(1600, 1200))
    assert quadrants(open_upright(path, draft=(150, 200))) == UPRIGHT


def test_open_upright_raises_what_the_callers_catch(tmp_path):
    """Every call site wraps this in `except OSError`. A helper that raised
    something else would turn one unreadable file into a dead run."""
    bad = tmp_path / "not-an-image.jpg"
    bad.write_bytes(b"this is not a JPEG")
    with pytest.raises(OSError):
        open_upright(bad)
    with pytest.raises(OSError):
        open_upright(tmp_path / "absent.jpg")


def test_a_truncated_file_raises_inside_the_helper_not_after_it(tmp_path):
    """The returned image must be fully decoded and detached from the file.
    If the decode were deferred, it would happen after the `with` had closed
    the handle and OUTSIDE every caller's `except OSError` - turning one
    truncated photo among 18,000 into a dead run rather than a counted one."""
    whole = tmp_path / "whole.jpg"
    write_oriented(whole, 6, size=(300, 400))
    truncated = tmp_path / "cut.jpg"
    data = whole.read_bytes()
    truncated.write_bytes(data[: len(data) // 2])
    with pytest.raises(OSError):
        open_upright(truncated)


# -------------------------------------------------------------- the renderer


@pytest.mark.parametrize("orientation", ALL_ORIENTATIONS)
def test_a_rendered_frame_is_the_right_way_up(tmp_path, orientation):
    """The assertion the earlier fix was missing. `fit_photo` on a canvas that
    matches the upright photo fills the frame exactly, so the frame IS the
    photo and its quadrants can be read directly."""
    path = tmp_path / f"o{orientation}.jpg"
    upright = write_oriented(path, orientation)
    frame, mode = fr.fit_photo(path, upright.size)
    assert mode == "downscale"
    assert frame.size == upright.size
    assert quadrants(frame) == UPRIGHT


@pytest.mark.parametrize("orientation", SWAPS_AXES)
def test_a_rotated_photo_is_not_letterboxed_into_the_other_orientation(tmp_path, orientation):
    """The symptom reported from the gallery: a portrait photo drawn into a
    canvas shaped for its raw landscape pixels. If the transpose is skipped
    the frame both pads and shows the picture on its side."""
    path = tmp_path / f"o{orientation}.jpg"
    write_oriented(path, orientation, size=(300, 400))  # raw 400x300
    frame, mode = fr.fit_photo(path, (300, 400))
    assert frame.size == (300, 400)
    assert mode == "downscale"
    # No black matte anywhere along the middle row or column.
    assert frame.getpixel((2, 200)) != (0, 0, 0)
    assert frame.getpixel((150, 2)) != (0, 0, 0)
    assert quadrants(frame) == UPRIGHT


@pytest.mark.parametrize("orientation", ALL_ORIENTATIONS)
def test_the_whole_frame_builder_keeps_a_photo_upright(tmp_path, orientation):
    """End to end through `build_frames`, not just `fit_photo`, so a future
    caption or backdrop stage cannot undo the rotation unnoticed."""
    from datetime import UTC, datetime

    from rekindle.memory.spec import FactSheet, MemorySpec, Shot
    from rekindle.models import MediaType, Photo, PhotoMeta

    t0 = datetime(2020, 5, 1, 12, 0, tzinfo=UTC)
    path = tmp_path / f"o{orientation}.jpg"
    upright = write_oriented(path, orientation)
    photo = Photo(
        file_hash="h",
        media_type=MediaType.IMAGE,
        paths=[path],
        meta=PhotoMeta(),
        first_seen=t0,
        last_seen=t0,
    )
    spec = MemorySpec(
        recipe="r",
        key="k",
        title="t",
        subtitle="s",
        shots=(Shot("h", "", "2020-05-01T12:00:00", True),),
        facts=FactSheet(title="t", recipe="r", photo_count=1),
        public_safe=False,
    )
    built, report = fr.build_frames(
        spec,
        upright.size,
        resolve=lambda _h: photo,
        locate=lambda _p: path,
        with_title=False,
    )
    assert report.rendered == 1
    assert quadrants(built[0]) == UPRIGHT


# ------------------------------------------------------ the fingerprint pass


@pytest.mark.parametrize("orientation", ALL_ORIENTATIONS)
def test_the_fingerprint_is_taken_of_the_upright_photo(tmp_path, orientation):
    """A dHash reads left-to-right gradients, so it is NOT invariant under
    rotation or mirroring. Comparing an oriented file against the identical
    upright one pins the pixels the pass measured - which `width`/`height`
    alone cannot do for orientations 2, 3 and 4."""
    reference = tmp_path / "upright.jpg"
    write_untagged(reference)
    oriented = tmp_path / f"o{orientation}.jpg"
    write_oriented(oriented, orientation)
    assert fp.fingerprint_file(oriented).phash == fp.fingerprint_file(reference).phash


@pytest.mark.parametrize("orientation", SWAPS_AXES)
def test_the_pass_records_the_upright_size(tmp_path, orientation):
    path = tmp_path / f"o{orientation}.jpg"
    write_oriented(path, orientation, size=(300, 400))  # raw 400x300
    got = fp.fingerprint_file(path)
    assert (got.width, got.height) == (300, 400)


# ------------------------------------------------------------- the embedder


@pytest.mark.parametrize("orientation", ALL_ORIENTATIONS)
def test_the_encoder_is_handed_an_upright_photo(tmp_path, orientation):
    """`semantic.embed._decode`. The vector for a sideways photo has a median
    cosine of 0.935 against its upright vector, where two entirely unrelated
    photos of this library sit at 0.553 - so the error is a large fraction of
    the distance the whole search index is built on."""
    from rekindle.semantic.embed import _decode

    path = tmp_path / f"o{orientation}.jpg"
    write_oriented(path, orientation, size=(800, 600))
    assert quadrants(_decode(path, 224)) == UPRIGHT


def test_the_encoder_gets_rgb(tmp_path):
    """The transpose must not cost the conversion the encoder relies on."""
    from rekindle.semantic.embed import _decode

    path = tmp_path / "o6.jpg"
    write_oriented(path, 6)
    assert _decode(path, 224).mode == "RGB"


# ------------------------------------------------------------ the face gate


class RecordingDetector:
    """Keeps the image it was shown, so a test can ask which way up it was."""

    size = 64

    def __init__(self) -> None:
        self.seen: list[Image.Image] = []

    def detect(self, image, **_):
        self.seen.append(image.copy())
        return (), 0.0


def _photo(path: Path) -> IndexedPhoto:
    return IndexedPhoto(
        file_hash="h",
        media_type="image",
        paths=(path,),
        albums=(),
        people=(),
        archived=False,
        trashed=False,
    )


@pytest.mark.parametrize("orientation", ALL_ORIENTATIONS)
def test_the_face_detector_is_shown_an_upright_photo(tmp_path, orientation):
    """The measurement that made this the headline defect: on 200 real
    orientation-5-8 photos, decoding without the transpose changed 14.5% of
    gate verdicts, and on 7% the detector saw NO face where the upright image
    has one. A face detector is not rotation invariant, and this gate's
    failure mode is a stranger's face on the internet."""
    path = tmp_path / f"o{orientation}.jpg"
    write_oriented(path, orientation, size=(800, 600))
    detector = RecordingDetector()
    _examine(
        _photo(path),
        detector,
        allowed=set(),
        detect_threshold=DEFAULT_DETECT_THRESHOLD,
        gate_threshold=DEFAULT_GATE_THRESHOLD,
    )
    assert len(detector.seen) == 1
    assert quadrants(detector.seen[0]) == UPRIGHT


@pytest.mark.parametrize("orientation", SWAPS_AXES)
def test_the_detection_reports_the_upright_image_size(tmp_path, orientation):
    """`Detection.image_size` scales every box back into pixel coordinates.
    Reporting the raw size would put a box on the wrong axis."""
    path = tmp_path / f"o{orientation}.jpg"
    write_oriented(path, orientation, size=(600, 800))  # upright portrait
    detector = RecordingDetector()
    detection = _examine(
        _photo(path),
        detector,
        allowed=set(),
        detect_threshold=DEFAULT_DETECT_THRESHOLD,
        gate_threshold=DEFAULT_GATE_THRESHOLD,
    )
    # Not an absolute size: `draft` legitimately decodes smaller than native.
    # What must hold is that the reported size is the size of the image the
    # detector was actually shown, and that it is the PORTRAIT way round.
    assert detection.image_size == detector.seen[0].size
    assert detection.image_size[0] < detection.image_size[1]
