import warnings
from datetime import datetime

import pytest

from rekindle.meta.exif import _dms_to_decimal, _rational, read_exif
from tests.fixtures.gen import (
    make_corrupt_exif_jpeg,
    make_jpeg,
    make_zero_denominator_gps_jpeg,
)


def test_reads_datetime_offset_and_camera(tmp_path):
    p = make_jpeg(
        tmp_path / "a.jpg",
        taken=datetime(2014, 3, 21, 17, 45),
        offset="+05:30",
        make="Canon",
        model="EOS R",
    )
    d = read_exif(p)
    assert d.taken_naive == datetime(2014, 3, 21, 17, 45)
    assert d.offset == "+05:30"
    assert d.camera_make == "Canon"
    assert d.camera_model == "EOS R"


def test_reads_dimensions(tmp_path):
    p = make_jpeg(tmp_path / "b.jpg", size=(120, 90))
    assert read_exif(p).width == 120
    assert read_exif(p).height == 90


def test_reads_gps_and_converts_dms_to_decimal(tmp_path):
    p = make_jpeg(tmp_path / "c.jpg", gps=(15.2993, 74.1240))
    d = read_exif(p)
    assert d.gps is not None
    assert abs(d.gps.lat - 15.2993) < 0.001
    assert abs(d.gps.lon - 74.1240) < 0.001


def test_southern_and_western_hemispheres_are_negative(tmp_path):
    p = make_jpeg(tmp_path / "d.jpg", gps=(-33.8688, -151.2093))
    d = read_exif(p)
    assert d.gps.lat < 0
    assert d.gps.lon < 0
    assert abs(d.gps.lat - (-33.8688)) < 0.001
    assert abs(d.gps.lon - (-151.2093)) < 0.001


def test_photo_without_exif_returns_empty_not_error(tmp_path):
    p = make_jpeg(tmp_path / "e.jpg")
    d = read_exif(p)
    assert d.taken_naive is None
    assert d.gps is None


def test_undecodable_file_is_flagged_not_silently_empty(tmp_path):
    """A corrupt image must be distinguishable from one that simply lacks EXIF."""
    p = tmp_path / "broken.jpg"
    p.write_bytes(b"\xff\xd8\xff" + b"garbage")
    d = read_exif(p)
    assert d.decode_ok is False
    assert d.error
    assert d.taken_naive is None


def test_healthy_file_without_exif_is_decode_ok(tmp_path):
    d = read_exif(make_jpeg(tmp_path / "plain.jpg"))
    assert d.decode_ok is True
    assert d.taken_naive is None


def test_corrupt_exif_produces_no_warning_output(tmp_path):
    """A real scan printed Pillow's 'Corrupt EXIF data' UserWarning straight
    to stderr. The file is still readable and already reported via
    decode_ok/error - the warning itself must be suppressed, not the
    information it carries."""
    p = make_corrupt_exif_jpeg(tmp_path / "corrupt_exif.jpg")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        read_exif(p)
    assert caught == []


def test_zero_denominator_rational_is_rejected_not_invented_as_zero():
    """`float(num) / float(den) if den else 0.0` turned a damaged rational
    into a real-looking 0, and because it never raised, _dms_to_decimal's
    ZeroDivisionError guard was unreachable dead code."""
    assert _rational((17, 2)) == 8.5
    with pytest.raises(ZeroDivisionError):
        _rational((17, 0))


def test_one_damaged_rational_rejects_the_whole_coordinate():
    """15deg 0min 57.57sec with the MINUTES damaged is not "15.016 degrees" -
    it is an unknown position. Substituting 0 for the broken part yields a
    coordinate that is wrong by up to half a degree and looks perfectly
    plausible, which no downstream check can catch."""
    intact = _dms_to_decimal(((15, 1), (17, 1), (5757, 100)), "N")
    assert intact is not None
    assert _dms_to_decimal(((15, 1), (17, 0), (5757, 100)), "N") is None


def test_gps_with_a_damaged_rational_yields_no_gps_at_all(tmp_path):
    """End to end, through a real JPEG. Silently-wrong GPS is the one failure
    this tool must never produce: a montage narrated with the wrong place is
    worse than one narrated with no place."""
    p = make_zero_denominator_gps_jpeg(tmp_path / "damaged_gps.jpg")
    d = read_exif(p)
    assert d.decode_ok is True  # the FILE is fine; only the coordinate is not
    assert d.gps is None
