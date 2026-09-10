from datetime import datetime

from rekindle.meta.exif import read_exif
from tests.fixtures.gen import make_jpeg


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
