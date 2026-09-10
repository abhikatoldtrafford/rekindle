from datetime import datetime

import pytest
from PIL import Image

from tests.fixtures.gen import build_library, make_jpeg, make_xmp_sidecar


def _gps_ifd_to_degrees(gps_ifd) -> tuple[float, float]:
    """Decode an EXIF GPS IFD (as produced by make_jpeg) back to decimal degrees."""

    def dms_to_deg(dms) -> float:
        d, m, s = dms
        return float(d) + float(m) / 60 + float(s) / 3600

    lat = dms_to_deg(gps_ifd[2])
    if gps_ifd[1] == "S":
        lat = -lat
    lon = dms_to_deg(gps_ifd[4])
    if gps_ifd[3] == "W":
        lon = -lon
    return lat, lon


def test_make_jpeg_is_a_real_readable_image(tmp_path):
    p = make_jpeg(tmp_path / "a.jpg", size=(32, 24))
    with Image.open(p) as im:
        assert im.size == (32, 24)
        assert im.format == "JPEG"


def test_make_jpeg_embeds_requested_exif(tmp_path):
    p = make_jpeg(
        tmp_path / "b.jpg",
        taken=datetime(2014, 3, 21, 17, 45, 0),
        offset="+05:30",
        make="Canon",
        model="EOS R",
    )
    with Image.open(p) as im:
        exif = im.getexif()
        ifd = exif.get_ifd(0x8769)
    assert ifd[0x9003] == "2014:03:21 17:45:00"
    assert ifd[0x9011] == "+05:30"
    assert exif[0x010F] == "Canon"


def test_make_jpeg_embeds_gps_northern_eastern_hemisphere(tmp_path):
    p = make_jpeg(tmp_path / "gps_ne.jpg", gps=(15.2993, 74.1240))
    with Image.open(p) as im:
        exif = im.getexif()
        gps_ifd = exif.get_ifd(0x8825)
    assert gps_ifd[1] == "N"
    assert gps_ifd[3] == "E"
    lat, lon = _gps_ifd_to_degrees(gps_ifd)
    assert lat == pytest.approx(15.2993, abs=0.001)
    assert lon == pytest.approx(74.1240, abs=0.001)


def test_make_jpeg_embeds_gps_southern_western_hemisphere(tmp_path):
    p = make_jpeg(tmp_path / "gps_sw.jpg", gps=(-33.8688, -151.2093))
    with Image.open(p) as im:
        exif = im.getexif()
        gps_ifd = exif.get_ifd(0x8825)
    assert gps_ifd[1] == "S"
    assert gps_ifd[3] == "W"
    lat, lon = _gps_ifd_to_degrees(gps_ifd)
    assert lat == pytest.approx(-33.8688, abs=0.001)
    assert lon == pytest.approx(-151.2093, abs=0.001)


def test_make_xmp_sidecar_writes_parseable_xml(tmp_path):
    img = make_jpeg(tmp_path / "c.jpg")
    side = make_xmp_sidecar(img, people=["Alice"], description="a day out")
    text = side.read_text(encoding="utf-8")
    assert side.name == "c.jpg.xmp"
    assert "Alice" in text
    assert "a day out" in text


def test_build_library_produces_the_expected_shape(tmp_path):
    root = build_library(tmp_path / "lib")
    names = {p.name for p in root.rglob("*") if p.is_file()}

    year = root / "Photos from 2014"
    album = root / "Goa Trip"

    # year folder and album folder contain the SAME file (duplicate by content)
    assert year.is_dir()
    assert album.is_dir()
    assert (year / "IMG_0001.jpg").read_bytes() == (album / "IMG_0001.jpg").read_bytes()
    # an edited variant exists alongside its original
    assert any(n.endswith("-edited.jpg") for n in names)
    # a JSON sidecar exists and must be ignored as media
    assert any(n.endswith(".json") for n in names)
    # an XMP sidecar exists
    assert any(n.endswith(".xmp") for n in names)
    # a file whose extension lies about its content
    assert "actually_jpeg.heic" in names
    # a non-media file
    assert "notes.txt" in names

    # --- shapes drawn from a real 4.3 GB Takeout export ---

    # Deleted photos live under Trash/ and must never be indexed.
    assert (root / "Trash" / "deleted.jpg").is_file()

    # A sidecar orphaned because its photo lives in an unextracted archive part.
    assert (year / "IMG_9999.jpg.supplemental-metadata.json").is_file()
    assert not (year / "IMG_9999.jpg").exists()

    # A motion photo: the .MP video sits beside its still and is a real
    # ISO-BMFF container (not JPEG bytes with a renamed extension) — Task 9's
    # sniff() types it by exactly these bytes.
    header = (album / "PXL_0001.MP").read_bytes()[:12]
    assert header[4:8] == b"ftyp"
    assert header[8:12] == b"isom"

    # The duplicate counter lands INSIDE the .json suffix, not after it.
    assert (year / "DSC_0880.JPG.supplemental-metadata(1).json").is_file()
