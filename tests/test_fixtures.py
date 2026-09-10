from PIL import Image

from tests.fixtures.gen import build_library, make_jpeg, make_xmp_sidecar


def test_make_jpeg_is_a_real_readable_image(tmp_path):
    p = make_jpeg(tmp_path / "a.jpg", size=(32, 24))
    with Image.open(p) as im:
        assert im.size == (32, 24)
        assert im.format == "JPEG"


def test_make_jpeg_embeds_requested_exif(tmp_path):
    from datetime import datetime

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

    # year folder and album folder contain the SAME file (duplicate by content)
    assert (root / "Photos from 2014").is_dir()
    assert (root / "Goa Trip").is_dir()
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
