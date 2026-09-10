from pathlib import Path

from rekindle.identity import file_hash, is_long_path, sniff
from rekindle.models import MediaType

JPEG = bytes.fromhex("ffd8ffe000104a46494600010100000100010000") + b"\x00" * 64
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
GIF = b"GIF89a" + b"\x00" * 64
WEBP = b"RIFF" + (100).to_bytes(4, "little") + b"WEBP" + b"\x00" * 64
HEIC = b"\x00\x00\x00\x18ftypheic" + b"\x00" * 64
AVIF = b"\x00\x00\x00\x18ftypavif" + b"\x00" * 64
MP4 = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64
MOV = b"\x00\x00\x00\x14ftypqt  " + b"\x00" * 64
CRX = b"\x00\x00\x00\x18ftypcrx " + b"\x00" * 64
JUNK = b"not a media file at all" + b"\x00" * 64


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


def test_hash_is_stable_and_content_addressed(tmp_path):
    a = _write(tmp_path, "a.jpg", JPEG)
    b = _write(tmp_path, "b.jpg", JPEG)
    c = _write(tmp_path, "c.jpg", PNG)
    assert file_hash(a) == file_hash(b)
    assert file_hash(a) != file_hash(c)
    assert len(file_hash(a)) == 32


def test_hash_is_invariant_to_chunk_size(tmp_path):
    big = _write(tmp_path, "big.bin", b"x" * (5 * 1024 * 1024))
    assert file_hash(big, chunk_size=1024) == file_hash(big)
    assert len(file_hash(big)) == 32


def test_sniff_identifies_formats_from_magic_bytes(tmp_path):
    cases = [
        ("a.jpg", JPEG, MediaType.IMAGE, "jpeg"),
        ("a.png", PNG, MediaType.IMAGE, "png"),
        ("a.gif", GIF, MediaType.IMAGE, "gif"),
        ("a.webp", WEBP, MediaType.IMAGE, "webp"),
        ("a.heic", HEIC, MediaType.IMAGE, "heic"),
        ("a.avif", AVIF, MediaType.IMAGE, "avif"),
        ("a.mp4", MP4, MediaType.VIDEO, "mp4"),
        ("a.mov", MOV, MediaType.VIDEO, "mov"),
    ]
    for name, data, want_type, want_fmt in cases:
        p = _write(tmp_path, name, data)
        assert sniff(p) == (want_type, want_fmt), name


def test_sniff_ignores_the_extension_entirely(tmp_path):
    """Takeout ships .heic files that are actually JPEG."""
    liar = _write(tmp_path, "actually_jpeg.heic", JPEG)
    assert sniff(liar) == (MediaType.IMAGE, "jpeg")


def test_sniff_returns_unknown_for_junk(tmp_path):
    p = _write(tmp_path, "notes.txt", JUNK)
    assert sniff(p) == (MediaType.UNKNOWN, "unknown")


def test_sniff_handles_empty_file(tmp_path):
    p = _write(tmp_path, "empty.jpg", b"")
    assert sniff(p) == (MediaType.UNKNOWN, "unknown")


def test_unknown_iso_bmff_brand_is_unknown_not_guessed_video(tmp_path):
    """Unrecognised ISO-BMFF brands must return UNKNOWN, never guessed video.

    Guessing video would file a raw camera format (crx) wrongly and silently.
    UNKNOWN gets it counted and surfaced instead.
    """
    p = _write(tmp_path, "photo.cr3", CRX)
    media_type, fmt = sniff(p)
    assert media_type is MediaType.UNKNOWN
    assert fmt == "ftyp:crx "


def test_long_path_detection():
    assert is_long_path(Path("C:/" + "a" * 300)) is True
    assert is_long_path(Path("short.jpg")) is False
