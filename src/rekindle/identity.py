"""File identity and type detection.

Identity is a hash of FILE bytes, never decoded pixels: decoding is expensive
and its output changes between Pillow/libjpeg versions, which would silently
invalidate the whole resume cache.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from rekindle.models import MediaType

_CHUNK = 1 << 20
_MAX_PATH = 260

# ISO-BMFF brands. Anything ftyp-based is decided by its brand, not extension.
_IMAGE_BRANDS = {"heic", "heix", "hevc", "hevx", "mif1", "msf1", "heif"}
_AVIF_BRANDS = {"avif", "avis"}
_VIDEO_BRANDS = {"isom", "iso2", "mp41", "mp42", "dash", "m4v ", "3gp4", "3gp5"}
_QT_BRANDS = {"qt  "}


def file_hash(path: Path, *, chunk_size: int = _CHUNK) -> str:
    """BLAKE2b-128 of the file's bytes, streamed."""
    h = hashlib.blake2b(digest_size=16)
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def sniff(path: Path) -> tuple[MediaType, str]:
    """Identify media type from magic bytes. The extension is never consulted.

    Raises OSError if the file cannot be read. Swallowing that into
    (UNKNOWN, "unknown") made a permission-denied or locked file
    indistinguishable from a text file, so callers filed it as "not media" -
    a positive claim about content that was never actually inspected. The
    caller must tell the two apart, so the IO failure propagates.
    """
    with path.open("rb") as f:
        head = f.read(32)

    if len(head) < 12:
        return (MediaType.UNKNOWN, "unknown")

    if head.startswith(b"\xff\xd8\xff"):
        return (MediaType.IMAGE, "jpeg")
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return (MediaType.IMAGE, "png")
    if head.startswith((b"GIF87a", b"GIF89a")):
        return (MediaType.IMAGE, "gif")
    if head.startswith(b"BM"):
        return (MediaType.IMAGE, "bmp")
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return (MediaType.IMAGE, "webp")
    if head.startswith(b"RIFF") and head[8:12] == b"AVI ":
        return (MediaType.VIDEO, "avi")
    if head.startswith((b"II*\x00", b"MM\x00*")):
        return (MediaType.IMAGE, "tiff")

    if head[4:8] == b"ftyp":
        brand = head[8:12].decode("ascii", errors="replace").lower()
        if brand in _AVIF_BRANDS:
            return (MediaType.IMAGE, "avif")
        if brand in _IMAGE_BRANDS:
            return (MediaType.IMAGE, "heic")
        if brand in _QT_BRANDS:
            return (MediaType.VIDEO, "mov")
        if brand in _VIDEO_BRANDS:
            return (MediaType.VIDEO, "mp4")
        # An unrecognised ISO-BMFF brand might be a raw camera format (crx) or
        # anything else. Guessing "video" would file it wrongly and silently;
        # UNKNOWN gets it counted and surfaced instead.
        return (MediaType.UNKNOWN, f"ftyp:{brand}")

    return (MediaType.UNKNOWN, "unknown")


def is_long_path(path: Path) -> bool:
    """Windows MAX_PATH check. pathlib does not solve this; we report it.

    Windows-only on purpose: Linux and macOS allow paths far longer than 260
    characters, so applying the limit everywhere made a deep-but-perfectly-
    healthy Linux tree trigger doctor's "On Windows, enable LongPathsEnabled"
    advice - advice that cannot possibly apply.
    """
    if sys.platform != "win32":
        return False
    return len(str(path)) >= _MAX_PATH
