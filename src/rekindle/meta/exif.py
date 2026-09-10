"""EXIF extraction via Pillow. Never raises - a broken file yields empty data."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

from rekindle.models import Gps

# Without this, Image.open on a HEIC raises, read_exif swallows it, and every
# iPhone photo indexes with no date, no GPS and no dimensions - silently.
# The `heic` extra installs pillow-heif; absence is a supported configuration.
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional extra
    HEIF_AVAILABLE = False

_EXIF_IFD = 0x8769
_GPS_IFD = 0x8825
_MAKE, _MODEL = 0x010F, 0x0110
_DATETIME_ORIGINAL, _OFFSET_ORIGINAL = 0x9003, 0x9011
_DATETIME_DIGITIZED, _OFFSET_DIGITIZED = 0x9004, 0x9012


@dataclass(frozen=True)
class ExifData:
    taken_naive: datetime | None = None
    offset: str | None = None
    gps: Gps | None = None
    camera_make: str | None = None
    camera_model: str | None = None
    width: int | None = None
    height: int | None = None
    # False when the image could not be decoded at all. Without this, a
    # truncated JPEG is indistinguishable from a photo that merely lacks EXIF,
    # and doctor reports perfect health on a library of corrupt files.
    decode_ok: bool = True
    error: str | None = None


def _rational(value: object) -> float:
    """Raises ZeroDivisionError on a damaged (zero-denominator) rational.

    It used to substitute 0.0, which turned one damaged component of a GPS
    triple into a wrong-but-entirely-plausible coordinate - off by up to half
    a degree, and indistinguishable downstream from a real reading. Because
    it never raised, _dms_to_decimal's ZeroDivisionError guard was
    unreachable. Silently-wrong location is the one failure this tool must
    not produce, so a damaged part now rejects the whole coordinate.
    """
    if isinstance(value, tuple) and len(value) == 2:
        num, den = value
        if not den:
            raise ZeroDivisionError("zero denominator in EXIF rational")
        return float(num) / float(den)
    return float(value)  # type: ignore[arg-type]


def _dms_to_decimal(dms: object, ref: object) -> float | None:
    try:
        d, m, s = (_rational(x) for x in dms)  # type: ignore[misc]
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    dec = d + m / 60.0 + s / 3600.0
    # Pillow does not hand a damaged rational back as a (num, den) tuple: it
    # returns an IFDRational that floats to NaN, so the guard above never
    # sees it. NaN propagates through every arithmetic and comparison
    # downstream without ever looking wrong, which is exactly how it reaches
    # the database. Reject anything non-finite, whatever produced it.
    if not math.isfinite(dec):
        return None
    if str(ref).upper().strip() in {"S", "W"}:
        dec = -dec
    return dec


def _parse_dt(raw: object) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        return datetime.strptime(raw.strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def read_exif(path: Path) -> ExifData:
    try:
        # A damaged-but-decodable EXIF block (seen on a real Takeout export)
        # makes Pillow emit a UserWarning straight to stderr. The file's
        # readability is already reported via decode_ok/error below, so this
        # scopes ONLY the Pillow calls - never a global filter, and nowhere
        # else in the codebase.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Image.open(path) as im:
                width, height = im.size
                exif = im.getexif()
                sub = exif.get_ifd(_EXIF_IFD)
                gps_ifd = exif.get_ifd(_GPS_IFD)
    except Image.DecompressionBombError as exc:
        return ExifData(decode_ok=False, error=f"decompression bomb: {exc}")
    except Exception as exc:
        return ExifData(decode_ok=False, error=f"{type(exc).__name__}: {exc}")

    taken = _parse_dt(sub.get(_DATETIME_ORIGINAL)) or _parse_dt(sub.get(_DATETIME_DIGITIZED))
    offset = sub.get(_OFFSET_ORIGINAL) or sub.get(_OFFSET_DIGITIZED)
    offset = offset.strip() if isinstance(offset, str) else None

    gps = None
    if gps_ifd:
        lat = _dms_to_decimal(gps_ifd.get(2), gps_ifd.get(1))
        lon = _dms_to_decimal(gps_ifd.get(4), gps_ifd.get(3))
        if lat is not None and lon is not None and not (lat == 0.0 and lon == 0.0):
            alt = None
            if gps_ifd.get(6) is not None:
                try:
                    alt = _rational(gps_ifd.get(6))
                except (TypeError, ValueError, ZeroDivisionError):
                    alt = None
                else:
                    alt = alt if math.isfinite(alt) else None
            gps = Gps(lat=lat, lon=lon, alt=alt)

    def _clean(v: object) -> str | None:
        return v.strip() or None if isinstance(v, str) else None

    return ExifData(
        taken_naive=taken,
        offset=offset,
        gps=gps,
        camera_make=_clean(exif.get(_MAKE)),
        camera_model=_clean(exif.get(_MODEL)),
        width=width,
        height=height,
    )
