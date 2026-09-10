"""Local time resolution.

Preference order, recorded in TzSource so doctor can report it:
    EXIF OffsetTimeOriginal -> GPS timezone -> naive EXIF (assumed UTC)
    -> file mtime -> nothing.

GPS lookup is injected rather than imported: timezonefinder is a large
optional dependency and must never be required.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from rekindle.models import Gps, TzSource

TzLookup = Callable[[Gps], str | None]

_OFFSET_RE = re.compile(r"^([+-])(\d{2}):?(\d{2})$")


def _parse_offset(raw: str | None) -> timezone | None:
    if not raw:
        return None
    m = _OFFSET_RE.match(raw.strip())
    if not m:
        return None
    sign, hours, minutes = m.groups()
    minutes_val = int(minutes)
    if not 0 <= minutes_val < 60:
        # e.g. "+05:99": matches the regex (any two digits) but is not a
        # valid minutes value. Treat it as malformed, same as "banana".
        return None
    delta = timedelta(hours=int(hours), minutes=minutes_val)
    try:
        return timezone(-delta if sign == "-" else delta)
    except ValueError:
        # e.g. "+30:00": matches the regex but is out of range for
        # datetime.timezone. Treat it as malformed, same as "banana".
        return None


def resolve(
    exif_taken: datetime | None,
    exif_offset: str | None,
    gps: Gps | None,
    file_mtime: datetime | None,
    tz_lookup: TzLookup | None = None,
) -> tuple[datetime | None, datetime | None, TzSource]:
    """Return (utc, local, source)."""
    if exif_taken is not None:
        tz = _parse_offset(exif_offset)
        if tz is not None:
            local = exif_taken.replace(tzinfo=tz)
            return local.astimezone(UTC), local, TzSource.EXIF_OFFSET

        if gps is not None and tz_lookup is not None:
            try:
                name = tz_lookup(gps)
            except Exception:
                # tz_lookup is an injected third-party callable (e.g. a
                # timezonefinder-backed lookup); it may raise on malformed
                # coordinates. Any failure here degrades to the naive rung,
                # same as a None return or an unresolvable zone name.
                name = None
            if name:
                try:
                    zone = ZoneInfo(name)
                except (ZoneInfoNotFoundError, ValueError):
                    zone = None
                if zone is not None:
                    local = exif_taken.replace(tzinfo=zone)
                    return local.astimezone(UTC), local, TzSource.GPS

        assumed = exif_taken.replace(tzinfo=UTC)
        return assumed, assumed, TzSource.EXIF_NAIVE

    if file_mtime is not None:
        stamped = file_mtime if file_mtime.tzinfo else file_mtime.replace(tzinfo=UTC)
        return stamped, stamped, TzSource.FILE_MTIME

    return None, None, TzSource.NONE
