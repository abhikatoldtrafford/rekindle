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


# Real UTC offsets run from -12:00 to +14:00 and are whole quarter-hours.
_MAX_OFFSET = timedelta(hours=14)
_OFFSET_QUANTUM = timedelta(minutes=15)
_QUANTUM_TOLERANCE = timedelta(seconds=90)


def is_plausible_offset(delta: timedelta) -> timedelta | None:
    """Round `delta` to a real UTC offset, or reject it.

    This is the guard that keeps a broken camera clock out of the timezone
    ladder. On the reference export the EXIF/Google difference is either
    exactly -05:30 (a timezone) or over a year (a reset clock); nothing in
    between. Rejecting anything that is not a quarter-hour within 14 hours
    separates the two cleanly.
    """
    if abs(delta) > _MAX_OFFSET:
        return None
    quanta = round(delta / _OFFSET_QUANTUM)
    rounded = _OFFSET_QUANTUM * quanta
    if abs(delta - rounded) > _QUANTUM_TOLERANCE:
        return None
    return rounded


def from_takeout(
    google_utc: datetime,
    existing_utc: datetime | None,
    existing_local: datetime | None,
    existing_tz_source: TzSource,
    gps: Gps | None = None,
    tz_lookup: TzLookup | None = None,
) -> tuple[datetime, datetime, TzSource]:
    """Derive (utc, local, tz_source) from Google's photoTakenTime.

    photoTakenTime is an INSTANT. It carries no offset, so local time has to
    come from somewhere else. Passing it to `resolve()` - which expects a
    naive wall-clock value - silently yields local == utc and relabels a 21:00
    photo as 15:30. That was v1's most damaging defect.

    Ladder, per spec section 5:
      1. an offset EXIF already established (OffsetTimeOriginal)
      2. a GPS timezone
      3. exif_naive - google_utc, which recovers +05:30 exactly
      4. UTC

    `TzSource.TAKEOUT` describes the DATE's provenance. When an offset or GPS
    resolved the zone, that source is kept: conflating the two loses
    information about how well local time is actually known.
    """
    # Rung 1 and 2: M0 already resolved a real zone and stored it on
    # taken_at_local. Reuse it rather than re-deriving - and note this keeps
    # `enrich` free of any file or timezonefinder access for these photos.
    if existing_tz_source in (TzSource.EXIF_OFFSET, TzSource.GPS) and existing_local is not None:
        offset = existing_local.utcoffset()
        if offset is not None:
            zone = timezone(offset)
            return google_utc, google_utc.astimezone(zone), existing_tz_source

    # Rung 2 for a photo M0 could not date at all but which has coordinates.
    if gps is not None and tz_lookup is not None:
        try:
            name = tz_lookup(gps)
        except Exception:
            # An injected third-party callable may raise on odd coordinates.
            # Any failure degrades to the next rung, exactly as in resolve().
            name = None
        if name:
            try:
                zone_info = ZoneInfo(name)
            except (ZoneInfoNotFoundError, ValueError):
                zone_info = None
            if zone_info is not None:
                return google_utc, google_utc.astimezone(zone_info), TzSource.GPS

    # Rung 3: M0 stored the naive EXIF wall clock stamped as UTC. The gap
    # between that and Google's true instant IS the photo's UTC offset.
    if existing_tz_source is TzSource.EXIF_NAIVE and existing_utc is not None:
        offset = is_plausible_offset(existing_utc - google_utc)
        if offset is not None:
            return google_utc, google_utc.astimezone(timezone(offset)), TzSource.TAKEOUT

    # Rung 4.
    return google_utc, google_utc, TzSource.TAKEOUT
