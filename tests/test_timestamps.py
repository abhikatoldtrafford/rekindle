from datetime import UTC, datetime

from rekindle.meta.timestamps import resolve
from rekindle.models import Gps, TzSource

NAIVE = datetime(2014, 3, 21, 17, 45)
MTIME = datetime(2020, 1, 2, 3, 4, tzinfo=UTC)


def test_exif_offset_is_preferred_and_converts_to_utc():
    utc, local, src = resolve(NAIVE, "+05:30", None, MTIME)
    assert src is TzSource.EXIF_OFFSET
    assert local.replace(tzinfo=None) == NAIVE
    assert utc == datetime(2014, 3, 21, 12, 15, tzinfo=UTC)


def test_negative_offset_converts_correctly():
    utc, _, src = resolve(NAIVE, "-08:00", None, MTIME)
    assert src is TzSource.EXIF_OFFSET
    assert utc == datetime(2014, 3, 22, 1, 45, tzinfo=UTC)


def test_gps_lookup_used_when_offset_missing():
    utc, local, src = resolve(
        NAIVE, None, Gps(15.3, 74.1), MTIME, tz_lookup=lambda _g: "Asia/Kolkata"
    )
    assert src is TzSource.GPS
    assert local.replace(tzinfo=None) == NAIVE
    assert utc == datetime(2014, 3, 21, 12, 15, tzinfo=UTC)


def test_naive_exif_treated_as_utc_when_nothing_else_known():
    utc, local, src = resolve(NAIVE, None, None, MTIME)
    assert src is TzSource.EXIF_NAIVE
    assert utc == NAIVE.replace(tzinfo=UTC)
    assert local == NAIVE.replace(tzinfo=UTC)


def test_gps_lookup_failure_falls_back_to_naive():
    utc, _, src = resolve(NAIVE, None, Gps(0.1, 0.1), MTIME, tz_lookup=lambda _g: None)
    assert src is TzSource.EXIF_NAIVE
    assert utc == NAIVE.replace(tzinfo=UTC)


def test_gps_present_but_no_tz_lookup_injected_falls_back_to_naive():
    """The ladder must skip the GPS rung entirely when no lookup is supplied,
    rather than raising or silently treating GPS as a timezone."""
    utc, _, src = resolve(NAIVE, None, Gps(15.3, 74.1), MTIME, tz_lookup=None)
    assert src is TzSource.EXIF_NAIVE
    assert utc == NAIVE.replace(tzinfo=UTC)


def test_gps_lookup_returns_unknown_zone_name_falls_back_to_naive():
    """A lookup can return a string that ZoneInfo cannot resolve; that must
    degrade to the naive rung, not raise."""
    utc, _, src = resolve(NAIVE, None, Gps(0.0, 0.0), MTIME, tz_lookup=lambda _g: "Not/AZone")
    assert src is TzSource.EXIF_NAIVE
    assert utc == NAIVE.replace(tzinfo=UTC)


def test_file_mtime_used_when_no_exif_date():
    utc, local, src = resolve(None, None, None, MTIME)
    assert src is TzSource.FILE_MTIME
    assert utc == MTIME
    assert local == MTIME


def test_no_information_at_all_yields_none():
    utc, local, src = resolve(None, None, None, None)
    assert (utc, local, src) == (None, None, TzSource.NONE)


def test_malformed_offset_falls_back_to_naive():
    utc, _, src = resolve(NAIVE, "banana", None, MTIME)
    assert src is TzSource.EXIF_NAIVE
    assert utc == NAIVE.replace(tzinfo=UTC)


def test_out_of_range_offset_falls_back_to_naive():
    """ "+30:00" matches the offset pattern but is not a valid UTC offset
    (datetime.timezone rejects anything >= 24h in magnitude); this must
    degrade to the naive rung, not raise."""
    utc, _, src = resolve(NAIVE, "+30:00", None, MTIME)
    assert src is TzSource.EXIF_NAIVE
    assert utc == NAIVE.replace(tzinfo=UTC)


def test_offset_minutes_out_of_range_falls_back_to_naive():
    """ "+05:99" matches the offset pattern (minutes is any two digits) but
    99 is not a valid minutes value; this must degrade to the naive rung,
    not be silently accepted as UTC+06:39."""
    utc, _, src = resolve(NAIVE, "+05:99", None, MTIME)
    assert src is TzSource.EXIF_NAIVE
    assert utc == NAIVE.replace(tzinfo=UTC)


def test_gps_lookup_raising_falls_back_to_naive():
    """An injected tz_lookup is a third-party callable (e.g. timezonefinder)
    that may raise on malformed coordinates. That must degrade to the naive
    rung, not propagate out of resolve()."""

    def _raising_lookup(_gps: Gps) -> str | None:
        raise RuntimeError("boom")

    utc, _, src = resolve(NAIVE, None, Gps(200.0, 200.0), MTIME, tz_lookup=_raising_lookup)
    assert src is TzSource.EXIF_NAIVE
    assert utc == NAIVE.replace(tzinfo=UTC)
