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


from datetime import timedelta as _timedelta  # noqa: E402
from datetime import timezone as _timezone  # noqa: E402

from rekindle.meta.timestamps import from_takeout, is_plausible_offset  # noqa: E402

IST = _timezone(_timedelta(hours=5, minutes=30))


def test_a_known_exif_offset_is_kept_not_discarded():
    """442 of 1,317 measured photos already had this and v1 threw it away."""
    google = datetime(2020, 3, 11, 15, 30, tzinfo=UTC)
    utc, local, src = from_takeout(
        google_utc=google,
        existing_utc=datetime(2020, 3, 11, 15, 30, tzinfo=UTC),
        existing_local=datetime(2020, 3, 11, 21, 0, tzinfo=IST),
        existing_tz_source=TzSource.EXIF_OFFSET,
    )
    assert utc == google
    assert local.hour == 21 and local.minute == 0
    assert local.utcoffset() == _timedelta(hours=5, minutes=30)
    assert src is TzSource.EXIF_OFFSET


def test_naive_exif_recovers_the_offset_from_the_difference():
    """811 of 1,317 measured photos. M0 stored 21:00 stamped UTC; Google says
    the instant was 15:30Z. The difference IS +05:30."""
    naive_stamped_utc = datetime(2020, 3, 11, 21, 0, tzinfo=UTC)
    google = datetime(2020, 3, 11, 15, 30, tzinfo=UTC)
    utc, local, src = from_takeout(
        google_utc=google,
        existing_utc=naive_stamped_utc,
        existing_local=naive_stamped_utc,
        existing_tz_source=TzSource.EXIF_NAIVE,
    )
    assert utc == google
    assert local.hour == 21 and local.minute == 0  # NOT 15:30
    assert local.utcoffset() == _timedelta(hours=5, minutes=30)
    assert src is TzSource.TAKEOUT


def test_a_broken_camera_clock_is_not_mistaken_for_an_offset():
    """EXIF says 2016-06-24, Google says 2020-03-14. That 1,359-day gap is a
    reset camera clock, not a timezone."""
    utc, local, src = from_takeout(
        google_utc=datetime(2020, 3, 14, 14, 17, 31, tzinfo=UTC),
        existing_utc=datetime(2016, 6, 24, 9, 47, 33, tzinfo=UTC),
        existing_local=datetime(2016, 6, 24, 9, 47, 33, tzinfo=UTC),
        existing_tz_source=TzSource.EXIF_NAIVE,
    )
    assert utc == datetime(2020, 3, 14, 14, 17, 31, tzinfo=UTC)
    assert local == utc
    assert src is TzSource.TAKEOUT


def test_no_prior_date_falls_back_to_utc():
    google = datetime(2020, 3, 11, 15, 30, tzinfo=UTC)
    utc, local, src = from_takeout(
        google_utc=google,
        existing_utc=datetime(2024, 1, 1, tzinfo=UTC),
        existing_local=datetime(2024, 1, 1, tzinfo=UTC),
        existing_tz_source=TzSource.FILE_MTIME,
    )
    assert utc == google and local == google
    assert src is TzSource.TAKEOUT


def test_gps_resolves_the_zone_when_there_is_no_exif_date():
    google = datetime(2020, 3, 11, 15, 30, tzinfo=UTC)
    utc, local, src = from_takeout(
        google_utc=google,
        existing_utc=None,
        existing_local=None,
        existing_tz_source=TzSource.NONE,
        gps=Gps(lat=22.5, lon=88.3),
        tz_lookup=lambda _g: "Asia/Kolkata",
    )
    assert utc == google
    assert local.hour == 21 and local.minute == 0
    assert src is TzSource.GPS


def test_a_raising_tz_lookup_degrades_instead_of_crashing():
    def boom(_g):
        raise RuntimeError("timezonefinder exploded")

    google = datetime(2020, 3, 11, 15, 30, tzinfo=UTC)
    utc, local, src = from_takeout(
        google_utc=google,
        existing_utc=None,
        existing_local=None,
        existing_tz_source=TzSource.NONE,
        gps=Gps(lat=22.5, lon=88.3),
        tz_lookup=boom,
    )
    assert utc == google and local == google and src is TzSource.TAKEOUT


def test_plausible_offsets():
    assert is_plausible_offset(_timedelta(hours=5, minutes=30)) == _timedelta(hours=5, minutes=30)
    assert is_plausible_offset(_timedelta(hours=-8)) == _timedelta(hours=-8)
    assert is_plausible_offset(_timedelta(hours=5, minutes=30, seconds=12)) == _timedelta(
        hours=5, minutes=30
    )
    assert is_plausible_offset(_timedelta(hours=15)) is None
    assert is_plausible_offset(_timedelta(days=1359)) is None
    assert is_plausible_offset(_timedelta(hours=5, minutes=37)) is None
