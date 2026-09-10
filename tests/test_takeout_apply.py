from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

from rekindle.enrich.takeout import EnrichReport, Sidecar, apply_sidecar
from rekindle.models import Gps, MediaType, Photo, PhotoMeta, TzSource

IST = timezone(timedelta(hours=5, minutes=30))
NOW = datetime(2020, 1, 1, tzinfo=UTC)


def _sidecar(**kw) -> Sidecar:
    base = dict(
        path=Path("x.json"),
        target_cf="x.jpg",
        title="x.jpg",
        taken_at_utc=None,
        people=(),
        gps=None,
        description=None,
        favorite=False,
        archived=False,
        trashed=False,
    )
    base.update(kw)
    return Sidecar(**base)


def _photo(meta: PhotoMeta) -> Photo:
    return Photo(
        file_hash="h",
        paths=[Path("x.jpg")],
        media_type=MediaType.IMAGE,
        meta=meta,
        first_seen=NOW,
        last_seen=NOW,
    )


def test_google_date_wins_and_local_keeps_the_wall_clock():
    naive = datetime(2020, 3, 11, 21, 0, tzinfo=UTC)
    photo = _photo(
        PhotoMeta(taken_at_utc=naive, taken_at_local=naive, tz_source=TzSource.EXIF_NAIVE)
    )
    report = EnrichReport()
    apply_sidecar(
        photo,
        _sidecar(taken_at_utc=datetime(2020, 3, 11, 15, 30, tzinfo=UTC)),
        frozenset(),
        report,
    )
    assert photo.meta.taken_at_utc == datetime(2020, 3, 11, 15, 30, tzinfo=UTC)
    assert photo.meta.taken_at_local.hour == 21
    assert photo.meta.exif_taken_at_utc == naive
    assert photo.metadata_conflict is False  # a timezone is not a conflict
    assert report.dates_corrected == 1


def test_a_known_exif_offset_that_agrees_is_not_a_conflict():
    """The EXIF instant is already true UTC here, so it must NOT have the
    offset subtracted before comparison. Getting that wrong invents a conflict
    on the 34% of photos that carry a real OffsetTimeOriginal."""
    google = datetime(2020, 3, 11, 15, 30, tzinfo=UTC)
    photo = _photo(
        PhotoMeta(
            taken_at_utc=google,
            taken_at_local=datetime(2020, 3, 11, 21, 0, tzinfo=IST),
            tz_source=TzSource.EXIF_OFFSET,
        )
    )
    report = EnrichReport()
    apply_sidecar(photo, _sidecar(taken_at_utc=google), frozenset(), report)
    assert photo.metadata_conflict is False
    assert report.conflicts == 0
    assert photo.meta.tz_source is TzSource.EXIF_OFFSET
    assert photo.meta.taken_at_local.hour == 21


def test_a_genuinely_different_instant_is_a_conflict_and_is_retained():
    exif = datetime(2016, 6, 24, 9, 47, 33, tzinfo=UTC)
    photo = _photo(PhotoMeta(taken_at_utc=exif, taken_at_local=exif, tz_source=TzSource.EXIF_NAIVE))
    report = EnrichReport()
    apply_sidecar(
        photo,
        _sidecar(taken_at_utc=datetime(2020, 3, 14, 14, 17, 31, tzinfo=UTC)),
        frozenset(),
        report,
    )
    assert photo.metadata_conflict is True
    assert photo.meta.exif_taken_at_utc == exif  # the promise v1 could not keep
    assert report.conflicts == 1


def test_a_clustered_google_date_does_not_override_a_real_exif_date():
    """618 real sidecars share one second. They did not fire together."""
    exif = datetime(2016, 6, 24, 9, 47, 33, tzinfo=UTC)
    google = datetime(2020, 3, 11, 7, 30, tzinfo=UTC)
    photo = _photo(PhotoMeta(taken_at_utc=exif, taken_at_local=exif, tz_source=TzSource.EXIF_NAIVE))
    report = EnrichReport()
    apply_sidecar(photo, _sidecar(taken_at_utc=google), frozenset({google}), report)
    assert photo.meta.taken_at_utc == exif
    assert photo.meta.tz_source is TzSource.EXIF_NAIVE
    assert report.clustered_dates_suppressed == 1
    assert report.dates_corrected == 0


def test_a_clustered_date_still_applies_when_there_is_no_real_exif_date():
    mtime = datetime(2024, 1, 1, tzinfo=UTC)
    google = datetime(2020, 3, 11, 7, 30, tzinfo=UTC)
    photo = _photo(
        PhotoMeta(taken_at_utc=mtime, taken_at_local=mtime, tz_source=TzSource.FILE_MTIME)
    )
    report = EnrichReport()
    apply_sidecar(photo, _sidecar(taken_at_utc=google), frozenset({google}), report)
    assert photo.meta.taken_at_utc == google
    assert report.clustered_dates_suppressed == 0


def test_people_replace_the_previous_takeout_set_and_keep_other_sources():
    photo = _photo(PhotoMeta(people=["FromXmp", "Ada", "Grace"], takeout_people=["Ada", "Grace"]))
    report = EnrichReport()
    apply_sidecar(photo, _sidecar(people=("Grace",)), frozenset(), report)
    assert photo.meta.people == ["FromXmp", "Grace"]  # Ada retracted
    assert photo.meta.takeout_people == ["Grace"]


def test_applying_twice_is_idempotent():
    photo = _photo(
        PhotoMeta(
            taken_at_utc=datetime(2020, 3, 11, 21, 0, tzinfo=UTC),
            taken_at_local=datetime(2020, 3, 11, 21, 0, tzinfo=UTC),
            tz_source=TzSource.EXIF_NAIVE,
            people=["Ada"],
        )
    )
    sidecar = _sidecar(
        taken_at_utc=datetime(2020, 3, 11, 15, 30, tzinfo=UTC),
        people=("Grace",),
        gps=Gps(lat=1.0, lon=2.0),
        description="hello",
        favorite=True,
    )
    apply_sidecar(photo, sidecar, frozenset(), EnrichReport())
    first = (
        photo.meta.taken_at_utc,
        photo.meta.taken_at_local,
        photo.meta.tz_source,
        list(photo.meta.people),
        list(photo.meta.takeout_people),
        photo.meta.exif_taken_at_utc,
        photo.metadata_conflict,
    )
    apply_sidecar(photo, sidecar, frozenset(), EnrichReport())
    assert first == (
        photo.meta.taken_at_utc,
        photo.meta.taken_at_local,
        photo.meta.tz_source,
        list(photo.meta.people),
        list(photo.meta.takeout_people),
        photo.meta.exif_taken_at_utc,
        photo.metadata_conflict,
    )


def test_gps_fills_only_when_absent():
    photo = _photo(PhotoMeta(gps=Gps(lat=10.0, lon=20.0)))
    report = EnrichReport()
    apply_sidecar(photo, _sidecar(gps=Gps(lat=1.0, lon=2.0)), frozenset(), report)
    assert photo.meta.gps.lat == 10.0
    assert report.gps_added == 0

    bare = _photo(PhotoMeta())
    apply_sidecar(bare, _sidecar(gps=Gps(lat=1.0, lon=2.0)), frozenset(), report)
    assert bare.meta.gps.lat == 1.0
    assert report.gps_added == 1


def test_longest_description_wins_and_flags_carry_over():
    photo = _photo(PhotoMeta(description="short"))
    report = EnrichReport()
    apply_sidecar(
        photo,
        _sidecar(description="a much longer caption", favorite=True, archived=True, trashed=True),
        frozenset(),
        report,
    )
    assert photo.meta.description == "a much longer caption"
    assert photo.meta.favorite is True
    assert photo.meta.archived is True
    assert photo.meta.trashed is True
    assert report.descriptions_added == 1
    assert report.favourites_added == 1


def test_sidecar_match_is_recorded():
    photo = _photo(PhotoMeta())
    apply_sidecar(photo, _sidecar(), frozenset(), EnrichReport())
    assert photo.sidecar_match == "exact"
