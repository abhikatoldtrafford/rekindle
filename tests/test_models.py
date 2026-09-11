# tests/test_models.py
from datetime import UTC, datetime

from rekindle.models import (
    FaceRegion,
    Gps,
    MediaType,
    Photo,
    PhotoMeta,
    SourceReport,
    TzSource,
    merge_meta,
)


def test_photo_meta_empty_has_no_data():
    m = PhotoMeta.empty()
    assert m.taken_at_utc is None
    assert m.gps is None
    assert m.people == []
    assert m.face_regions == []
    assert m.tz_source is TzSource.NONE


def test_photo_meta_lists_are_independent_between_instances():
    a = PhotoMeta.empty()
    b = PhotoMeta.empty()
    a.people.append("Alice")
    assert b.people == []


def test_photo_defaults():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    p = Photo(
        file_hash="abc",
        paths=[],
        media_type=MediaType.IMAGE,
        meta=PhotoMeta.empty(),
        first_seen=now,
        last_seen=now,
    )
    assert p.albums == []
    assert p.edited_of is None


def test_face_region_is_hashable_and_compares_by_value():
    # merge_meta calls dict.fromkeys(...) on face_regions to de-duplicate
    # while unioning. That only works if FaceRegion is frozen (hashable)
    # and compares by value. If it ever stopped being frozen, that call
    # would raise TypeError: unhashable type, breaking the union at
    # runtime - this test protects that invariant.
    a = FaceRegion(name="Alice", x=0.5, y=0.4, w=0.2, h=0.25)
    b = FaceRegion(name="Alice", x=0.5, y=0.4, w=0.2, h=0.25)
    c = FaceRegion(name="Bob", x=0.5, y=0.4, w=0.2, h=0.25)

    assert a == b
    assert hash(a) == hash(b)
    assert a != c

    deduped = {a, b, c}
    assert deduped == {a, c}

    as_dict_key = {a: "first", b: "second"}
    assert as_dict_key == {a: "second"}


def test_gps_is_frozen():
    import dataclasses

    import pytest

    g = Gps(lat=15.3, lon=74.0, alt=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        g.lat = 0.0


def test_source_report_counts_skips_by_reason():
    r = SourceReport()
    r.skip("not_media")
    r.skip("not_media")
    r.skip("unreadable")
    assert r.skipped == {"not_media": 2, "unreadable": 1}
    assert r.total_skipped == 3


# --- merge_meta ---
#
# merge_meta is the most defect-prone function in this milestone: an audit
# found the previous merge implementation destroyed metadata on every
# re-index. These tests are the regression guard for that failure mode.


def test_merge_meta_earliest_real_date_wins():
    earlier = datetime(2020, 1, 1, tzinfo=UTC)
    later = datetime(2021, 1, 1, tzinfo=UTC)
    old = PhotoMeta(taken_at_utc=later, tz_source=TzSource.EXIF_OFFSET)
    new = PhotoMeta(taken_at_utc=earlier, tz_source=TzSource.EXIF_OFFSET)

    merged, conflict = merge_meta(old, new)

    assert merged.taken_at_utc == earlier
    assert merged.tz_source is TzSource.EXIF_OFFSET
    assert conflict is True


def test_merge_meta_earliest_wins_when_old_is_the_earlier_side():
    """Residual 5: the test above puts the earlier date on `new`, where
    `keep = new` (the exact bug this tiebreak guards against) happens to
    pick the SAME side as the real earliest-wins rule - mutating the
    tiebreak to unconditional `keep = new` left the FULL SUITE green,
    258 passed. Swapping which side is earlier is the only way the two
    can be told apart.

    MUTATION (run, not assumed): replace `keep = old if old.taken_at_utc <=
    new.taken_at_utc else new` with `keep = new` and this fails on
    `merged.taken_at_utc == earlier` (reads `later` instead).
    """
    earlier = datetime(2020, 1, 1, tzinfo=UTC)
    later = datetime(2021, 1, 1, tzinfo=UTC)
    old = PhotoMeta(taken_at_utc=earlier, tz_source=TzSource.EXIF_OFFSET)
    new = PhotoMeta(taken_at_utc=later, tz_source=TzSource.EXIF_OFFSET)

    merged, conflict = merge_meta(old, new)

    assert merged.taken_at_utc == earlier
    assert merged.tz_source is TzSource.EXIF_OFFSET
    assert conflict is True


def test_merge_meta_an_enriched_new_side_wins_symmetrically():
    """Residual 5: `elif _enriched(new) and not _enriched(old): keep = new`
    is unreachable through either shipped call site (`db._merge` and
    `FolderSource.scan` always pass a folder-scanned `new`, and no shipped
    `Source` can make `_enriched` true) - replacing it with `elif False:`
    left the full suite green too. Called directly, bypassing both call
    sites, it is the exact mirror of the `_enriched(old)` branch just above
    it in `merge_meta`.

    MUTATION (run, not assumed): replace the branch condition with `elif
    False:` and this fails on `merged.taken_at_utc == enriched_instant`
    (falls through to the earliest-wins tiebreak instead, reading the
    camera's un-arbitrated date).
    """
    # `new` is the enriched side: Google's arbitrated instant, plus the EXIF
    # instant enrichment had already displaced. `old` is a re-scan of the
    # UNENRICHED side whose own EXIF instant has since moved (a camera clock
    # correction, or a different bytes-copy) - the exact disagreement
    # `_exif_instant(new) != old.taken_at_utc` exists to catch.
    displaced_exif = datetime(2016, 6, 24, 9, 47, 33, tzinfo=UTC)
    moved_exif = datetime(2016, 6, 25, 9, 47, 33, tzinfo=UTC)
    google_instant = datetime(2020, 3, 14, 14, 17, 31, tzinfo=UTC)
    old = PhotoMeta(taken_at_utc=moved_exif, tz_source=TzSource.EXIF_OFFSET)
    new = PhotoMeta(
        taken_at_utc=google_instant,
        tz_source=TzSource.TAKEOUT,
        exif_taken_at_utc=displaced_exif,
    )

    merged, conflict = merge_meta(old, new)

    assert merged.taken_at_utc == google_instant
    assert merged.tz_source is TzSource.TAKEOUT
    # The enriched side's displaced EXIF instant disagrees with the OTHER
    # side's raw date - a real conflict, not the crude exact-equality check
    # `_exif_instant`'s docstring warns against.
    assert conflict is True


def test_merge_meta_real_date_beats_file_mtime_fallback_even_if_earlier():
    # The folder source always populates taken_at_utc via mtime fallback.
    # A naive "whichever is set" or "earliest wins regardless of source"
    # rule would let that fallback overwrite a genuine EXIF capture date.
    real_date = datetime(2021, 6, 1, tzinfo=UTC)
    mtime_fallback = datetime(2019, 1, 1, tzinfo=UTC)  # earlier, but fake

    old = PhotoMeta(taken_at_utc=real_date, tz_source=TzSource.EXIF_OFFSET)
    new = PhotoMeta(taken_at_utc=mtime_fallback, tz_source=TzSource.FILE_MTIME)

    merged, conflict = merge_meta(old, new)

    assert merged.taken_at_utc == real_date
    assert merged.tz_source is TzSource.EXIF_OFFSET
    assert conflict is False

    # And symmetrically, when the real date is on the "new" side.
    old2 = PhotoMeta(taken_at_utc=mtime_fallback, tz_source=TzSource.FILE_MTIME)
    new2 = PhotoMeta(taken_at_utc=real_date, tz_source=TzSource.EXIF_OFFSET)

    merged2, conflict2 = merge_meta(old2, new2)

    assert merged2.taken_at_utc == real_date
    assert merged2.tz_source is TzSource.EXIF_OFFSET
    assert conflict2 is False


def test_merge_meta_unions_people_face_regions_and_keywords_preserving_order():
    region_a = FaceRegion(name="Alice", x=0.1, y=0.1, w=0.1, h=0.1)
    region_b = FaceRegion(name="Bob", x=0.2, y=0.2, w=0.1, h=0.1)

    old = PhotoMeta(
        people=["Alice", "Bob"],
        face_regions=[region_a],
        keywords=["trip", "beach"],
    )
    new = PhotoMeta(
        people=["Bob", "Carol"],
        face_regions=[region_a, region_b],
        keywords=["beach", "sunset"],
    )

    merged, _ = merge_meta(old, new)

    assert merged.people == ["Alice", "Bob", "Carol"]
    assert merged.face_regions == [region_a, region_b]
    assert merged.keywords == ["trip", "beach", "sunset"]


def test_merge_meta_description_longest_non_empty_wins():
    old = PhotoMeta(description="short")
    new = PhotoMeta(description="a much longer description")

    merged, conflict = merge_meta(old, new)

    assert merged.description == "a much longer description"
    assert conflict is True


def test_merge_meta_description_none_when_both_empty():
    old = PhotoMeta(description=None)
    new = PhotoMeta(description=None)

    merged, conflict = merge_meta(old, new)

    assert merged.description is None
    assert conflict is False


def test_merge_meta_description_no_conflict_when_only_one_side_has_it():
    old = PhotoMeta(description=None)
    new = PhotoMeta(description="only side")

    merged, conflict = merge_meta(old, new)

    assert merged.description == "only side"
    assert conflict is False


def test_merge_meta_scalar_fields_fall_back_to_whichever_side_has_a_value():
    gps = Gps(lat=1.0, lon=2.0, alt=None)
    old = PhotoMeta(
        gps=None,
        camera_make=None,
        camera_model="OldModel",
        width=None,
        height=100,
    )
    new = PhotoMeta(
        gps=gps,
        camera_make="Acme",
        camera_model=None,
        width=200,
        height=None,
    )

    merged, _ = merge_meta(old, new)

    assert merged.gps == gps
    assert merged.camera_make == "Acme"
    assert merged.camera_model == "OldModel"
    assert merged.width == 200
    assert merged.height == 100


def test_merge_meta_favorite_is_logical_or():
    assert merge_meta(PhotoMeta(favorite=False), PhotoMeta(favorite=False))[0].favorite is False
    assert merge_meta(PhotoMeta(favorite=True), PhotoMeta(favorite=False))[0].favorite is True
    assert merge_meta(PhotoMeta(favorite=False), PhotoMeta(favorite=True))[0].favorite is True
    assert merge_meta(PhotoMeta(favorite=True), PhotoMeta(favorite=True))[0].favorite is True


def test_merge_meta_clean_merge_has_no_conflict():
    now = datetime(2022, 3, 3, tzinfo=UTC)
    old = PhotoMeta(taken_at_utc=now, tz_source=TzSource.EXIF_OFFSET, description="a photo")
    new = PhotoMeta(taken_at_utc=now, tz_source=TzSource.EXIF_OFFSET, description="a photo")

    merged, conflict = merge_meta(old, new)

    assert merged.taken_at_utc == now
    assert merged.description == "a photo"
    assert conflict is False
