import json
from datetime import UTC, datetime, timedelta

import pytest

from rekindle.db import PhotoStore
from rekindle.enrich.takeout import EmptyIndexError, TakeoutEnricher
from rekindle.models import MediaType, Photo, PhotoMeta, TzSource
from rekindle.sources.folder import FolderSource
from tests.fixtures.gen import make_jpeg
from tests.fixtures.takeout import build_takeout


def _indexed(tmp_path):
    root = build_takeout(tmp_path / "Takeout")
    photos, _ = FolderSource().scan(root)
    store = PhotoStore(tmp_path / "data" / "rekindle.sqlite")
    store.upsert_many(photos)
    store.set_meta("index_root", str(root))
    return root, store


def test_enrich_populates_people_and_dates(tmp_path):
    root, store = _indexed(tmp_path)
    report = TakeoutEnricher().enrich(root, store)
    assert report.photos_enriched >= 5
    assert report.people_added >= 3
    by_name = {p.paths[0].name: p for p in store.iter_photos()}
    assert by_name["DSC00107.JPG"].meta.people == ["Grace"]
    assert by_name["DSC00107(1).JPG"].meta.people == ["Ada"]
    assert by_name["IMG_ALBUM.jpg"].meta.people == ["Ada"]
    store.close()


def test_the_accounting_identities_hold_end_to_end(tmp_path):
    root, store = _indexed(tmp_path)
    report = TakeoutEnricher().enrich(root, store)
    assert report.json_files_seen == report.files_accounted
    assert report.sidecars_seen == report.sidecars_accounted
    store.close()


def test_matched_equals_photos_enriched(tmp_path):
    """THE assertion that can fail.

    Both identities above partition sets that `account()` builds itself, so
    neither can see a sidecar parsed and then dropped somewhere downstream.
    This one ties the accounting to what actually reached the database: every
    sidecar counted as `matched` was written onto exactly one photo. Credit
    `matched` with `len(sidecars)` instead of the applied count and this fails
    immediately - by 3,358 on the reference export.
    """
    root, store = _indexed(tmp_path)
    report = TakeoutEnricher().enrich(root, store)
    assert report.matched == report.photos_enriched
    assert report.superseded >= 1  # AMBIG's second candidate
    store.close()


def test_a_distant_disagreeing_candidate_is_reported_as_an_override(tmp_path):
    """Two fixture photos hit this, both documented in
    `tests/fixtures/takeout.py`: AMBIG.jpg (a same-directory sidecar overrides
    a disagreeing one in Goa Trip), and the `Photos from 2011` copy of
    SHARED.jpg (same-directory sidecar overrides Goa Trip's disagreeing one;
    the fixture's OTHER copy, in Kolkata Trip, has no same-directory candidate
    at all and is refused as `ambiguous` instead - see
    `test_matched_equals_photos_enriched`). Traced by hand against `resolve()`:
    both are genuine directory-preference overrides, not a double-count.

    Rule 1 wins by design - but silently, doctor would print `ambiguous: 0`
    and imply nothing was overridden. 942 photos on the reference export.
    """
    root, store = _indexed(tmp_path)
    report = TakeoutEnricher().enrich(root, store)
    assert report.directory_preference_broke_a_tie == 2
    store.close()


def test_derivatives_are_enriched_and_counted(tmp_path):
    root, store = _indexed(tmp_path)
    report = TakeoutEnricher().enrich(root, store)
    assert report.derivatives_enriched >= 2
    by_name = {p.paths[0].name: p for p in store.iter_photos()}
    assert by_name["IMG_EDIT-edited.jpg"].meta.people == ["Grace"]
    assert by_name["IMG_EDIT-edited.jpg"].sidecar_match == "inherited"
    assert by_name["PXL_1.MP"].meta.taken_at_utc is not None
    store.close()


def test_enrich_is_idempotent(tmp_path):
    root, store = _indexed(tmp_path)
    first = TakeoutEnricher().enrich(root, store)
    snapshot = {
        p.file_hash: (
            p.meta.taken_at_utc,
            p.meta.taken_at_local,
            p.meta.tz_source,
            tuple(p.meta.people),
            tuple(p.meta.takeout_people),
            p.meta.exif_taken_at_utc,
            p.metadata_conflict,
            p.sidecar_match,
            tuple(p.albums),
        )
        for p in store.iter_photos()
    }
    second = TakeoutEnricher().enrich(root, store)
    assert {
        p.file_hash: (
            p.meta.taken_at_utc,
            p.meta.taken_at_local,
            p.meta.tz_source,
            tuple(p.meta.people),
            tuple(p.meta.takeout_people),
            p.meta.exif_taken_at_utc,
            p.metadata_conflict,
            p.sidecar_match,
            tuple(p.albums),
        )
        for p in store.iter_photos()
    } == snapshot
    assert second.matched == first.matched
    assert second.orphaned == first.orphaned
    store.close()


def test_re_enrich_of_an_exif_naive_photo_stays_idempotent(tmp_path):
    """Task 8's `exif_taken_at_utc` mechanism, exercised through an actual
    two-run `TakeoutEnricher().enrich()` rather than a bare `apply_sidecar()`
    call. `build_takeout`'s own fixture photos carry no EXIF taken time at
    all (tz_source starts at FILE_MTIME), so none of them touch the
    EXIF_NAIVE -> TAKEOUT reconstruction path; this test builds its own
    minimal root specifically to reach it at the integration level.

    Ladder rung 3 (`meta.timestamps.from_takeout`) recovers a real +05:30
    offset from the gap between the naive EXIF wall clock and Google's UTC
    instant. That recovery only survives a SECOND enrich if the pristine
    EXIF_NAIVE state is reconstructed from `exif_taken_at_utc` first - reading
    back the already-Takeout-derived `tz_source` instead silently regresses
    local time to UTC.
    """
    root = tmp_path / "Takeout"
    year = root / "Photos from 2019"
    naive_taken = datetime(2019, 5, 1, 15, 30)
    make_jpeg(year / "IST.jpg", taken=naive_taken)
    # 5:30 earlier than the naive wall clock - the IST offset, recoverable
    # only via rung 3.
    google_instant = datetime(2019, 5, 1, 10, 0, tzinfo=UTC)
    (year / "IST.jpg.supplemental-metadata.json").write_text(
        json.dumps(
            {
                "title": "IST.jpg",
                "photoTakenTime": {"timestamp": str(int(google_instant.timestamp()))},
            }
        ),
        encoding="utf-8",
    )

    photos, _ = FolderSource().scan(root)
    store = PhotoStore(tmp_path / "data" / "rekindle.sqlite")
    store.upsert_many(photos)
    digest = photos[0].file_hash

    TakeoutEnricher().enrich(root, store)
    after_first = store.get(digest)
    assert after_first.meta.tz_source == TzSource.TAKEOUT
    assert after_first.meta.taken_at_local.utcoffset() == timedelta(hours=5, minutes=30)
    assert after_first.metadata_conflict is False

    second = TakeoutEnricher().enrich(root, store)
    after_second = store.get(digest)

    # Aware-datetime `==` compares INSTANTS, not offsets - it would call
    # 10:00 UTC and 15:30+05:30 equal even though the second is UTC-flattened
    # local time, which is exactly what a lost reconstruction produces. The
    # offset itself is the assertion that cannot be fooled that way.
    assert after_second.meta.taken_at_local.utcoffset() == timedelta(hours=5, minutes=30)
    assert after_second.meta.taken_at_utc == after_first.meta.taken_at_utc
    assert after_second.meta.taken_at_local == after_first.meta.taken_at_local
    assert after_second.meta.tz_source == after_first.meta.tz_source
    assert after_second.meta.exif_taken_at_utc == after_first.meta.exif_taken_at_utc
    assert after_second.metadata_conflict is False
    assert second.dates_corrected == 0
    store.close()


def test_enrich_against_an_empty_index_refuses(tmp_path):
    root = build_takeout(tmp_path / "Takeout")
    store = PhotoStore(tmp_path / "data" / "rekindle.sqlite")
    with pytest.raises(EmptyIndexError):
        TakeoutEnricher().enrich(root, store)
    store.close()


def test_enrich_writes_photos_in_a_single_batch(tmp_path, monkeypatch):
    """The spec's 'one transaction per run': `update_many` batches every
    touched photo into one commit. A per-photo `update_photo` loop is orders
    of magnitude slower against 19,480 rows and is the exact anti-pattern
    `PhotoStore.update_photo`'s docstring warns is NOT `upsert_many`'s
    merge-based batch path. Pinned by call-counting, not timing: correctness
    of the final rows is identical either way, so nothing else here would
    ever catch the regression.
    """
    root, store = _indexed(tmp_path)
    calls = {"update_many": 0, "update_photo": 0}
    real_update_many = PhotoStore.update_many

    def counting_update_many(self, photos):
        calls["update_many"] += 1
        return real_update_many(self, photos)

    def counting_update_photo(self, photo):
        calls["update_photo"] += 1
        return None

    monkeypatch.setattr(PhotoStore, "update_many", counting_update_many)
    monkeypatch.setattr(PhotoStore, "update_photo", counting_update_photo)

    TakeoutEnricher().enrich(root, store)

    assert calls["update_many"] == 1
    assert calls["update_photo"] == 0
    store.close()


def test_enrich_records_when_and_where_it_ran(tmp_path):
    root, store = _indexed(tmp_path)
    TakeoutEnricher().enrich(root, store)
    assert store.get_meta("enrich_root") == str(root)
    assert store.get_meta("enriched_at")
    store.close()


def test_trash_sidecars_are_excluded_by_both_passes(tmp_path):
    """Delete the EXCLUDED_DIRS check in build_index and this fails.

    The first draft asserted `IMG_GONE.jpg` was absent from the store, which
    FolderSource guarantees on its own - that test passed with enrich()'s body
    deleted and said nothing about the enricher. This one pins the enricher's
    own exclusion, and that the excluded sidecar is not miscounted as an
    orphan (which would inflate doctor's INCOMPLETE EXPORT warning on a
    complete library).
    """
    root, store = _indexed(tmp_path)
    report = TakeoutEnricher().enrich(root, store)
    assert report.excluded_dirs == 1
    assert "img_gone.jpg" not in {p.paths[0].name.casefold() for p in store.iter_photos()}
    store.close()


def test_google_date_replaces_the_mtime_fallback(tmp_path):
    root, store = _indexed(tmp_path)
    TakeoutEnricher().enrich(root, store)
    by_name = {p.paths[0].name: p for p in store.iter_photos()}
    flags = by_name["IMG_FLAGS.jpg"]
    assert flags.meta.tz_source is not TzSource.FILE_MTIME
    assert flags.meta.favorite is True
    assert flags.meta.archived is True
    assert flags.meta.description == "a real caption"
    store.close()


# --- the paths[0] hazard -----------------------------------------------
#
# A previous review flagged that an earlier draft's `claimed`/`applied`
# bookkeeping keyed off `photo.paths[0].name.casefold()` alone. If a Photo
# ever carries paths with DIFFERENT casefolded filenames (the store's own
# dedup allows this: `FolderSource`/`PhotoStore._merge` union paths by
# content hash, with no requirement that the names match), a sidecar found
# via a non-first path would be attributed to the wrong (or no) target key
# in `claimed` and its whole group would fall through to `orphaned` in
# `account()`, even though it WAS applied. Not observed in real Takeout
# exports - album copies keep the filename - but nothing in the data model
# forbids it, so it is pinned here directly rather than left as an
# unverified assumption.
def test_a_sidecar_found_via_a_non_first_path_still_counts_as_matched(tmp_path):
    root = tmp_path / "Takeout"
    album = root / "Album"
    album.mkdir(parents=True)
    (album / "TWIN_NEW.jpg.supplemental-metadata.json").write_text(
        '{"title": "TWIN_NEW.jpg", "photoTakenTime": {"timestamp": "1400000000"}, '
        '"people": [{"name": "Zoe"}]}',
        encoding="utf-8",
    )

    store = PhotoStore(tmp_path / "data" / "rekindle.sqlite")
    now = datetime(2020, 1, 1, tzinfo=UTC)
    # paths[0] casefolds to "twin_old.jpg", which has NO sidecar at all - only
    # paths[1] ("twin_new.jpg") does. A `paths[0]`-only implementation would
    # never mark this target claimed, so its (successfully applied) sidecar
    # would be counted as `orphaned` rather than `matched`.
    twin = Photo(
        file_hash="twin-hash",
        paths=[album / "TWIN_OLD.jpg", album / "TWIN_NEW.jpg"],
        media_type=MediaType.IMAGE,
        meta=PhotoMeta(),
        first_seen=now,
        last_seen=now,
    )
    store.upsert_many([twin])

    report = TakeoutEnricher().enrich(root, store)

    assert report.matched == 1
    assert report.orphaned == 0
    assert report.photos_enriched == 1
    stored = store.get("twin-hash")
    assert stored.meta.people == ["Zoe"]
    assert stored.sidecar_match == "exact"
    store.close()
