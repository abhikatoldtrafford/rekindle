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

    # Broader than `matched`/`orphaned` alone: every count re-derived PURELY
    # from the filesystem and resolve()'s outcome - never from what happens
    # to already be stored - must be byte-identical between runs. These are
    # the accounting numbers `doctor` reports; if `enrich()` were secretly
    # order- or state-dependent, this would catch it where the original
    # two-field check would not.
    #
    # Comparing the WHOLE `EnrichReport` (`second == first`) would be
    # wrong, not stronger: `derivatives_enriched`, `people_added`,
    # `dates_corrected`, `gps_added`, `descriptions_added`,
    # `favourites_added`, `albums_retitled` and `conflicts` all count a
    # CHANGE this specific run made relative to what was already stored -
    # apply_sidecar's and propagate_to_derivatives's own idempotency
    # guarantees mean a correct second run makes NONE of these changes
    # again, so they are asserted to be exactly 0 below instead. Verified
    # empirically before writing this: on the real fixture, a correct
    # implementation's second run reads
    # derivatives_enriched=0/people_added=0/dates_corrected=0/gps_added=0/
    # descriptions_added=0/favourites_added=0 where the first run read
    # 2/7/10/1/1/1 - `first == second` would fail on CORRECT code.
    for field_name in (
        "json_files_seen",
        "sidecars_seen",
        "album_metadata",
        "other_json",
        "excluded_dirs",
        "unparseable",
        "matched",
        "orphaned",
        "ambiguous",
        "superseded",
        "photos_enriched",
        "directory_preference_broke_a_tie",
        "cross_photo_collisions",
        "title_disagreements",
        "album_title_collisions",
        "clustered_dates_suppressed",
    ):
        assert getattr(second, field_name) == getattr(first, field_name), field_name

    for field_name in (
        "derivatives_enriched",
        "people_added",
        "dates_corrected",
        "gps_added",
        "descriptions_added",
        "favourites_added",
        "albums_retitled",
        "conflicts",
    ):
        assert getattr(second, field_name) == 0, field_name
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


def test_a_sidecar_is_never_applied_to_two_distinct_photos(tmp_path):
    """Found by running the enricher against the real export (task 11's own
    instruction), not by inspection: `matched` came out 3 lower than
    `photos_enriched` (18,306 vs 18,309), for exactly this reason. 342
    filenames are shared by distinct photos even after content-hash dedup
    (`tests/fixtures/takeout.py`); when only ONE of them has a sidecar of
    its own, `resolve()` cannot see the OTHER, unrelated photo of the same
    name - its single-global-candidate fallback (rule 2) has nothing to
    disagree with, so it happily hands the SAME sidecar to both. That is a
    real misattribution (someone else's face tag and capture time written
    onto this photo), not just a bookkeeping gap - `resolve()` only ever
    sees one photo at a time, so only the enricher, which sees all of them,
    can catch it.

    A same-directory match is `resolve()`'s reliable branch (rule 1); the
    photo actually living beside the sidecar must win, and the coincidental
    namesake must be refused rather than silently tagged too.
    """
    root = tmp_path / "Takeout"
    (root / "A").mkdir(parents=True)
    (root / "B").mkdir(parents=True)
    # These exact sizes are load-bearing, not arbitrary: B's content hash
    # sorts LOWER than A's (verified directly against `identity.file_hash`).
    # A must still win, because it lives beside the sidecar - if the
    # same-directory preference were ever dropped in favour of a bare
    # lowest-file_hash tiebreak, B would wrongly win and this test would
    # catch it. Picking sizes at random would make that mutant pass by
    # accident roughly half the time.
    make_jpeg(root / "A" / "Photo0288.jpg", size=(38, 38))
    make_jpeg(root / "B" / "Photo0288.jpg", size=(40, 40))
    (root / "A" / "Photo0288.jpg.supplemental-metadata.json").write_text(
        json.dumps(
            {
                "title": "Photo0288.jpg",
                "photoTakenTime": {"timestamp": "1400000000"},
                "people": [{"name": "Priya"}],
            }
        ),
        encoding="utf-8",
    )

    photos, _ = FolderSource().scan(root)
    store = PhotoStore(tmp_path / "data" / "rekindle.sqlite")
    store.upsert_many(photos)

    report = TakeoutEnricher().enrich(root, store)

    assert report.matched == report.photos_enriched == 1
    assert report.cross_photo_collisions == 1
    by_dir = {p.paths[0].parent.name: p for p in store.iter_photos()}
    assert by_dir["A"].meta.people == ["Priya"]
    assert by_dir["A"].sidecar_match == "exact"
    # The namesake in B is a DIFFERENT photo (different bytes). It must not
    # receive A's face tag or capture time, and must not be reported as
    # enriched.
    assert by_dir["B"].meta.people == []
    # FolderSource always sets SOME taken_at_utc via the mtime fallback; the
    # thing that must NOT happen is Takeout's date landing on it.
    assert by_dir["B"].meta.tz_source == TzSource.FILE_MTIME
    assert by_dir["B"].sidecar_match == "ambiguous"
    store.close()


def test_reliable_requires_the_matching_name_not_just_the_directory(tmp_path):
    """A multi-path Photo can have ONE path sharing the sidecar's directory
    (under a DIFFERENT name) and ANOTHER path sharing the sidecar's name (in
    a DIFFERENT directory) - satisfying a directory-only check without
    either individual path actually being the file the sidecar describes.
    Checking directory alone would let this multi-path photo "win" over the
    genuine, single-path match living directly beside the sidecar - a worse
    outcome than not resolving the collision at all. Not hypothetical: 5
    real photos in the export have paths with differing casefolded
    filenames (e.g. `Dida\\IMG_20170927_184011(1).jpg` +
    `Dida(1)\\IMG_20170927_184011.jpg`).
    """
    root = tmp_path / "Takeout"
    d = root / "d"
    e = root / "e"
    d.mkdir(parents=True)
    e.mkdir(parents=True)
    (d / "Y.jpg.supplemental-metadata.json").write_text(
        json.dumps(
            {
                "title": "Y.jpg",
                "photoTakenTime": {"timestamp": "1400000000"},
                "people": [{"name": "Priya"}],
            }
        ),
        encoding="utf-8",
    )

    now = datetime(2020, 1, 1, tzinfo=UTC)
    # A multi-path photo with ONE path beside the sidecar (d/X.jpg - wrong
    # name) and ONE path sharing the sidecar's name (e/Y.jpg - wrong
    # directory). Neither path alone qualifies; a directory-only check would
    # wrongly let it qualify via the two paths together - and, under a
    # directory-only check, BOTH photos below would qualify as "reliable",
    # falling through to the lowest-file_hash tiebreak. The literal hashes
    # are chosen so that tiebreak picks the WRONG one ("aaa..." < "zzz..."),
    # so a regression to directory-only checking is caught here rather than
    # masked by an accidental hash ordering (the same trap the first draft
    # of `test_a_sidecar_is_never_applied_to_two_distinct_photos` fell into).
    wrong = Photo(
        file_hash="aaa-multipath-decoy",
        paths=[d / "X.jpg", e / "Y.jpg"],
        media_type=MediaType.IMAGE,
        meta=PhotoMeta(),
        first_seen=now,
        last_seen=now,
    )
    # The genuine match: one path, both the right directory AND the right
    # name.
    real = Photo(
        file_hash="zzz-genuine-match",
        paths=[d / "Y.jpg"],
        media_type=MediaType.IMAGE,
        meta=PhotoMeta(),
        first_seen=now,
        last_seen=now,
    )
    store = PhotoStore(tmp_path / "data" / "rekindle.sqlite")
    store.upsert_many([wrong, real])

    report = TakeoutEnricher().enrich(root, store)

    real_after = store.get("zzz-genuine-match")
    wrong_after = store.get("aaa-multipath-decoy")
    assert real_after.meta.people == ["Priya"]
    assert real_after.sidecar_match == "exact"
    assert wrong_after.meta.people == []
    assert wrong_after.sidecar_match == "ambiguous"
    assert report.matched == 1
    assert report.cross_photo_collisions == 1
    store.close()


def test_no_reliable_claimant_refuses_everyone_rather_than_guessing(tmp_path):
    """When NEITHER content-distinct namesake lives beside the sidecar,
    there is zero evidence favouring one over the other. Picking a
    "winner" anyway - even a deterministic one, like the lowest file_hash -
    is fabrication with extra steps: it writes Google's face tag and
    capture time onto a coin flip and calls it a match. `tests/fixtures/
    takeout.py` documents 8 real album folders that hold a sidecar and NO
    media at all (1,204 real photos are in that position); this reproduces
    exactly that shape, with two content-distinct namesakes elsewhere so
    that neither can claim the sidecar's own directory.
    """
    root = tmp_path / "Takeout"
    (root / "Album").mkdir(parents=True)
    (root / "C").mkdir(parents=True)
    (root / "D").mkdir(parents=True)
    (root / "Album" / "Z.jpg.supplemental-metadata.json").write_text(
        json.dumps(
            {
                "title": "Z.jpg",
                "photoTakenTime": {"timestamp": "1400000000"},
                "people": [{"name": "Priya"}],
            }
        ),
        encoding="utf-8",
    )
    make_jpeg(root / "C" / "Z.jpg", size=(44, 44))
    make_jpeg(root / "D" / "Z.jpg", size=(46, 46))

    photos, _ = FolderSource().scan(root)
    store = PhotoStore(tmp_path / "data" / "rekindle.sqlite")
    store.upsert_many(photos)

    report = TakeoutEnricher().enrich(root, store)

    assert report.matched == 0
    assert report.photos_enriched == 0
    assert report.ambiguous == 1
    assert report.cross_photo_collisions == 2
    for photo in store.iter_photos():
        assert photo.meta.people == []
        assert photo.sidecar_match == "ambiguous"
    store.close()


def test_enrich_against_an_empty_index_refuses(tmp_path):
    root = build_takeout(tmp_path / "Takeout")
    store = PhotoStore(tmp_path / "data" / "rekindle.sqlite")
    with pytest.raises(EmptyIndexError):
        TakeoutEnricher().enrich(root, store)
    store.close()


class _CommitCounter:
    """Forwards everything to a real `sqlite3.Connection` except `commit()`,
    which it counts. `sqlite3.Connection.commit` is a read-only attribute of
    a C-extension type - `monkeypatch.setattr(conn, "commit", ...)` raises
    `AttributeError: ... attribute 'commit' is read-only` - so a thin proxy
    swapped in for `store._conn` is the only way to observe real commits
    without changing `PhotoStore` itself.
    """

    def __init__(self, conn):
        self._real = conn
        self.commits = 0

    def commit(self):
        self.commits += 1
        return self._real.commit()

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_enrich_commits_exactly_once(tmp_path):
    """The spec's 'one transaction per run', measured the only way that
    actually proves it: counting real commits on the underlying connection,
    not calls to whichever store method happens to wrap them. A prior
    version of this test counted `update_many`/`update_photo` calls instead
    - it could not see that `enrich()` still called `set_meta` twice more
    afterward, each with its OWN commit, leaving a crash window between the
    photo batch and `enriched_at` where guard 3 (distinguish "ran and found
    nothing" from "never ran") could not tell the two apart. Photos and
    provenance must land in the SAME transaction.
    """
    root, store = _indexed(tmp_path)
    counter = _CommitCounter(store._conn)
    store._conn = counter

    TakeoutEnricher().enrich(root, store)

    assert counter.commits == 1
    store.close()


def test_enrich_records_when_and_where_it_ran(tmp_path):
    root, store = _indexed(tmp_path)
    TakeoutEnricher().enrich(root, store)
    assert store.get_meta("enrich_root") == str(root)
    raw = store.get_meta("enriched_at")
    assert raw
    # `assert raw` alone passes for any non-empty string, including a typo
    # like "yes". Guard 3 needs a REAL, UTC timestamp here - `doctor` (Task
    # 12) will compare it against wall-clock time to flag a stale enrich.
    parsed = datetime.fromisoformat(raw)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(0)
    store.close()


def test_enriched_at_is_recorded_even_when_nothing_is_found(tmp_path):
    """Half of guard 3's purpose: `sidecar_match == "none"` must be readable
    as "enrich ran and found nothing" rather than "enrich never ran" - which
    only works if `enriched_at` is written on a run that matches zero
    sidecars too. `_indexed()` always has matches (the shared fixture is
    full of them), so it can never exercise this path; this test indexes a
    photo with no Takeout JSON anywhere near it.
    """
    root = tmp_path / "Takeout"
    make_jpeg(root / "Photos from 2020" / "LONELY.jpg", size=(50, 50))
    photos, _ = FolderSource().scan(root)
    store = PhotoStore(tmp_path / "data" / "rekindle.sqlite")
    store.upsert_many(photos)

    report = TakeoutEnricher().enrich(root, store)

    assert report.matched == 0
    assert report.photos_enriched == 0
    only = next(iter(store.iter_photos()))
    assert only.sidecar_match == "none"
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
# `account()`, even though it WAS applied. This is not hypothetical: 5
# photos in the reference export have paths with differing casefolded
# filenames (e.g. `Dida\IMG_20170927_184011(1).jpg` +
# `Dida(1)\IMG_20170927_184011.jpg`) - album copies usually keep the
# filename, but not always - so this is pinned here directly rather than
# left as an unverified assumption.
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


def _reindex(root, store):
    """A second `rekindle index` over the same folder, exactly as the CLI
    does it: scan, then `upsert_many` (which merges via `models.merge_meta`)."""
    photos, _ = FolderSource().scan(root)
    store.upsert_many(photos)


def _dated_export(tmp_path, *, exif, google, name="CLOCK.jpg"):
    """A photo whose camera clock ran EARLY, with Google's correction."""
    root = tmp_path / "Takeout"
    year = root / "Photos from 2016"
    make_jpeg(year / name, taken=exif)
    (year / f"{name}.supplemental-metadata.json").write_text(
        json.dumps({"title": name, "photoTakenTime": {"timestamp": str(int(google.timestamp()))}}),
        encoding="utf-8",
    )
    return root


def test_a_re_index_does_not_revert_a_takeout_date_correction(tmp_path):
    """`index -> enrich -> index` must leave the enriched date in place.

    `upsert_many` merges through `models.merge_meta`, which encodes "earliest
    real date wins" - a tiebreak between sources of EQUAL authority. Takeout
    is not equal: it is Google's own record. Without the `_enriched` branch,
    a re-index pits Google's corrected 2020 date against the freshly re-read
    2016 EXIF date and the BROKEN CAMERA CLOCK wins.

    Measured on the reference export before this fix: 108 photos have
    `taken_at_utc > exif_taken_at_utc` and every one of them reverted on the
    next `rekindle index` (100 by more than a day). It self-heals on the next
    `enrich`, but `doctor --from-index` reports from exactly that window, and
    doctor's own orphan warning tells users to "Extract every part into the
    SAME folder and re-run" - which is this flow.

    MUTATION (run, not assumed): delete the `_enriched(old) and not
    _enriched(new)` branch from `models.merge_meta` and this test fails on
    `after_reindex.meta.taken_at_utc == google` - it comes back as the 2016
    EXIF instant with tz_source `exif_naive`.
    """
    exif_instant = datetime(2016, 3, 1, 12, 0)
    google = datetime(2020, 7, 4, 9, 30, tzinfo=UTC)
    root = _dated_export(tmp_path, exif=exif_instant, google=google)

    photos, _ = FolderSource().scan(root)
    store = PhotoStore(tmp_path / "data" / "rekindle.sqlite")
    store.upsert_many(photos)
    digest = photos[0].file_hash
    assert store.get(digest).meta.taken_at_utc == exif_instant.replace(tzinfo=UTC)

    TakeoutEnricher().enrich(root, store)
    after_enrich = store.get(digest)
    assert after_enrich.meta.taken_at_utc == google
    assert after_enrich.meta.tz_source is TzSource.TAKEOUT
    assert after_enrich.meta.exif_taken_at_utc == exif_instant.replace(tzinfo=UTC)
    assert after_enrich.metadata_conflict is True

    _reindex(root, store)
    after_reindex = store.get(digest)
    assert after_reindex.meta.taken_at_utc == google
    assert after_reindex.meta.tz_source is TzSource.TAKEOUT
    assert after_reindex.meta.taken_at_local == after_enrich.meta.taken_at_local
    # The displaced EXIF instant survives too - it is the flag's only payload.
    assert after_reindex.meta.exif_taken_at_utc == exif_instant.replace(tzinfo=UTC)
    # A real disagreement between Google and the re-read EXIF is still a real
    # disagreement: outranking the date must not retract the flag.
    assert after_reindex.metadata_conflict is True
    store.close()


def test_a_re_index_does_not_invent_a_conflict_out_of_a_timezone(tmp_path):
    """The other half of the `_enriched` branch.

    Once enrichment has run, the re-read EXIF instant is the one it ALREADY
    arbitrated (with offset normalisation and a one-minute tolerance) and
    recorded in `exif_taken_at_utc`. Asking the crude question here instead -
    `old.taken_at_utc != new.taken_at_utc`, exact, on a Google instant versus
    a naive wall clock - flags the photo's UTC offset as a disagreement.
    Measured on the reference export: a re-index raised `metadata_conflict`
    on 10,065 rows that `enrich` had deliberately left clear, all of them the
    +05:30 artefact, which `doctor --from-index` would print as "EXIF/Google
    date conflicts" until the next enrich recomputed them away.

    MUTATION (run, not assumed): change the enriched branch's `conflict` back
    to `old.taken_at_utc != new.taken_at_utc` and this test fails on
    `after_reindex.metadata_conflict is False`.
    """
    naive = datetime(2019, 5, 1, 15, 30)
    google = datetime(2019, 5, 1, 10, 0, tzinfo=UTC)  # 5:30 earlier: IST
    root = _dated_export(tmp_path, exif=naive, google=google, name="IST.jpg")

    photos, _ = FolderSource().scan(root)
    store = PhotoStore(tmp_path / "data" / "rekindle.sqlite")
    store.upsert_many(photos)
    digest = photos[0].file_hash

    TakeoutEnricher().enrich(root, store)
    assert store.get(digest).metadata_conflict is False

    _reindex(root, store)
    after_reindex = store.get(digest)
    assert after_reindex.metadata_conflict is False
    assert after_reindex.meta.taken_at_utc == google
    assert after_reindex.meta.taken_at_local.utcoffset() == timedelta(hours=5, minutes=30)
    store.close()


def test_a_resolved_conflict_is_retracted_and_a_live_one_is_not(tmp_path):
    """`metadata_conflict` was `photo.metadata_conflict or conflict`.

    `or` is monotonic, so a date the user later corrects in Google Photos
    stays flagged forever - 119 rows carry the flag on the reference export
    and none of them could ever lose it. This is NOT the same case as
    `favorite`/`archived`/`trashed`, whose deferral rested on "Takeout cannot
    encode an un-favourite, so there is no retraction event": a corrected
    date is an obvious, realistic retraction event.

    MUTATION (run, not assumed): restore `photo.metadata_conflict =
    photo.metadata_conflict or conflict` and this test fails on
    `fixed.metadata_conflict is False`.
    """
    root = tmp_path / "Takeout"
    year = root / "Photos from 2016"
    exif_instant = datetime(2016, 3, 1, 12, 0)
    wrong = datetime(2020, 7, 4, 9, 30, tzinfo=UTC)
    for name, size in (("FIXED.jpg", (24, 24)), ("STILLWRONG.jpg", (26, 26))):
        make_jpeg(year / name, taken=exif_instant, size=size)
        (year / f"{name}.supplemental-metadata.json").write_text(
            json.dumps(
                {"title": name, "photoTakenTime": {"timestamp": str(int(wrong.timestamp()))}}
            ),
            encoding="utf-8",
        )

    photos, _ = FolderSource().scan(root)
    store = PhotoStore(tmp_path / "data" / "rekindle.sqlite")
    store.upsert_many(photos)
    by_name = {p.paths[0].name: p.file_hash for p in photos}

    first = TakeoutEnricher().enrich(root, store)
    assert first.conflicts == 2
    assert first.conflicts_retracted == 0
    assert store.get(by_name["FIXED.jpg"]).metadata_conflict is True
    assert store.get(by_name["STILLWRONG.jpg"]).metadata_conflict is True

    # The user corrects FIXED.jpg's date in Google Photos and re-exports.
    (year / "FIXED.jpg.supplemental-metadata.json").write_text(
        json.dumps(
            {
                "title": "FIXED.jpg",
                "photoTakenTime": {
                    "timestamp": str(int(exif_instant.replace(tzinfo=UTC).timestamp()))
                },
            }
        ),
        encoding="utf-8",
    )
    second = TakeoutEnricher().enrich(root, store)

    fixed = store.get(by_name["FIXED.jpg"])
    assert fixed.metadata_conflict is False
    assert fixed.meta.taken_at_utc == exif_instant.replace(tzinfo=UTC)
    # A retraction is a real change to the record; it gets a counter of its
    # own rather than happening silently.
    assert second.conflicts_retracted == 1
    # The photo Google never corrected keeps its flag.
    assert store.get(by_name["STILLWRONG.jpg"]).metadata_conflict is True
    store.close()


def test_every_reported_total_equals_the_rows_actually_changed(tmp_path):
    """The printed numbers must reconcile with what was written.

    `apply_sidecar` credits `gps_added`/`people_added`/`dates_corrected`/
    `descriptions_added`/`favourites_added`; `propagate_to_derivatives` wrote
    the same fields and credited NONE of them, so only `derivatives_enriched`
    (a per-photo count) moved. Measured by instrumenting a real run: the
    report printed "GPS added: 38" against 337 rows actually written (299 via
    derivatives), "People added: 18,562" against 19,730 person tags, and
    "Dates corrected: 12,212" against 13,010. The GPS row was off by 8.9x,
    and a user can catch it with the product's own two commands -
    `doctor --from-index` reads `with_gps` 1,993 before and 2,330 after.

    `build_takeout` has three derivatives (IMG_EDIT-edited.jpg, PXL_1.MP and
    IMG_CAPTION-edited.jpg); PXL_1.MP.jpg's sidecar carries real coordinates,
    so the GPS row here is exactly the one that was wrong on the real
    export, and IMG_CAPTION.jpg's sidecar carries a description and
    `favorited: true` so its derivative's inherited copies make
    `descriptions_added`/`favourites_added` non-vacuous too - both totals
    read 142 and 3 before and after the wave with no derivative ever
    contributing to either.

    MUTATION (run, not assumed): drop `report.gps_added += 1` from
    `propagate_to_derivatives` and this fails with `gps_added 1 != 2`; drop
    `report.people_added += ...` there and it fails on the people total; drop
    `report.descriptions_added += 1` there and it fails with
    `descriptions_added 1 != 2`; drop `report.favourites_added += 1` there
    and it fails with `favourites_added 1 != 2`.
    """
    root, store = _indexed(tmp_path)

    def snapshot():
        return {
            p.file_hash: (
                p.meta.taken_at_utc,
                frozenset(p.meta.people),
                p.meta.gps,
                p.meta.description,
                p.meta.favorite,
            )
            for p in store.iter_photos()
        }

    before = snapshot()
    report = TakeoutEnricher().enrich(root, store)
    after = snapshot()

    assert report.derivatives_enriched >= 1, "fixture must exercise the derivative path"
    written_dates = sum(1 for h, a in after.items() if a[0] != before[h][0])
    written_people = sum(len(a[1] - before[h][1]) for h, a in after.items())
    written_gps = sum(1 for h, a in after.items() if a[2] is not None and before[h][2] is None)
    written_desc = sum(1 for h, a in after.items() if a[3] != before[h][3])
    written_favs = sum(1 for h, a in after.items() if a[4] and not before[h][4])

    assert report.dates_corrected == written_dates
    assert report.people_added == written_people
    assert report.gps_added == written_gps
    assert report.descriptions_added == written_desc
    assert report.favourites_added == written_favs
    # Not a vacuous pass: the derivative really did contribute to the GPS row,
    # and to the description/favourite rows too - the two other derivative
    # counters `propagate_to_derivatives` credits (Residual 2).
    assert written_gps == 2
    assert written_desc >= 1
    assert written_favs >= 1
    store.close()


def test_people_added_counts_a_retraction_and_an_addition_together(tmp_path):
    """`if len(people) > before` reported 0 new for a net-neutral change.

    Google drops "Ada" and adds "Grace" in the same re-export: the list is
    still one name long, the length guard sees no growth, and a real new face
    tag goes uncredited. "People added" is the headline number for this
    milestone's headline feature.

    MUTATION (run, not assumed): restore the `if len(meta.people) > before:`
    length guard in `apply_sidecar` and this fails with `people_added 0 != 1`.
    """
    root = tmp_path / "Takeout"
    year = root / "Photos from 2019"
    make_jpeg(year / "FACES.jpg")
    sidecar = year / "FACES.jpg.supplemental-metadata.json"

    def write(*names):
        sidecar.write_text(
            json.dumps({"title": "FACES.jpg", "people": [{"name": n} for n in names]}),
            encoding="utf-8",
        )

    write("Ada")
    photos, _ = FolderSource().scan(root)
    store = PhotoStore(tmp_path / "data" / "rekindle.sqlite")
    store.upsert_many(photos)
    digest = photos[0].file_hash
    assert TakeoutEnricher().enrich(root, store).people_added == 1

    write("Grace")
    second = TakeoutEnricher().enrich(root, store)
    assert store.get(digest).meta.people == ["Grace"]
    assert second.people_added == 1
    store.close()
