import json
from datetime import UTC, datetime
from pathlib import Path

from rekindle.db import PhotoStore
from rekindle.enrich.takeout import EnrichReport, TakeoutEnricher, build_index, resolve
from rekindle.models import MediaType, Photo, PhotoMeta
from rekindle.sources.folder import FolderSource
from tests.fixtures.takeout import build_takeout


def _write_sidecar_json(path: Path, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _photo(*paths: Path) -> Photo:
    now = datetime(2020, 1, 1, tzinfo=UTC)
    return Photo(
        file_hash="".join(p.name for p in paths),
        paths=list(paths),
        media_type=MediaType.IMAGE,
        meta=PhotoMeta(),
        first_seen=now,
        last_seen=now,
    )


def test_the_counter_variant_gets_its_own_sidecar(tmp_path):
    """v1's headline defect: both sidecars claimed DSC00107.JPG."""
    root = build_takeout(tmp_path / "Takeout")
    index = build_index(root, EnrichReport())
    year = root / "Photos from 2011"

    plain, match, _tie = resolve(_photo(year / "DSC00107.JPG"), index)
    assert match == "exact"
    assert plain.taken_at_utc == datetime.fromtimestamp(1323826707, UTC)
    assert plain.people == ("Grace",)

    counter, match, _tie = resolve(_photo(year / "DSC00107(1).JPG"), index)
    assert match == "exact"
    assert counter.taken_at_utc == datetime.fromtimestamp(1295183562, UTC)
    assert counter.people == ("Ada",)


def test_a_sidecar_in_a_media_less_album_still_matches(tmp_path):
    """Per-directory keying loses 1,204 real photos this way."""
    root = build_takeout(tmp_path / "Takeout")
    index = build_index(root, EnrichReport())
    found, match, _tie = resolve(_photo(root / "Photos from 2011" / "IMG_ALBUM.jpg"), index)
    assert match == "exact"
    assert found.people == ("Ada",)


def test_same_directory_wins_over_a_distant_candidate(tmp_path):
    root = build_takeout(tmp_path / "Takeout")
    index = build_index(root, EnrichReport())
    year = root / "Photos from 2011"
    found, match, tie = resolve(_photo(year / "AMBIG.jpg"), index)
    assert match == "exact"
    assert found.path.parent == year
    assert found.people == ("Ada",)
    # The Goa Trip candidate disagrees and was OVERRIDDEN, not refused. 942
    # photos on the reference export; silent, it is invisible.
    assert tie is True


def test_disagreeing_candidates_with_no_directory_tiebreak_are_refused(tmp_path):
    """Never let dictionary insertion order decide. 833 real colliding groups,
    100 with differing people, 90 spanning more than a day."""
    root = build_takeout(tmp_path / "Takeout")
    index = build_index(root, EnrichReport())
    # A path in NEITHER candidate's directory, so preference cannot resolve it.
    found, match, _tie = resolve(_photo(root / "Goa Trip" / "elsewhere" / "AMBIG.jpg"), index)
    assert match == "ambiguous"
    assert found is None


def test_a_photo_with_no_sidecar_resolves_to_none(tmp_path):
    root = build_takeout(tmp_path / "Takeout")
    index = build_index(root, EnrichReport())
    found, match, _tie = resolve(_photo(root / "Photos from 2011" / "IMG_EDIT-edited.jpg"), index)
    assert found is None
    assert match == "none"


def test_trash_sidecars_are_excluded_not_orphaned(tmp_path):
    root = build_takeout(tmp_path / "Takeout")
    report = EnrichReport()
    index = build_index(root, report)
    assert "img_gone.jpg" not in index.by_target
    assert report.excluded_dirs == 1


def test_build_index_buckets_every_json_file(tmp_path):
    root = build_takeout(tmp_path / "Takeout")
    report = EnrichReport()
    build_index(root, report)
    assert report.json_files_seen == len(list(root.rglob("*.json")))
    assert report.json_files_seen == 22
    assert report.files_accounted == 22
    # DSC00107 x2, AMBIG x2, SHARED x2, IMG_ALBUM, IMG_EDIT, PXL_1.MP.jpg,
    # IMG_MISSING, IMG_FLAGS, IMG_U0, IMG_U1
    assert report.sidecars_seen == 13
    # root, Goa Trip, Untitled, Untitled(1), No Name
    assert report.album_metadata == 5
    # user-generated-memory-titles.json, shared_album_comments.json
    assert report.other_json == 2
    assert report.excluded_dirs == 1  # Trash/IMG_GONE.jpg...json
    assert report.unparseable == 1  # broken.json


def test_the_sidecar_accounting_identity_holds(tmp_path):
    """v1 would have failed this by 984.

    Runs the SHIPPED algorithm: `TakeoutEnricher().enrich()`, not a
    re-implementation of it. The previous version of this test built its own
    claiming loop in the test body, including its own copy of the sticky
    refusal guard - so its docstring's claim that "deleting it reproduces the
    defect inside THIS test" was true only of the test's copy. Verified live
    at final review: deleting `claimed.get(key) != "ambiguous"` from
    `enrich()` left the whole suite green (241 passed). It was also keying on
    `photo.paths[0].name`, an algorithm production stopped using in Task 11,
    so it certified something that was no longer shipped.

    `claimed` is keyed by target, not by photo, and SHARED.jpg is two
    DISTINCT real photos here - one refused (Kolkata Trip, which `enrich()`
    reaches first: it sorts photos by their lexicographically first path and
    "Kolkata Trip" precedes "Photos from 2011"), one resolved exact (Photos
    from 2011). Without the sticky guard the second overwrites the first's
    refusal - 342 real filenames are shared by distinct photos even after
    content-hash dedup, and on the reference export that overwrite hid 13
    refusals behind a reported `ambiguous` of 0.

    MUTATION (run, not assumed): delete `claimed.get(key) != "ambiguous"`
    from `TakeoutEnricher.enrich` and this test fails on
    `assert report.ambiguous == 1` with ambiguous 0, superseded 2.
    """
    root = build_takeout(tmp_path / "Takeout")
    photos, _ = FolderSource().scan(root)
    store = PhotoStore(tmp_path / "data" / "rekindle.sqlite")
    store.upsert_many(photos)
    report = TakeoutEnricher().enrich(root, store)

    assert report.sidecars_seen == report.sidecars_accounted
    assert report.orphaned == 1  # IMG_MISSING
    # SHARED.jpg's Kolkata Trip photo is genuinely refused - and the refusal
    # SURVIVES even though a different photo (Photos from 2011) sharing the
    # same target name resolved exact for it.
    assert report.ambiguous == 1
    # AMBIG has two candidates; one is applied, the other is SUPERSEDED. The
    # first draft of this plan credited both to `matched`, which inflated it
    # by 3,358 on a real export.
    assert report.superseded == 1
    assert report.matched == report.photos_enriched
    assert report.matched == 10
    store.close()


def test_superseded_sidecars_are_reported_not_discarded(tmp_path):
    """Delete the `superseded` line from account() and this fails.

    The identity above would still pass, because the count would simply move
    into `matched`. That is the exact shape of v1's bug.
    """
    from rekindle.enrich.takeout import account

    root = build_takeout(tmp_path / "Takeout")
    report = EnrichReport()
    index = build_index(root, report)
    year = root / "Photos from 2011"
    sidecar, _match, _tie = resolve(_photo(year / "AMBIG.jpg"), index)
    account(index, {"ambig.jpg": "exact"}, {sidecar.path}, report)
    assert report.matched == 1
    assert report.superseded == 1


def test_matched_reflects_what_was_applied_not_merely_claimed(tmp_path):
    """`applied` is the entire correction over v1, and it is easy to leave
    undefended: every other test in this file builds `applied` FROM
    `resolve()`'s own output, so `matched == len(applied)` there is
    tautological for any implementation that merely counts claimed targets
    (e.g. `report.matched += 1` per claimed-exact target, never consulting
    `applied` at all - that mutant passes every other test here).

    This claims a target exact but applies NOTHING for it - `applied` is
    deliberately empty - so a per-target-counting mutant reports
    matched=1 where the honest answer, tied to actual writes, is matched=0.
    """
    from rekindle.enrich.takeout import account

    root = build_takeout(tmp_path / "Takeout")
    report = EnrichReport()
    index = build_index(root, report)
    year = root / "Photos from 2011"
    sidecar, match, _tie = resolve(_photo(year / "IMG_FLAGS.jpg"), index)
    assert match == "exact"
    assert sidecar is not None

    account(index, {"img_flags.jpg": "exact"}, set(), report)
    assert report.matched == 0
    assert report.superseded == 1


def test_same_directory_disagreement_is_refused_not_silently_first(tmp_path):
    """Line ~324 in resolve(): even a SAME-directory group must agree with
    itself, not just win on proximity. No real Takeout export can produce
    this naturally - the (N) counter always differentiates two same-directory
    sidecars for one photo - so this uses the one other way two sidecar
    files can legitimately collide on a target: a bare `.json` fallback
    sidecar (0 of 24,292 real files use it, but `sidecar_target` accepts it)
    alongside the standard `.supplemental-metadata.json` one, both naming the
    same photo with different stories. Without this fixture, coverage shows
    the branch as dead - which is exactly how the predecessor's ambiguity
    refusal went unexercised and then silently vacuous.
    """
    root = tmp_path / "Takeout"
    year = root / "Photos from 2011"
    _write_sidecar_json(
        year / "TWIN.jpg.supplemental-metadata.json",
        title="TWIN.jpg",
        photoTakenTime={"timestamp": "1000000000"},
        people=[{"name": "Ada"}],
    )
    _write_sidecar_json(
        year / "TWIN.jpg.json",
        title="TWIN.jpg",
        photoTakenTime={"timestamp": "2000000000"},
        people=[{"name": "Grace"}],
    )
    report = EnrichReport()
    index = build_index(root, report)
    found, match, tie = resolve(_photo(year / "TWIN.jpg"), index)
    assert match == "ambiguous"
    assert found is None
    assert tie is False


def test_a_gps_only_disagreement_is_detected_not_silently_overridden(tmp_path):
    """Important 4 residual: `_disagree` was widened to cover gps,
    description and favorite, but nothing pinned it - reverting the
    widening back to date+people alone passed the full suite unchanged.
    9 real disagreeing candidate pairs on the reference export differ ONLY
    on GPS (same capture time, same people, different coordinates); before
    the widening, `tie` was silently `False` for every one of them and
    `EnrichReport.gps_added` had no way to know a coordinate was ever
    contested.

    Same shape as `test_same_directory_wins_over_a_distant_candidate`: one
    same-directory candidate wins by directory preference, one distant
    candidate disagrees - but here the ONLY difference between them is GPS.
    """
    root = tmp_path / "Takeout"
    year = root / "Photos from 2011"
    _write_sidecar_json(
        year / "GPS_TIE.jpg.supplemental-metadata.json",
        title="GPS_TIE.jpg",
        photoTakenTime={"timestamp": "1400000000"},
        people=[{"name": "Ada"}],
        geoData={"latitude": 12.9716, "longitude": 77.5946, "altitude": 900.0},
        geoDataExif={"latitude": 12.9716, "longitude": 77.5946, "altitude": 900.0},
    )
    _write_sidecar_json(
        root / "Elsewhere" / "GPS_TIE.jpg.supplemental-metadata.json",
        title="GPS_TIE.jpg",
        photoTakenTime={"timestamp": "1400000000"},
        people=[{"name": "Ada"}],
        geoData={"latitude": 48.8566, "longitude": 2.3522, "altitude": 35.0},
        geoDataExif={"latitude": 48.8566, "longitude": 2.3522, "altitude": 35.0},
    )
    report = EnrichReport()
    index = build_index(root, report)
    found, match, tie = resolve(_photo(year / "GPS_TIE.jpg"), index)
    assert match == "exact"
    assert found.gps.lat == 12.9716
    assert tie is True


def test_a_description_only_disagreement_is_detected(tmp_path):
    """Same mechanism as the GPS case above, for `description`."""
    root = tmp_path / "Takeout"
    year = root / "Photos from 2011"
    _write_sidecar_json(
        year / "DESC_TIE.jpg.supplemental-metadata.json",
        title="DESC_TIE.jpg",
        photoTakenTime={"timestamp": "1400000000"},
        people=[{"name": "Ada"}],
        description="Caption A",
    )
    _write_sidecar_json(
        root / "Elsewhere" / "DESC_TIE.jpg.supplemental-metadata.json",
        title="DESC_TIE.jpg",
        photoTakenTime={"timestamp": "1400000000"},
        people=[{"name": "Ada"}],
        description="Caption B",
    )
    report = EnrichReport()
    index = build_index(root, report)
    _found, match, tie = resolve(_photo(year / "DESC_TIE.jpg"), index)
    assert match == "exact"
    assert tie is True


def test_a_favourite_only_disagreement_is_detected(tmp_path):
    """Same mechanism as the GPS case above, for `favorited`. 2 real
    disagreeing pairs on the reference export differ only on this flag."""
    root = tmp_path / "Takeout"
    year = root / "Photos from 2011"
    _write_sidecar_json(
        year / "FAV_TIE.jpg.supplemental-metadata.json",
        title="FAV_TIE.jpg",
        photoTakenTime={"timestamp": "1400000000"},
        people=[{"name": "Ada"}],
        favorited=True,
    )
    _write_sidecar_json(
        root / "Elsewhere" / "FAV_TIE.jpg.supplemental-metadata.json",
        title="FAV_TIE.jpg",
        photoTakenTime={"timestamp": "1400000000"},
        people=[{"name": "Ada"}],
    )
    report = EnrichReport()
    index = build_index(root, report)
    _found, match, tie = resolve(_photo(year / "FAV_TIE.jpg"), index)
    assert match == "exact"
    assert tie is True
