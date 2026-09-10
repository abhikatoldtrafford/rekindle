from datetime import UTC, datetime
from pathlib import Path

from rekindle.enrich.takeout import EnrichReport, build_index, resolve
from rekindle.models import MediaType, Photo, PhotoMeta
from tests.fixtures.takeout import build_takeout


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
    assert report.json_files_seen == 20
    assert report.files_accounted == 20
    assert report.sidecars_seen == 11
    # root, Goa Trip, Untitled, Untitled(1), No Name
    assert report.album_metadata == 5
    # user-generated-memory-titles.json, shared_album_comments.json
    assert report.other_json == 2
    assert report.excluded_dirs == 1  # Trash/IMG_GONE.jpg...json
    assert report.unparseable == 1  # broken.json


def test_the_sidecar_accounting_identity_holds(tmp_path):
    """v1 would have failed this by 984.

    On its own the identity is weak - `account()` partitions a set it builds
    itself, so it cannot see anything downstream of indexing. The assertion
    that actually bites is `matched == photos_enriched`, in
    tests/test_takeout_enricher.py. This one pins the partition; that one ties
    the partition to what reached the database.
    """
    from rekindle.enrich.takeout import account

    root = build_takeout(tmp_path / "Takeout")
    report = EnrichReport()
    index = build_index(root, report)
    photos = [
        _photo(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in {".jpg", ".mp"} and "Trash" not in p.parts
    ]
    claimed: dict[str, str] = {}
    applied: set[Path] = set()
    for photo in photos:
        sidecar, match, _tie = resolve(photo, index)
        if match != "none":
            claimed[photo.paths[0].name.casefold()] = match
        if sidecar is not None:
            applied.add(sidecar.path)

    account(index, claimed, applied, report)
    assert report.sidecars_seen == report.sidecars_accounted
    assert report.orphaned >= 1  # IMG_MISSING
    assert report.ambiguous == 0  # AMBIG resolved by directory preference
    # AMBIG has two candidates; one is applied, the other is SUPERSEDED. The
    # first draft of this plan credited both to `matched`, which inflated it
    # by 3,358 on a real export.
    assert report.superseded == 1
    assert report.matched == len(applied)


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
