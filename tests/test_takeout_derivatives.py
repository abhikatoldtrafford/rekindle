from datetime import UTC, datetime
from pathlib import Path

from rekindle.enrich.takeout import EnrichReport, propagate_to_derivatives
from rekindle.models import Gps, MediaType, Photo, PhotoMeta, TzSource

NOW = datetime(2020, 1, 1, tzinfo=UTC)
TAKEN = datetime(2020, 3, 11, 15, 30, tzinfo=UTC)


def _photo(digest: str, path: Path, meta: PhotoMeta, **kw) -> Photo:
    return Photo(
        file_hash=digest,
        paths=[path],
        media_type=kw.pop("media_type", MediaType.IMAGE),
        meta=meta,
        first_seen=NOW,
        last_seen=NOW,
        **kw,
    )


def _enriched() -> PhotoMeta:
    return PhotoMeta(
        taken_at_utc=TAKEN,
        taken_at_local=TAKEN,
        tz_source=TzSource.TAKEOUT,
        people=["Ada"],
        takeout_people=["Ada"],
        gps=Gps(lat=1.0, lon=2.0),
        description="a caption",
        favorite=True,
        archived=True,
    )


def test_an_edited_variant_inherits_from_its_original():
    lib = Path("/lib")
    original = _photo("orig", lib / "IMG.jpg", _enriched(), sidecar_match="exact")
    edited = _photo("ed", lib / "IMG-edited.jpg", PhotoMeta(), edited_of="orig")
    report = EnrichReport()
    changed = propagate_to_derivatives([original, edited], report)
    assert changed == [edited]
    assert edited.meta.taken_at_utc == TAKEN
    assert edited.meta.tz_source is TzSource.TAKEOUT
    assert edited.meta.people == ["Ada"]
    assert edited.meta.takeout_people == ["Ada"]
    assert edited.meta.gps.lat == 1.0
    assert edited.meta.description == "a caption"
    assert edited.meta.favorite is True
    assert edited.meta.archived is True
    assert edited.sidecar_match == "inherited"
    assert report.derivatives_enriched == 1


def test_a_motion_video_inherits_from_its_still():
    """M0 documents the rule: PXL_1.MP pairs with PXL_1.MP.jpg - the video's
    full NAME plus .jpg, not its stem."""
    lib = Path("/lib/Photos from 2011")
    still = _photo("still", lib / "PXL_1.MP.jpg", _enriched(), sidecar_match="exact")
    video = _photo("vid", lib / "PXL_1.MP", PhotoMeta(), media_type=MediaType.VIDEO)
    report = EnrichReport()
    propagate_to_derivatives([still, video], report)
    assert video.meta.taken_at_utc == TAKEN
    assert video.meta.taken_at_local == TAKEN
    assert video.meta.people == ["Ada"]
    assert video.sidecar_match == "inherited"
    assert report.derivatives_enriched == 1


def test_an_inherited_date_never_claims_an_exif_provenance():
    """A .MP video has no EXIF at all. Copying the still's `exif_offset`
    onto it made doctor report a provenance that cannot exist, on 655 files."""
    lib = Path("/lib/Photos from 2011")
    meta = _enriched()
    meta.tz_source = TzSource.EXIF_OFFSET
    still = _photo("still", lib / "PXL_1.MP.jpg", meta, sidecar_match="exact")
    video = _photo("vid", lib / "PXL_1.MP", PhotoMeta(), media_type=MediaType.VIDEO)
    propagate_to_derivatives([still, video], EnrichReport())
    assert video.meta.tz_source is TzSource.TAKEOUT
    assert still.meta.tz_source is TzSource.EXIF_OFFSET  # the donor is untouched
    assert video.meta.taken_at_utc == TAKEN


def test_motion_pairing_is_scoped_to_one_directory():
    still = _photo("still", Path("/lib/A/PXL_1.MP.jpg"), _enriched(), sidecar_match="exact")
    video = _photo("vid", Path("/lib/B/PXL_1.MP"), PhotoMeta(), media_type=MediaType.VIDEO)
    report = EnrichReport()
    assert propagate_to_derivatives([still, video], report) == []
    assert video.meta.taken_at_utc is None


def test_a_derivative_with_its_own_sidecar_is_left_alone():
    lib = Path("/lib")
    original = _photo("orig", lib / "IMG.jpg", _enriched(), sidecar_match="exact")
    own = PhotoMeta(taken_at_utc=datetime(2021, 1, 1, tzinfo=UTC), people=["Grace"])
    edited = _photo("ed", lib / "IMG-edited.jpg", own, edited_of="orig", sidecar_match="exact")
    report = EnrichReport()
    assert propagate_to_derivatives([original, edited], report) == []
    assert edited.meta.people == ["Grace"]


def test_an_unenriched_original_propagates_nothing():
    lib = Path("/lib")
    original = _photo("orig", lib / "IMG.jpg", PhotoMeta())
    edited = _photo("ed", lib / "IMG-edited.jpg", PhotoMeta(), edited_of="orig")
    report = EnrichReport()
    assert propagate_to_derivatives([original, edited], report) == []


def test_propagation_is_idempotent():
    lib = Path("/lib")
    original = _photo("orig", lib / "IMG.jpg", _enriched(), sidecar_match="exact")
    edited = _photo("ed", lib / "IMG-edited.jpg", PhotoMeta(), edited_of="orig")
    report = EnrichReport()
    propagate_to_derivatives([original, edited], report)
    assert propagate_to_derivatives([original, edited], EnrichReport()) == []
    assert edited.meta.people == ["Ada"]


def test_motion_pairing_matches_regardless_of_name_case():
    """Every other test in this file uses identically-cased names on both
    sides of the pair, which would still pass even with every naming
    `.casefold()` in propagate_to_derivatives deleted. A real export mixes
    .MP/.mp and .jpg/.JPG routinely - this is the case M0's original bug
    (0 pairs found on real data) actually lived in."""
    lib = Path("/lib/Photos from 2011")
    still = _photo("still2", lib / "PXL_2.mp.JPG", _enriched(), sidecar_match="exact")
    video = _photo("vid2", lib / "PXL_2.MP", PhotoMeta(), media_type=MediaType.VIDEO)
    report = EnrichReport()
    propagate_to_derivatives([still, video], report)
    assert video.meta.taken_at_utc == TAKEN
    assert video.sidecar_match == "inherited"
    assert report.derivatives_enriched == 1


def test_an_inherited_derivative_re_syncs_when_its_donor_is_corrected():
    """`takeout_people` exists precisely so a second enrich can retract a
    face tag the user corrected in Google Photos. A derivative must keep
    tracking that correction too - it is not a separate, permanently-frozen
    record just because its own enrichment came by inheritance."""
    lib = Path("/lib")
    original = _photo("orig", lib / "IMG.jpg", _enriched(), sidecar_match="exact")
    edited = _photo("ed", lib / "IMG-edited.jpg", PhotoMeta(), edited_of="orig")
    propagate_to_derivatives([original, edited], EnrichReport())
    assert edited.meta.people == ["Ada"]
    assert edited.sidecar_match == "inherited"

    # The donor's own second enrich retracts the face tag and corrects the
    # date - exactly what a corrected sidecar or a user edit in Google
    # Photos produces on a second run.
    original.meta.people = []
    original.meta.takeout_people = []
    original.meta.taken_at_utc = datetime(2020, 3, 12, 9, 0, tzinfo=UTC)
    original.meta.taken_at_local = datetime(2020, 3, 12, 9, 0, tzinfo=UTC)

    report = EnrichReport()
    changed = propagate_to_derivatives([original, edited], report)
    assert edited in changed
    assert edited.meta.people == []
    assert edited.meta.takeout_people == []
    assert edited.meta.taken_at_utc == datetime(2020, 3, 12, 9, 0, tzinfo=UTC)
    assert report.derivatives_enriched == 1


def test_a_motion_video_never_inherits_an_exif_instant():
    """A .MP video categorically has no EXIF, displaced or otherwise -
    unlike an -edited variant, which plausibly shares its original's EXIF
    lineage. Copying `exif_taken_at_utc` here is the milder twin of the
    tz_source bug this module already guards against."""
    lib = Path("/lib/Photos from 2011")
    meta = _enriched()
    meta.exif_taken_at_utc = datetime(2020, 3, 11, 10, 0, tzinfo=UTC)
    still = _photo("still", lib / "PXL_1.MP.jpg", meta, sidecar_match="exact")
    video = _photo("vid", lib / "PXL_1.MP", PhotoMeta(), media_type=MediaType.VIDEO)
    propagate_to_derivatives([still, video], EnrichReport())
    assert video.meta.exif_taken_at_utc is None


def test_an_edited_variant_still_inherits_an_exif_instant():
    """The motion-video exclusion above must not overreach: an -edited
    variant IS the same shot, re-encoded, so it plausibly shares its
    original's displaced EXIF instant."""
    lib = Path("/lib")
    meta = _enriched()
    meta.exif_taken_at_utc = datetime(2020, 3, 11, 10, 0, tzinfo=UTC)
    original = _photo("orig", lib / "IMG.jpg", meta, sidecar_match="exact")
    edited = _photo("ed", lib / "IMG-edited.jpg", PhotoMeta(), edited_of="orig")
    propagate_to_derivatives([original, edited], EnrichReport())
    assert edited.meta.exif_taken_at_utc == datetime(2020, 3, 11, 10, 0, tzinfo=UTC)
