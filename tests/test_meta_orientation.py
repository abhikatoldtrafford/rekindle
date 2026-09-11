"""Stale EXIF orientation tags: the decision rule, and what applying it moves.

Every assertion about a decode in here reads PIXELS, through
`tests.fixtures.oriented`. That is the lesson of `test_orientation.py`: a
400x300 file tagged 90 degrees is 300x400 whichever way the rotation went, so
a test that can only see width and height cannot tell a correction from its
inverse - and the correction this module applies is, by construction, exactly
the inverse of the tag.

The rule itself (`decide`) is pure arithmetic over detection scores and is
tested with no model at all, the way `semantic.faces.classify` is. There is
no detector in CI.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image

from rekindle.db import SCHEMA_VERSION, FingerprintRow, PhotoStore
from rekindle.meta import orientation as orient
from rekindle.meta.exif import open_upright, read_exif
from rekindle.models import MediaType, Photo, PhotoMeta, TzSource
from tests.fixtures.oriented import (
    ORIENTATION_TAG,
    UPRIGHT,
    quadrants,
    upright_marker,
    write_oriented,
)

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _no_overrides():
    """Every test starts with an empty override table and leaves one behind.

    The table is process-wide - `open_upright` takes a Path and no index
    handle - so a test that installs one and returns would change how every
    later test in the session decodes. That is exactly the failure mode this
    fixture exists to make impossible.
    """
    orient.clear_overrides()
    yield
    orient.clear_overrides()


def _stale(path: Path, orientation: int = 6, size=(400, 300)) -> Image.Image:
    """Write the file this whole module is about: pixels ALREADY UPRIGHT,
    plus an axis-swapping tag that lies about them.

    Not `write_oriented`, which writes an honest file (pixels transformed
    backwards through the tag). Here the pixels are the upright marker and the
    tag is simply wrong - which is what an earlier tool that rotated the
    pixels and forgot the tag leaves behind.
    """
    upright = upright_marker(size)
    exif = upright.getexif()
    exif[ORIENTATION_TAG] = orientation
    upright.save(path, "JPEG", quality=95, exif=exif)
    return upright


# --------------------------------------------------------------- the fixture
#
# A fixture that did not actually lie would let every test below pass while
# proving nothing.


def test_the_stale_fixture_really_is_stale(tmp_path):
    path = tmp_path / "stale.jpg"
    _stale(path)
    with Image.open(path) as im:
        assert im.getexif()[ORIENTATION_TAG] == 6
        # The raw pixels are ALREADY the upright picture...
        assert quadrants(im) == UPRIGHT
    # ...so applying the tag, which is the correct thing to do with an honest
    # file, turns this correct photograph on its side.
    assert quadrants(open_upright(path)) != UPRIGHT


# ------------------------------------------------------------ evidence, decide


def test_evidence_prefers_confidence_over_count():
    """The specific defect this replaces: ranking by how MANY boxes there are.

    Four detections at 0.20 are not better evidence than one at 0.85.
    """
    assert orient.evidence([0.2, 0.2, 0.2, 0.2]) < orient.evidence([0.85])
    # And the real pair that started it: 4 weak beat 2 strong under the old
    # count-first ranking.
    assert orient.evidence([0.669] * 4) > orient.evidence([0.75] * 2)  # count still speaks
    assert orient.evidence([0.2] * 4) < orient.evidence([0.75] * 2)  # but only when strong


def test_evidence_of_nothing_is_zero():
    assert orient.evidence([]) == 0.0


@pytest.mark.parametrize("tag", [1, 2, 3, 4, 0, None, 9])
def test_a_tag_that_does_not_swap_axes_is_never_second_guessed(tag):
    """The measured reason the rule is this narrow: an unrestricted detector
    that invents rotations for such files scored 0.22 precision by hand."""
    v = orient.decide(tag, [], [0.99, 0.99, 0.99])
    assert v.ignore_exif is False
    assert v.reason == orient.NO_TAG


@pytest.mark.parametrize("tag", sorted(orient.SWAPS_AXES))
def test_a_decisive_win_without_the_tag_calls_the_tag_stale(tag):
    v = orient.decide(tag, [0.10], [0.90])
    assert v.ignore_exif is True
    assert v.reason == orient.STALE
    assert v.tag == tag
    assert v.margin == pytest.approx(0.81 - 0.01)


def test_a_narrow_win_is_not_enough():
    """0.851 against 0.856 is noise, and noise must not overrule a file."""
    v = orient.decide(6, [0.851], [0.856])
    assert v.ignore_exif is False
    assert v.reason == orient.MARGIN_TOO_SMALL
    assert 0 < v.margin < orient.MIN_MARGIN


def test_the_margin_is_load_bearing():
    """Exactly at the threshold fires; a hair under it does not."""
    # e_raw - e_tag == MIN_MARGIN exactly.
    e_raw = orient.MIN_MARGIN + 0.45**2
    raw = (e_raw**0.5,)
    assert orient.decide(6, [0.45], list(raw)).ignore_exif is True
    assert orient.decide(6, [0.45], [raw[0] - 0.001]).ignore_exif is False


def test_no_face_without_the_tag_means_no_opinion():
    """Two piles of noise must not be compared against each other. The
    tag-ignored decode has the bigger number here and still loses."""
    v = orient.decide(6, [0.05], [0.44, 0.44, 0.44])
    assert v.ignore_exif is False
    assert v.reason == orient.NO_FACE
    # It wins on evidence by a mile and still loses, because nothing in it is
    # confident enough to be called a face.
    assert v.margin > orient.MIN_MARGIN


def test_the_face_floor_is_load_bearing():
    """Three boxes, so the evidence clears the margin either way and the ONLY
    thing that changes across these two lines is whether the best detection
    reaches `MIN_FACE`."""
    at = [orient.MIN_FACE] * 3
    under = [orient.MIN_FACE - 0.001] * 3
    assert orient.evidence(under) - 0.0 > orient.MIN_MARGIN  # the margin is not the guard here
    assert orient.decide(6, [0.0], at).ignore_exif is True
    assert orient.decide(6, [0.0], under).ignore_exif is False


def test_a_win_for_the_tag_never_fires():
    """The photograph agrees with its own metadata. This is the common case."""
    v = orient.decide(6, [0.95, 0.9], [0.5])
    assert v.ignore_exif is False
    assert v.margin < 0


def test_the_verdict_records_the_evidence_it_used():
    """A silent rotation is indistinguishable from a bug, so every field the
    report and the stored blob show has to survive the round trip."""
    v = orient.decide(6, [0.10], [0.90])
    blob = json.loads(v.as_json())
    assert blob["tag"] == 6
    assert blob["reason"] == orient.STALE
    assert blob["e_tag"] == pytest.approx(0.01)
    assert blob["e_raw"] == pytest.approx(0.81)
    assert blob["m_raw"] == pytest.approx(0.90)
    assert blob["margin"] == pytest.approx(0.80)


# ------------------------------------------------------------ the registry


def test_nothing_is_overridden_by_default(tmp_path):
    """The state CI is in, and the state a library nobody has run the pass on
    is in. It must decode exactly as it always did."""
    path = tmp_path / "o6.jpg"
    write_oriented(path, 6)
    assert orient.ignores_exif(path) is False
    assert quadrants(open_upright(path)) == UPRIGHT


def test_loading_replaces_rather_than_accumulates(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    _stale(a)
    _stale(b)
    assert orient.load_overrides([a]) == 1
    assert orient.ignores_exif(a) and not orient.ignores_exif(b)
    assert orient.load_overrides([b]) == 1
    # Opening a second index must not leave the first one's corrections behind.
    assert orient.ignores_exif(b) and not orient.ignores_exif(a)


def test_a_path_spelled_differently_is_the_same_file(tmp_path):
    path = tmp_path / "sub" / "x.jpg"
    path.parent.mkdir()
    _stale(path)
    orient.load_overrides([path])
    assert orient.ignores_exif(tmp_path / "sub" / ".." / "sub" / "x.jpg")


def test_clearing_puts_the_tag_back(tmp_path):
    path = tmp_path / "stale.jpg"
    _stale(path)
    orient.load_overrides([path])
    assert quadrants(open_upright(path)) == UPRIGHT
    orient.clear_overrides()
    assert quadrants(open_upright(path)) != UPRIGHT


# --------------------------------------------------- the chokepoint, in pixels


def test_an_overridden_file_decodes_to_its_stored_pixels(tmp_path):
    """The whole point. The file is upright on disk and its tag says rotate;
    with the override, `open_upright` hands back the photograph."""
    path = tmp_path / "stale.jpg"
    _stale(path)
    orient.load_overrides([path])
    assert quadrants(open_upright(path)) == UPRIGHT


@pytest.mark.parametrize("orientation", sorted(orient.SWAPS_AXES))
def test_the_override_is_exactly_not_applying_the_tag(tmp_path, orientation):
    """For every axis-swapping tag, an overridden decode equals the raw file.

    Asserted against Pillow's own untransposed decode rather than against a
    second hand-rolled expectation - the same discipline the oriented fixture
    uses.
    """
    path = tmp_path / f"o{orientation}.jpg"
    _stale(path, orientation)
    orient.load_overrides([path])
    with Image.open(path) as im:
        expected = quadrants(im)
    assert quadrants(open_upright(path)) == expected


def test_an_override_does_not_touch_any_other_file(tmp_path):
    stale, honest = tmp_path / "stale.jpg", tmp_path / "honest.jpg"
    _stale(stale)
    write_oriented(honest, 6)
    orient.load_overrides([stale])
    # The honest file still gets its tag applied, which is still correct.
    assert quadrants(open_upright(honest)) == UPRIGHT
    assert quadrants(open_upright(stale)) == UPRIGHT


def test_the_draft_hint_is_not_transposed_for_an_overridden_file(tmp_path):
    """`draft` arrives in the UPRIGHT axis order. When the tag is ignored the
    stored pixels already ARE that order, so transposing it here would ask
    libjpeg for the wrong scale - the defect the transpose exists to avoid,
    running backwards.
    """
    path = tmp_path / "stale.jpg"
    _stale(path, 6, size=(3200, 2400))
    orient.load_overrides([path])

    # This draft is chosen because it is one where the two behaviours actually
    # DIFFER. libjpeg only halves, so for most requests a transposed target
    # lands on the same scale and a test using one proves nothing; here the
    # correct decode cannot reduce at all (halving would give 1600x1200, and
    # 1200 is under the 1600 asked for) while a transposed target happily
    # reduces to exactly that - handing back an image SHORTER than requested,
    # which the caller must then upscale.
    got = open_upright(path, draft=(600, 1600))
    assert got.height >= 1600, "decoded smaller than the caller asked for"
    assert got.size == (3200, 2400)
    assert quadrants(got) == UPRIGHT

    # And the ordinary case still gets the reduction it asked for, rather than
    # the whole image: a hint that never fires is as wrong as one that fires
    # backwards, just more expensive.
    assert open_upright(path, draft=(1600, 600)).size == (1600, 1200)


def test_an_overridden_decode_still_raises_what_callers_catch(tmp_path):
    """`copy()` loads inside the `with`, so a truncated file raises OSError
    where every caller's try already is - exactly as `exif_transpose` does."""
    path = tmp_path / "trunc.jpg"
    _stale(path, 6, size=(800, 600))
    path.write_bytes(path.read_bytes()[: len(path.read_bytes()) // 3])
    orient.load_overrides([path])
    with pytest.raises(OSError):
        open_upright(path)


def test_the_index_dimensions_follow_the_override(tmp_path):
    """`read_exif` swaps width and height for an axis-swapping tag. If it went
    on doing that for an overridden file, a canvas sized from the index would
    not match the frame drawn from the file."""
    path = tmp_path / "stale.jpg"
    _stale(path, 6, size=(400, 300))
    assert (read_exif(path).width, read_exif(path).height) == (300, 400)
    orient.load_overrides([path])
    assert (read_exif(path).width, read_exif(path).height) == (400, 300)
    assert (read_exif(path).width, read_exif(path).height) == open_upright(path).size


# ---------------------------------------------------------------- the pass


class _FakeDetector:
    """A detector with no model behind it. `examine` only needs `.size` and
    `.detect`, which is what makes the pass testable where CI has neither
    onnxruntime nor weights."""

    size = 64

    def __init__(self, scores_by_landscape: dict[bool, list[float]]) -> None:
        self._scores = scores_by_landscape
        self.calls = 0

    def detect(self, image, **_kw):
        self.calls += 1
        scores = self._scores[image.width > image.height]
        return tuple(_Box(s) for s in scores), max(scores, default=0.0)


class _Box:
    def __init__(self, score: float) -> None:
        self.score = score


def test_examine_compares_the_two_decodes_of_one_file(tmp_path):
    path = tmp_path / "stale.jpg"
    _stale(path, 6, size=(400, 300))  # stored LANDSCAPE, tag says portrait
    det = _FakeDetector({True: [0.9], False: [0.1]})
    v = orient.examine(path, det)
    assert v.ignore_exif is True
    assert det.calls == 2


def test_examine_never_decodes_a_file_with_no_swapping_tag(tmp_path):
    """86% of a real library takes this path; it is what makes the pass
    minutes rather than hours."""
    path = tmp_path / "plain.jpg"
    write_oriented(path, 1)
    det = _FakeDetector({True: [0.9], False: [0.9]})
    assert orient.examine(path, det).reason == orient.NO_TAG
    assert det.calls == 0


def test_examine_does_not_go_through_the_override_it_writes(tmp_path):
    """A second run over an already-corrected library must see the file as it
    actually is. Reading it through `open_upright` would compare the corrected
    decode against itself and confirm whatever the first run decided."""
    path = tmp_path / "stale.jpg"
    _stale(path, 6, size=(400, 300))
    orient.load_overrides([path])
    det = _FakeDetector({True: [0.9], False: [0.1]})
    assert orient.examine(path, det).ignore_exif is True
    assert det.calls == 2


def _photo(path: Path, digest: str) -> Photo:
    return Photo(
        file_hash=digest,
        media_type=MediaType.IMAGE,
        paths=(path,),
        meta=PhotoMeta(tz_source=TzSource.NONE),
        first_seen=T0,
        last_seen=T0,
    )


def _store_with(tmp_path, files: dict[str, Path]) -> PhotoStore:
    store = PhotoStore(tmp_path / "db" / "rekindle.sqlite")
    store.upsert_many([_photo(p, h) for h, p in files.items()])
    return store


def test_the_pass_accounts_for_every_photo_and_writes_the_verdicts(tmp_path):
    stale, honest, plain = (tmp_path / n for n in ("s.jpg", "h.jpg", "p.jpg"))
    _stale(stale, 6, size=(400, 300))
    write_oriented(honest, 6)
    write_oriented(plain, 1)
    det = _FakeDetector({True: [0.9], False: [0.1]})
    with _store_with(tmp_path, {"a": stale, "b": honest, "c": plain}) as store:
        report = orient.run_orientation(store, det, workers=2)
        assert report.accounted
        assert report.examined == 3
        assert report.tag_ignored == 1
        assert report.no_tag == 1
        assert store.orientation_counts() == {"examined": 3, "ignoring_exif": 1}
        # And the correction is installed, so the next decode in this process
        # already uses it.
        assert orient.ignores_exif(stale)
    assert [p for p, _ in report.corrected] == [stale]


def test_the_pass_resumes_rather_than_restarting(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    write_oriented(a, 1)
    write_oriented(b, 1)
    det = _FakeDetector({True: [], False: []})
    with _store_with(tmp_path, {"a": a, "b": b}) as store:
        assert len(list(store.iter_unoriented())) == 2
        orient.run_orientation(store, det, workers=1)
        # Examined is finished work, whichever way it came out.
        assert list(store.iter_unoriented()) == []


def test_videos_are_out_of_scope_rather_than_failed(tmp_path):
    """A video has no EXIF orientation tag to call stale and is never decoded
    by `open_upright`. Because an error is deliberately not stored as a
    verdict, a video in scope would be re-opened on EVERY run and the pass
    could never report itself finished - which is what the first run over the
    reference library did to 1,117 of them.
    """
    photo = tmp_path / "p.jpg"
    write_oriented(photo, 1)
    clip = tmp_path / "v.mp4"
    clip.write_bytes(b"not an image")
    with PhotoStore(tmp_path / "db" / "rekindle.sqlite") as store:
        store.upsert_many(
            [
                _photo(photo, "img"),
                Photo(
                    file_hash="vid",
                    media_type=MediaType.VIDEO,
                    paths=(clip,),
                    meta=PhotoMeta(tz_source=TzSource.NONE),
                    first_seen=T0,
                    last_seen=T0,
                ),
            ]
        )
        assert [p.file_hash for p in store.iter_unoriented()] == ["img"]
        report = orient.run_orientation(store, _FakeDetector({True: [], False: []}), workers=1)
        assert report.errors == 0
        # Finished, and it stays finished.
        assert list(store.iter_unoriented()) == []


def test_an_unreadable_file_is_not_recorded_as_examined(tmp_path):
    """There is no column saying WHY a file failed, so writing 0 would be
    indistinguishable from 'examined, tag trusted' and the file would never be
    looked at again. A missing drive is a reason to retry."""
    missing = tmp_path / "gone.jpg"
    det = _FakeDetector({True: [], False: []})
    with _store_with(tmp_path, {"a": missing}) as store:
        report = orient.run_orientation(store, det, workers=1)
        assert report.errors == 1
        assert report.accounted
        assert len(list(store.iter_unoriented())) == 1


def test_recording_a_stale_tag_invalidates_that_photos_fingerprint(tmp_path):
    """Changing the decode changes the pixels, and phash/sharpness/colour are
    measurements OF those pixels. With no invalidation, 2,503 sideways vectors
    survived an earlier orientation fix."""
    stale, honest = tmp_path / "s.jpg", tmp_path / "h.jpg"
    _stale(stale, 6, size=(400, 300))
    write_oriented(honest, 6)
    with _store_with(tmp_path, {"a": stale, "b": honest}) as store:
        store.set_fingerprints(
            [
                FingerprintRow("a", phash=123, sharpness=0.5, brightness=0.5, colour="ab"),
                FingerprintRow("b", phash=456, sharpness=0.5, brightness=0.5, colour="cd"),
            ]
        )
        assert list(store.iter_unfingerprinted()) == []
        store.set_orientations([("a", True, "{}"), ("b", False, "{}")])
        redo = {p.file_hash for p in store.iter_unfingerprinted()}
        assert redo == {"a"}


def test_a_recorded_decode_failure_is_retried_after_a_correction(tmp_path):
    """`phash_error` is finished work - unless the decode it failed on is the
    one this verdict just changed."""
    stale = tmp_path / "s.jpg"
    _stale(stale, 6, size=(400, 300))
    with _store_with(tmp_path, {"a": stale}) as store:
        store.set_fingerprints([FingerprintRow("a", error="undecodable")])
        assert list(store.iter_unfingerprinted()) == []
        store.set_orientations([("a", True, "{}")])
        assert [p.file_hash for p in store.iter_unfingerprinted()] == ["a"]


# ------------------------------------------------------------------ the store


def test_opening_the_index_installs_its_corrections(tmp_path):
    """`open_upright` takes a Path and no index handle, so opening the index
    is the only moment the corrections can be installed."""
    stale = tmp_path / "s.jpg"
    _stale(stale, 6, size=(400, 300))
    db = tmp_path / "db" / "rekindle.sqlite"
    with PhotoStore(db) as store:
        store.upsert_many([_photo(stale, "a")])
        store.set_orientations([("a", True, "{}")])
    orient.clear_overrides()
    assert quadrants(open_upright(stale)) != UPRIGHT
    with PhotoStore(db):
        assert quadrants(open_upright(stale)) == UPRIGHT


def test_the_read_only_reader_installs_them_too(tmp_path):
    """The embedder and the face gate open the index through this reader, not
    through PhotoStore. If it did not load them they would keep seeing the
    photographs sideways - which is precisely the earlier defect."""
    from rekindle.semantic.photos import PhotoIndexReader

    stale = tmp_path / "s.jpg"
    _stale(stale, 6, size=(400, 300))
    db = tmp_path / "db" / "rekindle.sqlite"
    with PhotoStore(db) as store:
        store.upsert_many([_photo(stale, "a")])
        store.set_orientations([("a", True, "{}")])
    orient.clear_overrides()
    reader = PhotoIndexReader(db)
    try:
        assert quadrants(open_upright(stale)) == UPRIGHT
    finally:
        reader.close()


# --------------------------------------------------------------- the migration


_V5_CREATE = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE photos (
    file_hash TEXT PRIMARY KEY, media_type TEXT NOT NULL,
    paths TEXT NOT NULL, albums TEXT NOT NULL, edited_of TEXT,
    first_seen TEXT NOT NULL, last_seen TEXT NOT NULL,
    taken_at_utc TEXT, taken_at_local TEXT, tz_source TEXT NOT NULL,
    gps_lat REAL, gps_lon REAL, gps_alt REAL, people TEXT NOT NULL,
    face_regions TEXT NOT NULL, keywords TEXT NOT NULL,
    description TEXT, favorite INTEGER NOT NULL DEFAULT 0,
    camera_make TEXT, camera_model TEXT, width INTEGER, height INTEGER,
    source TEXT NOT NULL DEFAULT 'folder',
    metadata_conflict INTEGER NOT NULL DEFAULT 0,
    exif_taken_at_utc TEXT, takeout_people TEXT NOT NULL DEFAULT '[]',
    archived INTEGER NOT NULL DEFAULT 0, trashed INTEGER NOT NULL DEFAULT 0,
    sidecar_match TEXT NOT NULL DEFAULT 'none',
    phash INTEGER, sharpness REAL, phash_error TEXT, brightness REAL,
    colour TEXT
);
CREATE TABLE photo_paths (
    file_hash TEXT NOT NULL, path TEXT NOT NULL,
    name_cf TEXT NOT NULL, parent TEXT NOT NULL,
    PRIMARY KEY (file_hash, path)
);
"""


def _write_v5(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(_V5_CREATE)
    conn.execute("INSERT INTO meta VALUES ('schema_version', '5')")
    conn.execute(
        "INSERT INTO photos (file_hash, media_type, paths, albums, first_seen,"
        " last_seen, tz_source, people, face_regions, keywords, phash, colour)"
        " VALUES ('a','image','[]','[]','2026-01-01T12:00:00+00:00',"
        "'2026-01-01T12:00:00+00:00','none','[]','[]','[]', 99, 'ff')"
    )
    conn.commit()
    conn.close()


def test_a_v5_database_migrates_and_keeps_its_fingerprints(tmp_path):
    """v6 adds columns and destroys nothing. Unlike v5, no stored measurement
    changes meaning here: until the pass records a stale tag, every decode in
    the program is byte-for-byte what it was."""
    db = tmp_path / "rekindle.sqlite"
    _write_v5(db)
    with PhotoStore(db) as store:
        assert store.schema_version() == SCHEMA_VERSION == 6
        row = store._conn.execute("SELECT * FROM photos WHERE file_hash='a'").fetchone()
        assert row["phash"] == 99 and row["colour"] == "ff"
        # NULL, not 0: "never examined" is not "examined and fine".
        assert row["orient_ignore_exif"] is None
        assert [p.file_hash for p in store.iter_unoriented()] == ["a"]


def test_the_review_queue_holds_the_near_misses_and_nothing_else(tmp_path):
    """What the pass could not commit to, ranked by how close it came.

    The queue exists because the three files that prompted this whole
    mechanism are NOT corrected - they lean the right way by 0.10 to 0.23 and
    the threshold is 0.35. A photo whose picture AGREES with its tag is the
    ordinary case and must never appear here, or the queue is just a list of
    the library.
    """
    files = {}
    for name in ("near", "agrees", "plain"):
        p = tmp_path / f"{name}.jpg"
        _stale(p, 6, size=(400, 300))
        files[name] = p
    write_oriented(files["plain"], 1)
    with _store_with(tmp_path, files) as store:
        store.set_orientations(
            [
                ("near", False, orient.decide(6, [0.60], [0.66]).as_json()),
                ("agrees", False, orient.decide(6, [0.90], [0.10]).as_json()),
                ("plain", False, orient.decide(None, [], []).as_json()),
            ]
        )
        queue = store.orientation_review_queue()
    assert [p.stem for p, _ in queue] == ["near"]
    assert queue[0][1]["margin"] > 0


def test_the_review_queue_ranks_the_closest_call_first(tmp_path):
    files = {n: tmp_path / f"{n}.jpg" for n in ("a", "b", "c")}
    for p in files.values():
        _stale(p, 6, size=(400, 300))
    with _store_with(tmp_path, files) as store:
        store.set_orientations(
            [
                ("a", False, orient.decide(6, [0.10], [0.30]).as_json()),
                ("b", False, orient.decide(6, [0.10], [0.55]).as_json()),
                ("c", False, orient.decide(6, [0.10], [0.40]).as_json()),
            ]
        )
        assert [p.stem for p, _ in store.orientation_review_queue()] == ["b", "c", "a"]
        assert len(store.orientation_review_queue(limit=2)) == 2


def test_neither_orientation_module_imports_anything_heavy():
    """`meta.exif` imports `meta.orientation` at MODULE level, and `meta.exif`
    is reached by `rekindle index` on a default install with no extras at all.
    A module-level numpy or onnxruntime here would make the heaviest import in
    the project unconditional - and on a half-installed torch it would be a
    traceback on a command that has nothing to do with machine learning.

    A subprocess, because by the time this file runs pytest has already
    imported numpy for other modules and an in-process check would pass no
    matter what. The same reasoning, and the same mechanism, as
    `tests/test_semantic_imports.py`.
    """
    import subprocess
    import sys
    import textwrap

    heavy = ("numpy", "torch", "onnxruntime", "transformers", "PIL")
    for module in ("rekindle.meta.orientation", "rekindle.meta.exif"):
        code = textwrap.dedent(f"""
            import sys
            import {module}  # noqa: F401
            print(",".join(m for m in {heavy!r} if m in sys.modules))
        """)
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        found = [m for m in proc.stdout.strip().split(",") if m]
        # PIL is exempt for `meta.exif`, which cannot do its job without it -
        # it is a hard dependency, not an extra. `meta.orientation` must not
        # even pull that in: it is imported for `ignores_exif` on paths that
        # never decode anything.
        allowed = {"PIL"} if module.endswith("exif") else set()
        assert not (set(found) - allowed), f"{module} imported {found}"


def test_a_migrated_database_has_no_corrections_to_install(tmp_path):
    db = tmp_path / "rekindle.sqlite"
    _write_v5(db)
    orient.load_overrides([tmp_path / "leftover.jpg"])
    with PhotoStore(db):
        assert orient.override_count() == 0
