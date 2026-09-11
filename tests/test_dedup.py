"""Burst dedup.

The shapes here come from the real library: 30-second runs of visually
DISTINCT photos (the common case - median within-run dHash distance is 22),
photos sharing a timestamp to the second, and rows with NULL width/height.
"""

import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

from rekindle.memory.dedup import bursts, collapse, hamming
from rekindle.models import MediaType, Photo, PhotoMeta

T0 = datetime(2020, 5, 1, 12, 0, tzinfo=UTC)

# Two hashes 2 bits apart (below the default threshold of 6) and one 32 bits
# apart (far above it). Written as literals so the distances are visible.
H_A = 0b0000_0000
H_A2 = 0b0000_0011
H_FAR = 0xFFFFFFFF


def _p(
    name,
    *,
    at=0.0,
    phash=H_A,
    sharpness=1.0,
    size=(100, 100),
    media_type=MediaType.IMAGE,
) -> Photo:
    width, height = size if size else (None, None)
    return Photo(
        file_hash=name,
        paths=[Path(f"/lib/{name}.jpg")],
        media_type=media_type,
        meta=PhotoMeta(
            taken_at_utc=T0 + timedelta(seconds=at),
            taken_at_local=T0 + timedelta(seconds=at),
            phash=phash,
            sharpness=sharpness,
            width=width,
            height=height,
        ),
        first_seen=T0,
        last_seen=T0,
    )


# --------------------------------------------------------------------------
# hamming


def test_hamming_counts_differing_bits():
    assert hamming(0b1010, 0b1001) == 2
    assert hamming(0, 0) == 0
    assert hamming(0, 0xFFFFFFFFFFFFFFFF) == 64


# --------------------------------------------------------------------------
# the two signals, independently


def test_near_identical_frames_seconds_apart_collapse():
    kept, report = collapse([_p("a", at=0, phash=H_A), _p("b", at=5, phash=H_A2)])
    assert len(kept) == 1
    assert report.collapsed == 1


def test_identical_frames_HOURS_apart_both_survive():
    """The brief's explicit requirement: two similar photos taken hours apart
    must both survive. Only the time gate can save them - the hashes here are
    identical."""
    kept, _ = collapse([_p("a", at=0, phash=H_A), _p("b", at=3 * 3600, phash=H_A)])
    assert len(kept) == 2


def test_DIFFERENT_frames_seconds_apart_both_survive():
    """The case that makes time-only dedup unusable on this library: 59.2% of
    consecutive images are within 30s of the one before, and the median
    within-run hash distance is 22."""
    kept, _ = collapse([_p("a", at=0, phash=H_A), _p("b", at=5, phash=H_FAR)])
    assert len(kept) == 2


def _spread_hash(i: int) -> int:
    """Well-separated 64-bit values (golden-ratio multiplicative hashing).

    The minimum distance over all pairs of the first 80 is 19 bits, which is
    the point: it reproduces the real library, where the MEDIAN dHash distance
    between two photos in the same 30-second run is 22. An earlier version of
    this fixture used `1 << i`, whose consecutive values are 2 bits apart -
    genuinely near-identical by the hash, so collapsing them was correct and
    the test was asserting a fiction.
    """
    return (i * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF


def test_the_distinct_run_fixture_really_is_distinct():
    """Guards the fixture above against the mistake it was born from."""
    values = [_spread_hash(i) for i in range(80)]
    assert min(hamming(a, b) for i, a in enumerate(values) for b in values[i + 1 :]) > 6


def test_a_long_run_of_distinct_photos_is_not_collapsed():
    """The real shape: an 80-photo run at 30-second spacing, every frame
    different. Time-only chaining collapses this to ONE photo and silently
    destroys 79 memories."""
    photos = [_p(f"p{i:02d}", at=i * 30, phash=_spread_hash(i)) for i in range(80)]
    kept, report = collapse(photos)
    assert len(kept) == 80
    assert report.collapsed == 0


def test_a_genuine_long_burst_does_collapse():
    """The mirror image: 12 near-identical frames at 2-second spacing are one
    burst, however long the run is."""
    photos = [_p(f"p{i:02d}", at=i * 2, phash=H_A) for i in range(12)]
    kept, report = collapse(photos)
    assert len(kept) == 1
    assert report.bursts == 1


def test_time_chains_across_more_than_one_gap():
    """The gap is measured to the PREVIOUS photo, not the anchor, so a burst
    spanning 90 seconds at 30-second spacing stays one burst."""
    photos = [_p("a", at=0), _p("b", at=30), _p("c", at=60), _p("d", at=90)]
    assert len(collapse(photos)[0]) == 1


# --------------------------------------------------------------------------
# anchoring


def test_pixel_drift_starts_a_new_burst_instead_of_chaining():
    """A is near B, B is near C, but A is far from C.

    With a previous-photo comparison all three collapse to one and a distinct
    photo is lost. Anchored on A, C fails the test and starts its own burst.
    """
    a = _p("a", at=0, phash=0b0000_0000)
    b = _p("b", at=5, phash=0b0000_1111)  # 4 from a
    c = _p("c", at=10, phash=0b1111_1111)  # 4 from b, but 8 from a

    groups = bursts([a, b, c])

    assert [[p.file_hash for p in g] for g in groups] == [["a", "b"], ["c"]]


def test_a_new_anchor_is_adopted_when_a_burst_ends():
    a = _p("a", at=0, phash=0)
    b = _p("b", at=5, phash=H_FAR)
    c = _p("c", at=10, phash=H_FAR)
    assert [[p.file_hash for p in g] for g in bursts([a, b, c])] == [["a"], ["b", "c"]]


# --------------------------------------------------------------------------
# unknown is not similar


def test_a_photo_without_a_fingerprint_never_joins_a_burst():
    kept, report = collapse([_p("a", at=0, phash=H_A), _p("b", at=5, phash=None)])
    assert len(kept) == 2
    assert report.unfingerprinted == 1


def test_a_photo_without_a_fingerprint_never_absorbs_its_neighbours():
    """The anchor has no hash. Nothing may join it - otherwise one video or
    one undecodable file would swallow everything around it."""
    photos = [_p("a", at=0, phash=None), _p("b", at=5, phash=H_A), _p("c", at=10, phash=H_A)]
    groups = bursts(photos)
    assert [p.file_hash for p in groups[0]] == ["a"]
    assert len(kept_hashes(collapse(photos)[0])) == 2


def test_videos_never_collapse():
    """Videos are never fingerprinted, so they are structurally exempt. All
    1,117 of them in the reference library."""
    photos = [
        _p("v1", at=0, phash=None, size=None, media_type=MediaType.VIDEO),
        _p("v2", at=5, phash=None, size=None, media_type=MediaType.VIDEO),
    ]
    assert len(collapse(photos)[0]) == 2


def kept_hashes(photos):
    return [p.file_hash for p in photos]


# --------------------------------------------------------------------------
# choosing the survivor


def test_the_sharpest_frame_wins():
    photos = [_p("blurry", at=0, sharpness=1.0), _p("sharp", at=2, sharpness=9.0)]
    assert kept_hashes(collapse(photos)[0]) == ["sharp"]


def test_resolution_breaks_a_sharpness_tie():
    photos = [
        _p("small", at=0, sharpness=5.0, size=(100, 100)),
        _p("big", at=2, sharpness=5.0, size=(400, 400)),
    ]
    assert kept_hashes(collapse(photos)[0]) == ["big"]


def test_the_file_hash_breaks_a_full_tie():
    """The last tiebreak is the file CONTENT. Anything else - path, mtime,
    insertion order - gives different survivors on different machines and
    breaks the byte-for-byte reproducibility promise."""
    photos = [_p("zzz", at=0), _p("aaa", at=2)]
    assert kept_hashes(collapse(photos)[0]) == ["aaa"]


def test_null_dimensions_sort_last_and_do_not_crash():
    """All 1,117 video rows and any photo with dimension-less EXIF. An
    unknown size must never beat a known one."""
    photos = [
        _p("unknown", at=0, sharpness=5.0, size=None),
        _p("known", at=2, sharpness=5.0, size=(10, 10)),
    ]
    assert kept_hashes(collapse(photos)[0]) == ["known"]


def test_a_photo_with_no_sharpness_loses_to_one_with_any():
    photos = [_p("unmeasured", at=0, sharpness=None), _p("measured", at=2, sharpness=0.0)]
    assert kept_hashes(collapse(photos)[0]) == ["measured"]


# --------------------------------------------------------------------------
# determinism and accounting


def test_output_is_identical_under_input_shuffling():
    """The engine's central promise: same library in, same memories out. Any
    dependence on input order breaks it, and input order here comes from a
    SQLite scan whose stability is not guaranteed."""
    photos = [_p(f"p{i:02d}", at=i * 3, phash=H_A if i % 4 else H_FAR) for i in range(24)]
    baseline = kept_hashes(collapse(photos)[0])

    for seed in range(5):
        shuffled = photos[:]
        random.Random(seed).shuffle(shuffled)
        assert sorted(kept_hashes(collapse(shuffled)[0])) == sorted(baseline)


def test_survivors_keep_the_callers_order():
    """Dedup runs mid-pipeline; the recipe already chose an order and this
    stage must not silently re-sort it."""
    photos = [_p("c", at=100), _p("a", at=0), _p("b", at=3600)]
    assert kept_hashes(collapse(photos)[0]) == ["c", "a", "b"]


def test_everything_is_accounted_for():
    photos = [_p(f"p{i}", at=i * 2, phash=H_A) for i in range(5)] + [_p("far", at=99, phash=H_FAR)]
    kept, report = collapse(photos)
    assert report.accounted
    assert report.considered == 6
    assert report.kept == len(kept)


def test_an_empty_list_is_handled():
    kept, report = collapse([])
    assert kept == []
    assert report.accounted


def test_photos_sharing_a_timestamp_are_grouped_stably():
    """1,430 timestamps in the reference library are shared by more than one
    photo (one by 40), because Takeout dates are second-granular.

    Bursts are CONTIGUOUS runs of the sorted sequence, so with equal
    timestamps the file_hash tiebreak alone decides who is adjacent to whom -
    and therefore what groups exist at all. Sorted by hash the order is
    a(near) b(far) c(near), so `b` splits the two near-identical photos and
    all three survive. The assertion that matters is that this is STABLE: the
    same three photos in any input order give the same answer.
    """
    photos = [_p("b", at=0, phash=H_FAR), _p("a", at=0, phash=H_A), _p("c", at=0, phash=H_A)]
    expected = [["a"], ["b"], ["c"]]
    for seed in range(4):
        shuffled = photos[:]
        random.Random(seed).shuffle(shuffled)
        assert [[p.file_hash for p in g] for g in bursts(shuffled)] == expected


def test_the_threshold_is_configurable():
    photos = [_p("a", at=0, phash=0b0), _p("b", at=2, phash=0b1111_1111)]  # 8 apart
    assert len(collapse(photos)[0]) == 2
    assert len(collapse(photos, threshold=8)[0]) == 1


def test_the_gap_is_configurable():
    photos = [_p("a", at=0), _p("b", at=60)]
    assert len(collapse(photos)[0]) == 2
    assert len(collapse(photos, gap_seconds=120)[0]) == 1
