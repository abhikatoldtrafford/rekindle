"""Stratified selection: the allocator and the adaptive granularity.

Unit-level counterpart to the end-to-end tests in test_engine.py. The shapes
come from the real library's actual failure cases - notably
`on_this_day:12-22`, whose surviving buckets are 2019 with 126 photos, 2020
with 2 and 2022 with 1.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rekindle.memory import strata
from rekindle.memory.strata import allocate, bucket_key, resolve_level, stratify
from rekindle.models import MediaType, Photo, PhotoMeta


def _p(name, local, sharp=5.0) -> Photo:
    return Photo(
        file_hash=name,
        paths=[Path(f"/lib/{name}.jpg")],
        media_type=MediaType.IMAGE,
        meta=PhotoMeta(
            taken_at_utc=local.replace(tzinfo=UTC),
            taken_at_local=local,
            width=4000,
            height=3000,
            sharpness=sharp,
        ),
        first_seen=datetime(2020, 1, 1, tzinfo=UTC),
        last_seen=datetime(2020, 1, 1, tzinfo=UTC),
    )


def _rank(photos):
    """Stand-in for the engine's ranker: sharpest first, hash breaking ties."""
    return sorted(photos, key=lambda p: (-(p.meta.sharpness or 0), p.file_hash))


# --------------------------------------------------------------------------
# allocation


def test_every_bucket_gets_one_before_any_gets_two():
    """The floor. This is what stops a year with one photo vanishing."""
    assert allocate({(2019,): 126, (2020,): 2, (2022,): 1}, 3) == {
        (2019,): 1,
        (2020,): 1,
        (2022,): 1,
    }


def test_the_real_on_this_day_shape():
    """The exact case from the library: 126/2/1 into 24 slots.

    Before stratification this produced 24/0/0. Every year must appear, and
    the rich one must still carry the memory.
    """
    result = allocate({(2019,): 126, (2020,): 2, (2022,): 1}, 24)
    assert sum(result.values()) == 24
    assert result[(2020,)] >= 1
    assert result[(2022,)] >= 1
    assert result[(2019,)] >= 20


def test_a_bucket_is_never_given_more_than_it_has():
    """Otherwise slots evaporate and the memory is short."""
    result = allocate({(2019,): 100, (2020,): 2}, 24)
    assert result[(2020,)] <= 2
    assert sum(result.values()) == 24


def test_slots_are_fully_used_when_there_is_capacity():
    for slots in (3, 5, 12, 24):
        result = allocate({(2019,): 50, (2020,): 40, (2021,): 30}, slots)
        assert sum(result.values()) == slots, slots


def test_the_allocation_never_exceeds_the_total_available():
    """Three photos total cannot fill 24 slots, and the allocator must not
    pretend otherwise or the caller indexes past the end of a bucket."""
    result = allocate({(2019,): 2, (2020,): 1}, 24)
    assert sum(result.values()) == 3


def test_more_buckets_than_slots_keeps_the_LARGEST():
    """A 26-year span into 5 shots. The years with the most photos win, and
    the rest are reported rather than silently dropped."""
    sizes = {(2000 + i,): i + 1 for i in range(26)}
    result = allocate(sizes, 5)
    assert len(result) == 5
    assert set(result) == {(2025,), (2024,), (2023,), (2022,), (2021,)}


def test_an_equal_split_is_NOT_what_happens():
    """Pinning the choice: a bucket with 2 photos must not get the same share
    as one with 126, or the memory wastes slots on the thin period."""
    result = allocate({(2019,): 126, (2020,): 2}, 12)
    assert result[(2019,)] > result[(2020,)]


def test_allocation_is_deterministic_under_dict_order():
    forward = allocate({(2019,): 10, (2020,): 5, (2021,): 3}, 9)
    backward = allocate({(2021,): 3, (2020,): 5, (2019,): 10}, 9)
    assert forward == backward


def test_equal_buckets_split_evenly():
    result = allocate({(2019,): 10, (2020,): 10, (2021,): 10}, 9)
    assert set(result.values()) == {3}


def test_empty_input_and_zero_slots():
    assert allocate({}, 10) == {}
    assert allocate({(2019,): 5}, 0) == {}
    assert allocate({(2019,): 0}, 5) == {}


# --------------------------------------------------------------------------
# adaptive granularity


def test_a_one_week_trip_is_split_by_DAY():
    """510 Kashmir photos all in May 2015: only days separate them, so a
    year- or month-level split would be a single bucket and no spread."""
    photos = [_p(f"k{i}", datetime(2015, 5, 15, 9, 0) + timedelta(days=i % 8)) for i in range(40)]
    assert resolve_level(photos, strata.BY_SPAN, 24) == strata.BY_DAY


def test_a_multi_year_album_is_split_by_MONTH():
    """19 months of a child's life: too many days for 24 slots, so months."""
    photos = [_p(f"a{i}", datetime(2024, 3, 1) + timedelta(days=i * 9)) for i in range(60)]
    assert resolve_level(photos, strata.BY_SPAN, 24) == strata.BY_MONTH


def test_a_very_long_span_falls_back_to_YEAR():
    photos = [_p(f"y{i}", datetime(2000 + i % 26, (i % 12) + 1, (i % 28) + 1)) for i in range(600)]
    assert resolve_level(photos, strata.BY_SPAN, 24) == strata.BY_YEAR


def test_a_fixed_level_is_returned_unchanged():
    photos = [_p("a", datetime(2020, 5, 1))]
    assert resolve_level(photos, strata.BY_YEAR, 24) == strata.BY_YEAR
    assert resolve_level(photos, strata.BY_MONTH, 24) == strata.BY_MONTH
    assert resolve_level(photos, None, 24) is None


def test_bucket_keys_use_LOCAL_time():
    """13,116 rows in this library have a non-UTC local zone. Bucketing on the
    UTC instant would file a 00:30 photo under the previous day."""
    photo = _p("a", datetime(2020, 5, 1, 0, 30))
    photo.meta.taken_at_utc = datetime(2020, 4, 30, 19, 0, tzinfo=UTC)
    assert bucket_key(photo, strata.BY_DAY) == (2020, 5, 1)


# --------------------------------------------------------------------------
# stratify()


def test_stratify_spreads_and_ranks_within_each_bucket():
    photos = [_p(f"a{i}", datetime(2019, 1, 1), sharp=float(i)) for i in range(10)]
    photos += [_p(f"b{i}", datetime(2021, 1, 1), sharp=float(i)) for i in range(10)]

    chosen, report = stratify(photos, level=strata.BY_YEAR, slots=4, rank=_rank)

    years = {p.meta.taken_at_local.year for p in chosen}
    assert years == {2019, 2021}
    # The sharpest of each bucket is in, the dullest is not.
    assert "a9" in {p.file_hash for p in chosen}
    assert "a0" not in {p.file_hash for p in chosen}


def test_stratify_with_no_level_just_ranks():
    """`then_and_now` relies on this: no spread, only the recipe's own
    choice."""
    photos = [_p(f"a{i}", datetime(2019 + i, 1, 1), sharp=float(i)) for i in range(5)]
    chosen, report = stratify(photos, level=None, slots=2, rank=_rank)
    assert [p.file_hash for p in chosen] == ["a4", "a3"]
    assert report.dimension is None


def test_stratify_reports_buckets_the_GATES_removed():
    """A year whose every photo failed the quality gates legitimately cannot
    contribute - but it must be VISIBLE, or a lopsided memory looks like a
    bug."""
    offered = [_p(f"o{y}", datetime(y, 12, 22)) for y in (2015, 2018, 2019, 2020)]
    surviving = [p for p in offered if p.meta.taken_at_local.year in (2019, 2020)]

    _, report = stratify(surviving, level=strata.BY_YEAR, slots=24, rank=_rank, offered=offered)

    assert report.offered == 4
    assert report.surviving == 2
    assert report.lost_to_gates == 2
    assert report.keys_lost_to_gates == ["2015", "2018"]


def test_stratify_reports_buckets_that_got_no_slot():
    photos = [_p(f"y{y}", datetime(y, 1, 1)) for y in range(2000, 2010)]
    _, report = stratify(photos, level=strata.BY_YEAR, slots=3, rank=_rank)
    assert report.surviving == 10
    assert report.used == 3
    assert report.unslotted == 7
    assert len(report.keys_without_slots) == 7


def test_stratify_returns_at_most_the_slot_count():
    photos = [_p(f"a{i}", datetime(2019 + i % 3, 1, 1)) for i in range(30)]
    chosen, _ = stratify(photos, level=strata.BY_YEAR, slots=7, rank=_rank)
    assert len(chosen) == 7


def test_stratify_is_deterministic_under_input_shuffling():
    import random

    photos = [_p(f"a{i:02d}", datetime(2019 + i % 4, 1, 1), sharp=float(i % 7)) for i in range(40)]
    baseline = [p.file_hash for p in stratify(photos, level=strata.BY_YEAR, slots=9, rank=_rank)[0]]
    for seed in range(4):
        shuffled = photos[:]
        random.Random(seed).shuffle(shuffled)
        again = [
            p.file_hash for p in stratify(shuffled, level=strata.BY_YEAR, slots=9, rank=_rank)[0]
        ]
        assert again == baseline


def test_stratify_on_an_empty_list():
    chosen, report = stratify([], level=strata.BY_YEAR, slots=5, rank=_rank)
    assert chosen == []
    assert report.used == 0


@pytest.mark.parametrize("level", [strata.BY_YEAR, strata.BY_MONTH, strata.BY_DAY, None])
def test_stratify_never_returns_a_duplicate(level):
    photos = [_p(f"a{i:02d}", datetime(2019 + i % 3, (i % 12) + 1, 1)) for i in range(30)]
    chosen, _ = stratify(photos, level=level, slots=12, rank=_rank)
    assert len({p.file_hash for p in chosen}) == len(chosen)


# --------------------------------------------------------------------------
# diversity applies across buckets, not only within one


def test_diversity_reaches_ACROSS_buckets(tmp_path):
    """Otherwise filling a year's slot could hand back a near-identical shot
    from a day already represented in another year's slot.

    Bucket 2019 holds one distinctive photo. Bucket 2020's best candidate is
    its near-identical twin; its second choice is different. The twin must
    lose, which can only happen if the 2020 pick can see what 2019 chose.
    """
    photos = [
        _p("a2019", datetime(2019, 6, 1), sharp=9.0),
        _p("twin2020", datetime(2020, 6, 1), sharp=9.0),
        _p("fresh2020", datetime(2020, 6, 2), sharp=1.0),
    ]
    photos[0].meta.phash = 0
    photos[0].meta.colour = "ff" + "00" * 63
    photos[1].meta.phash = 0
    photos[1].meta.colour = "ff" + "00" * 63
    photos[2].meta.phash = (1 << 40) - 1
    photos[2].meta.colour = "00" * 40 + "ff" + "00" * 23

    chosen, _ = stratify(photos, level=strata.BY_YEAR, slots=2, rank=_rank)

    assert {p.file_hash for p in chosen} == {"a2019", "fresh2020"}


def test_the_allocation_is_BINDING_diversity_never_reallocates(tmp_path):
    """Stratification decides the SHAPE of the memory; diversity decides what
    fills each slot. A bucket never loses its slot to another because its
    photos happen to look alike."""
    photos = [_p(f"a{i}", datetime(2019, 6, 1, 9, i), sharp=9.0) for i in range(6)]
    photos += [_p("lonely", datetime(2022, 6, 1), sharp=1.0)]
    for photo in photos[:6]:
        photo.meta.phash = 0
        photo.meta.colour = "ff" + "00" * 63
    photos[-1].meta.phash = (1 << 40) - 1
    photos[-1].meta.colour = "00" * 40 + "ff" + "00" * 23

    chosen, report = stratify(photos, level=strata.BY_YEAR, slots=4, rank=_rank)

    years = {p.meta.taken_at_local.year for p in chosen}
    assert 2022 in years, "the thin bucket lost its slot to the rich one"
    assert len(chosen) == 4, "diversity left the memory short"
    assert report.diversity.restored >= 1
