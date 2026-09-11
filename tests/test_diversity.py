"""Diversity: content dissimilarity as a selection constraint.

The defect: selection ranked by quality and took the top N, so three good
photos of the same child on the same afternoon each won on their own merits
and the viewer saw the same picture three times. Measured on the rendered
output - `album_story-wedding_arnab_pics` spent 12 of 24 shots on one day, and
`album_story-avyan` had 3 shots of Avyan on 2025-09-17.

The criterion is CONTENT, not the calendar. These tests pin that: a single-day
memory of genuinely different moments is fine, and two near-identical photos
four hours apart are not.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rekindle.memory import diversity as dv
from rekindle.memory.diversity import (
    ColourSignal,
    CompositeSignal,
    DissimilaritySignal,
    PerceptualSignal,
    decode_colour,
    pick,
)
from rekindle.models import MediaType, Photo, PhotoMeta

T0 = datetime(2025, 8, 4, 10, 0, tzinfo=UTC)


def _hist(*, dominant: int = 0, spread: int = 1) -> str:
    """A 64-bin histogram with mass on `dominant` and its `spread` neighbours."""
    bins = [0] * 64
    for i in range(spread):
        bins[(dominant + i) % 64] = 255
    return "".join(f"{b:02x}" for b in bins)


def _p(name, *, phash=0, colour=None, at=0.0, sharp=5.0) -> Photo:
    return Photo(
        file_hash=name,
        paths=[Path(f"/lib/{name}.jpg")],
        media_type=MediaType.IMAGE,
        meta=PhotoMeta(
            taken_at_utc=T0 + timedelta(seconds=at),
            taken_at_local=(T0 + timedelta(seconds=at)).replace(tzinfo=None),
            phash=phash,
            colour=colour if colour is not None else _hist(dominant=0),
            sharpness=sharp,
            width=4000,
            height=3000,
        ),
        first_seen=T0,
        last_seen=T0,
    )


def _rank(photos):
    return sorted(photos, key=lambda p: (-(p.meta.sharpness or 0), p.file_hash))


# --------------------------------------------------------------------------
# the signals


def test_identical_photos_are_maximally_similar():
    a = _p("a", phash=0b1010, colour=_hist(dominant=8))
    b = _p("b", phash=0b1010, colour=_hist(dominant=8))
    assert CompositeSignal().between(a, b) == pytest.approx(0.0)


def test_structurally_different_photos_are_dissimilar():
    a = _p("a", phash=0, colour=_hist(dominant=0))
    b = _p("b", phash=(1 << 40) - 1, colour=_hist(dominant=40))
    assert CompositeSignal().between(a, b) > 0.9


def test_the_perceptual_signal_saturates_at_the_radius():
    near = _p("n", phash=0b111)  # 3 bits from 0
    far = _p("f", phash=(1 << 40) - 1)  # 40 bits from 0
    base = _p("b", phash=0)
    signal = PerceptualSignal()
    assert signal.between(base, near) == pytest.approx(3 / dv.PHASH_RADIUS)
    assert signal.between(base, far) == 1.0


def test_a_signal_ABSTAINS_rather_than_guessing_when_data_is_missing():
    """None is not 0 and not 1. It means this signal cannot judge, so the
    others decide - which is what stops one un-fingerprinted photo
    suppressing its neighbours."""
    a = _p("a", phash=None)
    b = _p("b", phash=5)
    assert PerceptualSignal().between(a, b) is None


def test_the_composite_falls_back_to_whichever_signal_can_judge():
    """A photo with a hash but no colour histogram is still judged."""
    a = _p("a", phash=0, colour=None)
    b = _p("b", phash=0, colour=None)
    a.meta.colour = None
    b.meta.colour = None
    assert ColourSignal().between(a, b) is None
    assert CompositeSignal().between(a, b) == pytest.approx(0.0)


def test_the_composite_abstains_when_nothing_can_judge():
    a = _p("a", phash=None)
    b = _p("b", phash=None)
    a.meta.colour = b.meta.colour = None
    assert CompositeSignal().between(a, b) is None


def test_a_malformed_colour_value_makes_the_signal_abstain_not_raise():
    a = _p("a", colour="not hex at all!")
    b = _p("b")
    assert ColourSignal().between(a, b) is None
    assert decode_colour("zz") is None
    assert decode_colour("abc") is None
    assert decode_colour(None) is None
    assert decode_colour("00" * 64) is None  # all-zero has no distribution


def test_the_colour_signal_separates_different_distributions():
    a = _p("a", colour=_hist(dominant=0))
    b = _p("b", colour=_hist(dominant=40))
    assert ColourSignal().between(a, b) == pytest.approx(1.0)


def test_the_signals_satisfy_the_protocol():
    """The seam an embedding implementation will plug into."""
    for signal in (PerceptualSignal(), ColourSignal(), CompositeSignal()):
        assert isinstance(signal, DissimilaritySignal)
        assert isinstance(signal.name, str) and signal.name
        assert 0.0 < signal.weight <= 1.0


def test_a_custom_signal_can_be_substituted():
    """The whole point of the seam: the semantic milestone must be able to add
    an implementation rather than rewrite selection."""

    class AlwaysIdentical:
        name = "fake"
        weight = 1.0

        def between(self, a, b):
            return 0.0

    assert isinstance(AlwaysIdentical(), DissimilaritySignal)
    photos = [_p(f"p{i}", phash=1 << i, at=i * 3600) for i in range(6)]
    picked, report = pick(photos, 4, rank=_rank, signal=AlwaysIdentical())
    # Everything looks identical to the fake signal, so every candidate after
    # the first is refused - and three are then restored, because a memory
    # must not be left short.
    assert len(picked) == 4
    assert report.rejected[dv.REJECT_TOO_SIMILAR] == 5
    assert report.restored == 3


# --------------------------------------------------------------------------
# the pick


def test_near_identical_photos_are_not_both_chosen():
    """The complaint, in miniature: two shots of the same scene, hours apart,
    both high quality. Only one should appear."""
    photos = [
        _p("twin_a", phash=0b1111, colour=_hist(dominant=1), at=0, sharp=9.0),
        _p("twin_b", phash=0b1111, colour=_hist(dominant=1), at=4 * 3600, sharp=8.9),
        _p("other", phash=(1 << 40) - 1, colour=_hist(dominant=40), at=7200, sharp=3.0),
    ]
    picked, _ = pick(photos, 2, rank=_rank)
    names = {p.file_hash for p in picked}
    assert "other" in names, "a worse but DIFFERENT photo must beat a duplicate"
    assert not {"twin_a", "twin_b"} <= names


def test_a_duplicate_is_REJECTED_when_it_would_otherwise_be_reached():
    """The graded penalty and the hard floor are different mechanisms.

    Above, the twin simply LOSES - it is never reached, so nothing is
    "rejected". Here there are more twins than slots, so the floor fires and
    the report says so. Asserting a rejection count in the losing case would
    have been asserting the wrong mechanism.
    """
    photos = [
        _p(f"twin{i}", phash=0b1111, colour=_hist(dominant=1), at=i * 3600, sharp=9.0 - i)
        for i in range(4)
    ]
    photos.append(_p("other", phash=(1 << 40) - 1, colour=_hist(dominant=40), sharp=1.0))
    picked, report = pick(photos, 2, rank=_rank)
    assert {p.file_hash for p in picked} == {"twin0", "other"}
    assert report.rejected[dv.REJECT_TOO_SIMILAR] == 1


def test_the_calendar_is_NOT_the_criterion(tmp_path):
    """A single-day memory of genuinely different moments is fine.

    This is why the per-day cap that was tried first is gone: it would have
    gutted a one-day album like `Diwali Kali Puja 22` while still permitting
    two near-identical photos taken four hours apart.
    """
    photos = [
        _p(f"m{i}", phash=(1 << (i * 7)), colour=_hist(dominant=i * 7), at=i * 600)
        for i in range(8)
    ]
    picked, report = pick(photos, 8, rank=_rank)
    assert len(picked) == 8
    assert report.total_rejected == 0


def test_a_worse_but_different_photo_beats_a_better_duplicate():
    best = _p("best", phash=0, colour=_hist(dominant=0), sharp=10.0)
    clone = _p("clone", phash=0, colour=_hist(dominant=0), sharp=9.9)
    different = _p("different", phash=(1 << 40) - 1, colour=_hist(dominant=40), sharp=1.0)
    picked, _ = pick([best, clone, different], 2, rank=_rank)
    assert {p.file_hash for p in picked} == {"best", "different"}


def test_diversity_applies_ACROSS_buckets_via_already():
    """Otherwise filling a year's slot could still hand back a near-identical
    shot from a day already represented in another bucket."""
    chosen = [_p("earlier", phash=0b1111, colour=_hist(dominant=1))]
    candidates = [
        _p("twin", phash=0b1111, colour=_hist(dominant=1), sharp=9.0),
        _p("fresh", phash=(1 << 40) - 1, colour=_hist(dominant=40), sharp=2.0),
    ]
    picked, _ = pick(candidates, 1, rank=_rank, already=chosen)
    assert [p.file_hash for p in picked] == ["fresh"]


def test_a_memory_is_never_left_SHORT_by_diversity():
    """A slightly repetitive memory beats a truncated one. The rejection stays
    in the report, so the restoration is visible rather than silent."""
    photos = [_p(f"same{i}", phash=0, colour=_hist(dominant=0), sharp=9.0 - i) for i in range(5)]
    picked, report = pick(photos, 4, rank=_rank)
    assert len(picked) == 4
    assert report.rejected[dv.REJECT_TOO_SIMILAR] == 4
    assert report.restored == 3


def test_restored_shots_are_the_best_of_the_refused():
    photos = [_p(f"s{i}", phash=0, colour=_hist(dominant=0), sharp=float(9 - i)) for i in range(5)]
    picked, _ = pick(photos, 3, rank=_rank)
    assert [p.file_hash for p in picked] == ["s0", "s1", "s2"]


def test_the_first_pick_is_decided_purely_on_quality():
    photos = [_p("mid", sharp=5.0), _p("best", sharp=9.0), _p("worst", sharp=1.0)]
    picked, _ = pick(photos, 1, rank=_rank)
    assert picked[0].file_hash == "best"


def test_time_is_a_weak_TIEBREAK(monkeypatch):
    """It breaks ties between ADJACENT ranks and nothing more.

    An earlier version of this test used two candidates and was vacuous: with
    two, the quality gap is 0.5 and the bonus is at most 0.05, so the top rank
    always wins and removing the bonus entirely changed nothing.

    The bonus can only matter when the quality step is smaller than it -
    i.e. more than 20 candidates. Here there are 24, so the step is 1/24 =
    0.042 and a temporally distant runner-up can overtake the leader. The same
    call with the bonus zeroed must choose differently, which is what proves
    the constant is doing the work.
    """
    chosen = [_p("anchor", phash=0, colour=_hist(dominant=0), at=0)]
    # Identical structure and colour, so dissimilarity is equal for all and
    # only quality and the time bonus separate them. Equal sharpness means
    # `_rank` orders them by name: n00 first, n01 second.
    candidates = [
        _p(f"n{i:02d}", phash=1 << 30, colour=_hist(dominant=30), at=60, sharp=5.0)
        for i in range(24)
    ]
    # The runner-up is five days away; everything else is a minute away.
    candidates[1] = _p("n01", phash=1 << 30, colour=_hist(dominant=30), at=5 * 86400, sharp=5.0)

    with_bonus, _ = pick(candidates, 1, rank=_rank, already=chosen)
    monkeypatch.setattr(dv, "TIME_TIEBREAK", 0.0)
    without_bonus, _ = pick(candidates, 1, rank=_rank, already=chosen)

    assert with_bonus[0].file_hash == "n01", "the time bonus did not break the tie"
    assert without_bonus[0].file_hash == "n00"


def test_time_never_REFUSES_anything():
    """It is a tiebreak, not a cap. Two photos a minute apart that LOOK
    different must both be available."""
    photos = [
        _p("a", phash=0, colour=_hist(dominant=0), at=0),
        _p("b", phash=(1 << 40) - 1, colour=_hist(dominant=40), at=60),
    ]
    picked, report = pick(photos, 2, rank=_rank)
    assert len(picked) == 2
    assert report.total_rejected == 0


def test_the_report_accounts_for_every_candidate():
    photos = [_p(f"p{i}", phash=1 << i, colour=_hist(dominant=i)) for i in range(6)]
    _, report = pick(photos, 3, rank=_rank)
    assert report.considered == 6
    assert report.picked == 3


def test_picking_is_deterministic_under_input_shuffling():
    import random

    photos = [
        _p(f"p{i:02d}", phash=(1 << (i % 50)), colour=_hist(dominant=i % 60), sharp=float(i % 7))
        for i in range(30)
    ]
    baseline = [p.file_hash for p in pick(photos, 8, rank=_rank)[0]]
    for seed in range(4):
        shuffled = photos[:]
        random.Random(seed).shuffle(shuffled)
        assert [p.file_hash for p in pick(shuffled, 8, rank=_rank)[0]] == baseline


def test_empty_and_zero_slot_cases():
    assert pick([], 5, rank=_rank)[0] == []
    assert pick([_p("a")], 0, rank=_rank)[0] == []


def test_an_unfingerprinted_photo_does_not_suppress_its_neighbours():
    """Unknown is not similar - the same rule dedup follows."""
    blank = _p("blank", phash=None)
    blank.meta.colour = None
    photos = [blank, _p("a", phash=1 << 5, colour=_hist(dominant=5))]
    picked, report = pick(photos, 2, rank=_rank)
    assert len(picked) == 2
    assert report.total_rejected == 0


# --------------------------------------------------------------------------
# gaps found by mutation


def test_the_PENALTY_itself_changes_which_photo_is_picked():
    """Without lambda, selection is plain top-N again and the whole module is
    inert. The earlier tests all happened to exercise the hard floor instead,
    so zeroing the penalty left them green.

    Here nothing crosses the floor - the twin is merely SIMILAR, not
    near-identical - so only the graded penalty can prefer the different
    photo.
    """
    photos = [
        _p("anchor", phash=0, colour=_hist(dominant=0), sharp=9.0),
        # Same colour distribution, 10 bits of structure away: composite
        # dissimilarity ~0.25 - clearly similar, but comfortably above the
        # 0.08 hard floor, so only the graded penalty can act on it.
        _p("similar", phash=0b1111111111, colour=_hist(dominant=0), sharp=8.0),
        _p("different", phash=(1 << 40) - 1, colour=_hist(dominant=40), sharp=7.0),
    ]
    with_penalty, _ = pick(photos, 2, rank=_rank)
    without_penalty, _ = pick(photos, 2, rank=_rank, lam=0.0)

    assert {p.file_hash for p in with_penalty} == {"anchor", "different"}
    assert {p.file_hash for p in without_penalty} == {"anchor", "similar"}


def test_similarity_is_measured_against_the_CLOSEST_chosen_photo():
    """Comparing against the furthest, or an average, would let a photo slip
    in because it differs from one pick while duplicating another."""
    chosen = [
        _p("far_anchor", phash=(1 << 40) - 1, colour=_hist(dominant=40)),
        _p("near_anchor", phash=0, colour=_hist(dominant=0)),
    ]
    candidates = [
        _p("twin_of_near", phash=0, colour=_hist(dominant=0), sharp=9.0),
        _p("novel", phash=(1 << 20) - 1, colour=_hist(dominant=20), sharp=1.0),
    ]
    picked, _ = pick(candidates, 1, rank=_rank, already=chosen)
    assert [p.file_hash for p in picked] == ["novel"]


def test_the_COLOUR_signal_contributes_to_the_decision():
    """Two photos with the SAME perceptual hash but different colour
    distributions - a change of room or of clothes with the same framing.
    Without the colour signal they are indistinguishable."""
    a = _p("a", phash=0b1010, colour=_hist(dominant=0))
    b = _p("b", phash=0b1010, colour=_hist(dominant=40))

    phash_only = PerceptualSignal().between(a, b)
    combined = CompositeSignal().between(a, b)

    assert phash_only == pytest.approx(0.0)
    assert combined > 0.3, "the colour signal is not reaching the composite"
