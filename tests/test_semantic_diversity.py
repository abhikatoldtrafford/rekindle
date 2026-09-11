"""The embedding as a diversity signal, and the trap it has to survive.

THE TRAP. A "durga puja over the years" memory is semantically homogeneous by
design: every photo is a Durga idol, so every pair has a high CLIP cosine.
`person_years` is one face repeatedly; an album story is one event. An
absolute threshold would reject the subject of the memory itself, and would do
it worse the better the memory is.

So the invariant these tests exist to pin is arithmetic rather than tuned:

    the MEDIAN pair of any pool maps to exactly 1.0 - no penalty at all -
    whatever its absolute cosine is.

Measured over the 399 candidate pools the reference library produces, the
per-pool median pairwise cosine runs from 0.460 to 0.930 and the 95th
percentile from 0.682 to 0.986. One number cannot mean the same thing in both,
which is why there is no number.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rekindle.models import MediaType, Photo, PhotoMeta

np = pytest.importorskip("numpy")

from rekindle.memory.diversity import (  # noqa: E402
    DEFAULT_SIGNAL,
    CompositeSignal,
    DissimilaritySignal,
)
from rekindle.semantic.diversity import (  # noqa: E402
    CEILING,
    MIN_SPREAD,
    SemanticSignal,
    _Cosines,
    calibrate,
)

T0 = datetime(2020, 1, 1, 12, 0, tzinfo=UTC)


def _p(name: str, *, at: float = 0.0) -> Photo:
    return Photo(
        file_hash=name,
        paths=[Path(f"/lib/{name}.jpg")],
        media_type=MediaType.IMAGE,
        meta=PhotoMeta(
            taken_at_utc=T0 + timedelta(seconds=at),
            taken_at_local=(T0 + timedelta(seconds=at)).replace(tzinfo=None),
            phash=0,
            sharpness=1.0,
            width=4000,
            height=3000,
        ),
        first_seen=T0,
        last_seen=T0,
    )


def _pool(cosine_matrix) -> tuple[list[Photo], _Cosines]:
    """Photos whose pairwise cosines are (very nearly) `cosine_matrix`.

    The fixture states the SIMILARITIES it wants rather than hand-picking
    vectors and hoping. A Gram matrix of unit vectors must be positive
    semi-definite and a hand-written similarity matrix need not be - several
    of the shapes below are not - so the nearest PSD matrix is taken by
    clipping the eigenvalues, and the rows are renormalised afterwards.

    That last step perturbs the result, which is why the fixture-trust tests
    below exist and state the tolerance out loud instead of assuming it is
    zero. A fixture nobody has checked is not evidence.
    """
    matrix = np.array(cosine_matrix, dtype="float64")
    values, directions = np.linalg.eigh(matrix)
    vectors = directions * np.sqrt(np.clip(values, 1e-6, None))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    names = tuple(f"p{i}" for i in range(len(matrix)))
    return [_p(n) for n in names], _Cosines(names, vectors.astype("float32"))


#: How far `_pool` may land from the matrix it was handed. See its docstring;
#: measured at up to 0.009 over the shapes used here.
FIXTURE_TOLERANCE = 0.01


def _uniform(n: int, value: float):
    m = np.full((n, n), value)
    np.fill_diagonal(m, 1.0)
    return m


def _graded(n: int, lo: float, hi: float):
    """A pool whose closest PAIR is `hi` and whose furthest is `lo`.

    Similarity falls off linearly with the gap between indices, so the pool
    has a real spread rather than two clumps - which is what a percentile is
    meant to describe.
    """
    gap = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    # Solved so that gap == 1 lands on `hi` and gap == n-1 on `lo`. The naive
    # version put `hi` on the diagonal, where no pair can reach it.
    slope = (hi - lo) * (n - 1) / (n - 2)
    m = (lo + slope) - (gap / (n - 1)) * slope
    np.fill_diagonal(m, 1.0)
    return m


# --------------------------------------------------------------------------
# the fixture itself must be trustworthy


@pytest.mark.parametrize("n,lo,hi", [(6, 0.40, 0.95), (9, 0.75, 0.99), (8, 0.20, 0.60)])
def test_the_fixture_spans_the_cosines_it_claims(n, lo, hi):
    """Every claim below rests on this. The first `_graded` put its highest
    value on the DIAGONAL, so no pair ever reached `hi` and the tests were
    quietly measuring a pool 0.11 less alike than they said they were."""
    photos, cosines = _pool(_graded(n, lo, hi))
    pairs = [cosines.between(a, b) for i, a in enumerate(photos) for b in photos[i + 1 :]]
    assert max(pairs) == pytest.approx(hi, abs=FIXTURE_TOLERANCE)
    assert min(pairs) == pytest.approx(lo, abs=FIXTURE_TOLERANCE)


def test_a_uniform_fixture_really_is_uniform():
    """The trap tests mean nothing if the "homogeneous" pool has a spread."""
    photos, cosines = _pool(_uniform(6, 0.93))
    pairs = [cosines.between(a, b) for i, a in enumerate(photos) for b in photos[i + 1 :]]
    assert max(pairs) - min(pairs) < FIXTURE_TOLERANCE
    assert max(pairs) == pytest.approx(0.93, abs=FIXTURE_TOLERANCE)


# --------------------------------------------------------------------------
# THE TRAP


@pytest.mark.parametrize("level", [0.95, 0.85, 0.70, 0.50])
def test_a_homogeneous_pool_penalises_nothing_at_any_absolute_similarity(level):
    """Every photo is the same kind of picture. That is what the memory IS.

    This is the durga-puja case reduced to its essentials, swept across the
    whole cosine range: whether the subject scores 0.95 or 0.50, a uniform
    pool has nothing that stands out, and the answer to "which of these is
    unusually alike?" is none of them.
    """
    photos, cosines = _pool(_uniform(6, level))
    signal = calibrate(photos, cosines)
    assert signal is not None
    for a in photos:
        for b in photos:
            if a is not b:
                assert signal.between(a, b) == pytest.approx(1.0)


def test_the_median_pair_is_never_penalised_however_alike_the_pool_is():
    """The invariant, stated as arithmetic. `low` IS the median, so the
    typical pair maps to exactly 1.0 by construction rather than by luck."""
    for lo, hi in ((0.30, 0.70), (0.75, 0.99), (0.88, 0.97)):
        photos, cosines = _pool(_graded(9, lo, hi))
        signal = calibrate(photos, cosines)
        pairs = sorted(cosines.between(a, b) for i, a in enumerate(photos) for b in photos[i + 1 :])
        median = float(np.percentile(pairs, 50))
        assert signal.low == pytest.approx(median, abs=1e-6)
        # A pair sitting exactly at this pool's median costs nothing.
        assert (signal.high - median) / (signal.high - signal.low) == pytest.approx(1.0)


def test_a_homogeneous_pool_and_a_varied_one_get_different_scales():
    """If one scale fitted both, the relative design would be pointless."""
    alike, alike_cos = _pool(_graded(8, 0.80, 0.98))
    varied, varied_cos = _pool(_graded(8, 0.20, 0.60))
    assert calibrate(alike, alike_cos).low > calibrate(varied, varied_cos).low


# --------------------------------------------------------------------------
# the two floors on `high`


def test_a_varied_pool_does_not_treat_a_near_random_pair_as_a_duplicate():
    """`on_this_day-11-21` has its 95th percentile at 0.682, and 0.682 is
    roughly the similarity of two RANDOM photographs in this library (median
    0.548, 99th percentile 0.787). Without the ceiling its most-alike pair
    would score the full penalty."""
    photos, cosines = _pool(_graded(8, 0.20, 0.60))
    signal = calibrate(photos, cosines)
    assert signal.high == pytest.approx(CEILING)
    # Nothing in a pool this varied is anywhere near maximally redundant.
    worst = min(signal.between(a, b) for i, a in enumerate(photos) for b in photos[i + 1 :])
    assert worst > 0.5


def test_a_uniform_pool_cannot_collapse_the_divisor():
    photos, cosines = _pool(_uniform(5, 0.99))
    signal = calibrate(photos, cosines)
    assert signal.high - signal.low >= MIN_SPREAD - 1e-9


def test_an_unusually_alike_pair_is_penalised_even_in_a_homogeneous_pool():
    """The signal is not inert on homogeneous sets - it still finds the pairs
    that stand out WITHIN them, which is the whole point of relativity."""
    matrix = _graded(8, 0.80, 0.92)
    matrix[0][1] = matrix[1][0] = 0.99
    photos, cosines = _pool(matrix)
    signal = calibrate(photos, cosines)
    assert signal.between(photos[0], photos[1]) < 0.35
    assert signal.between(photos[0], photos[4]) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# abstention and refusal to guess


def test_a_photo_with_no_vector_makes_the_signal_abstain():
    """None, not 0 and not 1: one un-embedded photo must not suppress its
    neighbours, and must not be declared different from everything either."""
    photos, cosines = _pool(_graded(5, 0.4, 0.9))
    signal = calibrate(photos, cosines)
    assert signal.between(photos[0], _p("never-embedded")) is None
    assert signal.between(_p("never-embedded"), photos[0]) is None


def test_a_pool_too_small_to_describe_a_distribution_gets_no_signal():
    """Two photos have exactly one pair, which would be its own median AND
    its own 95th percentile. Running on the pixel signals alone is the honest
    answer; inventing a scale from one number is not."""
    photos, cosines = _pool(_graded(4, 0.4, 0.9))
    assert calibrate(photos[:2], cosines) is None
    assert calibrate([], cosines) is None
    assert calibrate(photos[:3], cosines) is not None


def test_photos_outside_the_store_are_not_counted_towards_the_minimum():
    photos, cosines = _pool(_graded(4, 0.4, 0.9))
    mixed = [photos[0], photos[1], _p("x"), _p("y"), _p("z")]
    assert calibrate(mixed, cosines) is None


# --------------------------------------------------------------------------
# the seam, and determinism


def test_the_semantic_signal_answers_the_dissimilarity_protocol():
    photos, cosines = _pool(_graded(5, 0.4, 0.9))
    signal = calibrate(photos, cosines)
    assert isinstance(signal, DissimilaritySignal)
    assert isinstance(signal.name, str) and signal.name
    assert 0.0 < signal.weight <= 1.0
    # And drops into the composite without any change to it.
    composite = CompositeSignal(signals=(*DEFAULT_SIGNAL.signals, signal))
    assert isinstance(composite, DissimilaritySignal)
    assert composite.between(photos[0], photos[1]) is not None


def test_calibration_is_deterministic_and_order_independent():
    photos, cosines = _pool(_graded(9, 0.3, 0.95))
    first = calibrate(photos, cosines)
    second = calibrate(list(reversed(photos)), cosines)
    assert (first.low, first.high) == (second.low, second.high)


def test_a_large_pool_is_subsampled_by_a_fixed_stride_not_a_random_draw():
    """The same pool must calibrate to the same two numbers on every run, or
    the promise that one library gives one memory is gone."""
    from rekindle.semantic.diversity import CALIBRATION_ROWS

    n = CALIBRATION_ROWS + 50
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((n, 16)).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    names = tuple(f"p{i}" for i in range(n))
    cosines = _Cosines(names, vectors)
    photos = [_p(x) for x in names]
    runs = {(calibrate(photos, cosines).low, calibrate(photos, cosines).high) for _ in range(3)}
    assert len(runs) == 1


def test_a_pool_larger_than_the_precompute_limit_still_answers(monkeypatch):
    """Above the limit there is no cached matrix and `between` falls back to a
    dot product. The two paths must agree, or a big library would quietly get
    different memories from a small one."""
    from rekindle.semantic import diversity as sd

    photos, cached = _pool(_graded(6, 0.4, 0.95))
    monkeypatch.setattr(sd, "PRECOMPUTE_LIMIT", 1)
    bare = _Cosines(tuple(p.file_hash for p in photos), cached._data)
    assert bare._matrix is None, "the limit was not honoured"
    for i, a in enumerate(photos):
        for b in photos[i + 1 :]:
            assert bare.between(a, b) == pytest.approx(cached.between(a, b), abs=1e-6)


def test_the_signal_clamps_rather_than_returning_a_value_outside_the_range():
    """A pair below the pool's median maps above 1.0 before clamping, and one
    above `high` maps below 0. A dissimilarity outside [0, 1] would make the
    MMR penalty larger than the entire quality range."""
    a, b = _p("a"), _p("b")

    class _Stub:
        value = 0.0

        def between(self, x, y):
            return self.value

    stub = _Stub()
    signal = SemanticSignal(cosines=stub, low=0.5, high=0.9)
    stub.value = 0.1
    assert signal.between(a, b) == 1.0
    stub.value = 0.99
    assert signal.between(a, b) == 0.0
    stub.value = 0.7
    assert signal.between(a, b) == pytest.approx(0.5)
