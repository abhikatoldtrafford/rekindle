"""Embedding distance as a diversity signal, calibrated per memory.

THE TRAP, STATED FIRST
----------------------
**A "durga puja over the years" memory is semantically homogeneous by
design.** Every photo is a Durga idol, so every pair has a high CLIP cosine.
`person_years` is one face, repeatedly. `then_and_now` is one subject twice.
An album story of one event is one event.

An absolute cutoff - "reject any pair above 0.90" - would therefore reject the
subject of the memory itself, and would do it *worse the better the memory
is*: the more faithfully a recipe found photos of the thing the user asked
for, the more of them an absolute rule would throw away. That failure is
silent, it looks like a thin library rather than a broken rule, and it is the
single most likely way to get this wrong.

Measured, over the 399 candidate pools this library actually produces:

    pool median pairwise cosine    min 0.460   median 0.683   max 0.930
    pool 95th percentile           min 0.682   median 0.899   max 0.986

A pair at 0.90 is the *ninetieth percentile* of `on_this_day-11-27` and it is
**above the 99th** of `on_this_day-11-21`. One number cannot mean the same
thing in both. So the threshold is not a number, it is a **position in this
memory's own distribution**.

THE MAPPING
-----------
    low   = median of this pool's own pairwise cosines
    high  = max(95th percentile of the same, 0.90, low + 0.10)

    dissimilarity(a, b) = clamp01((high - cosine(a, b)) / (high - low))

The first line is the whole design. `low` is the median, so **the typical pair
of a memory always maps to exactly 1.0 and is never penalised at all** - that
is arithmetic, not tuning, and it holds whatever the absolute cosines are. A
pool of nothing but Durga idols has a typical idol pair; it costs nothing. Only
the pairs that are unusually alike *for this pool* move down the scale, and
only the ones at or above its own 95th percentile approach 0.

WHY `high` HAS TWO FLOORS
-------------------------
Both stop a purely relative rule from being absurd at an extreme, and neither
can touch the invariant above.

`0.90`, because a genuinely diverse pool can have its 95th percentile at 0.682
- `on_this_day-11-21` does - and without a floor a 0.68 pair there would be
declared maximally redundant when 0.68 is roughly the similarity of two
*random* photographs (the random median here is 0.548, the random 99th
percentile 0.787). Below 0.90 you have not earned the full penalty.

`low + 0.10`, because a pool can be so uniform that its median and its 95th
percentile coincide. Nothing stands out there - that is what uniform means -
and the divisor would otherwise collapse and amplify float noise into a
verdict. With the floor, a uniform pool penalises nothing, which is the honest
answer to "which of these is unusually alike?" when the answer is none.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It under-reacts to a pool that is mostly duplicates, because there the median
pair IS a duplicate and the rule calls duplicates typical. That is a real cost
and it is the right way round: the alternative miscalibrates every homogeneous
memory in the library to catch a few saturated ones, and the saturated case
already has a guardrail that is allowed to be absolute - burst dedup, inside
its 30-second window, where the question is factual. See `memory.dedup`.

WHAT THIS SIGNAL MAY NOT DO
---------------------------
It may not refuse a photo. `memory.diversity.pick` takes it as the ADVISORY
signal and keeps the pixel signals as the binding one, so the worst this can
do to a photograph is move it down the order. See that module for the
argument; it is the same trap, and the relative mapping alone is not a strong
enough guarantee against it to be the only one.

NUMPY IS IMPORTED LAZILY, like everywhere else under `rekindle.semantic`:
`tests/test_semantic_imports.py` pins that this module can be imported on a
default install, and `memory/` never imports it at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rekindle.config import CATALOGUE, active
from rekindle.memory.diversity import DissimilaritySignal
from rekindle.models import Photo
from rekindle.semantic.store import EmbeddingStore

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

#: `high` never falls below this. A pool whose own 95th percentile is lower
#: than this is a genuinely varied one, and nothing in it has earned the full
#: penalty - see the module docstring.
CEILING = CATALOGUE["semantic_diversity.ceiling"].default
#: `high` is never closer to `low` than this, so the divisor cannot collapse
#: on a uniform pool. 0.10 is a tenth of the whole cosine range these vectors
#: occupy in practice (random 0.55, same-moment 0.95).
MIN_SPREAD = CATALOGUE["semantic_diversity.min_spread"].default
#: Quantiles read off the pool. The median is "a typical pair here", and it is
#: the anchor of the scale; the 95th is "as alike as this memory's own most
#: alike pairs".
LOW_QUANTILE = CATALOGUE["semantic_diversity.low_quantile"].default
HIGH_QUANTILE = CATALOGUE["semantic_diversity.high_quantile"].default

#: How much of the candidate pool the calibration looks at. 600 rows is
#: 179,700 pairs, already far more than any quantile needs, and it bounds the
#: cost on a `year_in_review` pool of several thousand. The subsample is a
#: fixed stride, never a random draw: the same pool must calibrate to the same
#: two numbers on every run.
CALIBRATION_ROWS = 600

#: How much weight the embedding carries against the two pixel signals, whose
#: weights sum to 1.0. Equal billing: it is better than either at the question
#: they are all being asked, and worse than both at being sure.
WEIGHT = CATALOGUE["semantic_diversity.weight"].default

#: Pools smaller than this get their full cosine matrix precomputed - it is
#: what makes `between` a dict lookup instead of a dot product inside the
#: MMR loop. 2,000 rows is 16 MB of float32, and no pool in this library
#: comes close.
PRECOMPUTE_LIMIT = 2000


def _numpy():
    from rekindle.semantic.vectors import _numpy as load

    return load()


@dataclass(frozen=True)
class SemanticSignal:
    """`DissimilaritySignal` over CLIP cosines, calibrated to one pool.

    Abstains - returns None - for any photo without a vector, which is what
    keeps a video or an un-embedded file from suppressing its neighbours. See
    the module docstring for `low` and `high`.
    """

    cosines: _Cosines
    low: float
    high: float
    name: str = "semantic"
    #: Resolved by `calibrate`, which is the only thing that builds one
    #: of these in the pipeline. A default argument would freeze the
    #: shipped weight at import.
    weight: float = WEIGHT

    def between(self, a: Photo, b: Photo) -> float | None:
        value = self.cosines.between(a, b)
        if value is None:
            return None
        span = self.high - self.low
        # Guaranteed >= MIN_SPREAD by `calibrate`, but a caller can construct
        # this class directly and a zero divisor would be a crash in a render.
        if span <= 0.0:  # pragma: no cover - unreachable via `calibrate`
            return 1.0 if value < self.high else 0.0
        return max(0.0, min(1.0, (self.high - value) / span))


class _Cosines:
    """Cosine between any two photos of one store, or None.

    Vectors are L2-normalised once on load, so a cosine is a dot product. The
    store's own `normalized` flag is not trusted: renormalising a unit vector
    costs one pass and removes a whole class of silent wrongness.
    """

    def __init__(self, hashes: tuple[str, ...], data: np.ndarray) -> None:
        self._row = {h: i for i, h in enumerate(hashes)}
        self._data = data
        self._matrix: np.ndarray | None = None
        if len(hashes) <= PRECOMPUTE_LIMIT:
            self._matrix = data @ data.T

    def __len__(self) -> int:
        return len(self._row)

    def between(self, a: Photo, b: Photo) -> float | None:
        i = self._row.get(a.file_hash)
        j = self._row.get(b.file_hash)
        if i is None or j is None:
            return None
        if self._matrix is not None:
            return float(self._matrix[i, j])
        return float(self._data[i] @ self._data[j])

    def pairwise(self, rows: list[int]) -> np.ndarray:
        np = _numpy()
        block = self._data[rows]
        sim = block @ block.T
        return sim[np.triu_indices(len(rows), 1)]

    def rows_for(self, hashes: list[str]) -> list[int]:
        return [self._row[h] for h in hashes if h in self._row]


class Support:
    """`memory.diversity.SemanticSupport` backed by a real embedding store.

    Built once per `rekindle memory` run: loading the matrix is 60 MB off disk
    and every memory in the batch shares it.
    """

    def __init__(self, store: EmbeddingStore) -> None:
        from rekindle.semantic.vectors import load_matrix, normalise

        matrix = load_matrix(store)
        self._cosines = _Cosines(matrix.hashes, normalise(matrix.data))

    def __len__(self) -> int:
        return len(self._cosines)

    def cosine(self, a: Photo, b: Photo) -> float | None:
        """The raw number, for burst dedup. No calibration: inside a 30-second
        window the question is factual and the absolute value is the answer."""
        return self._cosines.between(a, b)

    def signal_for(self, photos: list[Photo]) -> DissimilaritySignal | None:
        return calibrate(photos, self._cosines)


def calibrate(photos: list[Photo], cosines: _Cosines) -> SemanticSignal | None:
    """A signal scaled to this pool, or None when the pool cannot scale one.

    None - not a default scale - when fewer than three photos of the pool have
    vectors, because two points do not describe a distribution and one pair's
    own cosine would become both its median and its 95th percentile. The
    caller then runs on the pixel signals alone, which is the correct
    behaviour for an un-embedded corner of a library rather than a degenerate
    one.
    """
    np = _numpy()
    rows = cosines.rows_for([p.file_hash for p in photos])
    if len(rows) < 3:
        return None
    if len(rows) > CALIBRATION_ROWS:
        rows = rows[:: (len(rows) + CALIBRATION_ROWS - 1) // CALIBRATION_ROWS]
    pairs = cosines.pairwise(rows)
    # `low` is the median and nothing is allowed to move it: that is what
    # makes a typical pair cost exactly nothing, in every memory, whatever its
    # subject. The floors go on `high`, where they cannot reach the invariant.
    cfg = active().semantic_diversity
    low = float(np.percentile(pairs, cfg.low_quantile))
    high = max(float(np.percentile(pairs, cfg.high_quantile)), cfg.ceiling, low + cfg.min_spread)
    return SemanticSignal(cosines=cosines, low=low, high=high, weight=cfg.weight)


def open_support(data_dir, model_key: str | None = None) -> Support | None:
    """A `Support` for this library, or None when there is nothing to support.

    None covers every "no embeddings here" case - the extra not installed, the
    store never written, the store empty - because to this caller they are one
    outcome: run on pixels. The CLI is what tells the user WHICH of them it
    was, since only there is a sentence naming a command useful.
    """
    from pathlib import Path

    from rekindle.semantic.availability import probe
    from rekindle.semantic.registry import embed_model
    from rekindle.semantic.store import store_root

    if not probe().any:
        return None
    spec = embed_model(model_key)
    root = store_root(Path(data_dir), spec.key)
    if not (root / "manifest.sqlite").is_file():
        return None
    store = EmbeddingStore(root, dim=spec.dim, model_key=spec.key)
    try:
        if store.count() < 3:
            return None
        return Support(store)
    finally:
        store.close()
