"""Bulk vector maths over an `EmbeddingStore`. The only module that needs numpy.

Brute force, on purpose. 19,480 x 768 float32 is 60 MB; a query is one
matrix-vector product and takes about 12 ms on this CPU. An ANN index (faiss,
hnswlib) would add a C++ dependency, a second artifact to keep in sync with
the store, and a recall/latency knob to explain - to make a 12 ms query into a
1 ms query. The design spec already contains an explicit escalation gate for
vector search; until a library is large enough to trip it, brute force is the
correct answer and the honest one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rekindle.extras import install_command
from rekindle.semantic.availability import SemanticUnavailable
from rekindle.semantic.store import ITEM_SIZE, EmbeddingStore, StoreError

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise SemanticUnavailable(
            f"vector maths needs numpy, from the 'semantic' extra ({install_command('semantic')})"
        ) from exc
    return np


@dataclass(frozen=True)
class Matrix:
    """Every vector in a store, in row order, with the hashes beside them."""

    hashes: tuple[str, ...]
    data: np.ndarray  # (n, dim) float32, rows L2-normalised

    def __len__(self) -> int:
        return len(self.hashes)

    @property
    def dim(self) -> int:
        return int(self.data.shape[1]) if len(self.data) else 0


def load_matrix(store: EmbeddingStore) -> Matrix:
    """Read the whole matrix. Validates row contiguity before trusting it.

    The store assigns rows 0..n-1 with no gaps, so `ordered_hashes()[i]` owns
    row i. That invariant is what lets the file be read as one block instead
    of n seeks - so it is CHECKED here rather than assumed, because if it ever
    breaks, every result silently belongs to the wrong photo.
    """
    np = _numpy()
    rows = list(store.iter_rows())
    expected_bytes = len(rows) * store.dim * ITEM_SIZE
    raw = store.raw_bytes()
    if len(raw) < expected_bytes:
        raise StoreError(
            f"{store.vectors_path} holds {len(raw)} bytes, manifest needs "
            f"{expected_bytes}. The store is truncated."
        )
    for i, (_, row) in enumerate(rows):
        if row != i:
            raise StoreError(
                f"{store.manifest_path} row numbering is not contiguous: "
                f"position {i} holds row {row}. Delete {store.root} and re-embed."
            )
    data = np.frombuffer(raw[:expected_bytes], dtype="<f4").reshape(len(rows), store.dim)
    return Matrix(tuple(h for h, _ in rows), np.ascontiguousarray(data))


def normalise(array: np.ndarray) -> np.ndarray:
    np = _numpy()
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    # A zero row would divide to nan and poison every later comparison.
    norms = np.where(norms == 0, 1.0, norms)
    return (array / norms).astype(np.float32, copy=False)


def cosine_scores(matrix: Matrix, query: list[float] | np.ndarray) -> np.ndarray:
    """Cosine similarity of every row against `query`.

    Both sides are unit vectors, so the dot product IS the cosine. `normalise`
    is applied to the query anyway: a caller passing a raw text embedding
    would otherwise get scores scaled by its magnitude, which changes nothing
    about the ranking but makes the reported numbers meaningless.
    """
    np = _numpy()
    q = normalise(np.asarray(query, dtype=np.float32))
    if q.shape[-1] != matrix.dim:
        raise StoreError(
            f"query has {q.shape[-1]} dims, store holds {matrix.dim}. "
            "These are different embedding spaces."
        )
    return matrix.data @ q


def top_k(scores: np.ndarray, k: int) -> list[int]:
    """Indices of the k highest scores, best first. Ties broken by index."""
    np = _numpy()
    if k <= 0 or not len(scores):
        return []
    k = min(k, len(scores))
    part = np.argpartition(-scores, k - 1)[:k]
    return [int(i) for i in part[np.argsort(-scores[part], kind="stable")]]
