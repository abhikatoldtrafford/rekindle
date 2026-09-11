"""Aesthetic ranking: choose 40 photos out of Kashmir's 508.

THE PROBLEM THIS SOLVES
-----------------------
Sharpness-and-exposure scoring ranks a crisp photo of a wall above a slightly
soft photo of someone laughing. It measures whether the camera worked, not
whether the photo is worth looking at.

WHAT THIS IS
------------
The LAION improved-aesthetic-predictor V2 head: a five-layer MLP regressing a
1-10 rating from a CLIP ViT-L/14 image embedding. It is a LINEAR chain - the
published architecture has its ReLUs commented out, which is why the checkpoint
is called `linearMSE` - so scoring is five matrix multiplies and needs neither
torch nor a GPU. Embeddings are already in the store; scoring the whole library
costs about a second.

WHAT IT IS NOT
--------------
It is a model of average human preference on SAC/AVA/LAION-Logos, not of this
user's preference. It reliably likes sunsets, bokeh and symmetry, and reliably
undervalues a blurry photo of a person that matters. So:

  * it RANKS WITHIN a candidate set, never filters across the library;
  * `pick_best` enforces a spread so a recipe cannot return forty photographs
    of the same sunset, which is exactly what pure top-k does;
  * the technical score below is combined with it, not replaced by it.

THE HEAD ONLY WORKS ON THE SPACE IT WAS TRAINED ON. `score_store` refuses a
store whose model key is not `clip-vit-l14`, rather than silently multiplying a
1152-dim SigLIP vector by a 768-wide matrix and producing numbers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rekindle.semantic.availability import SemanticUnavailable
from rekindle.semantic.registry import HeadModel, aesthetic_model
from rekindle.semantic.store import EmbeddingStore
from rekindle.semantic.torchfile import load_state_dict
from rekindle.semantic.vectors import Matrix, load_matrix, normalise

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

#: The published rating scale. Used only to clamp obviously-broken output.
SCORE_MIN = 0.0
SCORE_MAX = 10.0


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise SemanticUnavailable(
            "aesthetic scoring needs numpy, from the 'semantic' extra"
        ) from exc
    return np


class AestheticError(RuntimeError):
    """The head cannot be applied as asked."""


@dataclass(frozen=True)
class Scored:
    file_hash: str
    score: float


class AestheticHead:
    """A chain of affine layers over a unit-norm CLIP embedding."""

    def __init__(self, weights: Sequence, biases: Sequence, spec: HeadModel | None = None) -> None:
        self._w = list(weights)
        self._b = list(biases)
        self.spec = spec
        self.in_dim = int(self._w[0].shape[1])

    @classmethod
    def from_file(cls, path: Path, spec: HeadModel | None = None) -> AestheticHead:
        np = _numpy()
        state = load_state_dict(path)
        # Keys are layers.<n>.weight / layers.<n>.bias with gaps where the
        # Dropout modules sit. Sorting by that integer is what keeps the chain
        # in order; sorting the strings would put layers.10 before layers.2.
        pairs: dict[int, dict[str, object]] = {}
        for key, value in state.items():
            parts = key.split(".")
            if len(parts) != 3 or not parts[1].isdigit():
                continue
            pairs.setdefault(int(parts[1]), {})[parts[2]] = value
        if not pairs:
            raise AestheticError(
                f"{path} has no layers.<n>.weight entries; keys are "
                f"{sorted(state)[:8]}. This is not the LAION aesthetic head."
            )
        weights, biases = [], []
        for index in sorted(pairs):
            layer = pairs[index]
            if "weight" not in layer or "bias" not in layer:
                raise AestheticError(f"{path}: layer {index} is missing weight or bias")
            weights.append(np.asarray(layer["weight"], dtype=np.float32))
            biases.append(np.asarray(layer["bias"], dtype=np.float32))
        head = cls(weights, biases, spec)
        if spec is not None and head.in_dim != spec.in_dim:
            raise AestheticError(
                f"{path} takes {head.in_dim}-dim input but the registry declares "
                f"{spec.in_dim} for '{spec.key}'."
            )
        return head

    def score(self, embeddings) -> np.ndarray:
        """Ratings for a (n, dim) array of embeddings, in order."""
        np = _numpy()
        x = normalise(np.asarray(embeddings, dtype=np.float32))
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.in_dim:
            raise AestheticError(
                f"this head takes {self.in_dim}-dim embeddings, got {x.shape[1]}. "
                "The aesthetic head is trained on ONE embedding space and is "
                "meaningless on another."
            )
        for w, b in zip(self._w, self._b, strict=True):
            x = x @ w.T + b
        return np.clip(x.reshape(-1), SCORE_MIN, SCORE_MAX)


def load_head(cache_dir: Path, key: str | None = None) -> AestheticHead:
    from rekindle.semantic.setup import snapshot_dir

    spec = aesthetic_model(key)
    root = snapshot_dir(spec.pin, cache_dir, offline=True)
    return AestheticHead.from_file(root / spec.pin.files[0], spec)


def score_store(
    store: EmbeddingStore,
    head: AestheticHead,
    *,
    matrix: Matrix | None = None,
) -> list[Scored]:
    """Score every vector in `store`, refusing a mismatched embedding space."""
    if head.spec is not None and store.model_key != head.spec.embed_key:
        raise AestheticError(
            f"'{head.spec.key}' scores '{head.spec.embed_key}' embeddings, but "
            f"{store.root} holds '{store.model_key}'. Embed with "
            f"--model {head.spec.embed_key}, or pick a head trained on "
            f"'{store.model_key}'."
        )
    mat = matrix if matrix is not None else load_matrix(store)
    if not len(mat):
        return []
    scores = head.score(mat.data)
    return [Scored(h, float(s)) for h, s in zip(mat.hashes, scores, strict=True)]


def pick_best(
    candidates: Sequence[Scored],
    k: int,
    *,
    vectors: dict[str, Sequence[float]] | None = None,
    max_similarity: float = 0.92,
) -> list[Scored]:
    """The k best, with near-duplicates suppressed.

    Plain top-k on an aesthetic score is a near-duplicate machine: burst shots
    differ by hundredths of a rating point, so the top 40 of a 508-photo trip
    is routinely eight scenes. Greedy selection with a similarity ceiling is
    the cheapest fix that actually works, and it degrades to plain top-k when
    no vectors are supplied rather than silently doing nothing.
    """
    ordered = sorted(candidates, key=lambda s: (-s.score, s.file_hash))
    if vectors is None:
        return ordered[:k]
    np = _numpy()
    chosen: list[Scored] = []
    kept: list[np.ndarray] = []
    for cand in ordered:
        if len(chosen) == k:
            break
        vec = vectors.get(cand.file_hash)
        if vec is None:
            chosen.append(cand)
            continue
        v = normalise(np.asarray(vec, dtype=np.float32))
        if kept and max(float(v @ other) for other in kept) > max_similarity:
            continue
        chosen.append(cand)
        kept.append(v)
    if len(chosen) < k:
        # Falling short because everything left was a near-duplicate is worse
        # than returning duplicates: a recipe asked for k photos.
        seen = {c.file_hash for c in chosen}
        for cand in ordered:
            if len(chosen) == k:
                break
            if cand.file_hash not in seen:
                chosen.append(cand)
    return chosen[:k]
