"""Scene clustering: turn the untagged, un-albumed majority into memory material.

WHY THIS IS THE MILESTONE'S MAIN JUSTIFICATION
----------------------------------------------
Measured on the reference index (19,480 photos, 19,318 live):

    8,433  live photos have NO face tag at all
    7,037  of those also have no album that is not a "Photos from YYYY"
           Takeout year bucket (6,483 of them are images)

Every recipe in the design selects on people, albums, GPS or time. Those 7,037
are invisible to three of the four. Clustering them by scene is what makes
them addressable.

WHY SPHERICAL K-MEANS, AND WHY IT IS WRITTEN HERE
-------------------------------------------------
Embeddings are L2-normalised, so cosine similarity is the metric and the
natural centroid is the normalised mean - spherical k-means, forty lines of
numpy. The alternatives were weighed:

  * **scikit-learn KMeans** - Euclidean on unit vectors is monotone in cosine,
    so it would work, but it is a 30 MB dependency for an algorithm shorter
    than the code that would configure it, and its k-means++ is seeded by a
    global RNG whose determinism across versions is not promised. Cluster ids
    that move between runs would make the review queue useless.
  * **HDBSCAN / DBSCAN** - density clustering is the theoretically better fit
    (it can say "this photo is in no scene") and it is another dependency with
    two parameters that need tuning per library. Deferred, with a note: the
    `noise` concept is genuinely missing here and `min_cosine` below is a
    poor substitute.
  * **Agglomerative** - O(n^2) memory at 19k photos is 1.5 GB of distances.

Determinism is a requirement, not a nicety: `rekindle cluster` writes cluster
ids that a user will later refer to ("publish cluster 12"), so the same store
and the same k must produce the same partition. Seeded `numpy.random.Generator`
plus a stable tie-break gives that; it is pinned by a test that clusters twice.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rekindle.semantic.availability import SemanticUnavailable
from rekindle.semantic.vectors import Matrix, normalise

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

#: k for a library of n photos, when the caller does not say. sqrt(n/2) is the
#: usual rule of thumb; capped because a review UI cannot show 400 clusters and
#: floored because fewer than 2 is not a clustering.
K_MIN = 2
K_MAX = 120


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise SemanticUnavailable(
            "clustering needs numpy, from the 'semantic' extra (uv sync --extra semantic)"
        ) from exc
    return np


def suggest_k(n: int) -> int:
    if n <= K_MIN:
        return max(1, n)
    return max(K_MIN, min(K_MAX, int(round((n / 2) ** 0.5))))


@dataclass(frozen=True)
class Cluster:
    cluster_id: int
    members: tuple[str, ...]
    centroid: tuple[float, ...]
    #: Mean cosine of a member to the centroid. High = a tight, coherent scene.
    cohesion: float
    #: Mean cosine of the centroid to the NEXT-nearest centroid. Low = this
    #: cluster is well separated from its neighbours.
    separation: float
    label: str = ""
    label_score: float = 0.0

    def __len__(self) -> int:
        return len(self.members)


@dataclass(frozen=True)
class Clustering:
    clusters: tuple[Cluster, ...]
    k: int
    n: int
    iterations: int
    converged: bool
    seed: int
    #: Hashes left unassigned because no centroid was within `min_cosine`.
    outliers: tuple[str, ...] = field(default=())

    def of(self, file_hash: str) -> int | None:
        for c in self.clusters:
            if file_hash in c.members:
                return c.cluster_id
        return None


def spherical_kmeans(
    matrix: Matrix,
    k: int,
    *,
    seed: int = 0,
    max_iter: int = 100,
    min_cosine: float | None = None,
) -> Clustering:
    """Partition `matrix` into `k` clusters by cosine similarity.

    `min_cosine`, when given, holds back photos whose best centroid is further
    than that away. They come back in `outliers` rather than being forced into
    the least-bad cluster: a "scene" made of everything that matched nothing is
    worse than no scene, and this is the closest this algorithm gets to the
    noise label DBSCAN would give for free.
    """
    np = _numpy()
    n = len(matrix)
    if n == 0:
        return Clustering((), 0, 0, 0, True, seed)
    k = max(1, min(k, n))
    data = matrix.data

    centres = _kmeanspp(np, data, k, seed)
    assign = np.full(n, -1, dtype=np.int64)
    iterations = 0
    converged = False
    for step in range(1, max_iter + 1):
        iterations = step
        sims = data @ centres.T
        new = sims.argmax(axis=1)
        if np.array_equal(new, assign):
            converged = True
            break
        assign = new
        for j in range(k):
            members = data[assign == j]
            if len(members):
                centres[j] = members.mean(axis=0)
            else:
                # An empty cluster would stay empty forever. Re-seed it on the
                # point currently worst served by any centroid - the standard
                # repair, and the one that keeps k meaning what it says.
                worst = int((data @ centres.T).max(axis=1).argmin())
                centres[j] = data[worst]
        centres = normalise(centres)

    sims = data @ centres.T
    best = sims.argmax(axis=1)
    best_sim = sims.max(axis=1)

    outliers: list[str] = []
    clusters: list[Cluster] = []
    between = centres @ centres.T
    np.fill_diagonal(between, -2.0)
    for j in range(k):
        mask = best == j
        if min_cosine is not None:
            mask = mask & (best_sim >= min_cosine)
        members = [matrix.hashes[i] for i in np.flatnonzero(mask)]
        if not members:
            continue
        clusters.append(
            Cluster(
                cluster_id=j,
                members=tuple(members),
                centroid=tuple(float(v) for v in centres[j]),
                cohesion=float(best_sim[mask].mean()),
                separation=float(between[j].max()),
            )
        )
    if min_cosine is not None:
        outliers = [matrix.hashes[i] for i in np.flatnonzero(best_sim < min_cosine)]

    # Largest first, then by cluster id: a stable order a human can read down.
    clusters.sort(key=lambda c: (-len(c.members), c.cluster_id))
    return Clustering(
        clusters=tuple(clusters),
        k=k,
        n=n,
        iterations=iterations,
        converged=converged,
        seed=seed,
        outliers=tuple(outliers),
    )


def _kmeanspp(np, data: np.ndarray, k: int, seed: int) -> np.ndarray:
    """k-means++ seeding on cosine distance, with an explicit Generator.

    `default_rng(seed)` rather than the legacy global RNG: the legacy one is
    process-global state any other library can disturb, and this function's
    output has to be reproducible for a user who wrote down a cluster id.
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    first = int(rng.integers(n))
    centres = [data[first]]
    for _ in range(k - 1):
        closest = (data @ np.asarray(centres, dtype=np.float32).T).max(axis=1)
        d2 = np.clip(1.0 - closest, 0.0, None) ** 2
        total = float(d2.sum())
        if total <= 0.0:
            # Every remaining point coincides with a centre (duplicates).
            centres.append(data[int(rng.integers(n))])
            continue
        centres.append(data[int(rng.choice(n, p=d2 / total))])
    return normalise(np.asarray(centres, dtype=np.float32))


def label_clusters(
    clustering: Clustering,
    vocabulary: Sequence[str],
    text_vectors: Sequence[Sequence[float]],
    *,
    min_score: float = 0.0,
) -> Clustering:
    """Name each cluster with its best-matching phrase from `vocabulary`.

    The label is cosmetic and is presented as such. It is the cosine of the
    cluster CENTROID against each phrase - not a vote over members - because a
    centroid is the thing the cluster actually is, and a per-member vote makes
    a 600-photo cluster 600 times more expensive to label for no more signal.
    """
    np = _numpy()
    if not vocabulary or not clustering.clusters:
        return clustering
    texts = normalise(np.asarray(text_vectors, dtype=np.float32))
    labelled = []
    for c in clustering.clusters:
        scores = texts @ np.asarray(c.centroid, dtype=np.float32)
        best = int(scores.argmax())
        score = float(scores[best])
        labelled.append(
            Cluster(
                cluster_id=c.cluster_id,
                members=c.members,
                centroid=c.centroid,
                cohesion=c.cohesion,
                separation=c.separation,
                label=vocabulary[best] if score >= min_score else "",
                label_score=score,
            )
        )
    return Clustering(
        clusters=tuple(labelled),
        k=clustering.k,
        n=clustering.n,
        iterations=clustering.iterations,
        converged=clustering.converged,
        seed=clustering.seed,
        outliers=clustering.outliers,
    )


#: Candidate scene names, used to label clusters and nothing else. Chosen to
#: span what a personal library from this part of the world actually contains,
#: NOT copied from an ImageNet class list: "tench, goldfish, great white
#: shark" labels nothing in anybody's holiday photos.
SCENE_VOCABULARY: tuple[str, ...] = (
    "snow-covered mountains",
    "a lake or river",
    "the sea or a beach",
    "a forest or green countryside",
    "a desert or dry landscape",
    "a sunset or sunrise",
    "a city street",
    "tall buildings and skyscrapers",
    "the inside of a house or a room",
    "a restaurant or cafe table",
    "a plate of food",
    "a group of people posing",
    "a wedding",
    "a religious ceremony or temple",
    "a festival with lights and decorations",
    "a baby or small child",
    "a birthday party with a cake",
    "a dog or a cat",
    "flowers or a garden",
    "a car, a bus or a train",
    "an aeroplane or an airport",
    "a boat on water",
    "a screenshot of a phone or a computer screen",
    "a document, a receipt or a piece of paper",
    "a shop or a market",
    "a swimming pool",
    "a sports match or a playing field",
    "a concert or a stage",
    "a museum or an art gallery",
    "a hospital or a clinic",
    "a night scene with artificial light",
    "a close-up of an object",
)
