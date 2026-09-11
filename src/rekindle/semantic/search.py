"""Text -> photos, and photo -> similar photos, over an embedding store."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rekindle.semantic.encoder import ImageTextEncoder
from rekindle.semantic.photos import IndexedPhoto, PhotoIndexReader, ReadFilter
from rekindle.semantic.store import EmbeddingStore
from rekindle.semantic.vectors import Matrix, cosine_scores, load_matrix, top_k

#: CLIP and SigLIP were both trained on caption-like text, not search queries.
#: Wrapping a bare query in these templates and averaging the resulting vectors
#: is the standard prompt ensemble from the CLIP paper; on the reference
#: library it is the difference between "sunset" returning orange walls and
#: returning sunsets. Measured in docs/audit-m3-semantic.md.
PROMPT_TEMPLATES = (
    "a photo of {}",
    "a photograph of {}",
    "{}",
)


@dataclass(frozen=True)
class Hit:
    file_hash: str
    score: float
    photo: IndexedPhoto | None

    @property
    def path_str(self) -> str:
        return str(self.photo.path) if self.photo else "(not in index)"


def embed_query(
    encoder: ImageTextEncoder,
    query: str,
    *,
    templates: Sequence[str] = PROMPT_TEMPLATES,
) -> list[float]:
    """One unit vector for `query`, averaged over the prompt ensemble.

    Averaging then renormalising, not averaging alone: the mean of k unit
    vectors is shorter than a unit vector, which would scale every reported
    similarity down by a factor nobody could interpret.
    """
    text = query.strip()
    if not text:
        raise ValueError("empty query")
    prompts = [t.format(text) for t in templates] or [text]
    vectors = encoder.encode_texts(prompts)
    dim = len(vectors[0])
    summed = [sum(v[i] for v in vectors) for i in range(dim)]
    norm = sum(v * v for v in summed) ** 0.5
    if norm == 0.0:  # pragma: no cover - would require exactly opposed prompts
        return vectors[0]
    return [v / norm for v in summed]


class SemanticSearch:
    """Brute-force cosine search, with the index consulted only for hits."""

    def __init__(
        self,
        store: EmbeddingStore,
        reader: PhotoIndexReader | None = None,
        *,
        matrix: Matrix | None = None,
    ) -> None:
        self.store = store
        self.reader = reader
        self.matrix = matrix if matrix is not None else load_matrix(store)

    def search_vector(
        self,
        query: list[float],
        *,
        k: int = 20,
        min_score: float | None = None,
        allowed: set[str] | None = None,
    ) -> list[Hit]:
        if not len(self.matrix):
            return []
        scores = cosine_scores(self.matrix, query)
        # Over-fetch when filtering, so a filter that rejects most of the top-k
        # still returns k results rather than however many survived.
        want = k if allowed is None else min(len(self.matrix), max(k * 20, k))
        picks = top_k(scores, want)
        hits: list[tuple[str, float]] = []
        for i in picks:
            file_hash = self.matrix.hashes[i]
            if allowed is not None and file_hash not in allowed:
                continue
            score = float(scores[i])
            if min_score is not None and score < min_score:
                break
            hits.append((file_hash, score))
            if len(hits) == k:
                break
        return self._resolve(hits)

    def search_text(
        self,
        encoder: ImageTextEncoder,
        query: str,
        *,
        k: int = 20,
        min_score: float | None = None,
        allowed: set[str] | None = None,
    ) -> list[Hit]:
        return self.search_vector(
            embed_query(encoder, query), k=k, min_score=min_score, allowed=allowed
        )

    def similar_to(self, file_hash: str, *, k: int = 20) -> list[Hit]:
        """Nearest neighbours of a photo already in the store, itself excluded."""
        vector = self.store.get(file_hash)
        if vector is None:
            raise KeyError(f"{file_hash} has no embedding in {self.store.root}")
        scores = cosine_scores(self.matrix, vector)
        hits = []
        for i in top_k(scores, min(k + 1, len(self.matrix))):
            if self.matrix.hashes[i] == file_hash:
                continue
            hits.append((self.matrix.hashes[i], float(scores[i])))
            if len(hits) == k:
                break
        return self._resolve(hits)

    def allowed_hashes(self, where: ReadFilter) -> set[str]:
        """The hashes a filter admits - used to scope a search to an album etc."""
        if self.reader is None:
            raise ValueError("filtering a search needs a PhotoIndexReader")
        return {p.file_hash for p in self.reader.iter_photos(where)}

    def _resolve(self, hits: list[tuple[str, float]]) -> list[Hit]:
        photos: dict[str, IndexedPhoto] = {}
        if self.reader is not None and hits:
            photos = self.reader.get_many([h for h, _ in hits])
        return [Hit(h, s, photos.get(h)) for h, s in hits]
