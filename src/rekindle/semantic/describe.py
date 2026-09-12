"""Scoring a photograph against the caption vocabulary, in percentiles.

`memory/vocab.py` owns the vocabulary and the rules that pick from it. This
module owns the one thing those rules cannot compute without a model: how far
into a probe's own distribution a given photograph sits.

**Why a percentile and not the cosine.** Every other part of this project
already refuses to read an absolute cosine, and the reason applies here in its
sharpest form. Choosing between "a plate of food" and "a sandy beach" for one
photograph means comparing two DIFFERENT queries, whose score scales are not
the same: `prompt.py` records that `scuba diving underwater` outscores
`durga puja` on a library with zero scuba photographs in it. Take the larger
number and you take whichever phrase CLIP is warm on in general.

So each probe is scored against the WHOLE embedded library once, and a
photograph's percentile within that distribution is what the rules read. Every
probe is then measured against the same 18,201 photographs and the numbers are
comparable. The cost is one matrix product per probe - 23 probes over
18,201 x 768 float32 is about 0.3 s and 56 MB already resident - paid once per
process and then answered from memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rekindle.memory import vocab
from rekindle.semantic.search import embed_query
from rekindle.semantic.vectors import Matrix, cosine_scores, load_matrix

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

    from rekindle.semantic.encoder import ImageTextEncoder
    from rekindle.semantic.store import EmbeddingStore


@dataclass
class Describer:
    """Percentile scores for one library, against one vocabulary.

    Built once and reused: the expensive part is the library-wide score
    distribution for each probe, and it depends on nothing but the store and
    the vocabulary.
    """

    matrix: Matrix
    #: probe -> the library's scores for it, SORTED ascending. A photograph's
    #: percentile is `searchsorted` into this.
    reference: dict[str, np.ndarray]
    #: probe -> this library's score for every photograph, in matrix row order.
    scores: dict[str, np.ndarray]
    index_of: dict[str, int]

    def scores_for(self, file_hash: str) -> dict[str, float] | None:
        """Probe -> percentile in [0, 1], or None when the photo is unembedded.

        A video, or a photograph added since the last `rekindle semantic
        embed`, has no vector and therefore no grounding - which is a blank
        caption, not a guessed one.
        """
        row = self.index_of.get(file_hash)
        if row is None:
            return None
        import numpy as np

        out: dict[str, float] = {}
        for probe, column in self.scores.items():
            reference = self.reference[probe]
            # `side="left"` counts the number of library photographs scoring
            # STRICTLY below this one, so a photograph tied with the whole
            # library scores 0 rather than 1. The conservative direction: a
            # probe that scores identically everywhere describes nothing.
            below = int(np.searchsorted(reference, column[row], side="left"))
            out[probe] = below / len(reference)
        return out


def build(
    store: EmbeddingStore,
    encoder: ImageTextEncoder,
    *,
    matrix: Matrix | None = None,
    path: Path | None = None,
) -> Describer:
    """Embed every probe and score it against the whole library.

    The probes are embedded through `search.embed_query`, the same prompt
    ensemble the rest of the semantic layer uses, so a vocabulary term and a
    search for the same words land in the same place.
    """
    import numpy as np

    grid = matrix if matrix is not None else load_matrix(store)
    if not len(grid):
        raise ValueError("the embedding store holds no vectors")
    scores: dict[str, np.ndarray] = {}
    reference: dict[str, np.ndarray] = {}
    for probe in vocab.probes(path):
        column = cosine_scores(grid, embed_query(encoder, probe))
        scores[probe] = column
        reference[probe] = np.sort(column)
    return Describer(
        matrix=grid,
        reference=reference,
        scores=scores,
        index_of={h: i for i, h in enumerate(grid.hashes)},
    )
