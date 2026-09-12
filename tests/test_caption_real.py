"""Opt-in re-measurement of the caption layer against the reference library.

Skipped everywhere except a machine that has the index, an embedding store and
the semantic extra. CI has none of them and must stay green.

**Why this file exists.** Every number in `memory/vocab.py` came from
hand-grading photographs on one library, and this project's own history says a
number nobody re-measures is a number that quietly goes wrong - seven figures
have now failed re-measurement here, three of them in this milestone. These
tests cannot re-do the hand-grading; what they can do is fail the day the
SHAPE of the measurement changes: coverage collapsing to nothing, the subject
facet suddenly firing everywhere, a capitalised word reaching a caption.

The assertions are RANGES, for the same reason `test_prompt_real.py` uses
them: re-embedding the library is legitimately a new store and a new answer,
and a test that pinned exact captions would fail for a correct reason and
teach everyone to ignore it.

Point it at another library with REKINDLE_REAL_DATA_DIR.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

from rekindle.db import PhotoStore
from rekindle.memory import vocab
from rekindle.memory.index import MemoryIndex
from rekindle.memory.policy import load_policy

DATA = Path(os.environ.get("REKINDLE_REAL_DATA_DIR", r"D:\google_photos\data"))
DB = DATA / "rekindle.sqlite"
STORE = DATA / "semantic" / "clip-vit-l14"

pytestmark = pytest.mark.skipif(
    not (DB.is_file() and (STORE / "manifest.sqlite").is_file()),
    reason="needs the reference index and a clip-vit-l14 embedding store",
)

#: The sample the numbers in `vocab.py` were graded on. Fixed seed, so the
#: same 36 photographs come back and a change in the numbers is a change in
#: the code rather than in the draw.
SEED = 7
SAMPLE = 36


@pytest.fixture(scope="module")
def described():
    pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from rekindle.semantic import describe
    from rekindle.semantic.encoder import load_encoder
    from rekindle.semantic.registry import embed_model
    from rekindle.semantic.setup import cache_dir_for
    from rekindle.semantic.store import EmbeddingStore, store_root

    spec = embed_model(None)
    root = store_root(DATA, spec.key)
    store = EmbeddingStore(
        root,
        dim=spec.dim,
        model_key=spec.key,
        model_revision=spec.pin("torch").revision if spec.torch else "",
    )
    encoder = load_encoder(spec.key, device="auto", cache_dir=cache_dir_for(DATA)).encoder
    return describe.build(store, encoder)


@pytest.fixture(scope="module")
def index():
    store = PhotoStore(DB)
    try:
        yield MemoryIndex.open(store, load_policy(DATA))
    finally:
        store.close()


def _sample(index, described):
    pool = [p for p in index.images() if p.file_hash in described.index_of]
    return random.Random(SEED).sample(pool, SAMPLE)


def _captions(index, described):
    out = []
    for photo in _sample(index, described):
        grounding = vocab.choose(described.scores_for(photo.file_hash))
        out.append((photo, grounding, vocab.phrase(grounding, "")))
    return out


def test_most_photographs_get_no_caption_and_that_is_correct(index, described):
    """Measured: 11 of 36 at seed 7, 7 of 36 at seed 23 - about a quarter.

    The blanks are the indoor portraits and the family groups, which is
    exactly the population where a wrong caption is worst. A run that
    captions most of the library has had its gates loosened; a run that
    captions none has lost the vocabulary.
    """
    captioned = sum(1 for _, _, line in _captions(index, described) if line)
    assert 3 <= captioned <= SAMPLE // 2, f"{captioned}/{SAMPLE} captioned"


def test_the_subject_facet_stays_rarer_than_the_setting_facet(index, described):
    """The measurement the per-facet floors came from.

    At one floor of 0.97 the subject facet fired on a third of captions and
    every clear error was one of them - a vehicle on an empty lake shore, a
    statue on a framed print. At 0.995 it is the rarest facet by some way. If
    that ever inverts, the floors have drifted.
    """
    counts = dict.fromkeys(vocab.FACETS, 0)
    for _, grounding, _ in _captions(index, described):
        for term in grounding.terms:
            counts[term.facet] += 1
    assert counts[vocab.SUBJECT] <= counts[vocab.SETTING], counts


def test_no_caption_ever_contains_a_word_the_vocabulary_did_not_supply(index, described):
    """The closed-vocabulary guarantee, checked against real photographs
    rather than against a hand-built score mapping."""
    allowed: set[str] = {"a", "an", "the"}
    for term in vocab.load():
        allowed |= {w.strip(",").casefold() for w in term.says.split()}
    for photo, _, line in _captions(index, described):
        for word in line.split():
            assert word.strip(",").casefold() in allowed, (line, word, photo.file_hash)


def test_the_same_photograph_gives_the_same_caption_twice(index, described):
    """The engine's byte-for-byte promise, at the caption layer, with no cache
    involved - the CLIP path is deterministic on its own and the cache is an
    optimisation there rather than the mechanism."""
    first = [line for _, _, line in _captions(index, described)]
    second = [line for _, _, line in _captions(index, described)]
    assert first == second


def test_a_photograph_with_no_embedding_is_described_as_nothing(index, described):
    """Videos are never embedded. `scores_for` must say so rather than
    guessing, or every video in a memory gets whatever the zero vector is
    nearest to."""
    assert described.scores_for("no-such-hash-at-all") is None


def test_every_probe_scores_something_and_nothing_scores_everything(index, described):
    """A probe at the 100th percentile for a third of the library is a probe
    that describes the library rather than a photograph.

    This is the failure mode `known-limitations.md` records for the cluster
    labels - a phrase absorbing whatever is nearest - measured directly.
    """
    for photo in _sample(index, described):
        scores = described.scores_for(photo.file_hash)
        assert len(scores) == len(vocab.probes())
        top = sum(1 for v in scores.values() if v >= 0.995)
        assert top <= len(vocab.probes()) // 2, (photo.file_hash, top)
