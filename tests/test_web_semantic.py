"""The semantic seam, exercised the way a threaded server exercises it.

THE BUG THESE EXIST FOR
-----------------------
The embedding layer is loaded lazily by whichever request thread asks for it
first, and `EmbeddingStore` holds a `sqlite3` connection - which belongs to
the thread that created it. `SemanticSearch.similar_to` begins with
`store.get(file_hash)`, so "more like this" worked once and then died with
`sqlite3.ProgrammingError` on the next request, closing the socket with no
response at all. Text search hid it completely, because `search_vector` reads
only the in-memory matrix.

It was found by running the real UI against the real 18,201-vector store, not
by reading. These tests reproduce the shape of it - load on one thread, query
on several others - with `ToyEncoder`, which is a genuine (if weak) joint
embedding function rather than a mock, so the ANSWERS are asserted too.
"""

from __future__ import annotations

import threading

import pytest

from rekindle.semantic.store import EmbeddingStore
from rekindle.web.library import Library, semantic_callables
from tests.fixtures.semantic import TOY_DIM, ToyEncoder, solid_image, write_photo
from tests.fixtures.web import make_library, open_library

pytest.importorskip("numpy")

COLOURS = ("red", "green", "blue")


@pytest.fixture
def library(tmp_path):
    """A real `Library` whose semantic layer is a real store of toy vectors.

    The two callables come from `library.semantic_callables`, which is the
    function `Library._load_semantic` itself calls - so these tests drive the
    shipped code, not a second copy of it written in the fixture. All that is
    substituted is where the store and the encoder come from.
    """
    from rekindle.semantic.search import SemanticSearch

    data_dir, _ = make_library(tmp_path)
    encoder = ToyEncoder()
    store = EmbeddingStore(tmp_path / "store", dim=TOY_DIM, model_key="toy")

    index = open_library(data_dir).require_index()
    targets = [p.file_hash for p in index.all()[:6]]
    vectors = []
    for i, file_hash in enumerate(targets):
        colour = COLOURS[i % len(COLOURS)]
        write_photo(tmp_path / "lib2", f"{colour}{i}.jpg", colour)
        vectors.append((file_hash, encoder.encode_images([solid_image(colour)])[0]))
    store.add_many(vectors)

    lib = Library(data_dir)
    lib.load()
    # THE PRODUCTION FUNCTION, not a copy of it. `_load_semantic` differs only
    # in where the store and the encoder come from - the model registry and an
    # ONNX or torch runtime, neither of which CI installs - and the pair of
    # callables it installs is built by exactly this call.
    lib._retriever, lib._similar = semantic_callables(SemanticSearch(store, None), encoder)
    return lib, targets, store


def test_neighbours_do_not_touch_the_stores_connection(library):
    """The regression test, stated as the property rather than the symptom.

    A `sqlite3` connection is usable only from its creating thread, so calling
    the neighbour lookup from another thread would raise `ProgrammingError`
    if it read the manifest. Ten threads, none of them the one that built the
    store.
    """
    lib, targets, _store = library
    errors: list[BaseException] = []
    answers: list[list] = []

    def ask() -> None:
        try:
            answers.append(lib.similar()(targets[0], 3))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=ask) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == [], f"the neighbour lookup is not thread-safe: {errors[0]!r}"
    assert all(a == answers[0] for a in answers), "the same query gave different answers"
    assert answers[0], "the fixture must produce neighbours, or this proves nothing"


def test_the_store_connection_really_is_thread_bound(library):
    """Proves the test above is not vacuous.

    If `EmbeddingStore` were thread-safe, the regression test would pass
    whatever the implementation did. It is not, and this is what that looks
    like.
    """
    import sqlite3

    _lib, targets, store = library
    caught: list[BaseException] = []

    def touch() -> None:
        try:
            store.get(targets[0])
        except BaseException as exc:  # noqa: BLE001
            caught.append(exc)

    thread = threading.Thread(target=touch)
    thread.start()
    thread.join()
    assert caught and isinstance(caught[0], sqlite3.ProgrammingError)


def test_a_photo_is_not_its_own_neighbour(library):
    lib, targets, _store = library
    hits = lib.similar()(targets[0], 4)
    assert targets[0] not in {h for h, _ in hits}
    assert len(hits) == 4


def test_a_photo_with_no_embedding_has_no_neighbours(library):
    """Half the fixture library was never embedded. That is the normal state
    of a library mid-`rekindle semantic embed`, and it must not raise."""
    lib, targets, _store = library
    unembedded = next(p.file_hash for p in lib.require_index().all() if p.file_hash not in targets)
    assert lib.similar()(unembedded, 4) == []


def test_neighbours_are_ranked_by_the_embedding_not_by_chance(library):
    """`ToyEncoder` embeds colour, so the nearest neighbour of a red photo is
    the other red photo. An answer this specific cannot come from a stub."""
    lib, targets, _store = library
    # targets[0] and targets[3] were both written as "red" (i % 3 == 0).
    hits = lib.similar()(targets[0], 5)
    assert hits[0][0] == targets[3]
    assert hits[0][1] > 0.99


def test_searching_from_many_threads_gives_one_answer(library):
    lib, _targets, _store = library
    answers: list[list] = []
    errors: list[BaseException] = []

    def ask() -> None:
        try:
            answers.append(lib.retriever()("red", 3))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=ask) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert answers and all(a == answers[0] for a in answers)
