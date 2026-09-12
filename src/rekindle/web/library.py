"""The open library: one guardrailed index, shared by every request.

`MemoryIndex.open` reads every row and applies the policy once. On the
reference library that is 19,480 rows; doing it per request would make the
first click on every page wait for it. So it happens once, on a background
thread started when the server binds, and requests that arrive before it
finishes are told it is still loading rather than blocked forever.

THREADS AND SQLITE
------------------
`PhotoStore` wraps a `sqlite3` connection, and a connection belongs to the
thread that created it. Nothing here holds one open: the index is built in the
loader thread and the store is closed immediately afterwards, and the two
operations that must WRITE - dismissing a memory, recording that one was
surfaced - open a fresh store inside the calling thread and close it again.

The index does hold a READ-ONLY connection of its own, because it loads photos
lazily and the store it was opened from is gone by the time a request arrives.
That connection is thread-local and its LRU cache is behind a lock, both
inside `MemoryIndex`, so sharing one index across request threads is still
safe - but it is safe because that class arranges it, not because the object
is inert. See `memory.index`.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rekindle import config
from rekindle.db import PhotoStore
from rekindle.memory.index import MemoryIndex
from rekindle.memory.policy import CONFIG_NAME, PolicyError, load_policy
from rekindle.models import Photo

DB_NAME = "rekindle.sqlite"

#: `f(text_or_hash, k) -> [(file_hash, score), ...]`, best first. The same
#: shape `memory.prompt.Retriever` expects, so the prompt path takes one
#: unchanged.
Retrieve = Callable[[str, int], list[tuple[str, float]]]

#: Distinguishes "never looked" from "looked and there is none", so the
#: 60 MB matrix load is attempted exactly once.
_UNSET = object()

STATE_LOADING = "loading"
STATE_READY = "ready"
STATE_FAILED = "failed"


class LibraryError(RuntimeError):
    """Something the user has to fix, phrased as the sentence to show them."""


@dataclass(frozen=True)
class SearchHit:
    file_hash: str
    score: float
    why: str


class Library:
    """Everything a request needs, and nothing a request may bypass."""

    def __init__(self, data_dir: Path, *, public_safe: bool = False) -> None:
        self.data_dir = data_dir
        self.public_safe = public_safe
        self.state = STATE_LOADING
        self.error = ""
        self.index: MemoryIndex | None = None
        self._ready = threading.Event()
        self._semantic_lock = threading.Lock()
        self._retriever: Retrieve | None = None
        self._support: object = _UNSET
        self._similar: Retrieve | None = None
        self.semantic_note = ""
        self.semantic_error = ""

    # ---- lifecycle

    @property
    def db_path(self) -> Path:
        return self.data_dir / DB_NAME

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self.load, name="rekindle-index", daemon=True)
        thread.start()
        return thread

    def load(self) -> None:
        try:
            if not self.db_path.is_file():
                raise LibraryError(
                    f"No index at {self.db_path}. Run `rekindle index <folder>` first."
                )
            # Thresholds first, so the index and every later request see
            # the same numbers. `open_index` does this too; the UI has its own
            # entry point and must not be the one that skips it.
            config.activate_from(self.data_dir)
            policy = load_policy(self.data_dir / CONFIG_NAME)
            store = PhotoStore(self.db_path)
            try:
                # Dismissals are merged into the file policy here for exactly
                # the reason `memory/cli.py:open_index` does it: a person
                # excluded with `rekindle exclude` and one excluded in
                # exclusions.toml must be enforced by the same code.
                from rekindle.memory.history import MemoryState

                policy = MemoryState(store).apply_to(policy).with_public_safe(self.public_safe)
                self.index = MemoryIndex.open(store, policy)
            finally:
                store.close()
            self.state = STATE_READY
        except (LibraryError, PolicyError, config.ConfigError, ValueError, OSError) as exc:
            self.state = STATE_FAILED
            self.error = str(exc)
        finally:
            self._ready.set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._ready.wait(timeout)

    def require_index(self) -> MemoryIndex:
        if self.state == STATE_LOADING:
            raise LibraryError("The library is still being read. Try again in a moment.")
        if self.index is None:
            raise LibraryError(self.error or "The library could not be opened.")
        return self.index

    # ---- writes, each in its own connection

    def with_state(self, work: Callable[[object], object]) -> object:
        """Run `work(MemoryState)` against a freshly opened store.

        Every mutation goes through here so that no `sqlite3` connection is
        ever touched from a thread other than the one that made it.
        """
        from rekindle.memory.history import MemoryState

        if not self.db_path.is_file():
            raise LibraryError(f"No index at {self.db_path}.")
        store = PhotoStore(self.db_path)
        try:
            return work(MemoryState(store))
        finally:
            store.close()

    # ---- metadata search: always available, no model, no extra

    def search_metadata(self, query: str, *, limit: int = 60) -> list[SearchHit]:
        """Find photos by what the index already knows about them.

        Album title, person name, description, file name and an ISO date
        prefix, matched case-insensitively as substrings. This is not a
        semantic search and does not pretend to be - it is the half of "search
        the library" that works on an install with no extras, and on a library
        with face tags it is the fastest way to find the shot the recipe
        missed.

        Ordered by capture time, newest first, with `file_hash` breaking ties,
        so the same query always returns the same page.
        """
        index = self.require_index()
        needle = query.strip().casefold()
        if not needle:
            return []
        hits: list[tuple[Photo, str]] = []
        # Streamed: only the MATCHES are kept, so a search over a large
        # library costs the hits rather than the library.
        for photo in index.iter_all():
            why = _metadata_match(photo, needle)
            if why:
                hits.append((photo, why))
        hits.sort(key=lambda pw: (pw[0].meta.taken_at_utc, pw[0].file_hash), reverse=True)
        return [SearchHit(p.file_hash, 0.0, why) for p, why in hits[:limit]]

    # ---- semantic search: optional, and honest when it is missing

    def semantic_available(self) -> bool:
        from rekindle.semantic.availability import probe

        return probe().any

    def semantic_support(self):
        """The embedding store as a diversity/dedup signal, or None.

        Distinct from `retriever`: that one needs the TEXT encoder, which is a
        1.7 GB CLIP model. This needs only the stored image vectors, so a
        library that was embedded on another machine can still be calibrated
        and its memories re-selected here. Never raises - "no embeddings" is a
        supported configuration.
        """
        with self._semantic_lock:
            if self._support is _UNSET:
                try:
                    from rekindle.semantic.diversity import open_support

                    self._support = open_support(self.data_dir)
                except Exception:  # noqa: BLE001 - any failure means "not available"
                    self._support = None
            return self._support

    def retriever(self) -> Retrieve:
        """The embedding-backed retriever, loaded once, on first use.

        Raises `LibraryError` carrying the install or embed command. The whole
        point is that the UI without the extra is a working UI with one panel
        greyed out, not a traceback.
        """
        with self._semantic_lock:
            if self._retriever is None:
                self._load_semantic()
            assert self._retriever is not None
            return self._retriever

    def similar(self) -> Retrieve:
        with self._semantic_lock:
            if self._similar is None:
                self._load_semantic()
            assert self._similar is not None
            return self._similar

    def _load_semantic(self) -> None:
        """Open the embedding store and the text encoder.

        Deliberately mirrors `memory/cli.py:_retriever_for` rather than
        importing it: that function is a CLI helper that raises typer's own
        exceptions and prints through a rich Console, neither of which belongs
        in an HTTP handler. The two differences are the exception type and the
        added photo-to-photo neighbour lookup the UI's "more like this" needs.
        """
        from rekindle.semantic.availability import SemanticUnavailable, require
        from rekindle.semantic.encoder import load_encoder
        from rekindle.semantic.registry import embed_model
        from rekindle.semantic.search import SemanticSearch
        from rekindle.semantic.setup import cache_dir_for
        from rekindle.semantic.store import EmbeddingStore, store_root

        try:
            require(feature="Prompt memories")
        except SemanticUnavailable as exc:
            self.semantic_error = str(exc)
            raise LibraryError(str(exc)) from exc

        spec = embed_model(None)
        root = store_root(self.data_dir, spec.key)
        if not (root / "manifest.sqlite").is_file():
            message = (
                f"No embeddings for '{spec.key}' at {root}. Run `rekindle semantic embed` first."
            )
            self.semantic_error = message
            raise LibraryError(message)
        store = EmbeddingStore(
            root,
            dim=spec.dim,
            model_key=spec.key,
            model_revision=spec.pin("torch").revision if spec.torch else "",
        )
        search = SemanticSearch(store, None)
        if not len(search.matrix):
            message = (
                f"The embedding store at {root} holds no vectors. "
                "Run `rekindle semantic embed` first."
            )
            self.semantic_error = message
            raise LibraryError(message)
        loaded = load_encoder(spec.key, device="auto", cache_dir=cache_dir_for(self.data_dir))

        self._retriever, self._similar = semantic_callables(search, loaded.encoder)
        self.semantic_note = (
            f"{len(search.matrix)} vectors, model {spec.key}, "
            f"{loaded.runtime} on {loaded.device.device}"
        )
        self.semantic_error = ""


def semantic_callables(search, encoder) -> tuple[Retrieve, Retrieve]:
    """`(retrieve, neighbours)` over an already-open search.

    A free function, not a closure buried in `_load_semantic`, so that
    `tests/test_web_semantic.py` can drive THIS code with a toy encoder and a
    toy store. Building the pair inside the loader would leave the test
    asserting against its own copy of the logic, which is the shape of a test
    that cannot fail.

    **`SemanticSearch.similar_to` is deliberately not used here**, and the
    reason is a bug found by running the real UI against the real 18,201-vector
    store. `similar_to` begins with `store.get(file_hash)`, which reads the
    manifest over the `EmbeddingStore`'s own `sqlite3` connection - and a
    `sqlite3` connection belongs to the thread that created it. The semantic
    layer is loaded lazily by whichever request thread asks first, so the
    second "more like this" landed on a different thread, raised
    `ProgrammingError`, and closed the socket with no response at all. Text
    search hid it completely, because `search_vector` reads only the in-memory
    matrix.

    So the vector is taken from that same in-memory matrix - whose rows are the
    store's rows by construction, an invariant `load_matrix` validates - and
    handed to the very function `similar_to` would have handed it to. Nothing
    below touches a connection, so every request thread is safe.
    """
    from rekindle.semantic.search import embed_query

    rows = {file_hash: i for i, file_hash in enumerate(search.matrix.hashes)}

    def retrieve(text: str, k: int) -> list[tuple[str, float]]:
        hits = search.search_vector(embed_query(encoder, text), k=k)
        return [(h.file_hash, h.score) for h in hits]

    def neighbours(file_hash: str, k: int) -> list[tuple[str, float]]:
        row = rows.get(file_hash)
        if row is None:
            return []
        hits = search.search_vector(search.matrix.data[row].tolist(), k=k + 1)
        # A photo is its own nearest neighbour at cosine 1.0.
        return [(h.file_hash, h.score) for h in hits if h.file_hash != file_hash][:k]

    return retrieve, neighbours


def _metadata_match(photo: Photo, needle: str) -> str:
    """The first field of `photo` that contains `needle`, named, or ""."""
    for album in photo.albums:
        if needle in album.casefold():
            return f"album {album}"
    for person in photo.meta.people:
        if person and needle in person.casefold():
            return f"person {person}"
    description = photo.meta.description
    if description and needle in description.casefold():
        return "description"
    local = photo.meta.taken_at_local
    if local and local.isoformat().startswith(needle):
        return f"date {local.date().isoformat()}"
    for path in photo.paths:
        if needle in path.name.casefold():
            return f"file {path.name}"
    return ""
