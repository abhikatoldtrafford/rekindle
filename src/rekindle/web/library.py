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
The index itself is immutable in-memory data, so every reader thread can share
it without a lock.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rekindle.db import PhotoStore
from rekindle.memory.index import MemoryIndex
from rekindle.memory.policy import CONFIG_NAME, PolicyError, load_policy
from rekindle.models import Photo

DB_NAME = "rekindle.sqlite"

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
        self._retriever: Callable[[str, int], list[tuple[str, float]]] | None = None
        self._similar: Callable[[str, int], list[tuple[str, float]]] | None = None
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
        except (LibraryError, PolicyError, ValueError, OSError) as exc:
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
        for photo in index.all():
            why = _metadata_match(photo, needle)
            if why:
                hits.append((photo, why))
        hits.sort(key=lambda pw: (pw[0].meta.taken_at_utc, pw[0].file_hash), reverse=True)
        return [SearchHit(p.file_hash, 0.0, why) for p, why in hits[:limit]]

    # ---- semantic search: optional, and honest when it is missing

    def semantic_available(self) -> bool:
        from rekindle.semantic.availability import probe

        return probe().any

    def retriever(self) -> Callable[[str, int], list[tuple[str, float]]]:
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

    def similar(self) -> Callable[[str, int], list[tuple[str, float]]]:
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
        from rekindle.semantic.search import SemanticSearch, embed_query
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
        encoder = load_encoder(spec.key, device="auto", cache_dir=cache_dir_for(self.data_dir))

        def retrieve(text: str, k: int) -> list[tuple[str, float]]:
            hits = search.search_vector(embed_query(encoder.encoder, text), k=k)
            return [(h.file_hash, h.score) for h in hits]

        def neighbours(file_hash: str, k: int) -> list[tuple[str, float]]:
            try:
                hits = search.similar_to(file_hash, k=k)
            except KeyError:
                return []
            return [(h.file_hash, h.score) for h in hits]

        self._retriever = retrieve
        self._similar = neighbours
        self.semantic_note = (
            f"{len(search.matrix)} vectors, model {spec.key}, "
            f"{encoder.runtime} on {encoder.device.device}"
        )
        self.semantic_error = ""


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
