"""The embedding store: a flat float32 matrix plus its own SQLite manifest.

WHY THIS IS NOT A COLUMN ON `photos`
------------------------------------
A 1152-dim float32 vector is 4.6 KB. Nineteen thousand of them is 88 MB of
BLOBs in a table that `iter_photos()` streams in full on every enrich run, and
every similarity query would have to deserialise all of them through sqlite3.
That is the performance argument, and it is real but secondary.

The binding argument is concurrency of PEOPLE, not processes. M2 is adding
columns and bumping `SCHEMA_VERSION` in the same file at the same time. Two
independent stores merge as two independent additions; two sets of columns on
one table plus two migrations racing for `SCHEMA_VERSION = 3` is a conflict
someone has to resolve by hand, on a schema migration, which is the worst
place to resolve one. So this milestone touches neither `db.py` nor the
`photos` table at all: it keeps its own directory, its own schema, and its own
version counter.

ON-DISK LAYOUT
--------------
    <data-dir>/semantic/<store_id>/
        vectors.f32      row-major float32, little-endian, `dim` per row
        manifest.sqlite  which file_hash owns which row, and what wrote it

`store_id` is the model key, so two models can be embedded side by side and
compared - which is exactly how this milestone chose its model.

THE STALENESS PROBLEM, AND WHY THE FILE HASH DOES NOT SOLVE IT
--------------------------------------------------------------
This docstring used to say "NO STALENESS PROBLEM": `file_hash` is BLAKE2b-128
of the file bytes, so the content behind a hash cannot change, so a vector is
valid forever for its (hash, model, revision) triple.

That is false, and it cost two full re-embeds. **A vector is computed from
DECODED PIXELS, not from bytes.** 2,503 of the reference library's images
carry an axis-swapping EXIF orientation tag, and `meta.orientation` can prove
such a tag STALE from image content - at which point `open_upright` stops
applying it and hands back different pixels for *byte-identical* input. The
hash does not move. Neither does the model, nor its revision. The vector is
nevertheless wrong, and nothing in this store could say so: `embed_photos`
skips any hash the store already holds, so 13.8% of the live store stayed
sideways through the fix that was supposed to repair it.

So each row now records a **decode key**: an opaque short string, supplied by
the caller, naming everything the decode depended on that the file hash does
NOT already pin down. `semantic.embed.decode_key` builds it; see there for
what goes in it and why the EXIF tag itself does not need to. This store only
compares the strings - it has no opinion about decoding, and must not, since
it is the stdlib-only module.

`decode_key` is NULLABLE, and NULL means "written before this store recorded
provenance", not "fresh". `plan()` reports those separately as `unverified`
rather than pretending either way; redoing them is the caller's explicit
choice, because on a contributor's CPU a full re-embed is five hours.

Adding the column did NOT bump `STORE_SCHEMA_VERSION`. That counter means
"an older reader would MISREAD this store", and an older reader would not:
`vectors.f32` is byte-identical, and the old manifest statements
(`INSERT INTO vectors(file_hash, row, embedded_at)`, `SELECT file_hash, row`)
are all still valid against the wider table. Bumping it would instead have
made every existing store refuse to open and demand the very full rebuild
this feature exists to avoid.

The manifest still records the model revision so that swapping a model
invalidates everything by landing in a different directory, not by silently
mixing two embedding spaces.

STDLIB ONLY. numpy is an optional extra; this module is how `rekindle` reports
on a store, and reporting must work on a default install. Bulk maths lives in
`vectors.py`, which does need numpy.
"""

from __future__ import annotations

import sqlite3
import sys
from array import array
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

#: Bumped when the on-disk layout changes in a way an older reader would
#: misread. Independent of `db.SCHEMA_VERSION` on purpose - see module docstring.
STORE_SCHEMA_VERSION = 1

ITEM_SIZE = 4  # float32

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS vectors (
    file_hash   TEXT PRIMARY KEY,
    row         INTEGER NOT NULL UNIQUE,
    embedded_at TEXT NOT NULL,
    -- What the vector was computed FROM, beyond the bytes the hash pins.
    -- NULL on rows written before this column existed: unknown, not fresh.
    decode_key  TEXT
);

CREATE INDEX IF NOT EXISTS idx_vectors_row ON vectors(row);
"""


class StoreError(RuntimeError):
    """The store on disk cannot be used as asked."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _to_le(values: Sequence[float]) -> bytes:
    """Pack floats as little-endian float32, whatever this CPU prefers.

    Every platform rekindle supports today is little-endian, so the byteswap
    branch is unreachable in practice - but a store written on one machine and
    read on another is an explicit use case (it is a file in a data directory,
    not a process-local cache), and a silently byte-reversed vector produces
    plausible-looking garbage rather than an error.
    """
    buf = array("f", values)
    if sys.byteorder != "little":
        buf.byteswap()
    return buf.tobytes()


def _from_le(raw: bytes) -> list[float]:
    buf = array("f")
    buf.frombytes(raw)
    if sys.byteorder != "little":
        buf.byteswap()
    return buf.tolist()


@dataclass
class EmbedPlan:
    """What a re-embed would have to do, split by WHY.

    Three buckets, not two, because "has no vector" and "has a vector computed
    from pixels we no longer produce" are different costs, and "has a vector
    and we cannot tell" is a third thing that must not be quietly folded into
    either.
    """

    #: No vector at all.
    missing: list[str] = field(default_factory=list)
    #: A vector exists and its recorded decode key DISAGREES with the current one.
    stale: list[str] = field(default_factory=list)
    #: A vector exists but predates provenance recording. Unknown, not fresh.
    unverified: list[str] = field(default_factory=list)
    #: A vector exists and its recorded decode key matches. Nothing to do.
    fresh: int = 0
    #: Hashes offered more than once in one call.
    duplicates: int = 0

    def to_embed(self, *, include_unverified: bool = False) -> list[str]:
        """The work, in a stable order: missing first, then stale."""
        out = [*self.missing, *self.stale]
        if include_unverified:
            out.extend(self.unverified)
        return out


class EmbeddingStore:
    """Append-only vectors keyed by `file_hash`, with a manifest beside them."""

    def __init__(
        self,
        root: Path,
        *,
        dim: int | None = None,
        model_key: str | None = None,
        model_revision: str | None = None,
        normalized: bool = True,
    ) -> None:
        """Open (or create) the store at `root`.

        `dim` and `model_key` are required to CREATE a store and are verified
        against the manifest when one already exists. Passing a different dim
        or model to an existing directory is an error, not a silent rewrite:
        two embedding spaces in one matrix is a bug whose only symptom is bad
        search results.
        """
        self.root = root
        self.vectors_path = root / "vectors.f32"
        self.manifest_path = root / "manifest.sqlite"
        root.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.manifest_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()
        try:
            self._init_meta(dim, model_key, model_revision, normalized)
            self._reconcile()
        except BaseException:
            # Same reasoning as PhotoStore: raising from __init__ leaves the
            # caller no handle to close, and on Windows an open sqlite handle
            # keeps the file locked until GC.
            self._conn.close()
            raise

    # ---------------------------------------------------------------- setup

    def _migrate(self) -> None:
        """Add columns a store written by an older rekindle does not have.

        Additive only, and idempotent. `executescript(_SCHEMA)` above uses
        `CREATE TABLE IF NOT EXISTS`, which does exactly nothing to a table
        that already exists - so a column added to `_SCHEMA` never reaches a
        store somebody already has. That is the bug this method exists to
        avoid, and it is the same shape as the one `db.py` solved with its
        `_COLUMN_ADDITIONS` ladder.

        Deliberately NOT a version bump: see the module docstring.
        """
        found = {r["name"] for r in self._conn.execute("PRAGMA table_info(vectors)")}
        if "decode_key" not in found:
            self._conn.execute("ALTER TABLE vectors ADD COLUMN decode_key TEXT")

    def _init_meta(
        self,
        dim: int | None,
        model_key: str | None,
        model_revision: str | None,
        normalized: bool,
    ) -> None:
        stored_version = self.get_meta("store_schema_version")
        if stored_version is None:
            if dim is None or model_key is None:
                raise StoreError(
                    f"{self.root} holds no embedding store yet, and creating one "
                    "needs both `dim` and `model_key`."
                )
            self._set_meta_many(
                {
                    "store_schema_version": str(STORE_SCHEMA_VERSION),
                    "dim": str(dim),
                    "model_key": model_key,
                    "model_revision": model_revision or "",
                    "normalized": "1" if normalized else "0",
                    "created_at": _now(),
                }
            )
            return

        found = int(stored_version)
        if found != STORE_SCHEMA_VERSION:
            raise StoreError(
                f"{self.manifest_path} uses embedding-store schema v{found}, this "
                f"rekindle expects v{STORE_SCHEMA_VERSION}. Delete "
                f"{self.root} and re-run `rekindle embed`."
            )
        if dim is not None and dim != self.dim:
            raise StoreError(
                f"{self.root} stores {self.dim}-dim vectors, but the caller asked "
                f"for {dim}. Two embedding spaces cannot share one matrix."
            )
        if model_key is not None and model_key != self.model_key:
            raise StoreError(
                f"{self.root} was written by model '{self.model_key}', not "
                f"'{model_key}'. Each model gets its own store directory."
            )
        if model_revision and self.model_revision and model_revision != self.model_revision:
            raise StoreError(
                f"{self.root} was written at model revision "
                f"{self.model_revision}, but the caller is at {model_revision}. "
                "Vectors from two revisions are not comparable. Delete "
                f"{self.root} and re-run `rekindle embed`."
            )

    def _reconcile(self) -> None:
        """Make the matrix file agree with the manifest, or refuse to open.

        A crash between `_append_vector` (bytes to disk) and the manifest
        commit leaves a tail of orphan rows nothing references. That tail is
        harmless data but it is also the exact shape of a corrupt store, so it
        is truncated rather than tolerated: the next write would otherwise
        allocate row N over bytes already occupied by the orphan.

        The reverse - manifest rows past the end of the file - is not
        recoverable and raises, because some hash would silently read another
        photo's vector (or off the end).
        """
        expected = self.count() * self.dim * ITEM_SIZE
        if not self.vectors_path.exists():
            if expected:
                raise StoreError(
                    f"{self.manifest_path} lists {self.count()} vectors but "
                    f"{self.vectors_path} does not exist. Delete {self.root} "
                    "and re-run `rekindle embed`."
                )
            self.vectors_path.touch()
            return
        actual = self.vectors_path.stat().st_size
        if actual < expected:
            raise StoreError(
                f"{self.vectors_path} is {actual} bytes but "
                f"{self.manifest_path} lists {self.count()} vectors of "
                f"{self.dim} dims ({expected} bytes). The store is truncated. "
                f"Delete {self.root} and re-run `rekindle embed`."
            )
        if actual > expected:
            # Orphan tail from an interrupted run.
            with self.vectors_path.open("r+b") as fh:
                fh.truncate(expected)

    # ----------------------------------------------------------- meta access

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def _set_meta_many(self, values: dict[str, str]) -> None:
        self._conn.executemany(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            list(values.items()),
        )
        self._conn.commit()

    def set_meta(self, key: str, value: str) -> None:
        self._set_meta_many({key: value})

    @property
    def dim(self) -> int:
        raw = self.get_meta("dim")
        if raw is None:
            raise StoreError(f"{self.manifest_path} has no `dim` - it is not a store.")
        return int(raw)

    @property
    def model_key(self) -> str:
        return self.get_meta("model_key") or ""

    @property
    def model_revision(self) -> str:
        return self.get_meta("model_revision") or ""

    @property
    def normalized(self) -> bool:
        return self.get_meta("normalized") == "1"

    # ---------------------------------------------------------------- counts

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) AS n FROM vectors").fetchone()["n"])

    def hashes(self) -> set[str]:
        return {r["file_hash"] for r in self._conn.execute("SELECT file_hash FROM vectors")}

    def has(self, file_hash: str) -> bool:
        return (
            self._conn.execute("SELECT 1 FROM vectors WHERE file_hash = ?", (file_hash,)).fetchone()
            is not None
        )

    def missing(self, hashes: Iterable[str]) -> list[str]:
        """Which of `hashes` have no vector yet, in the order given, deduped.

        Order-preserving on purpose: `rekindle embed` reports progress against
        it, and a set would shuffle the work differently on every run, which
        makes a partial run's throughput numbers incomparable.
        """
        present = self.hashes()
        seen: set[str] = set()
        out = []
        for h in hashes:
            if h not in present and h not in seen:
                seen.add(h)
                out.append(h)
        return out

    # ------------------------------------------------------------ provenance

    def decode_key_of(self, file_hash: str) -> str | None:
        """What this vector was computed from, or None if it predates the column."""
        row = self._conn.execute(
            "SELECT decode_key FROM vectors WHERE file_hash = ?", (file_hash,)
        ).fetchone()
        return row["decode_key"] if row else None

    def decode_keys(self) -> dict[str, str | None]:
        """Every held hash mapped to its decode key. One query, not one per hash."""
        return {
            r["file_hash"]: r["decode_key"]
            for r in self._conn.execute("SELECT file_hash, decode_key FROM vectors")
        }

    def unverified_hashes(self) -> set[str]:
        """Hashes whose vector predates provenance recording."""
        return {
            r["file_hash"]
            for r in self._conn.execute("SELECT file_hash FROM vectors WHERE decode_key IS NULL")
        }

    def plan(self, wanted: Iterable[tuple[str, str]]) -> EmbedPlan:
        """Sort `(file_hash, decode_key)` into what must be recomputed and what must not.

        `decode_key` is what the CALLER would compute for that photo right
        now. A row whose recorded key differs was computed from different
        pixels and is `stale`; a row with no recorded key at all is
        `unverified` and is neither claimed fresh nor silently redone.

        Order-preserving and deduped, for the same reason `missing` is.
        """
        held = self.decode_keys()
        plan = EmbedPlan()
        seen: set[str] = set()
        for file_hash, key in wanted:
            if file_hash in seen:
                plan.duplicates += 1
                continue
            seen.add(file_hash)
            if file_hash not in held:
                plan.missing.append(file_hash)
                continue
            recorded = held[file_hash]
            if recorded is None:
                plan.unverified.append(file_hash)
            elif recorded != key:
                plan.stale.append(file_hash)
            else:
                plan.fresh += 1
        return plan

    # ----------------------------------------------------------------- rows

    def _next_row(self) -> int:
        row = self._conn.execute("SELECT MAX(row) AS m FROM vectors").fetchone()
        return 0 if row["m"] is None else int(row["m"]) + 1

    def _row_of(self, file_hash: str) -> int | None:
        row = self._conn.execute(
            "SELECT row FROM vectors WHERE file_hash = ?", (file_hash,)
        ).fetchone()
        return int(row["row"]) if row else None

    def add_many(
        self,
        items: Iterable[tuple[str, Sequence[float]]],
        *,
        decode_keys: Mapping[str, str] | None = None,
    ) -> int:
        """Write vectors for `items`, returning how many rows were written.

        Bytes first, then the manifest, in ONE manifest transaction. The order
        matters: a crash after the bytes and before the commit leaves an
        orphan tail that `_reconcile` truncates on the next open. The reverse
        order would leave a manifest row pointing past the end of the file,
        which `_reconcile` can only refuse.

        `decode_keys` says what each vector was computed from. A hash absent
        from it is stored with a NULL key, INCLUDING when it overwrites a row
        that had one: a caller who will not say what it decoded has made the
        row unverified, and carrying the old key forward would assert a
        provenance nobody vouched for. That is precisely the failure this
        column exists to prevent, so it is not weakened here for convenience.
        """
        batch = list(items)
        if not batch:
            return 0
        dim = self.dim
        for file_hash, vec in batch:
            if len(vec) != dim:
                raise StoreError(f"vector for {file_hash} has {len(vec)} dims, store holds {dim}")

        next_row = self._next_row()
        assignments: list[tuple[str, int, bytes]] = []
        for file_hash, vec in batch:
            existing = self._row_of(file_hash)
            if existing is None:
                assignments.append((file_hash, next_row, _to_le(vec)))
                next_row += 1
            else:
                assignments.append((file_hash, existing, _to_le(vec)))

        stride = dim * ITEM_SIZE
        with self.vectors_path.open("r+b") as fh:
            for _, row, raw in assignments:
                fh.seek(row * stride)
                fh.write(raw)
            fh.flush()
        stamp = _now()
        keys = decode_keys or {}
        self._conn.executemany(
            "INSERT INTO vectors(file_hash, row, embedded_at, decode_key) VALUES(?, ?, ?, ?) "
            "ON CONFLICT(file_hash) DO UPDATE SET embedded_at = excluded.embedded_at, "
            "decode_key = excluded.decode_key",
            [(h, row, stamp, keys.get(h)) for h, row, _ in assignments],
        )
        self._conn.commit()
        return len(assignments)

    def get(self, file_hash: str) -> list[float] | None:
        row = self._row_of(file_hash)
        if row is None:
            return None
        stride = self.dim * ITEM_SIZE
        with self.vectors_path.open("rb") as fh:
            fh.seek(row * stride)
            raw = fh.read(stride)
        if len(raw) != stride:
            raise StoreError(f"{self.vectors_path} ended early reading row {row} for {file_hash}")
        return _from_le(raw)

    def iter_rows(self) -> Iterator[tuple[str, int]]:
        """(file_hash, row) for every vector, ordered by row.

        Ordered so a caller can zip it against a matrix read straight off disk
        without a per-row seek.
        """
        for r in self._conn.execute("SELECT file_hash, row FROM vectors ORDER BY row"):
            yield r["file_hash"], int(r["row"])

    def ordered_hashes(self) -> list[str]:
        return [h for h, _ in self.iter_rows()]

    def raw_bytes(self) -> bytes:
        return self.vectors_path.read_bytes()

    # ------------------------------------------------------------- lifecycle

    def __enter__(self) -> EmbeddingStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()


def store_root(data_dir: Path, store_id: str) -> Path:
    return data_dir / "semantic" / store_id
