"""A narrow, READ-ONLY view of the photo index, for the semantic pipeline.

WHY NOT `PhotoStore`
--------------------
Three reasons, in order of how much they bind.

1. `PhotoStore.__init__` WRITES. It runs `executescript(_SCHEMA)`, an
   `INSERT OR IGNORE` into `meta`, and `commit()` before it does anything
   else, and it creates the file if it is absent. The semantic pipeline reads
   a library it has no business modifying - `rekindle embed` is a read of the
   index and a write of a store beside it - and pointing a writing opener at
   somebody's index to answer "which photos need embedding?" is how an index
   gets damaged by a tool that only meant to look.

2. `PhotoStore` refuses any schema version but its own. That is correct for
   the writer. It is wrong here: this milestone and M2 are being built at the
   same time against the same index, M2 bumps `SCHEMA_VERSION`, and a reader
   that needs eleven columns has no reason to care which of them is current.
   The columns this module reads have existed since v1 (or v2, for the
   Takeout flags) and are additive-only, so pinning the exact version buys
   nothing and costs the ability to read a library the user already has.

3. It keeps `db.py` untouched. Nothing in this milestone edits the shared
   store, which is the whole reason the embeddings live in their own
   directory too.

The cost is that a column rename in `photos` breaks this module without
breaking a type check. `require_columns` therefore checks the column set at
open time and names what is missing, rather than dying later inside a SELECT.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType

#: Google Takeout names its per-year dump folders "Photos from 2019". `index`
#: turns every containing folder into an album, so 100% of a Takeout library
#: has an album and "has an album" means nothing. Measured on the reference
#: export: 19,480/19,480 photos have >=1 album, 15,388 of the 19,318 live ones
#: have NO album that is not one of these buckets.
DATE_BUCKET_PREFIX = "Photos from "

_COLUMNS = (
    "file_hash",
    "media_type",
    "paths",
    "albums",
    "people",
    "takeout_people",
    "archived",
    "trashed",
    "taken_at_utc",
    "taken_at_local",
    "width",
    "height",
)


class IndexUnavailable(RuntimeError):
    """The index is missing, or is not a rekindle index."""


@dataclass(frozen=True)
class IndexedPhoto:
    file_hash: str
    media_type: str
    paths: tuple[Path, ...]
    albums: tuple[str, ...]
    people: tuple[str, ...]
    archived: bool
    trashed: bool
    taken_at_utc: str | None = None
    taken_at_local: str | None = None
    width: int | None = None
    height: int | None = None

    @property
    def path(self) -> Path:
        return self.paths[0]

    @property
    def real_albums(self) -> tuple[str, ...]:
        """Albums that are not a Takeout year bucket."""
        return tuple(a for a in self.albums if not a.startswith(DATE_BUCKET_PREFIX))

    def existing_path(self) -> Path | None:
        """The first path that is actually on disk, or None.

        A photo indexed from two folders keeps both; one of them may since
        have been deleted or be on an unmounted drive. Embedding must skip
        that photo with a reason, not raise.
        """
        for p in self.paths:
            if p.is_file():
                return p
        return None


@dataclass
class ReadFilter:
    """Which rows the semantic pipeline is allowed to touch.

    `archived` defaults to EXCLUDED. A Google-archived photo is one the user
    deliberately hid; 162 of them exist on the reference export. Nothing in
    this milestone may resurface them - not a search hit, not a cluster
    member, not a face-gate candidate - so the exclusion lives at the reader,
    where every consumer inherits it, rather than in each consumer.
    """

    images_only: bool = True
    include_archived: bool = False
    include_trashed: bool = False
    albums: tuple[str, ...] = ()
    without_people: bool = False
    limit: int | None = None
    _: field(default=None, init=False, repr=False) = None  # type: ignore[valid-type]


class PhotoIndexReader:
    """Read-only access to the `photos` table. Never writes, never migrates."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        if not db_path.is_file():
            raise IndexUnavailable(f"No index at {db_path}. Run `rekindle index <folder>` first.")
        # mode=ro, not immutable=1: the index may legitimately be open for
        # writing by another rekindle process, and `immutable` promises SQLite
        # the file cannot change, which would let it serve a stale page cache.
        uri = f"file:{db_path.as_posix()}?mode=ro"
        try:
            self._conn = sqlite3.connect(uri, uri=True)
        except sqlite3.OperationalError as exc:  # pragma: no cover - OS dependent
            raise IndexUnavailable(f"Cannot open {db_path} read-only: {exc}") from exc
        self._conn.row_factory = sqlite3.Row
        try:
            self._require_columns()
        except BaseException:
            self._conn.close()
            raise

    def _require_columns(self) -> None:
        try:
            found = {r["name"] for r in self._conn.execute("PRAGMA table_info(photos)")}
        except sqlite3.DatabaseError as exc:
            raise IndexUnavailable(f"{self.db_path} is not a SQLite database: {exc}") from exc
        if not found:
            raise IndexUnavailable(f"{self.db_path} has no `photos` table. Is it a rekindle index?")
        missing = [c for c in _COLUMNS if c not in found]
        if missing:
            raise IndexUnavailable(
                f"{self.db_path} is missing column(s) {', '.join(missing)} from "
                "`photos`. It was written by a rekindle too old for semantic "
                "features; re-run `rekindle index` and `rekindle enrich`."
            )

    def schema_version(self) -> int | None:
        """Reported, never enforced. See the module docstring."""
        try:
            row = self._conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        except sqlite3.DatabaseError:
            return None
        if row is None:
            return None
        try:
            return int(row["value"])
        except (TypeError, ValueError):
            return None

    def index_root(self) -> str | None:
        try:
            row = self._conn.execute("SELECT value FROM meta WHERE key='index_root'").fetchone()
        except sqlite3.DatabaseError:
            return None
        return row["value"] if row else None

    def count(self, where: ReadFilter | None = None) -> int:
        sql, params = self._where(where or ReadFilter())
        return int(
            self._conn.execute(f"SELECT COUNT(*) AS n FROM photos WHERE {sql}", params).fetchone()[
                "n"
            ]
        )

    @staticmethod
    def _where(f: ReadFilter) -> tuple[str, list[object]]:
        clauses = ["1=1"]
        params: list[object] = []
        if f.images_only:
            clauses.append("media_type = 'image'")
        if not f.include_archived:
            clauses.append("archived = 0")
        if not f.include_trashed:
            clauses.append("trashed = 0")
        if f.without_people:
            # `people` is the merged list; `takeout_people` is the subset the
            # last enrich contributed. A photo is "untagged" only when BOTH
            # are empty - checking `people` alone would be enough today but
            # silently wrong for any source that writes one and not the other.
            clauses.append("people = '[]' AND takeout_people = '[]'")
        return " AND ".join(clauses), params

    def iter_photos(self, where: ReadFilter | None = None) -> Iterator[IndexedPhoto]:
        f = where or ReadFilter()
        sql, params = self._where(f)
        cols = ", ".join(_COLUMNS)
        query = f"SELECT {cols} FROM photos WHERE {sql} ORDER BY file_hash"
        if f.limit is not None:
            query += " LIMIT ?"
            params = [*params, f.limit]
        for row in self._conn.execute(query, params):
            photo = self._to_photo(row)
            if f.albums and not (set(f.albums) & set(photo.albums)):
                continue
            yield photo

    def get(self, file_hash: str) -> IndexedPhoto | None:
        cols = ", ".join(_COLUMNS)
        row = self._conn.execute(
            f"SELECT {cols} FROM photos WHERE file_hash = ?", (file_hash,)
        ).fetchone()
        return self._to_photo(row) if row else None

    def get_many(self, hashes: list[str]) -> dict[str, IndexedPhoto]:
        """One SELECT per 500 hashes, not one per hash.

        Search resolves a top-k of hashes to photos; 19,480 individual SELECTs
        to render 20 results would be absurd, but so would loading the whole
        table to find 20 rows.
        """
        out: dict[str, IndexedPhoto] = {}
        cols = ", ".join(_COLUMNS)
        for i in range(0, len(hashes), 500):
            chunk = hashes[i : i + 500]
            marks = ",".join("?" * len(chunk))
            for row in self._conn.execute(
                f"SELECT {cols} FROM photos WHERE file_hash IN ({marks})", chunk
            ):
                photo = self._to_photo(row)
                out[photo.file_hash] = photo
        return out

    @staticmethod
    def _to_photo(row: sqlite3.Row) -> IndexedPhoto:
        return IndexedPhoto(
            file_hash=row["file_hash"],
            media_type=row["media_type"],
            paths=tuple(Path(p) for p in json.loads(row["paths"])),
            albums=tuple(json.loads(row["albums"])),
            people=tuple(json.loads(row["people"])),
            archived=bool(row["archived"]),
            trashed=bool(row["trashed"]),
            taken_at_utc=row["taken_at_utc"],
            taken_at_local=row["taken_at_local"],
            width=row["width"],
            height=row["height"],
        )

    def __enter__(self) -> PhotoIndexReader:
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
