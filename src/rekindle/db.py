"""SQLite store. The relational source of truth for photos.

Writes are batched into a single transaction per run: per-photo commits are
orders of magnitude slower and generate enormous IO.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path
from types import TracebackType

from rekindle.models import (
    FaceRegion,
    Gps,
    MediaType,
    Photo,
    PhotoMeta,
    TzSource,
    merge_meta,
)

SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS photos (
    file_hash     TEXT PRIMARY KEY,
    media_type    TEXT NOT NULL,
    paths         TEXT NOT NULL,
    albums        TEXT NOT NULL,
    edited_of     TEXT,
    first_seen    TEXT NOT NULL,
    last_seen     TEXT NOT NULL,
    taken_at_utc  TEXT,
    taken_at_local TEXT,
    tz_source     TEXT NOT NULL,
    gps_lat       REAL,
    gps_lon       REAL,
    gps_alt       REAL,
    people        TEXT NOT NULL,
    face_regions  TEXT NOT NULL,
    keywords      TEXT NOT NULL,
    description   TEXT,
    favorite      INTEGER NOT NULL DEFAULT 0,
    camera_make   TEXT,
    camera_model  TEXT,
    width         INTEGER,
    height        INTEGER,
    source        TEXT NOT NULL DEFAULT 'folder',
    metadata_conflict INTEGER NOT NULL DEFAULT 0,
    exif_taken_at_utc TEXT,
    takeout_people TEXT NOT NULL DEFAULT '[]',
    archived      INTEGER NOT NULL DEFAULT 0,
    trashed       INTEGER NOT NULL DEFAULT 0,
    sidecar_match TEXT NOT NULL DEFAULT 'none'
);

-- `paths` is a JSON blob and therefore unqueryable. The enricher must look a
-- photo up by filename 20,000 times; without this table that is a full table
-- scan per lookup.
CREATE TABLE IF NOT EXISTS photo_paths (
    file_hash TEXT NOT NULL,
    path      TEXT NOT NULL,
    name_cf   TEXT NOT NULL,
    parent    TEXT NOT NULL,
    PRIMARY KEY (file_hash, path)
);

CREATE INDEX IF NOT EXISTS idx_photos_taken ON photos(taken_at_utc);
CREATE INDEX IF NOT EXISTS idx_photos_type  ON photos(media_type);
CREATE INDEX IF NOT EXISTS idx_photos_edited ON photos(edited_of);
CREATE INDEX IF NOT EXISTS idx_photo_paths_name ON photo_paths(name_cf);
"""

# Column additions, in order, applied to a database created before them.
# ALTER TABLE ADD COLUMN is the only schema change SQLite does cheaply and it
# is all this migration needs.
_V2_COLUMNS = (
    ("exif_taken_at_utc", "TEXT"),
    ("takeout_people", "TEXT NOT NULL DEFAULT '[]'"),
    ("archived", "INTEGER NOT NULL DEFAULT 0"),
    ("trashed", "INTEGER NOT NULL DEFAULT 0"),
    ("sidecar_match", "TEXT NOT NULL DEFAULT 'none'"),
)


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _undt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class PhotoStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()
        try:
            self._migrate()
        except BaseException:
            # Same reasoning as the version-mismatch branch below: a raise
            # from __init__ leaves the caller no handle to close the
            # connection, which on Windows keeps the file locked until GC or
            # process exit. Close it ourselves so a retry sees the real error
            # (e.g. corrupt JSON) instead of "database is locked".
            self._conn.close()
            raise
        # CREATE TABLE IF NOT EXISTS silently keeps an old table, so a version
        # check is the only thing standing between a schema change and a
        # baffling OperationalError on the next write.
        found = self.schema_version()
        if found != SCHEMA_VERSION:
            # A raise here means __init__ never returns, so the caller gets no
            # handle to close the connection - it would otherwise stay open
            # (and, on Windows, keep the file locked) until GC or process
            # exit. Close it ourselves before propagating.
            self._conn.close()
            if found > SCHEMA_VERSION:
                raise RuntimeError(
                    f"{db_path} was written by a newer rekindle "
                    f"(schema v{found} > v{SCHEMA_VERSION}). Upgrade rekindle."
                )
            raise RuntimeError(
                f"{db_path} uses schema v{found}, this rekindle expects "
                f"v{SCHEMA_VERSION}, and no migration exists for that step. "
                "Delete the file and re-index."
            )

    def _migrate(self) -> None:
        """v1 -> v2. Additive only: no row is ever rewritten or dropped.

        Guards on the EXACT starting version (1), not `< SCHEMA_VERSION`: a
        `>=` guard plus an unconditional bump to `SCHEMA_VERSION` would stamp
        any older database - including a hypothetical pre-v1 schema this
        step knows nothing about - straight to v2 and declare success. That
        leaves the friendly "no migration exists" error in __init__
        unreachable now, and means a future v3 step would silently skip
        migrating a v1 database that was never bumped. A database at any
        version other than 1 is left untouched for __init__'s version check
        to reject with its explicit error message.
        """
        if self.schema_version() != 1:
            return
        existing = {r["name"] for r in self._conn.execute("PRAGMA table_info(photos)")}
        for name, decl in _V2_COLUMNS:
            if name not in existing:
                self._conn.execute(f"ALTER TABLE photos ADD COLUMN {name} {decl}")
        # photo_paths was created empty by _SCHEMA above; backfill it from the
        # JSON blob that was the only path record in v1.
        # fetchall(), not a live cursor: writing through the same connection
        # while iterating a SELECT on it is undefined behaviour in SQLite.
        rows = self._conn.execute("SELECT file_hash, paths FROM photos").fetchall()
        for row in rows:
            for raw in json.loads(row["paths"]):
                self._index_path(self._conn, row["file_hash"], Path(raw))
        self._conn.execute(
            "UPDATE meta SET value = ? WHERE key = 'schema_version'",
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()

    @staticmethod
    def _index_path(cur: sqlite3.Connection | sqlite3.Cursor, digest: str, path: Path) -> None:
        cur.execute(
            "INSERT OR REPLACE INTO photo_paths(file_hash, path, name_cf, parent)"
            " VALUES (?, ?, ?, ?)",
            (digest, str(path), path.name.casefold(), str(path.parent)),
        )

    def __enter__(self) -> PhotoStore:
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

    def schema_version(self) -> int:
        row = self._conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        return int(row["value"])

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) AS n FROM photos").fetchone()["n"])

    def all_hashes(self) -> set[str]:
        return {r["file_hash"] for r in self._conn.execute("SELECT file_hash FROM photos")}

    def iter_photos(self) -> Iterator[Photo]:
        """Every row, streamed. The enricher needs all of them and `get()` per
        hash would be one SELECT per photo - 19,480 of them on a real export.

        WARNING: this yields from a LIVE cursor. Writing to this store while
        iterating it is undefined behaviour in SQLite - the same hazard
        `_migrate` calls `.fetchall()` to avoid. Callers that write must
        materialise first: `photos = list(store.iter_photos())`, as any
        writing caller must - including the planned Takeout enricher.
        """
        for row in self._conn.execute("SELECT * FROM photos"):
            yield self._row_to_photo(row)

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._conn.commit()

    def hashes_for_filename(self, name: str) -> set[str]:
        """Every photo having a path whose filename casefolds to `name`.

        Casefolds `name` itself: a caller passing "X.JPG" must match a path
        stored as "x.jpg" exactly as one passing "x.jpg" would, or this
        becomes a silent "no match" trap for whichever caller doesn't happen
        to pre-casefold its input.
        """
        return {
            r["file_hash"]
            for r in self._conn.execute(
                "SELECT file_hash FROM photo_paths WHERE name_cf = ?", (name.casefold(),)
            )
        }

    def upsert_many(self, photos: Iterable[Photo]) -> tuple[int, int]:
        """Insert or merge. Returns (inserted, updated). One transaction."""
        inserted = updated = 0
        cur = self._conn.cursor()
        for p in photos:
            existing = self.get(p.file_hash)
            if existing is None:
                self._insert(cur, p)
                inserted += 1
            else:
                self._insert(cur, self._merge(existing, p))
                updated += 1
        self._conn.commit()
        return inserted, updated

    def update_photo(self, photo: Photo) -> None:
        """Write a photo AS GIVEN. No merge, no conflict detection.

        Deliberately not upsert_many: merge_meta encodes "earliest real date
        wins", a tiebreak between sources of EQUAL authority. Takeout is not
        equal - it is Google's own record - and routing enrichment through
        that rule would silently discard every date correction.
        """
        self._insert(self._conn.cursor(), photo)
        self._conn.commit()

    def update_many(self, photos: Iterable[Photo]) -> int:
        """update_photo for a batch, in one transaction."""
        cur = self._conn.cursor()
        n = 0
        for photo in photos:
            self._insert(cur, photo)
            n += 1
        self._conn.commit()
        return n

    def update_many_with_meta(self, photos: Iterable[Photo], meta: dict[str, str]) -> int:
        """`update_many`, plus `meta` keys, in ONE transaction.

        For a caller (the Takeout enricher) whose meta keys are PROVENANCE
        for this exact write - `enriched_at` recorded in a separate, later
        transaction would leave a crash window where photos are enriched but
        the row a reader needs to tell "ran and found nothing" apart from
        "never ran" (guard 3) does not exist yet. `set_meta`'s own commit is
        correct for its other callers (e.g. `index_root`, written once with
        nothing else pending); it is wrong here specifically because two
        MORE commits after the photo batch is exactly the "one transaction
        per run" constraint this method exists to restore.
        """
        cur = self._conn.cursor()
        n = 0
        for photo in photos:
            self._insert(cur, photo)
            n += 1
        for key, value in meta.items():
            cur.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
        self._conn.commit()
        return n

    @staticmethod
    def _merge(old: Photo, new: Photo) -> Photo:
        """Union paths and albums; keep the widest time window."""
        paths = list(dict.fromkeys([*old.paths, *new.paths]))
        albums = list(dict.fromkeys([*old.albums, *new.albums]))
        meta, conflict = merge_meta(old.meta, new.meta)
        return Photo(
            file_hash=old.file_hash,
            paths=paths,
            media_type=new.media_type,
            meta=meta,
            first_seen=min(old.first_seen, new.first_seen),
            last_seen=max(old.last_seen, new.last_seen),
            albums=albums,
            edited_of=new.edited_of or old.edited_of,
            source=new.source or old.source,
            metadata_conflict=old.metadata_conflict or new.metadata_conflict or conflict,
            sidecar_match=old.sidecar_match if old.sidecar_match != "none" else new.sidecar_match,
        )

    @staticmethod
    def _insert(cur: sqlite3.Cursor, p: Photo) -> None:
        m = p.meta
        # Columns are named explicitly. Positional VALUES(...) breaks silently
        # the moment a column is added or reordered.
        cur.execute(
            """INSERT OR REPLACE INTO photos
               (file_hash, media_type, paths, albums, edited_of, first_seen,
                last_seen, taken_at_utc, taken_at_local, tz_source, gps_lat,
                gps_lon, gps_alt, people, face_regions, keywords, description,
                favorite, camera_make, camera_model, width, height, source,
                metadata_conflict, exif_taken_at_utc, takeout_people, archived,
                trashed, sidecar_match)
               VALUES
               (:file_hash,:media_type,:paths,:albums,:edited_of,:first_seen,
                :last_seen,:taken_at_utc,:taken_at_local,:tz_source,:gps_lat,
                :gps_lon,:gps_alt,:people,:face_regions,:keywords,:description,
                :favorite,:camera_make,:camera_model,:width,:height,:source,
                :metadata_conflict,:exif_taken_at_utc,:takeout_people,:archived,
                :trashed,:sidecar_match)""",
            {
                "file_hash": p.file_hash,
                "media_type": str(p.media_type),
                "paths": json.dumps([str(x) for x in p.paths]),
                "albums": json.dumps(p.albums),
                "edited_of": p.edited_of,
                "first_seen": _dt(p.first_seen),
                "last_seen": _dt(p.last_seen),
                "taken_at_utc": _dt(m.taken_at_utc),
                "taken_at_local": _dt(m.taken_at_local),
                "tz_source": str(m.tz_source),
                "gps_lat": m.gps.lat if m.gps else None,
                "gps_lon": m.gps.lon if m.gps else None,
                "gps_alt": m.gps.alt if m.gps else None,
                "people": json.dumps(m.people),
                "face_regions": json.dumps(
                    [
                        {"name": r.name, "x": r.x, "y": r.y, "w": r.w, "h": r.h}
                        for r in m.face_regions
                    ]
                ),
                "keywords": json.dumps(m.keywords),
                "description": m.description,
                "favorite": int(m.favorite),
                "camera_make": m.camera_make,
                "camera_model": m.camera_model,
                "width": m.width,
                "height": m.height,
                "source": p.source,
                "metadata_conflict": int(p.metadata_conflict),
                "exif_taken_at_utc": _dt(m.exif_taken_at_utc),
                "takeout_people": json.dumps(m.takeout_people),
                "archived": int(m.archived),
                "trashed": int(m.trashed),
                "sidecar_match": p.sidecar_match,
            },
        )
        cur.execute("DELETE FROM photo_paths WHERE file_hash = ?", (p.file_hash,))
        for path in p.paths:
            PhotoStore._index_path(cur, p.file_hash, path)

    def get(self, file_hash: str) -> Photo | None:
        row = self._conn.execute(
            "SELECT * FROM photos WHERE file_hash = ?", (file_hash,)
        ).fetchone()
        return self._row_to_photo(row) if row else None

    @staticmethod
    def _row_to_photo(row: sqlite3.Row) -> Photo:
        gps = None
        if row["gps_lat"] is not None and row["gps_lon"] is not None:
            gps = Gps(lat=row["gps_lat"], lon=row["gps_lon"], alt=row["gps_alt"])
        meta = PhotoMeta(
            taken_at_utc=_undt(row["taken_at_utc"]),
            taken_at_local=_undt(row["taken_at_local"]),
            tz_source=TzSource(row["tz_source"]),
            gps=gps,
            people=json.loads(row["people"]),
            face_regions=[FaceRegion(**r) for r in json.loads(row["face_regions"])],
            keywords=json.loads(row["keywords"]),
            description=row["description"],
            favorite=bool(row["favorite"]),
            camera_make=row["camera_make"],
            camera_model=row["camera_model"],
            width=row["width"],
            height=row["height"],
            exif_taken_at_utc=_undt(row["exif_taken_at_utc"]),
            takeout_people=json.loads(row["takeout_people"]),
            archived=bool(row["archived"]),
            trashed=bool(row["trashed"]),
        )
        return Photo(
            file_hash=row["file_hash"],
            paths=[Path(x) for x in json.loads(row["paths"])],
            media_type=MediaType(row["media_type"]),
            meta=meta,
            first_seen=_undt(row["first_seen"]),
            last_seen=_undt(row["last_seen"]),
            albums=json.loads(row["albums"]),
            edited_of=row["edited_of"],
            source=row["source"],
            metadata_conflict=bool(row["metadata_conflict"]),
            sidecar_match=row["sidecar_match"],
        )
