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

SCHEMA_VERSION = 1

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
    metadata_conflict INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_photos_taken ON photos(taken_at_utc);
CREATE INDEX IF NOT EXISTS idx_photos_type  ON photos(media_type);
CREATE INDEX IF NOT EXISTS idx_photos_edited ON photos(edited_of);
"""


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
        # CREATE TABLE IF NOT EXISTS silently keeps an old table, so a version
        # check is the only thing standing between a schema change and a
        # baffling OperationalError on the next write.
        found = self.schema_version()
        if found > SCHEMA_VERSION:
            raise RuntimeError(
                f"{db_path} was written by a newer rekindle "
                f"(schema v{found} > v{SCHEMA_VERSION}). Upgrade rekindle."
            )
        if found < SCHEMA_VERSION:
            raise RuntimeError(
                f"{db_path} uses schema v{found}, this rekindle expects "
                f"v{SCHEMA_VERSION}, and no migration exists yet. Delete the "
                "file and re-index."
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
                metadata_conflict)
               VALUES
               (:file_hash,:media_type,:paths,:albums,:edited_of,:first_seen,
                :last_seen,:taken_at_utc,:taken_at_local,:tz_source,:gps_lat,
                :gps_lon,:gps_alt,:people,:face_regions,:keywords,:description,
                :favorite,:camera_make,:camera_model,:width,:height,:source,
                :metadata_conflict)""",
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
            },
        )

    def get(self, file_hash: str) -> Photo | None:
        row = self._conn.execute(
            "SELECT * FROM photos WHERE file_hash = ?", (file_hash,)
        ).fetchone()
        return self._row_to_photo(row) if row else None

    def iter_photos(self) -> Iterator[Photo]:
        for row in self._conn.execute("SELECT * FROM photos"):
            yield self._row_to_photo(row)

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
        )
