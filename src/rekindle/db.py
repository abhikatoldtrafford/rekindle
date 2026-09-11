"""SQLite store. The relational source of truth for photos.

Writes are batched into a single transaction per run: per-photo commits are
orders of magnitude slower and generate enormous IO.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
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

SCHEMA_VERSION = 5

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
    sidecar_match TEXT NOT NULL DEFAULT 'none',
    phash         INTEGER,
    sharpness     REAL,
    phash_error   TEXT,
    brightness    REAL,
    colour        TEXT
);

-- `paths` is a JSON blob and therefore unqueryable, so this table exists to
-- make "which photo has a path named X?" answerable in SQL.
--
-- MAINTAINED BUT NOT YET READ BY ANY PRODUCTION CODE PATH. The claim this
-- comment used to make - that the enricher needs it 20,000 times - is false:
-- `TakeoutEnricher.enrich()` materialises `list(store.iter_photos())` once
-- and builds its lookups in memory, and `hashes_for_filename` below has no
-- caller outside the tests. Every `_insert` still pays for it (roughly 19k
-- deletes and 24k inserts per enrich run on the reference export). Kept
-- deliberately for one more milestone: dropping it is a v3 schema migration,
-- and shipping a migration to delete infrastructure one milestone after
-- adding it is worse than carrying it. See docs/known-limitations.md - M2
-- must either wire `resolve()` to it or drop it.
CREATE TABLE IF NOT EXISTS photo_paths (
    file_hash TEXT NOT NULL,
    path      TEXT NOT NULL,
    name_cf   TEXT NOT NULL,
    parent    TEXT NOT NULL,
    PRIMARY KEY (file_hash, path)
);

-- Persisted state that decides what a user SEES. One store, deliberately:
-- dismissal and the resurfacing cooldown both gate the same thing, and two
-- mechanisms in two places are two mechanisms that can disagree.
--
-- New TABLES, unlike new columns, are safe in this script: CREATE TABLE IF NOT
-- EXISTS creates them on an old database too, so no migration rung is needed.
--
-- `kind` is 'memory' | 'person' | 'album' | 'dates'. The first is checked when
-- offers are generated; the other three are merged into the ExclusionPolicy
-- and enforced at the MemoryIndex chokepoint like any other exclusion, so a
-- dismissal cannot be honoured in one place and forgotten in another.
CREATE TABLE IF NOT EXISTS memory_exclusions (
    kind       TEXT NOT NULL,
    value      TEXT NOT NULL,
    reason     TEXT NOT NULL DEFAULT 'dismissed',
    created_at TEXT NOT NULL,
    PRIMARY KEY (kind, value)
);

-- When each memory was last surfaced, for the resurfacing cooldown. Keyed by
-- the STABLE memory id (recipe + subject + period), never by its photo set -
-- see memory.history.memory_id.
CREATE TABLE IF NOT EXISTS memory_history (
    memory_id   TEXT PRIMARY KEY,
    surfaced_at TEXT NOT NULL,
    title       TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_photos_taken ON photos(taken_at_utc);
CREATE INDEX IF NOT EXISTS idx_photos_type  ON photos(media_type);
CREATE INDEX IF NOT EXISTS idx_photos_edited ON photos(edited_of);
CREATE INDEX IF NOT EXISTS idx_photo_paths_name ON photo_paths(name_cf);
"""

# Indexes over columns that a pre-migration database does not have yet.
#
# These CANNOT live in `_SCHEMA`. That script runs first thing in `__init__`,
# before `_migrate`, and on an OLD database `CREATE TABLE IF NOT EXISTS photos`
# is a no-op that leaves the v1 column set in place - so a
# `CREATE INDEX ... ON photos(phash)` in `_SCHEMA` raises "no such column:
# phash" and every v1 database becomes unopenable. Found by the existing v1
# migration tests, which is exactly what they are for.
_LATE_INDEXES = """
-- `rekindle fingerprint` resumes by asking for the rows it has not done yet.
-- Without this it is a full scan of every row on every resume.
CREATE INDEX IF NOT EXISTS idx_photos_phash ON photos(phash, phash_error);
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

# v3: the perceptual fingerprint that burst dedup needs. All nullable - a row
# that has never been fingerprinted is a normal, expected state, not a defect.
_V3_COLUMNS = (
    ("phash", "INTEGER"),
    ("sharpness", "REAL"),
    ("phash_error", "TEXT"),
    ("brightness", "REAL"),
)

# v4: a coarse 4x4x4 RGB histogram per image, hex-encoded. Feeds the colour
# half of the diversity signal - see memory.diversity.ColourSignal. 128 bytes
# a row, about 2.5 MB across this library.
_V4_COLUMNS = (("colour", "TEXT"),)


_SIGN_BIT = 1 << 63
_U64 = 1 << 64


def _signed64(value: int | None) -> int | None:
    """Reinterpret an unsigned 64-bit value as SQLite's SIGNED 64-bit INTEGER.

    A dHash is 64 unsigned bits and roughly half of all images set the top
    one, so `phash >= 2**63` is the ordinary case, not an exotic one - and
    handing such a value to sqlite3 raises `OverflowError: Python int too
    large to convert to SQLite INTEGER`. Storing the hash as TEXT would avoid
    the conversion but costs the integer index and 16 bytes a row for nothing.

    In-memory, `PhotoMeta.phash` is ALWAYS unsigned: Hamming distance is
    computed as `bin(a ^ b).count("1")`, and Python's arbitrary-precision
    negative integers have a notional infinite run of sign bits that makes
    that expression silently wrong. The reinterpretation therefore lives
    here, at the storage boundary, and nowhere else.
    """
    if value is None:
        return None
    return value - _U64 if value & _SIGN_BIT else value


def _unsigned64(value: int | None) -> int | None:
    if value is None:
        return None
    return value + _U64 if value < 0 else value


@dataclass(frozen=True)
class FingerprintRow:
    """One photo's measured pixels, as `set_fingerprints` writes them.

    A dataclass rather than a tuple: it grew from three fields to seven, and a
    positional tuple that long is exactly how `sharpness` and `brightness`
    end up swapped with nothing to notice.
    """

    file_hash: str
    phash: int | None = None
    sharpness: float | None = None
    brightness: float | None = None
    width: int | None = None
    height: int | None = None
    colour: str | None = None
    error: str | None = None


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
        if found == SCHEMA_VERSION:
            # Only now are the v3 columns guaranteed to exist - on a fresh
            # database because _SCHEMA created them, on an old one because
            # _migrate just added them. Guarded on the version so a database
            # about to be REJECTED below is not written to on the way out.
            self._conn.executescript(_LATE_INDEXES)
            self._conn.commit()
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
        """Walk the version LADDER, one known step at a time.

        v2 shipped a single `if self.schema_version() != 1: return` step. That
        guard was right about the thing it was defending - a `< SCHEMA_VERSION`
        guard plus an unconditional bump to `SCHEMA_VERSION` would stamp any
        older database, including a hypothetical pre-v1 schema this code knows
        nothing about, straight to the current version and declare success -
        but its own comment predicted what would break next: "a future v3 step
        would silently skip migrating a v1 database". v3 is that step, and a v1
        database opened by this code would have exited the old guard at v2 with
        `__init__` then rejecting it as unmigratable.

        The ladder keeps the property and drops the defect. Each entry migrates
        FROM its key TO the next version, and only versions with an entry are
        touched: a database at an unknown version matches no step, the loop
        stops immediately, and `__init__`'s explicit error message rejects it
        exactly as before. Each step commits its own bump, so an interrupted
        two-step migration resumes from the rung it reached rather than
        restarting or, worse, re-running an already-applied step.
        """
        steps = {1: self._to_v2, 2: self._to_v3, 3: self._to_v4, 4: self._to_v5}
        while (version := self.schema_version()) != SCHEMA_VERSION:
            step = steps.get(version)
            if step is None:
                return
            step()
            # A step that does not advance the version would spin forever.
            # Cheaper to assert than to debug a hung `rekindle index`.
            if self.schema_version() == version:
                raise RuntimeError(f"migration step from v{version} did not advance the version")

    def _bump(self, to: int) -> None:
        self._conn.execute("UPDATE meta SET value = ? WHERE key = 'schema_version'", (str(to),))
        self._conn.commit()

    def _add_columns(self, columns: tuple[tuple[str, str], ...]) -> None:
        """ALTER TABLE ADD COLUMN is the only schema change SQLite does cheaply.

        Additive only: no row is ever rewritten or dropped. The `existing`
        check matters because `_SCHEMA` above uses CREATE TABLE IF NOT EXISTS
        with the CURRENT column list, so a freshly created database already has
        every column while an old one does not.
        """
        existing = {r["name"] for r in self._conn.execute("PRAGMA table_info(photos)")}
        for name, decl in columns:
            if name not in existing:
                self._conn.execute(f"ALTER TABLE photos ADD COLUMN {name} {decl}")

    def _to_v2(self) -> None:
        self._add_columns(_V2_COLUMNS)
        # photo_paths was created empty by _SCHEMA above; backfill it from the
        # JSON blob that was the only path record in v1.
        # fetchall(), not a live cursor: writing through the same connection
        # while iterating a SELECT on it is undefined behaviour in SQLite.
        rows = self._conn.execute("SELECT file_hash, paths FROM photos").fetchall()
        for row in rows:
            for raw in json.loads(row["paths"]):
                self._index_path(self._conn, row["file_hash"], Path(raw))
        self._bump(2)

    def _to_v3(self) -> None:
        """Fingerprint columns. Nothing to backfill: every existing row is
        legitimately "not fingerprinted yet", which is what NULL already says.
        `rekindle fingerprint` fills them in and is resumable precisely so
        this migration does not have to do six minutes of work inside an
        `__init__`."""
        self._add_columns(_V3_COLUMNS)
        self._bump(3)

    def _to_v4(self) -> None:
        """The colour histogram. Nothing to backfill: NULL correctly says
        "not measured yet", and `rekindle fingerprint` fills it in on its next
        run - which is resumable precisely so a schema step never has to do
        minutes of decoding inside an `__init__`."""
        self._add_columns(_V4_COLUMNS)
        self._bump(4)

    def _to_v5(self) -> None:
        """No new column - the MEANING of two existing ones changed.

        v5 is the one migration that DELETES measurements, deliberately.
        `memory.fingerprint.sharpness` was a mean gradient roughly in [0, 25];
        it is now a reblur ratio in [0, 1]. Every comparison the engine makes
        on that column - the burst survivor, the quality percentile, the
        `MIN_SHARPNESS` gate - is a comparison BETWEEN rows, so a database
        holding both scales would not fail, it would silently rank every old
        row above every new one. There is no conversion: the old number cannot
        be recovered from the new one or the other way round.

        `colour` and `phash` go with it. The pass now drafts in "RGB" rather
        than "L" - the old grayscale draft made every stored colour histogram
        a luminance histogram - and it drafts at 1024 rather than 64, which
        moves the hash's input resolution. Mixing two scales of hash across
        one dedup comparison has the same failure mode as mixing two scales of
        sharpness.

        So every SUCCESSFUL fingerprint is cleared and `rekindle fingerprint`
        recomputes it. Rows that recorded an ERROR are left alone: the reason
        a file could not be decoded does not change with the measure, and
        re-attempting them on every upgrade is what `phash_error` exists to
        prevent. `width` and `height` are kept for the same reason - the v3
        pass already repaired them and the values do not depend on the
        measure, so clearing them would blind the resolution floor until the
        re-run finished.

        Cost on the reference library: ~18,363 photos at ~59 ms, about 18
        minutes, once. It is resumable exactly as a first run is.
        """
        self._conn.execute(
            "UPDATE photos SET phash = NULL, sharpness = NULL, brightness = NULL,"
            " colour = NULL WHERE phash IS NOT NULL"
        )
        self._bump(5)

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

    def iter_unfingerprinted(self) -> Iterator[Photo]:
        """Rows with neither a phash nor a recorded reason they lack one.

        This is what makes `rekindle fingerprint` resumable: a row that has
        been done, and a row that was TRIED and failed, are both excluded, so
        an interrupted six-minute pass resumes instead of restarting and a
        permanently undecodable file is not re-decoded on every run.

        Same live-cursor hazard as `iter_photos`: a caller that writes must
        materialise first.
        """
        # Two shapes need work: a row never attempted, AND a row that was
        # fingerprinted before a later schema added a measurement. The second
        # is what makes a new signal (the v4 colour histogram) reach an index
        # that already has hashes, without forcing a full re-index.
        for row in self._conn.execute(
            "SELECT * FROM photos WHERE (phash IS NULL AND phash_error IS NULL) "
            "   OR (phash IS NOT NULL AND colour IS NULL)"
        ):
            yield self._row_to_photo(row)

    def set_fingerprints(self, rows: Iterable[FingerprintRow]) -> int:
        """Write the fingerprint columns for a batch, in one transaction.

        A targeted UPDATE, deliberately not `update_many`: that path rewrites
        every column of the row plus its whole `photo_paths` block, so a
        fingerprint pass would rewrite 19,480 full rows to store three numbers
        each - and would silently clobber any enrichment written since the
        Photo objects were loaded. This touches only what it computed.

        Width and height are written only when the pass actually measured
        them, because it also REPAIRS them: M0 stored `Image.size` without
        applying the EXIF orientation tag, leaving roughly 2,475 rows with the
        two swapped. The fingerprint pass decodes every image anyway, so it
        can correct them without the full re-index that would otherwise be the
        only route. A None leaves the stored value untouched rather than
        nulling a dimension the decode simply could not read.
        """
        cur = self._conn.cursor()
        n = 0
        for row in rows:
            fields = [
                "phash = ?",
                "sharpness = ?",
                "phash_error = ?",
                "brightness = ?",
                "colour = ?",
            ]
            values: list[object] = [
                _signed64(row.phash),
                row.sharpness,
                row.error,
                row.brightness,
                row.colour,
            ]
            if row.width is not None and row.height is not None:
                fields += ["width = ?", "height = ?"]
                values += [row.width, row.height]
            values.append(row.file_hash)
            cur.execute(f"UPDATE photos SET {', '.join(fields)} WHERE file_hash = ?", values)
            n += cur.rowcount
        self._conn.commit()
        return n

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
                trashed, sidecar_match, phash, sharpness, phash_error,
                brightness, colour)
               VALUES
               (:file_hash,:media_type,:paths,:albums,:edited_of,:first_seen,
                :last_seen,:taken_at_utc,:taken_at_local,:tz_source,:gps_lat,
                :gps_lon,:gps_alt,:people,:face_regions,:keywords,:description,
                :favorite,:camera_make,:camera_model,:width,:height,:source,
                :metadata_conflict,:exif_taken_at_utc,:takeout_people,:archived,
                :trashed,:sidecar_match,:phash,:sharpness,:phash_error,
                :brightness,:colour)""",
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
                "phash": _signed64(m.phash),
                "sharpness": m.sharpness,
                "phash_error": m.phash_error,
                "brightness": m.brightness,
                "colour": m.colour,
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
            phash=_unsigned64(row["phash"]),
            sharpness=row["sharpness"],
            phash_error=row["phash_error"],
            brightness=row["brightness"],
            colour=row["colour"],
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
