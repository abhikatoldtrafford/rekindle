# Takeout Enrichment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read Google Takeout's JSON sidecars and enrich the `Photo` rows that `rekindle index` already created with face-tag names, Google's capture time, GPS, descriptions and real album titles.

**Architecture:** A second pass, not a `Source`. `TakeoutEnricher.enrich(root, store)` walks the export's `.json` files, derives each sidecar's target **filename** (relocating Takeout's `(N)` counter), builds one global `filename -> [sidecar]` index, resolves each indexed photo against it with same-directory preference and an explicit refusal on ambiguity, then writes metadata back through a new non-merging store API. Enrichment propagates to `-edited` and `.MP` derivatives, which Google never gives a sidecar. Every JSON file lands in exactly one counter, enforced by an accounting-invariant test.

**Tech Stack:** Python 3.12+, `uv`, stdlib `json` + `sqlite3` + `pathlib`, `typer` + `rich` (CLI), `pytest`, `ruff`. No new runtime dependencies.

**Spec:** docs/superpowers/specs/2026-09-10-takeout-enrichment-design.md

## Global Constraints

- **Python floor is 3.12.** Do not use 3.13-only syntax.
- **No new runtime dependencies.** Everything here is stdlib plus what M0 already installs.
- **No network calls anywhere.** A test that needs the network is a broken test.
- **Cross-platform: Windows, macOS, Linux.** Use `pathlib` exclusively. Never hardcode `/` or `\`.
- **Match on the sidecar's own FILENAME, never on its `title` field.** The property that holds is COLLISION REDUCTION, not a higher raw match count: measured across 24,248 real sidecars, keying on `title` leaves 4,737 sidecars beyond the first claiming one photo, keying on the filename leaves 3,772 — the 965 that `title` mis-pairs. Raw matches actually fall by 4 (five `(N)` sidecars correctly become orphans because their photo is in an un-extracted archive part). `title` is a cross-check only.
- **The counter is RELOCATED, not stripped.** `DSC00107.JPG.supplemental-metadata(1).json` targets `DSC00107(1).JPG`, not `DSC00107.JPG`.
- **Never let dictionary insertion order resolve a collision.** If two sidecars claim one photo and disagree on capture time or people, refuse to enrich and count it.
- **Filename lookups casefold.** Directory identity is the `Path`.
- **`photoTakenTime` is a UTC instant, never a wall-clock value.** It must never be passed to `timestamps.resolve()`, which expects naive local time.
- **Report, never silently drop.** `json_files_seen == sidecars_seen + album_metadata + excluded_dirs + unparseable` and `sidecars_seen == matched + orphaned + ambiguous`. Both are pinned by a test.
- **One orphan definition.** `FolderSource` and `TakeoutEnricher` call the same function.
- **`EXCLUDED_DIRS` is inherited from `folder.py`.** The enricher must not parse `Trash/`'s sidecars.
- **A schema change ships with a migration.** Deleting a user's index is a data-loss event, not an upgrade.
- **All timestamps stored UTC**, ISO 8601 strings in SQLite.
- **Nothing regresses.** M0 is 122 passing tests, `ruff` clean, CI green on Windows/macOS/Linux x Python 3.12/3.13.

## Two places this plan makes the spec concrete

The spec leaves two things underspecified. Both are resolved here, deliberately:

1. **§7's accounting identity has no bucket** for album `metadata.json`, for
   `shared_album_comments.json` / `user-generated-memory-titles.json`, or for a
   JSON file that fails to parse. This plan splits it into two identities
   (above) so every `.json` file under the root is accounted for, not only the
   per-photo sidecars.
2. **§5's "union on first enrich, replace on re-enrich"** cannot be implemented
   against a flat `people` list — there is nothing to subtract. This plan adds
   `PhotoMeta.takeout_people`, holding exactly what the last enrich wrote, and
   computes `people = (existing - previous_takeout) + new_takeout`. That is
   retractable and idempotent; a flat union is neither.

---

### Task 1: Models and schema migration to v2

The migration is the highest-risk change in this plan: `db.py` currently *raises*
on a version mismatch with "Delete the file and re-index", so shipping new
columns without a migration destroys every existing user's index and forces a
re-hash of their whole library. This task exists on its own, first, so a reviewer
can reject it in isolation.

Note the second trap: `models.merge_meta` builds a **new** `PhotoMeta` from an
explicit field list. Any field added to `PhotoMeta` and not added there is
silently dropped on the next `rekindle index` — which is exactly the defect M0's
revision history records as "`_merge` contradicted the spec's merge policy on
every field and destroyed enrichment on every re-index".

**Files:**
- Modify: `src/rekindle/models.py`
- Modify: `src/rekindle/db.py`
- Test: `tests/test_db.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `PhotoMeta`, `Photo`, `TzSource`, `merge_meta` (M0)
- Produces:
  - `TzSource.TAKEOUT: str` (value `"takeout"`)
  - `PhotoMeta.archived: bool`, `PhotoMeta.trashed: bool`
  - `PhotoMeta.exif_taken_at_utc: datetime | None`
  - `PhotoMeta.takeout_people: list[str]`
  - `Photo.sidecar_match: str` (`"exact"` | `"ambiguous"` | `"none"`)
  - `db.SCHEMA_VERSION == 2`
  - `photo_paths(file_hash, path, name_cf, parent)` table with index `idx_photo_paths_name`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db.py  (append)
import json
import sqlite3
from datetime import UTC, datetime

from rekindle.db import SCHEMA_VERSION, PhotoStore
from rekindle.models import MediaType, Photo, PhotoMeta, TzSource


def _write_v1_database(path):
    """An M0-schema database, byte-for-byte as v1 wrote it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE photos (
            file_hash TEXT PRIMARY KEY, media_type TEXT NOT NULL,
            paths TEXT NOT NULL, albums TEXT NOT NULL, edited_of TEXT,
            first_seen TEXT NOT NULL, last_seen TEXT NOT NULL,
            taken_at_utc TEXT, taken_at_local TEXT, tz_source TEXT NOT NULL,
            gps_lat REAL, gps_lon REAL, gps_alt REAL, people TEXT NOT NULL,
            face_regions TEXT NOT NULL, keywords TEXT NOT NULL,
            description TEXT, favorite INTEGER NOT NULL DEFAULT 0,
            camera_make TEXT, camera_model TEXT, width INTEGER, height INTEGER,
            source TEXT NOT NULL DEFAULT 'folder',
            metadata_conflict INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    conn.execute("INSERT INTO meta(key, value) VALUES('schema_version', '1')")
    conn.execute(
        "INSERT INTO photos (file_hash, media_type, paths, albums, first_seen,"
        " last_seen, tz_source, people, face_regions, keywords) VALUES"
        " ('abc', 'image', ?, ?, '2020-01-01T00:00:00+00:00',"
        " '2020-01-01T00:00:00+00:00', 'exif_naive', ?, '[]', '[]')",
        (
            json.dumps(["/lib/Photos from 2014/IMG_0001.jpg"]),
            json.dumps(["Photos from 2014"]),
            json.dumps(["Ada"]),
        ),
    )
    conn.commit()
    conn.close()


def test_v1_database_migrates_without_losing_rows(tmp_path):
    """A data-loss regression test. M0 users must not be told to re-index."""
    db_path = tmp_path / "data" / "rekindle.sqlite"
    _write_v1_database(db_path)

    with PhotoStore(db_path) as store:
        assert store.schema_version() == SCHEMA_VERSION == 2
        assert store.count() == 1
        photo = store.get("abc")
        assert photo is not None
        assert photo.meta.people == ["Ada"]
        assert photo.albums == ["Photos from 2014"]
        # New columns arrive with their defaults, not with NULLs that crash
        # deserialisation.
        assert photo.sidecar_match == "none"
        assert photo.meta.archived is False
        assert photo.meta.trashed is False
        assert photo.meta.exif_taken_at_utc is None
        assert photo.meta.takeout_people == []


def test_migration_backfills_the_photo_paths_table(tmp_path):
    db_path = tmp_path / "data" / "rekindle.sqlite"
    _write_v1_database(db_path)
    with PhotoStore(db_path) as store:
        hits = store.hashes_for_filename("img_0001.jpg")
    assert hits == {"abc"}


def test_new_meta_fields_survive_a_reindex_merge():
    """merge_meta enumerates fields; one left out is enrichment destroyed."""
    enriched = PhotoMeta(
        taken_at_utc=datetime(2014, 6, 1, tzinfo=UTC),
        tz_source=TzSource.TAKEOUT,
        people=["Ada"],
        takeout_people=["Ada"],
        archived=True,
        trashed=False,
        exif_taken_at_utc=datetime(2010, 1, 1, tzinfo=UTC),
    )
    fresh_from_disk = PhotoMeta(
        taken_at_utc=datetime(2020, 1, 1, tzinfo=UTC),
        tz_source=TzSource.FILE_MTIME,
    )
    merged, _ = merge_meta(enriched, fresh_from_disk)
    assert merged.takeout_people == ["Ada"]
    assert merged.archived is True
    assert merged.exif_taken_at_utc == datetime(2010, 1, 1, tzinfo=UTC)


def test_takeout_is_a_real_tz_source():
    assert TzSource("takeout") is TzSource.TAKEOUT
```

Add `merge_meta` to the existing `rekindle.models` import line at the top of
`tests/test_db.py`, or import it inside the test module alongside the others.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_db.py -v`
Expected: FAIL — `AttributeError: TAKEOUT` on `TzSource`, and
`RuntimeError: ... uses schema v1, this rekindle expects v2, and no migration exists yet`.

- [ ] **Step 3: Write the implementation**

In `src/rekindle/models.py`, add the enum member, the `PhotoMeta` fields, the
`Photo` field, and — critically — the four new lines in `merge_meta`:

```python
class TzSource(StrEnum):
    """How local time was determined. Recorded so doctor can report it."""

    EXIF_OFFSET = "exif_offset"
    EXIF_NAIVE = "exif_naive"
    GPS = "gps"
    FILE_MTIME = "file_mtime"
    # Google's photoTakenTime supplied the INSTANT. It says nothing about the
    # zone: local time is derived separately (meta.timestamps.from_takeout).
    TAKEOUT = "takeout"
    NONE = "none"
```

```python
@dataclass
class PhotoMeta:
    taken_at_utc: datetime | None = None
    taken_at_local: datetime | None = None
    tz_source: TzSource = TzSource.NONE
    gps: Gps | None = None
    people: list[str] = field(default_factory=list)
    face_regions: list[FaceRegion] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    description: str | None = None
    favorite: bool = False
    camera_make: str | None = None
    camera_model: str | None = None
    width: int | None = None
    height: int | None = None
    # The EXIF instant that Google's photoTakenTime displaced. Without this,
    # `metadata_conflict` is a flag with no payload: the spec promised the
    # record was "flagged rather than silently overwritten" while there was
    # exactly one date field to overwrite.
    exif_taken_at_utc: datetime | None = None
    # Exactly what the last Takeout enrich contributed to `people`. Needed so a
    # second enrich can RETRACT a face tag the user corrected in Google Photos.
    # A flat union can only ever grow.
    takeout_people: list[str] = field(default_factory=list)
    # Google's own flags. Archived means the user deliberately hid this photo;
    # it must never surface in a montage. 162 rows on the reference export.
    archived: bool = False
    # EXPECTED TO BE PERMANENTLY ZERO FOR TAKEOUT. All 12 `trashed: true`
    # sidecars on the reference export live under `Trash/`, which both
    # FolderSource and build_index exclude, so the photo never reaches the
    # store to be flagged. The field exists for sources that expose a
    # soft-delete flag WITHOUT segregating the files - Immich and Apple Photos
    # both do. If a Takeout run ever reports a non-zero count here, Google has
    # changed the export layout and the exclusion needs revisiting.
    trashed: bool = False
```

```python
@dataclass
class Photo:
    ...
    metadata_conflict: bool = False
    # "exact"  - a sidecar was resolved for this photo
    # "ambiguous" - candidates disagreed and enrichment was REFUSED
    # "none"   - no sidecar, or enrich has never run. Distinguish the two by
    #            the `enriched_at` key in the meta table, not by this field.
    sidecar_match: str = "none"
```

In `merge_meta`, extend the constructed `PhotoMeta` (the four new lines are the
whole point of `test_new_meta_fields_survive_a_reindex_merge`):

```python
    merged = PhotoMeta(
        taken_at_utc=keep.taken_at_utc,
        taken_at_local=keep.taken_at_local,
        tz_source=keep.tz_source,
        gps=old.gps or new.gps,
        people=list(dict.fromkeys([*old.people, *new.people])),
        face_regions=list(dict.fromkeys([*old.face_regions, *new.face_regions])),
        keywords=list(dict.fromkeys([*old.keywords, *new.keywords])),
        description=max(descs, key=len) if descs else None,
        favorite=old.favorite or new.favorite,
        camera_make=old.camera_make or new.camera_make,
        camera_model=old.camera_model or new.camera_model,
        width=old.width or new.width,
        height=old.height or new.height,
        # A re-index must never destroy enrichment. `new` here is the folder
        # source, which knows none of these, so `old` wins on all four.
        exif_taken_at_utc=old.exif_taken_at_utc or new.exif_taken_at_utc,
        takeout_people=list(old.takeout_people or new.takeout_people),
        archived=old.archived or new.archived,
        trashed=old.trashed or new.trashed,
    )
```

In `src/rekindle/db.py`, bump the version, add the columns and the table, and add
the migration:

```python
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
```

Replace the version check in `__init__` with a migrate-then-check:

```python
        self._conn.executescript(_SCHEMA)
        self._conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()
        self._migrate()
        # CREATE TABLE IF NOT EXISTS silently keeps an old table, so a version
        # check is the only thing standing between a schema change and a
        # baffling OperationalError on the next write.
        found = self.schema_version()
        if found != SCHEMA_VERSION:
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
        """v1 -> v2. Additive only: no row is ever rewritten or dropped."""
        if self.schema_version() >= SCHEMA_VERSION:
            return
        existing = {
            r["name"] for r in self._conn.execute("PRAGMA table_info(photos)")
        }
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
```

Teach `_insert` to keep `photo_paths` in step, and to persist the new columns.
Add to the `INSERT OR REPLACE INTO photos` column list and `VALUES` clause:
`exif_taken_at_utc, takeout_people, archived, trashed, sidecar_match` /
`:exif_taken_at_utc,:takeout_people,:archived,:trashed,:sidecar_match`, with these
parameter entries:

```python
                "exif_taken_at_utc": _dt(m.exif_taken_at_utc),
                "takeout_people": json.dumps(m.takeout_people),
                "archived": int(m.archived),
                "trashed": int(m.trashed),
                "sidecar_match": p.sidecar_match,
```

and, immediately after the `cur.execute(...)` in `_insert`:

```python
        cur.execute("DELETE FROM photo_paths WHERE file_hash = ?", (p.file_hash,))
        for path in p.paths:
            PhotoStore._index_path(cur, p.file_hash, path)
```

Teach `_row_to_photo` to read them back:

```python
        meta = PhotoMeta(
            ...
            height=row["height"],
            exif_taken_at_utc=_undt(row["exif_taken_at_utc"]),
            takeout_people=json.loads(row["takeout_people"]),
            archived=bool(row["archived"]),
            trashed=bool(row["trashed"]),
        )
        return Photo(
            ...
            metadata_conflict=bool(row["metadata_conflict"]),
            sidecar_match=row["sidecar_match"],
        )
```

Carry `sidecar_match` through `PhotoStore._merge` so a re-index does not reset it:

```python
            metadata_conflict=old.metadata_conflict or new.metadata_conflict or conflict,
            sidecar_match=old.sidecar_match if old.sidecar_match != "none" else new.sidecar_match,
        )
```

Add the lookup this task's second test needs. Note honestly that the enricher
loads every photo into memory (Task 11) and so does not call this method; the
`photo_paths` table exists because spec §8 requires a queryable path index and
because the incremental scan `PhotoStore.all_hashes()` was built for will need
one. It is tested, not speculative machinery — but if a reviewer wants it gone,
the table must stay and only this accessor goes.

```python
    def hashes_for_filename(self, name_cf: str) -> set[str]:
        """Every photo having a path whose filename casefolds to `name_cf`."""
        return {
            r["file_hash"]
            for r in self._conn.execute(
                "SELECT file_hash FROM photo_paths WHERE name_cf = ?", (name_cf,)
            )
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_db.py tests/test_models.py -v`
Expected: all pass. Then `uv run pytest -v` — 122 existing tests must still pass.

- [ ] **Step 5: Commit**

```bash
git add src/rekindle/models.py src/rekindle/db.py tests/test_db.py tests/test_models.py
git commit -m "feat: schema v2 with an additive migration and a photo_paths index"
```

---

### Task 2: Store API for enumeration and non-merging writes

The spec's central loop — "for each `Photo`, for each path it has" — is
unimplementable against M0's `PhotoStore`, which exposes only `count`,
`all_hashes`, `upsert_many`, `get` and `schema_version`. There is no way to
enumerate, and the only writer routes through `merge_meta`, whose "earliest real
date wins" would silently reject Google's date.

**Files:**
- Modify: `src/rekindle/db.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: `PhotoStore` (Task 1), `Photo`, `PhotoMeta`
- Produces:
  - `PhotoStore.iter_photos() -> Iterator[Photo]`
  - `PhotoStore.update_photo(photo: Photo) -> None` — writes without merging
  - `PhotoStore.update_many(photos: Iterable[Photo]) -> int` — the batched form, consumed by Task 11
  - `PhotoStore.get_meta(key: str) -> str | None`
  - `PhotoStore.set_meta(key: str, value: str) -> None`

**Deviation from the spec, recorded deliberately:** spec §8 names this method
`update_meta(file_hash, meta, ...)`. It ships as `update_photo(photo)` instead,
because enrichment also changes `albums`, `sidecar_match` and
`metadata_conflict`, none of which live on `PhotoMeta`. A `meta`-only signature
would need three more parameters to say the same thing.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db.py  (append)
def test_iter_photos_yields_every_row(tmp_path):
    store = PhotoStore(tmp_path / "d" / "r.sqlite")
    now = datetime(2020, 1, 1, tzinfo=UTC)
    store.upsert_many(
        [
            Photo(
                file_hash=h,
                paths=[tmp_path / f"{h}.jpg"],
                media_type=MediaType.IMAGE,
                meta=PhotoMeta(),
                first_seen=now,
                last_seen=now,
            )
            for h in ("aa", "bb", "cc")
        ]
    )
    assert {p.file_hash for p in store.iter_photos()} == {"aa", "bb", "cc"}
    store.close()


def test_update_photo_does_not_merge(tmp_path):
    """merge_meta keeps the EARLIEST real date. Enrichment must be able to
    move a date FORWARD when Google says so, which upsert_many cannot do."""
    store = PhotoStore(tmp_path / "d" / "r.sqlite")
    now = datetime(2020, 1, 1, tzinfo=UTC)
    original = Photo(
        file_hash="aa",
        paths=[tmp_path / "aa.jpg"],
        media_type=MediaType.IMAGE,
        meta=PhotoMeta(
            taken_at_utc=datetime(2010, 1, 1, tzinfo=UTC),
            tz_source=TzSource.EXIF_NAIVE,
            people=["Ada", "Grace"],
        ),
        first_seen=now,
        last_seen=now,
    )
    store.upsert_many([original])

    original.meta.taken_at_utc = datetime(2015, 6, 1, tzinfo=UTC)
    original.meta.tz_source = TzSource.TAKEOUT
    original.meta.people = ["Grace"]
    original.sidecar_match = "exact"
    store.update_photo(original)

    reloaded = store.get("aa")
    assert reloaded.meta.taken_at_utc == datetime(2015, 6, 1, tzinfo=UTC)
    assert reloaded.meta.tz_source is TzSource.TAKEOUT
    assert reloaded.meta.people == ["Grace"]  # not unioned back to two
    assert reloaded.sidecar_match == "exact"
    store.close()


def test_meta_keys_round_trip(tmp_path):
    store = PhotoStore(tmp_path / "d" / "r.sqlite")
    assert store.get_meta("enriched_at") is None
    store.set_meta("enriched_at", "2026-09-11T10:00:00+00:00")
    store.set_meta("enriched_at", "2026-09-12T10:00:00+00:00")
    assert store.get_meta("enriched_at") == "2026-09-12T10:00:00+00:00"
    store.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_db.py -k "iter_photos or update_photo or meta_keys" -v`
Expected: FAIL — `AttributeError: 'PhotoStore' object has no attribute 'iter_photos'`

- [ ] **Step 3: Write the implementation**

```python
# src/rekindle/db.py  (add to PhotoStore; `Iterator` joins the
# collections.abc import alongside Iterable)

    def iter_photos(self) -> Iterator[Photo]:
        """Every row, streamed. The enricher needs all of them and `get()` per
        hash would be one SELECT per photo - 19,480 of them on a real export.

        WARNING: this yields from a LIVE cursor. Writing to this store while
        iterating it is undefined behaviour in SQLite - the same hazard
        `_migrate` calls `.fetchall()` to avoid. Callers that write must
        materialise first: `photos = list(store.iter_photos())`, which is what
        TakeoutEnricher does.
        """
        for row in self._conn.execute("SELECT * FROM photos"):
            yield self._row_to_photo(row)

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_db.py -v && uv run ruff check .`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/rekindle/db.py tests/test_db.py
git commit -m "feat: iter_photos, update_photo and meta key access on PhotoStore"
```

---

### Task 3: Shared sidecar naming, with the counter relocated

This changes a function M0 already ships with passing tests. **One existing
assertion changes**, at `tests/test_folder_source.py:177`:

```python
assert _sidecar_target("DSC_0880.JPG.supplemental-metadata(1).json") == "DSC_0880.JPG"
```

becomes `== "DSC_0880(1).JPG"`. That assertion *is* the v1 bug, written down.
Every other `_sidecar_target` assertion is unchanged and must still pass, as must
`test_orphan_sidecars_are_counted` — the fixture has no `DSC_0880(1).JPG` either,
so that sidecar stays orphaned and the count stays 2.

The function moves to its own module so `FolderSource` and `TakeoutEnricher`
share one definition. §4 of the spec: "One function, used by both."

**Files:**
- Create: `src/rekindle/sidecars.py`
- Modify: `src/rekindle/sources/folder.py`
- Create: `tests/test_sidecars.py`
- Modify: `tests/test_folder_source.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `rekindle.sidecars.sidecar_target(name: str) -> str`
  - `rekindle.sidecars.is_album_metadata(name: str) -> bool`
  - `rekindle.sources.folder._sidecar_target` re-exported, so M0's import path keeps working

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sidecars.py
from rekindle.sidecars import is_album_metadata, sidecar_target


def test_plain_supplemental_metadata_yields_the_filename():
    assert sidecar_target("IMG_1234.jpg.supplemental-metadata.json") == "IMG_1234.jpg"


def test_truncated_supplemental_suffix_still_matches():
    assert sidecar_target("IMG_1234.jpg.supplemental-metad.json") == "IMG_1234.jpg"


def test_bare_json_suffix_yields_the_filename():
    assert sidecar_target("IMG_1234.jpg.json") == "IMG_1234.jpg"


def test_counter_is_relocated_into_the_stem_not_stripped():
    """THE v1 BUG. Google puts the counter in the SIDECAR's name; the photo
    carries it inside the stem. Measured on a real export: 963 sidecars would
    otherwise be paired with a different photo of the same name."""
    assert sidecar_target("DSC00107.JPG.supplemental-metadata(1).json") == "DSC00107(1).JPG"
    assert sidecar_target("Photo0007.jpg.supplemental-metadata(2).json") == "Photo0007(2).jpg"


def test_counter_relocation_preserves_spaces_and_dots_in_the_stem():
    assert (
        sidecar_target("Photo0549 - Copy.jpg.supplemental-metadata(1).json")
        == "Photo0549 - Copy(1).jpg"
    )
    assert (
        sidecar_target("PXL_1.MP.jpg.supplemental-metadata(1).json") == "PXL_1.MP(1).jpg"
    )


def test_extensionless_target_appends_the_counter():
    assert sidecar_target("README.supplemental-metadata(1).json") == "README(1)"


def test_unrecognised_name_is_returned_unchanged():
    assert sidecar_target("IMG_7000.aae") == "IMG_7000.aae"


def test_album_metadata_is_recognised_case_insensitively():
    assert is_album_metadata("metadata.json")
    assert is_album_metadata("Metadata.JSON")
    assert not is_album_metadata("IMG_1234.jpg.supplemental-metadata.json")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sidecars.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rekindle.sidecars'`

- [ ] **Step 3: Write the implementation**

```python
# src/rekindle/sidecars.py
"""Naming rules for Google Takeout's JSON sidecars.

One module, imported by both `sources.folder` (for its orphan count) and
`enrich.takeout` (for its match key). Two implementations of this rule
produced two different orphan numbers for the same export, both rendered by
`doctor` - which is why it lives here.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

# Verified against a real 24,248-sidecar export: exactly two schemes,
# `.supplemental-metadata` (23,255) and `.supplemental-metadata(N)` (993),
# with no truncated forms present. The `met[a-z]*` wildcard covers truncation
# anyway - it is well documented elsewhere and costs nothing.
_SUPPLEMENTAL_RE = re.compile(
    r"^(?P<base>.+?)\.supplemental-met[a-z]*(?:\((?P<counter>\d+)\))?\.json$",
    re.IGNORECASE,
)
_PLAIN_JSON_RE = re.compile(
    r"^(?P<base>.+?)(?:\((?P<counter>\d+)\))?\.json$",
    re.IGNORECASE,
)


def sidecar_target(name: str) -> str:
    """The media filename a Takeout sidecar refers to.

    THE COUNTER IS RELOCATED, NOT STRIPPED. Google disambiguates two photos
    sharing a filename by numbering the SIDECAR:

        DSC00107.JPG.supplemental-metadata.json     -> DSC00107.JPG
        DSC00107.JPG.supplemental-metadata(1).json  -> DSC00107(1).JPG

    The sidecar's own `title` field says "DSC00107.JPG" in BOTH cases, which
    is why matching on `title` mis-paired 963 photos on the reference export.

    A name this does not recognise is returned unchanged, so a non-Google
    sidecar (`.aae`, `.thm`) never silently becomes a bogus target.
    """
    match = _SUPPLEMENTAL_RE.match(name) or _PLAIN_JSON_RE.match(name)
    if match is None:
        return name
    base = match.group("base")
    counter = match.group("counter")
    if counter is None:
        return base
    # PurePosixPath, not PurePath: a bare filename can legally contain a
    # backslash on Linux, and PureWindowsPath would treat it as a separator.
    # It can never contain a forward slash, so the posix flavour is safe on
    # every platform and gives identical results on all three.
    pure = PurePosixPath(base)
    return f"{pure.stem}({counter}){pure.suffix}"


def is_album_metadata(name: str) -> bool:
    """`metadata.json` describes the ALBUM, never a photo."""
    return name.casefold() == "metadata.json"
```

In `src/rekindle/sources/folder.py`:

1. Delete the local `_sidecar_target` function and its two module-level regexes.
2. **Delete `import re`** — that function held its last use, and leaving it is
   `F401 imported but unused`, which fails the Definition of Done.
3. Add the re-export **immediately after** `from rekindle.models import (...)`,
   not merely "near the other imports": `rekindle.sidecars` sorts after
   `rekindle.models` and before `rekindle.sources`, and anywhere else is
   `I001 un-sorted imports`.

```python
from rekindle.sidecars import sidecar_target as _sidecar_target
```

Update the one existing assertion in `tests/test_folder_source.py`:

```python
def test_sidecar_target_strips_supplemental_metadata_and_counter():
    from rekindle.sources.folder import _sidecar_target

    assert _sidecar_target("IMG_1234.jpg.supplemental-metadata.json") == "IMG_1234.jpg"
    # CHANGED from "DSC_0880.JPG". The counter belongs to the PHOTO's stem;
    # stripping it made two sidecars claim one file. See tests/test_sidecars.py.
    assert _sidecar_target("DSC_0880.JPG.supplemental-metadata(1).json") == "DSC_0880(1).JPG"
    assert _sidecar_target("IMG_1234.jpg.supplemental-metad.json") == "IMG_1234.jpg"
    assert _sidecar_target("IMG_1234.jpg.json") == "IMG_1234.jpg"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sidecars.py tests/test_folder_source.py -v`
Expected: all pass, including the unchanged `test_orphan_sidecars_are_counted`
(still exactly 2) and `test_json_sidecars_are_ignored_but_counted` (still 4).

- [ ] **Step 5: Commit**

```bash
git add src/rekindle/sidecars.py src/rekindle/sources/folder.py tests/test_sidecars.py tests/test_folder_source.py
git commit -m "fix: relocate the Takeout sidecar counter instead of stripping it"
```

---

### Task 4: Sidecar parsing

Reads one JSON file into a typed record. Never raises: a malformed file is
counted, not a crash. Classification matters — the export root and album folders
hold `metadata.json`, `shared_album_comments.json` and
`user-generated-memory-titles.json`, and the last of those has a `title` that is
a **list**, not a string.

> **Post-implementation correction (2026-09-11):** the `classify_json` and
> `_geo` snippets below were revised from what this plan originally proposed.
> The original `classify_json` classified PHOTO only when one of
> `photoTakenTime`/`creationTime`/`geoData`/`imageViews` was present — which
> misfiles a real (if rare) marker-less sidecar carrying only `title` and
> `people` as OTHER, caught by this task's own
> `test_a_sidecar_with_no_taken_time_still_parses`. The fix adds a
> string-valued `title` as an additional PHOTO signal: that is exactly the
> field `user-generated-memory-titles.json` (title is a LIST) and
> `shared_album_comments.json` (no `title` key at all) both lack, so neither
> is misclassified. The original `_geo` also defaulted a missing `latitude`
> or `longitude` to `0.0`, which could fabricate a wrong-but-plausible
> coordinate from a partial block; the fix requires both keys present before
> reading either. Both corrections and their mutation evidence are recorded
> in `task-4-report.md`.

**Files:**
- Create: `src/rekindle/enrich/__init__.py`
- Create: `src/rekindle/enrich/takeout.py`
- Create: `tests/test_takeout_parse.py`

**Interfaces:**
- Consumes: `rekindle.sidecars.sidecar_target`, `rekindle.models.Gps`
- Produces:
  - `Sidecar` frozen dataclass with fields `path, target_cf, title, taken_at_utc, people, gps, description, favorite, archived, trashed`
  - `JsonKind` StrEnum: `PHOTO`, `ALBUM`, `OTHER`, `UNPARSEABLE`
  - `classify_json(path: Path) -> tuple[JsonKind, dict | None]`
  - `parse_sidecar(path: Path, payload: dict) -> Sidecar`
  - `album_title(payload: dict) -> str | None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_takeout_parse.py
import json
from datetime import UTC, datetime

from rekindle.enrich.takeout import JsonKind, album_title, classify_json, parse_sidecar


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_photo_sidecar_parses_every_field_we_use(tmp_path):
    p = _write(
        tmp_path / "IMG_1.jpg.supplemental-metadata.json",
        {
            "title": "IMG_1.jpg",
            "description": "  a caption  ",
            "photoTakenTime": {"timestamp": "1774082363"},
            "creationTime": {"timestamp": "1774085809"},
            "geoData": {"latitude": 22.5, "longitude": 88.3, "altitude": 9.0},
            "people": [{"name": "Ada"}, {"name": "Grace"}],
            "favorited": True,
            "archived": True,
        },
    )
    kind, payload = classify_json(p)
    assert kind is JsonKind.PHOTO
    s = parse_sidecar(p, payload)
    assert s.target_cf == "img_1.jpg"
    assert s.title == "IMG_1.jpg"
    assert s.taken_at_utc == datetime.fromtimestamp(1774082363, UTC)
    assert s.people == ("Ada", "Grace")
    assert s.gps.lat == 22.5 and s.gps.lon == 88.3 and s.gps.alt == 9.0
    assert s.description == "a caption"
    assert s.favorite is True
    assert s.archived is True
    assert s.trashed is False


def test_zeroed_geodata_is_absent_not_a_coordinate(tmp_path):
    """(0,0) is the Gulf of Guinea. Measured: 89.6% of real sidecars."""
    p = _write(
        tmp_path / "IMG_2.jpg.supplemental-metadata.json",
        {
            "title": "IMG_2.jpg",
            "photoTakenTime": {"timestamp": "1"},
            "geoData": {"latitude": 0.0, "longitude": 0.0, "altitude": 0.0},
        },
    )
    _, payload = classify_json(p)
    assert parse_sidecar(p, payload).gps is None


def test_geodataexif_is_used_when_geodata_is_zero(tmp_path):
    p = _write(
        tmp_path / "IMG_3.jpg.supplemental-metadata.json",
        {
            "title": "IMG_3.jpg",
            "photoTakenTime": {"timestamp": "1"},
            "geoData": {"latitude": 0.0, "longitude": 0.0, "altitude": 0.0},
            "geoDataExif": {"latitude": 1.5, "longitude": 2.5, "altitude": 0.0},
        },
    )
    _, payload = classify_json(p)
    assert parse_sidecar(p, payload).gps.lat == 1.5


def test_the_counter_is_relocated_when_deriving_the_target(tmp_path):
    p = _write(
        tmp_path / "DSC00107.JPG.supplemental-metadata(1).json",
        {"title": "DSC00107.JPG", "photoTakenTime": {"timestamp": "1"}},
    )
    _, payload = classify_json(p)
    s = parse_sidecar(p, payload)
    assert s.target_cf == "dsc00107(1).jpg"
    assert s.title == "DSC00107.JPG"  # title still disagrees; kept as a cross-check


def test_empty_and_absent_optional_fields(tmp_path):
    p = _write(
        tmp_path / "IMG_4.jpg.supplemental-metadata.json",
        {"title": "IMG_4.jpg", "description": "", "photoTakenTime": {"timestamp": "1"}},
    )
    _, payload = classify_json(p)
    s = parse_sidecar(p, payload)
    assert s.description is None
    assert s.people == ()
    assert s.favorite is False


def test_album_metadata_is_classified_as_album(tmp_path):
    p = _write(tmp_path / "Goa Trip" / "metadata.json", {"title": "Goa/ Trip"})
    kind, payload = classify_json(p)
    assert kind is JsonKind.ALBUM
    assert album_title(payload) == "Goa/ Trip"


def test_album_metadata_with_a_null_or_empty_title_yields_none(tmp_path):
    for payload in ({"title": None}, {"title": ""}, {"title": "   "}, {}):
        p = _write(tmp_path / f"a{id(payload)}" / "metadata.json", payload)
        _, parsed = classify_json(p)
        assert album_title(parsed) is None


def test_a_list_valued_title_is_not_an_album_title(tmp_path):
    """user-generated-memory-titles.json really does carry title: [...]."""
    p = _write(
        tmp_path / "user-generated-memory-titles.json",
        {"title": ["happy birthday ", "Leh Ladakh"]},
    )
    kind, payload = classify_json(p)
    assert kind is JsonKind.OTHER
    assert album_title(payload) is None


def test_shared_album_comments_is_other(tmp_path):
    p = _write(tmp_path / "shared_album_comments.json", {"comments": []})
    assert classify_json(p)[0] is JsonKind.OTHER


def test_malformed_json_is_unparseable_not_an_exception(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text("{not json", encoding="utf-8")
    kind, payload = classify_json(p)
    assert kind is JsonKind.UNPARSEABLE
    assert payload is None


def test_a_json_array_at_the_top_level_is_unparseable(tmp_path):
    p = _write(tmp_path / "arr.json", [1, 2, 3])
    assert classify_json(p)[0] is JsonKind.UNPARSEABLE


def test_a_sidecar_with_no_taken_time_still_parses(tmp_path):
    p = _write(tmp_path / "IMG_5.jpg.json", {"title": "IMG_5.jpg", "people": [{"name": "Ada"}]})
    kind, payload = classify_json(p)
    assert kind is JsonKind.PHOTO
    assert parse_sidecar(p, payload).taken_at_utc is None


def test_people_entries_without_a_usable_name_are_dropped(tmp_path):
    p = _write(
        tmp_path / "IMG_6.jpg.json",
        {
            "title": "IMG_6.jpg",
            "people": [{"name": "Ada"}, {"name": "  "}, {"name": None}, {}, {"name": "Ada"}],
        },
    )
    _, payload = classify_json(p)
    assert parse_sidecar(p, payload).people == ("Ada",)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_takeout_parse.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rekindle.enrich'`

- [ ] **Step 3: Write the implementation**

```python
# src/rekindle/enrich/__init__.py
"""Second-pass enrichment of already-indexed photos."""
```

```python
# src/rekindle/enrich/takeout.py
"""Google Takeout JSON sidecar enrichment.

Deliberately NOT a `Source`: that protocol returns (list[Photo], SourceReport)
for something producing records from pixels. This updates existing records
addressed by content hash. It does inherit the protocol's real contract -
count everything - which `EnrichReport` enforces in Task 6.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from rekindle.models import Gps
from rekindle.sidecars import is_album_metadata, sidecar_target


class JsonKind(StrEnum):
    PHOTO = "photo"
    ALBUM = "album"
    OTHER = "other"
    UNPARSEABLE = "unparseable"


@dataclass(frozen=True)
class Sidecar:
    path: Path
    # Casefolded filename of the photo this sidecar describes, with Takeout's
    # (N) counter relocated into the stem. THE MATCH KEY.
    target_cf: str
    # Google's own record of the filename. It omits the counter, so it is a
    # cross-check only - never the key. See spec section 0.
    title: str
    taken_at_utc: datetime | None
    people: tuple[str, ...]
    gps: Gps | None
    description: str | None
    favorite: bool
    archived: bool
    trashed: bool

    @property
    def title_disagrees(self) -> bool:
        """True when `title` names something other than the derived target.

        Expected for every `(N)` sidecar. Anything else is worth reporting.
        """
        return self.title.casefold() != self.target_cf


def classify_json(path: Path) -> tuple[JsonKind, dict | None]:
    """Read and classify one .json file. Never raises."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return JsonKind.UNPARSEABLE, None
    if not isinstance(payload, dict):
        # A top-level array is not something we know how to read.
        return JsonKind.UNPARSEABLE, None
    if is_album_metadata(path.name):
        return JsonKind.ALBUM, payload
    # A per-photo sidecar always carries a string `title` (Google's own name
    # for the file) and/or one of these markers. Checking `title`'s TYPE, not
    # just its presence, is what keeps `user-generated-memory-titles.json`
    # (title is a LIST) out of PHOTO; checking the markers too is what lets a
    # sidecar with no `title` at all - or a genuine export we have not seen -
    # still classify correctly. `shared_album_comments.json` has neither and
    # correctly falls through to OTHER.
    has_marker = any(
        k in payload for k in ("photoTakenTime", "creationTime", "geoData", "imageViews")
    )
    if isinstance(payload.get("title"), str) or has_marker:
        return JsonKind.PHOTO, payload
    return JsonKind.OTHER, payload


def _timestamp(block: object) -> datetime | None:
    if not isinstance(block, dict):
        return None
    raw = block.get("timestamp")
    try:
        return datetime.fromtimestamp(int(raw), UTC)
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def _geo(block: object) -> Gps | None:
    """A (0,0) pair is the Gulf of Guinea, not a location. 89.6% of real
    sidecars carry exactly that, and `geoDataExif` is key-ABSENT on them
    rather than zeroed."""
    if not isinstance(block, dict):
        return None
    # latitude and longitude must BOTH be present. Defaulting a missing half
    # to 0.0 would fabricate a wrong-but-plausible coordinate rather than
    # report the absence - not observed in the real export, but the
    # constraint is "never fabricate a value", not "never seen yet".
    # Altitude stays optional: it is genuinely optional in the format.
    if "latitude" not in block or "longitude" not in block:
        return None
    try:
        lat = float(block["latitude"])
        lon = float(block["longitude"])
        alt = float(block.get("altitude", 0.0))
    except (TypeError, ValueError):
        return None
    if abs(lat) < 1e-9 and abs(lon) < 1e-9:
        return None
    return Gps(lat=lat, lon=lon, alt=alt)


def parse_sidecar(path: Path, payload: dict) -> Sidecar:
    """Never raises. A field we cannot read is absent, never fabricated."""
    people: list[str] = []
    raw_people = payload.get("people")
    if isinstance(raw_people, list):
        for entry in raw_people:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if isinstance(name, str) and name.strip() and name not in people:
                people.append(name)

    description = payload.get("description")
    description = description.strip() if isinstance(description, str) else ""

    title = payload.get("title")
    title = title if isinstance(title, str) else ""

    return Sidecar(
        path=path,
        target_cf=sidecar_target(path.name).casefold(),
        title=title,
        taken_at_utc=_timestamp(payload.get("photoTakenTime")),
        people=tuple(people),
        # geoData first: measured on 24,248 sidecars, the two never disagree
        # and geoDataExif never carries data geoData lacks. The fallback costs
        # nothing and covers exports we have not seen.
        gps=_geo(payload.get("geoData")) or _geo(payload.get("geoDataExif")),
        description=description or None,
        # `favorited` is absent when false - it is never written as false.
        favorite=payload.get("favorited") is True,
        archived=payload.get("archived") is True,
        trashed=payload.get("trashed") is True,
    )


def album_title(payload: dict | None) -> str | None:
    """The album's real, pre-sanitisation name, or None.

    Returns None for a null, empty, whitespace-only or LIST-valued title - the
    export root's metadata.json has `title: null` and
    user-generated-memory-titles.json has a list.
    """
    if not isinstance(payload, dict):
        return None
    title = payload.get("title")
    if not isinstance(title, str):
        return None
    return title.strip() or None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_takeout_parse.py -v && uv run ruff check .`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/rekindle/enrich tests/test_takeout_parse.py
git commit -m "feat: parse and classify Takeout JSON sidecars"
```

---

### Task 5: Takeout fixture generator

M0's motion-photo bug came from a fixture that invented a convention Google does
not use. Every structure below was **measured** on a real 45,900-file export, not
assumed. This tree is the behavioural spec for Tasks 6-12; changing it breaks
several tasks at once.

**Files:**
- Create: `tests/fixtures/takeout.py`
- Create: `tests/test_takeout_fixtures.py`

**Interfaces:**
- Consumes: `tests.fixtures.gen.make_jpeg` (M0)
- Produces: `build_takeout(root: Path) -> Path` — creates and returns the export root

- [ ] **Step 1: Write the failing test**

```python
# tests/test_takeout_fixtures.py
import json

from tests.fixtures.takeout import build_takeout


def test_the_tree_reproduces_every_measured_pathology(tmp_path):
    """This test verifies NO production code. It is a fixture asserting the
    fixture, and it earns its place only as a drift guard: build_takeout() is
    the behavioural spec for Tasks 6, 9, 11 and 12, so a change to the tree
    that quietly removes a pathology would otherwise weaken four task suites
    at once with all of them still green. Keep it, and do not mistake it for
    coverage."""
    root = build_takeout(tmp_path / "Takeout")
    year = root / "Photos from 2011"

    # Both sidecar filename schemes, and a real (N) collision: two sidecars
    # whose `title` is identical, belonging to two different photos.
    assert (year / "DSC00107.JPG").is_file()
    assert (year / "DSC00107(1).JPG").is_file()
    assert (year / "DSC00107.JPG.supplemental-metadata.json").is_file()
    assert (year / "DSC00107.JPG.supplemental-metadata(1).json").is_file()
    a = json.loads((year / "DSC00107.JPG.supplemental-metadata.json").read_text())
    b = json.loads((year / "DSC00107.JPG.supplemental-metadata(1).json").read_text())
    assert a["title"] == b["title"] == "DSC00107.JPG"
    assert a["photoTakenTime"] != b["photoTakenTime"]
    assert a["people"] != b["people"]

    # An album folder holding sidecars and ZERO media - 8 of these in the
    # reference export, hundreds of sidecars each.
    empty_album = root / "wedding_anniversary"
    assert (empty_album / "IMG_ALBUM.jpg.supplemental-metadata.json").is_file()
    assert not any(p.suffix.lower() == ".jpg" for p in empty_album.iterdir())

    # A derivative with no sidecar of its own, and its original with one.
    assert (year / "IMG_EDIT-edited.jpg").is_file()
    assert not (year / "IMG_EDIT-edited.jpg.supplemental-metadata.json").exists()
    assert (year / "IMG_EDIT.jpg.supplemental-metadata.json").is_file()

    # A motion photo: the .MP video has no sidecar, the .MP.jpg still does.
    assert (year / "PXL_1.MP").is_file()
    assert (year / "PXL_1.MP.jpg").is_file()
    assert not (year / "PXL_1.MP.supplemental-metadata.json").exists()
    assert (year / "PXL_1.MP.jpg.supplemental-metadata.json").is_file()

    # Non-photo JSON that must not be mistaken for a sidecar.
    assert json.loads((root / "metadata.json").read_text())["title"] is None
    assert isinstance(
        json.loads((root / "user-generated-memory-titles.json").read_text())["title"], list
    )
    assert (root / "Goa Trip" / "shared_album_comments.json").is_file()

    # geoDataExif key-ABSENT rather than zeroed, favorited absent when false.
    plain = json.loads((year / "IMG_EDIT.jpg.supplemental-metadata.json").read_text())
    assert "geoDataExif" not in plain
    assert "favorited" not in plain

    # Trash: excluded by FolderSource, and the enricher must exclude it too.
    assert (root / "Trash" / "IMG_GONE.jpg.supplemental-metadata.json").is_file()

    # A malformed JSON file, so `unparseable` is exercised.
    assert (year / "broken.json").read_text() == "{not json"


def test_album_titles_cover_differing_empty_and_colliding(tmp_path):
    root = build_takeout(tmp_path / "Takeout")
    assert json.loads((root / "Goa Trip" / "metadata.json").read_text())["title"] == "Goa/ Trip"
    assert json.loads((root / "Untitled" / "metadata.json").read_text())["title"] == "Untitled"
    assert json.loads((root / "Untitled(1)" / "metadata.json").read_text())["title"] == "Untitled"
    assert json.loads((root / "No Name" / "metadata.json").read_text())["title"] == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_takeout_fixtures.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tests.fixtures.takeout'`

- [ ] **Step 3: Write the implementation**

```python
# tests/fixtures/takeout.py
"""A Takeout export reproducing structure MEASURED on a real 45,900-file
export. Nothing here is invented: every pathology below was counted.

    (N) collisions with differing people   833 groups, 100 with differing people
    album folders holding zero media       8
    -edited files with no sidecar          177
    .MP halves with no sidecar             692
    geoDataExif key-absent                 89.6% of sidecars
    favorited absent when false            all but 7 sidecars
    non-photo JSON at album level          metadata.json, shared_album_comments,
                                           user-generated-memory-titles
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.fixtures.gen import make_jpeg


def _sidecar(path: Path, **overrides: object) -> None:
    payload: dict[str, object] = {
        "title": overrides.pop("title"),
        "description": "",
        "imageViews": "1",
        "creationTime": {"timestamp": "1500000000"},
        "photoTakenTime": {"timestamp": "1400000000"},
        "geoData": {"latitude": 0.0, "longitude": 0.0, "altitude": 0.0},
        "url": "https://photos.google.com/photo/x",
    }
    payload.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_takeout(root: Path) -> Path:
    year = root / "Photos from 2011"
    year.mkdir(parents=True, exist_ok=True)

    # --- the (N) collision -------------------------------------------------
    # Two DIFFERENT photos, two sidecars, one `title`. Matching on `title`
    # pairs both sidecars with DSC00107.JPG and leaves DSC00107(1).JPG bare.
    make_jpeg(year / "DSC00107.JPG", size=(24, 24))
    make_jpeg(year / "DSC00107(1).JPG", size=(25, 25))
    _sidecar(
        year / "DSC00107.JPG.supplemental-metadata.json",
        title="DSC00107.JPG",
        photoTakenTime={"timestamp": "1323826707"},
        people=[{"name": "Grace"}],
    )
    _sidecar(
        year / "DSC00107.JPG.supplemental-metadata(1).json",
        title="DSC00107.JPG",
        photoTakenTime={"timestamp": "1295183562"},
        people=[{"name": "Ada"}],
    )

    # --- an ambiguous pair: same target, disagreeing, no directory tiebreak -
    make_jpeg(year / "AMBIG.jpg", size=(26, 26))
    _sidecar(
        year / "AMBIG.jpg.supplemental-metadata.json",
        title="AMBIG.jpg",
        photoTakenTime={"timestamp": "1000000000"},
        people=[{"name": "Ada"}],
    )
    _sidecar(
        root / "Goa Trip" / "AMBIG.jpg.supplemental-metadata.json",
        title="AMBIG.jpg",
        photoTakenTime={"timestamp": "1200000000"},
        people=[{"name": "Grace"}],
    )

    # --- a photo whose only sidecar lives in an album with no media --------
    # 1,204 real photos are in this position; per-directory keying loses them.
    make_jpeg(year / "IMG_ALBUM.jpg", size=(27, 27))
    _sidecar(
        root / "wedding_anniversary" / "IMG_ALBUM.jpg.supplemental-metadata.json",
        title="IMG_ALBUM.jpg",
        photoTakenTime={"timestamp": "1600000000"},
        people=[{"name": "Ada"}],
    )

    # --- derivatives Google never writes a sidecar for ---------------------
    make_jpeg(year / "IMG_EDIT.jpg", size=(28, 28))
    make_jpeg(year / "IMG_EDIT-edited.jpg", size=(29, 29))
    _sidecar(
        year / "IMG_EDIT.jpg.supplemental-metadata.json",
        title="IMG_EDIT.jpg",
        photoTakenTime={"timestamp": "1400000000"},
        people=[{"name": "Grace"}],
        geoData={"latitude": 22.5, "longitude": 88.3, "altitude": 9.0},
    )
    make_jpeg(year / "PXL_1.MP.jpg", size=(30, 30))
    (year / "PXL_1.MP").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32)
    _sidecar(
        year / "PXL_1.MP.jpg.supplemental-metadata.json",
        title="PXL_1.MP.jpg",
        photoTakenTime={"timestamp": "1700000000"},
        people=[{"name": "Ada"}],
    )

    # --- an orphan: a sidecar whose photo is in an un-extracted part -------
    _sidecar(
        year / "IMG_MISSING.jpg.supplemental-metadata.json", title="IMG_MISSING.jpg"
    )

    # --- flags, and a bare `.json` sidecar --------------------------------
    make_jpeg(year / "IMG_FLAGS.jpg", size=(31, 31))
    _sidecar(
        year / "IMG_FLAGS.jpg.json",
        title="IMG_FLAGS.jpg",
        photoTakenTime={"timestamp": "1450000000"},
        description="a real caption",
        favorited=True,
        archived=True,
    )

    # --- malformed JSON ---------------------------------------------------
    (year / "broken.json").write_text("{not json", encoding="utf-8")

    # --- non-photo JSON ---------------------------------------------------
    (root / "metadata.json").write_text(json.dumps({"title": None}), encoding="utf-8")
    (root / "user-generated-memory-titles.json").write_text(
        json.dumps({"title": ["happy birthday ", "Leh Ladakh"]}), encoding="utf-8"
    )
    goa = root / "Goa Trip"
    goa.mkdir(parents=True, exist_ok=True)
    (goa / "metadata.json").write_text(json.dumps({"title": "Goa/ Trip"}), encoding="utf-8")
    (goa / "shared_album_comments.json").write_text(
        json.dumps({"comments": []}), encoding="utf-8"
    )

    # --- album title pathologies ------------------------------------------
    for folder, title in (("Untitled", "Untitled"), ("Untitled(1)", "Untitled"), ("No Name", "")):
        (root / folder).mkdir(parents=True, exist_ok=True)
        (root / folder / "metadata.json").write_text(
            json.dumps({"title": title}), encoding="utf-8"
        )
    make_jpeg(root / "Untitled" / "IMG_U0.jpg", size=(32, 32))
    make_jpeg(root / "Untitled(1)" / "IMG_U1.jpg", size=(33, 33))
    _sidecar(root / "Untitled" / "IMG_U0.jpg.supplemental-metadata.json", title="IMG_U0.jpg")
    _sidecar(root / "Untitled(1)" / "IMG_U1.jpg.supplemental-metadata.json", title="IMG_U1.jpg")

    # --- Trash: never parsed ----------------------------------------------
    trash = root / "Trash"
    trash.mkdir(parents=True, exist_ok=True)
    make_jpeg(trash / "IMG_GONE.jpg", size=(34, 34))
    _sidecar(trash / "IMG_GONE.jpg.supplemental-metadata.json", title="IMG_GONE.jpg")

    return root
```

If `make_jpeg` in `tests/fixtures/gen.py` does not accept a `size` keyword, call
it with whatever signature it has — every photo simply needs distinct bytes so
each gets its own `file_hash`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_takeout_fixtures.py -v && uv run ruff format tests/`
Expected: all pass. `ruff format` rewrites `tests/fixtures/takeout.py` as
transcribed above (line lengths in `_sidecar` call sites); run it before
committing or `ruff format --check .` fails at the Definition of Done.

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/takeout.py tests/test_takeout_fixtures.py
git commit -m "test: Takeout fixture tree built from measured export structure"
```

---
### Task 6: The sidecar index, resolution, and the accounting invariant

This is the task that makes v1's bug class unshippable. The index is **global**,
not per-directory — per-directory keying discarded 3,927 sidecars (16.2%), 3,026
of them carrying people, because Takeout writes a sidecar into every album a
photo belongs to without duplicating the pixels there.

`EnrichReport`'s two identities are checked by a test. v1 inherited the slogan
"report, never silently drop" and none of the machinery, and 984 sidecars
vanished into `dict.__setitem__`.

**Files:**
- Modify: `src/rekindle/enrich/takeout.py`
- Create: `tests/test_takeout_index.py`

**Interfaces:**
- Consumes: `Sidecar`, `JsonKind`, `classify_json`, `parse_sidecar` (Task 4); `Photo` (Task 1); `folder.EXCLUDED_DIRS` (M0)
- Produces:
  - `EnrichReport` dataclass with the counters listed below, plus properties `files_accounted` and `sidecars_accounted`
  - `SidecarIndex` dataclass wrapping `by_target: dict[str, list[Sidecar]]`
  - `build_index(root: Path, report: EnrichReport) -> SidecarIndex`
  - `resolve(photo: Photo, index: SidecarIndex) -> tuple[Sidecar | None, str]` returning `(sidecar, sidecar_match)` where `sidecar_match` is `"exact"`, `"ambiguous"` or `"none"`
  - `account(index: SidecarIndex, claimed: dict[str, str], report: EnrichReport) -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_takeout_index.py
from datetime import UTC, datetime
from pathlib import Path

from rekindle.enrich.takeout import EnrichReport, build_index, resolve
from rekindle.models import MediaType, Photo, PhotoMeta
from tests.fixtures.takeout import build_takeout


def _photo(*paths: Path) -> Photo:
    now = datetime(2020, 1, 1, tzinfo=UTC)
    return Photo(
        file_hash="".join(p.name for p in paths),
        paths=list(paths),
        media_type=MediaType.IMAGE,
        meta=PhotoMeta(),
        first_seen=now,
        last_seen=now,
    )


def test_the_counter_variant_gets_its_own_sidecar(tmp_path):
    """v1's headline defect: both sidecars claimed DSC00107.JPG."""
    root = build_takeout(tmp_path / "Takeout")
    index = build_index(root, EnrichReport())
    year = root / "Photos from 2011"

    plain, match, _tie = resolve(_photo(year / "DSC00107.JPG"), index)
    assert match == "exact"
    assert plain.taken_at_utc == datetime.fromtimestamp(1323826707, UTC)
    assert plain.people == ("Grace",)

    counter, match, _tie = resolve(_photo(year / "DSC00107(1).JPG"), index)
    assert match == "exact"
    assert counter.taken_at_utc == datetime.fromtimestamp(1295183562, UTC)
    assert counter.people == ("Ada",)


def test_a_sidecar_in_a_media_less_album_still_matches(tmp_path):
    """Per-directory keying loses 1,204 real photos this way."""
    root = build_takeout(tmp_path / "Takeout")
    index = build_index(root, EnrichReport())
    found, match, _tie = resolve(_photo(root / "Photos from 2011" / "IMG_ALBUM.jpg"), index)
    assert match == "exact"
    assert found.people == ("Ada",)


def test_same_directory_wins_over_a_distant_candidate(tmp_path):
    root = build_takeout(tmp_path / "Takeout")
    index = build_index(root, EnrichReport())
    year = root / "Photos from 2011"
    found, match, tie = resolve(_photo(year / "AMBIG.jpg"), index)
    assert match == "exact"
    assert found.path.parent == year
    assert found.people == ("Ada",)
    # The Goa Trip candidate disagrees and was OVERRIDDEN, not refused. 942
    # photos on the reference export; silent, it is invisible.
    assert tie is True


def test_disagreeing_candidates_with_no_directory_tiebreak_are_refused(tmp_path):
    """Never let dictionary insertion order decide. 833 real colliding groups,
    100 with differing people, 90 spanning more than a day."""
    root = build_takeout(tmp_path / "Takeout")
    index = build_index(root, EnrichReport())
    # A path in NEITHER candidate's directory, so preference cannot resolve it.
    found, match, _tie = resolve(_photo(root / "Goa Trip" / "elsewhere" / "AMBIG.jpg"), index)
    assert match == "ambiguous"
    assert found is None


def test_a_photo_with_no_sidecar_resolves_to_none(tmp_path):
    root = build_takeout(tmp_path / "Takeout")
    index = build_index(root, EnrichReport())
    found, match, _tie = resolve(_photo(root / "Photos from 2011" / "IMG_EDIT-edited.jpg"), index)
    assert found is None
    assert match == "none"


def test_trash_sidecars_are_excluded_not_orphaned(tmp_path):
    root = build_takeout(tmp_path / "Takeout")
    report = EnrichReport()
    index = build_index(root, report)
    assert "img_gone.jpg" not in index.by_target
    assert report.excluded_dirs == 1


def test_build_index_buckets_every_json_file(tmp_path):
    root = build_takeout(tmp_path / "Takeout")
    report = EnrichReport()
    build_index(root, report)
    assert report.json_files_seen == len(list(root.rglob("*.json")))
    assert report.json_files_seen == 20
    assert report.files_accounted == 20
    assert report.sidecars_seen == 11
    # root, Goa Trip, Untitled, Untitled(1), No Name
    assert report.album_metadata == 5
    # user-generated-memory-titles.json, shared_album_comments.json
    assert report.other_json == 2
    assert report.excluded_dirs == 1  # Trash/IMG_GONE.jpg...json
    assert report.unparseable == 1  # broken.json


def test_the_sidecar_accounting_identity_holds(tmp_path):
    """v1 would have failed this by 984.

    On its own the identity is weak - `account()` partitions a set it builds
    itself, so it cannot see anything downstream of indexing. The assertion
    that actually bites is `matched == photos_enriched`, in
    tests/test_takeout_enricher.py. This one pins the partition; that one ties
    the partition to what reached the database.
    """
    from rekindle.enrich.takeout import account

    root = build_takeout(tmp_path / "Takeout")
    report = EnrichReport()
    index = build_index(root, report)
    photos = [
        _photo(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in {".jpg", ".mp"} and "Trash" not in p.parts
    ]
    claimed: dict[str, str] = {}
    applied: set[Path] = set()
    for photo in photos:
        sidecar, match, _tie = resolve(photo, index)
        if match != "none":
            claimed[photo.paths[0].name.casefold()] = match
        if sidecar is not None:
            applied.add(sidecar.path)

    account(index, claimed, applied, report)
    assert report.sidecars_seen == report.sidecars_accounted
    assert report.orphaned >= 1  # IMG_MISSING
    assert report.ambiguous == 0  # AMBIG resolved by directory preference
    # AMBIG has two candidates; one is applied, the other is SUPERSEDED. The
    # first draft of this plan credited both to `matched`, which inflated it
    # by 3,358 on a real export.
    assert report.superseded == 1
    assert report.matched == len(applied)


def test_superseded_sidecars_are_reported_not_discarded(tmp_path):
    """Delete the `superseded` line from account() and this fails.

    The identity above would still pass, because the count would simply move
    into `matched`. That is the exact shape of v1's bug.
    """
    from rekindle.enrich.takeout import account

    root = build_takeout(tmp_path / "Takeout")
    report = EnrichReport()
    index = build_index(root, report)
    year = root / "Photos from 2011"
    sidecar, _match, _tie = resolve(_photo(year / "AMBIG.jpg"), index)
    account(index, {"ambig.jpg": "exact"}, {sidecar.path}, report)
    assert report.matched == 1
    assert report.superseded == 1
```

The 11 sidecars are: `DSC00107` x2, `AMBIG` x2, `IMG_ALBUM`, `IMG_EDIT`,
`PXL_1.MP.jpg`, `IMG_MISSING`, `IMG_FLAGS`, `IMG_U0`, `IMG_U1`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_takeout_index.py -v`
Expected: FAIL — `ImportError: cannot import name 'EnrichReport' from 'rekindle.enrich.takeout'`

- [ ] **Step 3: Write the implementation**

```python
# src/rekindle/enrich/takeout.py  (append; add these imports at the top)
# from dataclasses import dataclass, field
# from rekindle.sources.folder import EXCLUDED_DIRS


@dataclass
class EnrichReport:
    """What the enrichment pass found. Two identities, both pinned by a test.

        json_files_seen == sidecars_seen + album_metadata
                           + other_json + excluded_dirs + unparseable
        sidecars_seen   == matched + superseded + orphaned + ambiguous

    The spec's section 7 gives only the second, and without `superseded` in it
    3,358 sidecars on the reference export are parsed and discarded with
    nothing reporting it.

    Neither identity can catch a bug downstream of indexing on its own - both
    partition sets that `account()` builds. The check that CAN fail is
    `matched == photos_enriched`, asserted end to end in Task 11: it ties the
    accounting to what was actually written to the database.
    """

    json_files_seen: int = 0
    sidecars_seen: int = 0
    album_metadata: int = 0
    other_json: int = 0
    excluded_dirs: int = 0
    unparseable: int = 0

    matched: int = 0
    orphaned: int = 0
    ambiguous: int = 0
    # Sidecars that named a photo we DID enrich but were not the one applied -
    # a second or third candidate for the same file. 3,358 of them on the
    # reference export; without this bucket they are discarded unreported.
    superseded: int = 0

    photos_enriched: int = 0
    # Photos where a same-directory candidate was preferred over a DISTANT one
    # that disagreed with it. Spec-conformant (rule 1 outranks rule 3) but
    # 942 on the reference export against an `ambiguous` count of 2, so
    # leaving it unreported makes the hazard look negligible.
    directory_preference_broke_a_tie: int = 0
    derivatives_enriched: int = 0
    people_added: int = 0
    dates_corrected: int = 0
    gps_added: int = 0
    descriptions_added: int = 0
    favourites_added: int = 0
    albums_retitled: int = 0
    conflicts: int = 0
    clustered_dates_suppressed: int = 0
    title_disagreements: int = 0
    album_title_collisions: list[tuple[str, str]] = field(default_factory=list)

    @property
    def files_accounted(self) -> int:
        return (
            self.sidecars_seen
            + self.album_metadata
            + self.other_json
            + self.excluded_dirs
            + self.unparseable
        )

    @property
    def sidecars_accounted(self) -> int:
        return self.matched + self.superseded + self.orphaned + self.ambiguous


@dataclass
class SidecarIndex:
    """Global, keyed on the casefolded target filename.

    v1 keyed on (directory, title). That is wrong twice over: `title` omits
    the (N) counter, and Takeout writes a sidecar into every album a photo
    belongs to WITHOUT duplicating the pixels there - 8 album folders in the
    reference export hold sidecars and zero media.
    """

    by_target: dict[str, list[Sidecar]] = field(default_factory=dict)
    albums: dict[Path, str] = field(default_factory=dict)

    def add(self, sidecar: Sidecar) -> None:
        self.by_target.setdefault(sidecar.target_cf, []).append(sidecar)


def build_index(root: Path, report: EnrichReport) -> SidecarIndex:
    """Walk every .json under `root`, classify it, and bucket it.

    EXCLUDED_DIRS is inherited from FolderSource: parsing Trash/'s sidecars
    and counting them against an index that correctly excludes those photos
    would inflate the orphan count on a perfectly complete library.
    """
    index = SidecarIndex()
    for path in sorted(root.rglob("*.json")):
        report.json_files_seen += 1
        if any(part.casefold() in EXCLUDED_DIRS for part in path.relative_to(root).parts[:-1]):
            report.excluded_dirs += 1
            continue
        kind, payload = classify_json(path)
        if kind is JsonKind.UNPARSEABLE:
            report.unparseable += 1
            continue
        if kind is JsonKind.ALBUM:
            report.album_metadata += 1
            title = album_title(payload)
            if title:
                index.albums[path.parent] = title
            continue
        if kind is JsonKind.OTHER:
            report.other_json += 1
            continue
        report.sidecars_seen += 1
        sidecar = parse_sidecar(path, payload)
        if sidecar.title_disagrees:
            report.title_disagreements += 1
        index.add(sidecar)
    return index


def _disagree(a: Sidecar, b: Sidecar) -> bool:
    """Do two candidates for one photo tell different stories?

    Only the fields that would change the photo's record. Two sidecars
    agreeing on date and people are interchangeable, however many there are.
    """
    return a.taken_at_utc != b.taken_at_utc or set(a.people) != set(b.people)


def resolve(photo: Photo, index: SidecarIndex) -> tuple[Sidecar | None, str, bool]:
    """Find the one sidecar describing `photo`, or refuse.

    Returns (sidecar, outcome, preference_broke_a_tie).

    1. A candidate in the same directory as one of the photo's paths wins.
    2. Otherwise, if the remaining candidates all agree, take the first.
    3. Otherwise REFUSE, and say so. Never let insertion order decide.

    Rule 1 deliberately outranks rule 3, so a DISTANT candidate that disagrees
    is overridden rather than refused. Measured on the reference export that
    happens to 942 photos, and every disagreeing group has its candidates in
    different directories - so rule 3 fires only 2 times. Left silent, doctor
    prints an `ambiguous` count of 2 and the hazard looks negligible when it
    is not. The third return value is what makes the override visible.
    """
    candidates: list[Sidecar] = []
    parents = {p.parent for p in photo.paths}
    for path in photo.paths:
        for sidecar in index.by_target.get(path.name.casefold(), ()):
            if sidecar not in candidates:
                candidates.append(sidecar)
    if not candidates:
        return None, "none", False

    same_dir = [s for s in candidates if s.path.parent in parents]
    if same_dir:
        chosen = same_dir[0]
        if any(_disagree(chosen, s) for s in same_dir[1:]):
            return None, "ambiguous", False
        tie = any(
            _disagree(chosen, s) for s in candidates if s.path.parent not in parents
        )
        return chosen, "exact", tie

    if all(not _disagree(candidates[0], s) for s in candidates[1:]):
        return candidates[0], "exact", False
    return None, "ambiguous", False


def account(
    index: SidecarIndex,
    claimed: dict[str, str],
    applied: set[Path],
    report: EnrichReport,
) -> None:
    """Partition every indexed sidecar into exactly one bucket.

    `claimed` maps a casefolded target to the resolution outcome; `applied`
    holds the paths of the sidecars actually WRITTEN onto a photo.

    The `superseded` bucket is the whole point. Crediting `matched` with
    `len(sidecars)` for every claimed target - as the first draft of this plan
    did - inflated it by 3,358 on the reference export, because a target with
    three candidates contributes one application and two discards. Those 3,358
    were parsed and thrown away with nothing reporting it, which is precisely
    v1's failure mode surviving inside the machinery built to prevent it.
    """
    for target, sidecars in index.by_target.items():
        outcome = claimed.get(target)
        if outcome == "ambiguous":
            report.ambiguous += len(sidecars)
        elif outcome == "exact":
            used = sum(1 for s in sidecars if s.path in applied)
            report.matched += used
            report.superseded += len(sidecars) - used
        else:
            report.orphaned += len(sidecars)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_takeout_index.py -v && uv run ruff check .`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/rekindle/enrich/takeout.py tests/test_takeout_index.py
git commit -m "feat: global sidecar index with directory preference and an accounting invariant"
```

---

### Task 7: Deriving time from a UTC instant

**The single most damaging error in v1.** `photoTakenTime` is a UTC instant;
`timestamps.resolve()` expects a naive wall-clock value and stamps a zone onto
it. Feeding one to the other yields `local = utc`, so a photo taken at 21:00 IST
is relabelled 15:30 — and v1 did this to *every* EXIF-dated photo. Measured on
1,317: 811 shifted by exactly −330 minutes, and 442 more that already carried a
correct `EXIF_OFFSET` had it thrown away.

The enricher reads the index, not the files, so the EXIF offset must be recovered
from what M0 already stored. That is also what keeps `enrich` fast: no file is
opened, which is the promise §3 makes about re-running after adding archive parts.

**Files:**
- Modify: `src/rekindle/meta/timestamps.py`
- Test: `tests/test_timestamps.py`

**Interfaces:**
- Consumes: `TzSource`, `Gps`, `TzLookup` (M0)
- Produces: `from_takeout(google_utc, existing_utc, existing_local, existing_tz_source, gps=None, tz_lookup=None) -> tuple[datetime, datetime, TzSource]`
- Produces: `is_plausible_offset(delta: timedelta) -> timedelta | None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_timestamps.py  (append)
from datetime import UTC, datetime, timedelta, timezone

from rekindle.meta.timestamps import from_takeout, is_plausible_offset
from rekindle.models import Gps, TzSource

IST = timezone(timedelta(hours=5, minutes=30))


def test_a_known_exif_offset_is_kept_not_discarded():
    """442 of 1,317 measured photos already had this and v1 threw it away."""
    google = datetime(2020, 3, 11, 15, 30, tzinfo=UTC)
    utc, local, src = from_takeout(
        google_utc=google,
        existing_utc=datetime(2020, 3, 11, 15, 30, tzinfo=UTC),
        existing_local=datetime(2020, 3, 11, 21, 0, tzinfo=IST),
        existing_tz_source=TzSource.EXIF_OFFSET,
    )
    assert utc == google
    assert local.hour == 21 and local.minute == 0
    assert local.utcoffset() == timedelta(hours=5, minutes=30)
    assert src is TzSource.EXIF_OFFSET


def test_naive_exif_recovers_the_offset_from_the_difference():
    """811 of 1,317 measured photos. M0 stored 21:00 stamped UTC; Google says
    the instant was 15:30Z. The difference IS +05:30."""
    naive_stamped_utc = datetime(2020, 3, 11, 21, 0, tzinfo=UTC)
    google = datetime(2020, 3, 11, 15, 30, tzinfo=UTC)
    utc, local, src = from_takeout(
        google_utc=google,
        existing_utc=naive_stamped_utc,
        existing_local=naive_stamped_utc,
        existing_tz_source=TzSource.EXIF_NAIVE,
    )
    assert utc == google
    assert local.hour == 21 and local.minute == 0  # NOT 15:30
    assert local.utcoffset() == timedelta(hours=5, minutes=30)
    assert src is TzSource.TAKEOUT


def test_a_broken_camera_clock_is_not_mistaken_for_an_offset():
    """EXIF says 2016-06-24, Google says 2020-03-14. That 1,359-day gap is a
    reset camera clock, not a timezone."""
    utc, local, src = from_takeout(
        google_utc=datetime(2020, 3, 14, 14, 17, 31, tzinfo=UTC),
        existing_utc=datetime(2016, 6, 24, 9, 47, 33, tzinfo=UTC),
        existing_local=datetime(2016, 6, 24, 9, 47, 33, tzinfo=UTC),
        existing_tz_source=TzSource.EXIF_NAIVE,
    )
    assert utc == datetime(2020, 3, 14, 14, 17, 31, tzinfo=UTC)
    assert local == utc
    assert src is TzSource.TAKEOUT


def test_no_prior_date_falls_back_to_utc():
    google = datetime(2020, 3, 11, 15, 30, tzinfo=UTC)
    utc, local, src = from_takeout(
        google_utc=google,
        existing_utc=datetime(2024, 1, 1, tzinfo=UTC),
        existing_local=datetime(2024, 1, 1, tzinfo=UTC),
        existing_tz_source=TzSource.FILE_MTIME,
    )
    assert utc == google and local == google
    assert src is TzSource.TAKEOUT


def test_gps_resolves_the_zone_when_there_is_no_exif_date():
    google = datetime(2020, 3, 11, 15, 30, tzinfo=UTC)
    utc, local, src = from_takeout(
        google_utc=google,
        existing_utc=None,
        existing_local=None,
        existing_tz_source=TzSource.NONE,
        gps=Gps(lat=22.5, lon=88.3),
        tz_lookup=lambda _g: "Asia/Kolkata",
    )
    assert utc == google
    assert local.hour == 21 and local.minute == 0
    assert src is TzSource.GPS


def test_a_raising_tz_lookup_degrades_instead_of_crashing():
    def boom(_g):
        raise RuntimeError("timezonefinder exploded")

    google = datetime(2020, 3, 11, 15, 30, tzinfo=UTC)
    utc, local, src = from_takeout(
        google_utc=google,
        existing_utc=None,
        existing_local=None,
        existing_tz_source=TzSource.NONE,
        gps=Gps(lat=22.5, lon=88.3),
        tz_lookup=boom,
    )
    assert utc == google and local == google and src is TzSource.TAKEOUT


def test_plausible_offsets():
    assert is_plausible_offset(timedelta(hours=5, minutes=30)) == timedelta(hours=5, minutes=30)
    assert is_plausible_offset(timedelta(hours=-8)) == timedelta(hours=-8)
    assert is_plausible_offset(timedelta(hours=5, minutes=30, seconds=12)) == timedelta(
        hours=5, minutes=30
    )
    assert is_plausible_offset(timedelta(hours=15)) is None
    assert is_plausible_offset(timedelta(days=1359)) is None
    assert is_plausible_offset(timedelta(hours=5, minutes=37)) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_timestamps.py -v`
Expected: FAIL — `ImportError: cannot import name 'from_takeout' from 'rekindle.meta.timestamps'`

- [ ] **Step 3: Write the implementation**

```python
# src/rekindle/meta/timestamps.py  (append)

# Real UTC offsets run from -12:00 to +14:00 and are whole quarter-hours.
_MAX_OFFSET = timedelta(hours=14)
_OFFSET_QUANTUM = timedelta(minutes=15)
_QUANTUM_TOLERANCE = timedelta(seconds=90)


def is_plausible_offset(delta: timedelta) -> timedelta | None:
    """Round `delta` to a real UTC offset, or reject it.

    This is the guard that keeps a broken camera clock out of the timezone
    ladder. On the reference export the EXIF/Google difference is either
    exactly -05:30 (a timezone) or over a year (a reset clock); nothing in
    between. Rejecting anything that is not a quarter-hour within 14 hours
    separates the two cleanly.
    """
    if abs(delta) > _MAX_OFFSET:
        return None
    quanta = round(delta / _OFFSET_QUANTUM)
    rounded = _OFFSET_QUANTUM * quanta
    if abs(delta - rounded) > _QUANTUM_TOLERANCE:
        return None
    return rounded


def from_takeout(
    google_utc: datetime,
    existing_utc: datetime | None,
    existing_local: datetime | None,
    existing_tz_source: TzSource,
    gps: Gps | None = None,
    tz_lookup: TzLookup | None = None,
) -> tuple[datetime, datetime, TzSource]:
    """Derive (utc, local, tz_source) from Google's photoTakenTime.

    photoTakenTime is an INSTANT. It carries no offset, so local time has to
    come from somewhere else. Passing it to `resolve()` - which expects a
    naive wall-clock value - silently yields local == utc and relabels a 21:00
    photo as 15:30. That was v1's most damaging defect.

    Ladder, per spec section 5:
      1. an offset EXIF already established (OffsetTimeOriginal)
      2. a GPS timezone
      3. exif_naive - google_utc, which recovers +05:30 exactly
      4. UTC

    `TzSource.TAKEOUT` describes the DATE's provenance. When an offset or GPS
    resolved the zone, that source is kept: conflating the two loses
    information about how well local time is actually known.
    """
    # Rung 1 and 2: M0 already resolved a real zone and stored it on
    # taken_at_local. Reuse it rather than re-deriving - and note this keeps
    # `enrich` free of any file or timezonefinder access for these photos.
    if existing_tz_source in (TzSource.EXIF_OFFSET, TzSource.GPS) and existing_local is not None:
        offset = existing_local.utcoffset()
        if offset is not None:
            zone = timezone(offset)
            return google_utc, google_utc.astimezone(zone), existing_tz_source

    # Rung 2 for a photo M0 could not date at all but which has coordinates.
    if gps is not None and tz_lookup is not None:
        try:
            name = tz_lookup(gps)
        except Exception:
            # An injected third-party callable may raise on odd coordinates.
            # Any failure degrades to the next rung, exactly as in resolve().
            name = None
        if name:
            try:
                zone_info = ZoneInfo(name)
            except (ZoneInfoNotFoundError, ValueError):
                zone_info = None
            if zone_info is not None:
                return google_utc, google_utc.astimezone(zone_info), TzSource.GPS

    # Rung 3: M0 stored the naive EXIF wall clock stamped as UTC. The gap
    # between that and Google's true instant IS the photo's UTC offset.
    if existing_tz_source is TzSource.EXIF_NAIVE and existing_utc is not None:
        offset = is_plausible_offset(existing_utc - google_utc)
        if offset is not None:
            return google_utc, google_utc.astimezone(timezone(offset)), TzSource.TAKEOUT

    # Rung 4.
    return google_utc, google_utc, TzSource.TAKEOUT
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_timestamps.py -v && uv run ruff check .`
Expected: all pass, including M0's 13 existing timestamp tests

- [ ] **Step 5: Commit**

```bash
git add src/rekindle/meta/timestamps.py tests/test_timestamps.py
git commit -m "feat: derive local time from Google's UTC instant instead of round-tripping it"
```

---

### Task 8: Applying a sidecar to a photo

The field rules from spec §5, in one pure function. Two of them are load-bearing:

- **`people` replaces rather than unions.** A flat union can only grow, so a face
  tag corrected in Google Photos could never be fixed by re-exporting. The
  subtraction is possible only because `PhotoMeta.takeout_people` records what
  the *last* enrich wrote.
- **`metadata_conflict` compares instants after offset normalisation.** v1's rule
  fired on 65.8% of comparable photos, almost all of it the −330 minute timezone
  artefact. After normalisation it means what a reader assumes: a genuinely
  different capture time.

**Files:**
- Modify: `src/rekindle/enrich/takeout.py`
- Create: `tests/test_takeout_apply.py`

**Interfaces:**
- Consumes: `Sidecar` (Task 4), `EnrichReport` (Task 6), `from_takeout` (Task 7), `PhotoMeta`, `Photo`
- Produces:
  - `CLUSTER_MIN: int = 10`
  - `CONFLICT_TOLERANCE: timedelta = timedelta(minutes=1)`
  - `apply_sidecar(photo: Photo, sidecar: Sidecar, clustered: frozenset[datetime], report: EnrichReport, tz_lookup: TzLookup | None = None) -> None` — mutates `photo` in place
  - `clustered_timestamps(index: SidecarIndex) -> frozenset[datetime]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_takeout_apply.py
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

from rekindle.enrich.takeout import EnrichReport, Sidecar, apply_sidecar
from rekindle.models import Gps, MediaType, Photo, PhotoMeta, TzSource

IST = timezone(timedelta(hours=5, minutes=30))
NOW = datetime(2020, 1, 1, tzinfo=UTC)


def _sidecar(**kw) -> Sidecar:
    base = dict(
        path=Path("x.json"),
        target_cf="x.jpg",
        title="x.jpg",
        taken_at_utc=None,
        people=(),
        gps=None,
        description=None,
        favorite=False,
        archived=False,
        trashed=False,
    )
    base.update(kw)
    return Sidecar(**base)


def _photo(meta: PhotoMeta) -> Photo:
    return Photo(
        file_hash="h",
        paths=[Path("x.jpg")],
        media_type=MediaType.IMAGE,
        meta=meta,
        first_seen=NOW,
        last_seen=NOW,
    )


def test_google_date_wins_and_local_keeps_the_wall_clock():
    naive = datetime(2020, 3, 11, 21, 0, tzinfo=UTC)
    photo = _photo(
        PhotoMeta(taken_at_utc=naive, taken_at_local=naive, tz_source=TzSource.EXIF_NAIVE)
    )
    report = EnrichReport()
    apply_sidecar(
        photo,
        _sidecar(taken_at_utc=datetime(2020, 3, 11, 15, 30, tzinfo=UTC)),
        frozenset(),
        report,
    )
    assert photo.meta.taken_at_utc == datetime(2020, 3, 11, 15, 30, tzinfo=UTC)
    assert photo.meta.taken_at_local.hour == 21
    assert photo.meta.exif_taken_at_utc == naive
    assert photo.metadata_conflict is False  # a timezone is not a conflict
    assert report.dates_corrected == 1


def test_a_known_exif_offset_that_agrees_is_not_a_conflict():
    """The EXIF instant is already true UTC here, so it must NOT have the
    offset subtracted before comparison. Getting that wrong invents a conflict
    on the 34% of photos that carry a real OffsetTimeOriginal."""
    google = datetime(2020, 3, 11, 15, 30, tzinfo=UTC)
    photo = _photo(
        PhotoMeta(
            taken_at_utc=google,
            taken_at_local=datetime(2020, 3, 11, 21, 0, tzinfo=IST),
            tz_source=TzSource.EXIF_OFFSET,
        )
    )
    report = EnrichReport()
    apply_sidecar(photo, _sidecar(taken_at_utc=google), frozenset(), report)
    assert photo.metadata_conflict is False
    assert report.conflicts == 0
    assert photo.meta.tz_source is TzSource.EXIF_OFFSET
    assert photo.meta.taken_at_local.hour == 21


def test_a_genuinely_different_instant_is_a_conflict_and_is_retained():
    exif = datetime(2016, 6, 24, 9, 47, 33, tzinfo=UTC)
    photo = _photo(PhotoMeta(taken_at_utc=exif, taken_at_local=exif, tz_source=TzSource.EXIF_NAIVE))
    report = EnrichReport()
    apply_sidecar(
        photo,
        _sidecar(taken_at_utc=datetime(2020, 3, 14, 14, 17, 31, tzinfo=UTC)),
        frozenset(),
        report,
    )
    assert photo.metadata_conflict is True
    assert photo.meta.exif_taken_at_utc == exif  # the promise v1 could not keep
    assert report.conflicts == 1


def test_a_clustered_google_date_does_not_override_a_real_exif_date():
    """618 real sidecars share one second. They did not fire together."""
    exif = datetime(2016, 6, 24, 9, 47, 33, tzinfo=UTC)
    google = datetime(2020, 3, 11, 7, 30, tzinfo=UTC)
    photo = _photo(PhotoMeta(taken_at_utc=exif, taken_at_local=exif, tz_source=TzSource.EXIF_NAIVE))
    report = EnrichReport()
    apply_sidecar(photo, _sidecar(taken_at_utc=google), frozenset({google}), report)
    assert photo.meta.taken_at_utc == exif
    assert photo.meta.tz_source is TzSource.EXIF_NAIVE
    assert report.clustered_dates_suppressed == 1
    assert report.dates_corrected == 0


def test_a_clustered_date_still_applies_when_there_is_no_real_exif_date():
    mtime = datetime(2024, 1, 1, tzinfo=UTC)
    google = datetime(2020, 3, 11, 7, 30, tzinfo=UTC)
    photo = _photo(
        PhotoMeta(taken_at_utc=mtime, taken_at_local=mtime, tz_source=TzSource.FILE_MTIME)
    )
    report = EnrichReport()
    apply_sidecar(photo, _sidecar(taken_at_utc=google), frozenset({google}), report)
    assert photo.meta.taken_at_utc == google
    assert report.clustered_dates_suppressed == 0


def test_people_replace_the_previous_takeout_set_and_keep_other_sources():
    photo = _photo(PhotoMeta(people=["FromXmp", "Ada", "Grace"], takeout_people=["Ada", "Grace"]))
    report = EnrichReport()
    apply_sidecar(photo, _sidecar(people=("Grace",)), frozenset(), report)
    assert photo.meta.people == ["FromXmp", "Grace"]  # Ada retracted
    assert photo.meta.takeout_people == ["Grace"]


def test_applying_twice_is_idempotent():
    photo = _photo(
        PhotoMeta(
            taken_at_utc=datetime(2020, 3, 11, 21, 0, tzinfo=UTC),
            taken_at_local=datetime(2020, 3, 11, 21, 0, tzinfo=UTC),
            tz_source=TzSource.EXIF_NAIVE,
            people=["Ada"],
        )
    )
    sidecar = _sidecar(
        taken_at_utc=datetime(2020, 3, 11, 15, 30, tzinfo=UTC),
        people=("Grace",),
        gps=Gps(lat=1.0, lon=2.0),
        description="hello",
        favorite=True,
    )
    apply_sidecar(photo, sidecar, frozenset(), EnrichReport())
    first = (
        photo.meta.taken_at_utc,
        photo.meta.taken_at_local,
        photo.meta.tz_source,
        list(photo.meta.people),
        list(photo.meta.takeout_people),
        photo.meta.exif_taken_at_utc,
        photo.metadata_conflict,
    )
    apply_sidecar(photo, sidecar, frozenset(), EnrichReport())
    assert first == (
        photo.meta.taken_at_utc,
        photo.meta.taken_at_local,
        photo.meta.tz_source,
        list(photo.meta.people),
        list(photo.meta.takeout_people),
        photo.meta.exif_taken_at_utc,
        photo.metadata_conflict,
    )


def test_gps_fills_only_when_absent():
    photo = _photo(PhotoMeta(gps=Gps(lat=10.0, lon=20.0)))
    report = EnrichReport()
    apply_sidecar(photo, _sidecar(gps=Gps(lat=1.0, lon=2.0)), frozenset(), report)
    assert photo.meta.gps.lat == 10.0
    assert report.gps_added == 0

    bare = _photo(PhotoMeta())
    apply_sidecar(bare, _sidecar(gps=Gps(lat=1.0, lon=2.0)), frozenset(), report)
    assert bare.meta.gps.lat == 1.0
    assert report.gps_added == 1


def test_longest_description_wins_and_flags_carry_over():
    photo = _photo(PhotoMeta(description="short"))
    report = EnrichReport()
    apply_sidecar(
        photo,
        _sidecar(description="a much longer caption", favorite=True, archived=True, trashed=True),
        frozenset(),
        report,
    )
    assert photo.meta.description == "a much longer caption"
    assert photo.meta.favorite is True
    assert photo.meta.archived is True
    assert photo.meta.trashed is True
    assert report.descriptions_added == 1
    assert report.favourites_added == 1


def test_sidecar_match_is_recorded():
    photo = _photo(PhotoMeta())
    apply_sidecar(photo, _sidecar(), frozenset(), EnrichReport())
    assert photo.sidecar_match == "exact"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_takeout_apply.py -v`
Expected: FAIL — `ImportError: cannot import name 'apply_sidecar' from 'rekindle.enrich.takeout'`

- [ ] **Step 3: Write the implementation**

```python
# src/rekindle/enrich/takeout.py  (append; add
# `from collections import Counter`, `from datetime import timedelta` and
# `from rekindle.meta.timestamps import TzLookup, from_takeout`,
# `from rekindle.models import TzSource` to the imports)

# 20 timestamps in the reference export are shared by >=10 sidecars, covering
# 907 records - one value on 618 of them. No 618 photos fire in one second;
# these are Google's day- or year-granular guesses.
CLUSTER_MIN = 10

# After offset normalisation the -05:30 artefact is gone, so a tight bound now
# means genuine disagreement. A day-wide tolerance would hide exactly the
# broken-camera-clock cases (2016 EXIF vs 2020 Google) the flag exists for.
CONFLICT_TOLERANCE = timedelta(minutes=1)


def clustered_timestamps(index: SidecarIndex) -> frozenset[datetime]:
    """photoTakenTime values shared by CLUSTER_MIN or more sidecars."""
    counts: Counter[datetime] = Counter()
    for sidecars in index.by_target.values():
        for sidecar in sidecars:
            if sidecar.taken_at_utc is not None:
                counts[sidecar.taken_at_utc] += 1
    return frozenset(ts for ts, n in counts.items() if n >= CLUSTER_MIN)


def _has_real_date(meta: PhotoMeta) -> bool:
    """A filesystem mtime is not knowing when a photo was taken."""
    return meta.taken_at_utc is not None and meta.tz_source not in (
        TzSource.FILE_MTIME,
        TzSource.NONE,
        TzSource.TAKEOUT,
    )


def apply_sidecar(
    photo: Photo,
    sidecar: Sidecar,
    clustered: frozenset[datetime],
    report: EnrichReport,
    tz_lookup: TzLookup | None = None,
) -> None:
    """Write `sidecar` onto `photo`, in place, per spec section 5.

    Idempotent: running it twice with the same sidecar produces the same
    record. That is why `people` replaces rather than unions and why
    `exif_taken_at_utc` is written once and then left alone.
    """
    meta = photo.meta

    if sidecar.taken_at_utc is not None:
        suppress = sidecar.taken_at_utc in clustered and _has_real_date(meta)
        if suppress:
            report.clustered_dates_suppressed += 1
        else:
            # Was the EXIF value a naive WALL CLOCK stamped UTC, or a true
            # instant? Only the first needs its offset subtracted before it can
            # be compared. Getting this wrong invents a conflict on every photo
            # that already carried a correct OffsetTimeOriginal - 34% of them.
            was_naive = meta.tz_source is TzSource.EXIF_NAIVE or (
                meta.tz_source is TzSource.TAKEOUT and meta.exif_taken_at_utc is not None
            )
            # Keep the displaced EXIF instant BEFORE overwriting it, and only
            # on the first enrich - on a re-run taken_at_utc is already
            # Google's, and copying that here would erase the real one.
            if meta.exif_taken_at_utc is None and _has_real_date(meta):
                meta.exif_taken_at_utc = meta.taken_at_utc

            utc, local, tz_source = from_takeout(
                google_utc=sidecar.taken_at_utc,
                existing_utc=meta.exif_taken_at_utc or meta.taken_at_utc,
                existing_local=meta.taken_at_local,
                existing_tz_source=(
                    TzSource.EXIF_NAIVE
                    if meta.tz_source is TzSource.TAKEOUT and meta.exif_taken_at_utc
                    else meta.tz_source
                ),
                gps=meta.gps or sidecar.gps,
                tz_lookup=tz_lookup,
            )
            if meta.taken_at_utc != utc:
                report.dates_corrected += 1
            meta.taken_at_utc = utc
            meta.taken_at_local = local
            meta.tz_source = tz_source

            # Instants, after normalisation. from_takeout has already folded a
            # pure timezone difference away, so anything left is real.
            if meta.exif_taken_at_utc is not None:
                normalised = meta.exif_taken_at_utc
                if was_naive and local is not None:
                    offset = local.utcoffset()
                    if offset is not None:
                        normalised = meta.exif_taken_at_utc - offset
                conflict = abs(normalised - utc) > CONFLICT_TOLERANCE
                if conflict and not photo.metadata_conflict:
                    report.conflicts += 1
                photo.metadata_conflict = photo.metadata_conflict or conflict

    # People: REPLACE the previous Takeout contribution, keep everything else.
    # A union can only grow, so a face tag corrected in Google Photos could
    # never be retracted - and this tool holds 40 real people's names.
    previous = set(meta.takeout_people)
    kept = [name for name in meta.people if name not in previous]
    before = len(meta.people)
    meta.people = list(dict.fromkeys([*kept, *sidecar.people]))
    meta.takeout_people = list(sidecar.people)
    if len(meta.people) > before:
        report.people_added += len(meta.people) - before

    if meta.gps is None and sidecar.gps is not None:
        meta.gps = sidecar.gps
        report.gps_added += 1

    if sidecar.description and (
        meta.description is None or len(sidecar.description) > len(meta.description)
    ):
        meta.description = sidecar.description
        report.descriptions_added += 1

    if sidecar.favorite and not meta.favorite:
        meta.favorite = True
        report.favourites_added += 1

    # Google's own flags. Archived means the user deliberately hid this photo.
    meta.archived = meta.archived or sidecar.archived
    meta.trashed = meta.trashed or sidecar.trashed

    photo.sidecar_match = "exact"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_takeout_apply.py -v && uv run ruff check .`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/rekindle/enrich/takeout.py tests/test_takeout_apply.py
git commit -m "feat: sidecar merge rules with retractable people and instant-based conflicts"
```

---

### Task 9: Album retitling

Album `metadata.json` holds the pre-sanitisation name: folder
`Abhirup Birthday- Sudipta Saad` has title `Abhirup Birthday/ Sudipta Saad`.
Applied **wherever the photo lives**, because 8 album folders in the reference
export hold sidecars and no media at all, so a per-directory rule would never
fire for them.

Two titles collapsing to one name (`Untitled`, `Untitled(1)` → `Untitled`) is
detected and reported, not merged — spec §11.

**Files:**
- Modify: `src/rekindle/enrich/takeout.py`
- Create: `tests/test_takeout_albums.py`

**Interfaces:**
- Consumes: `SidecarIndex.albums` (Task 6), `EnrichReport` (Task 6), `Photo`
- Produces:
  - `album_renames(index: SidecarIndex, report: EnrichReport) -> dict[str, str]` — folder name to real title
  - `retitle_albums(photo: Photo, renames: dict[str, str], report: EnrichReport) -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_takeout_albums.py
from datetime import UTC, datetime
from pathlib import Path

from rekindle.enrich.takeout import (
    EnrichReport,
    SidecarIndex,
    album_renames,
    build_index,
    retitle_albums,
)
from rekindle.models import MediaType, Photo, PhotoMeta
from tests.fixtures.takeout import build_takeout

NOW = datetime(2020, 1, 1, tzinfo=UTC)


def _photo(*albums: str) -> Photo:
    return Photo(
        file_hash="h",
        paths=[Path("x.jpg")],
        media_type=MediaType.IMAGE,
        meta=PhotoMeta(),
        first_seen=NOW,
        last_seen=NOW,
        albums=list(albums),
    )


def test_a_differing_title_replaces_the_sanitised_folder_name(tmp_path):
    root = build_takeout(tmp_path / "Takeout")
    report = EnrichReport()
    index = build_index(root, report)
    renames = album_renames(index, report)
    assert renames["Goa Trip"] == "Goa/ Trip"

    photo = _photo("Goa Trip", "Photos from 2011")
    retitle_albums(photo, renames, report)
    assert photo.albums == ["Goa/ Trip", "Photos from 2011"]
    assert report.albums_retitled == 1


def test_an_empty_or_null_title_leaves_the_folder_name_in_place(tmp_path):
    root = build_takeout(tmp_path / "Takeout")
    report = EnrichReport()
    renames = album_renames(build_index(root, report), report)
    assert "No Name" not in renames
    photo = _photo("No Name")
    retitle_albums(photo, renames, report)
    assert photo.albums == ["No Name"]


def test_a_title_matching_its_own_folder_is_not_a_rename(tmp_path):
    root = build_takeout(tmp_path / "Takeout")
    report = EnrichReport()
    renames = album_renames(build_index(root, report), report)
    assert "Untitled" not in renames


def test_colliding_titles_are_reported_and_not_applied(tmp_path):
    """Untitled(1) has title 'Untitled', which is already another album."""
    root = build_takeout(tmp_path / "Takeout")
    report = EnrichReport()
    renames = album_renames(build_index(root, report), report)
    assert "Untitled(1)" not in renames
    assert ("Untitled(1)", "Untitled") in report.album_title_collisions


def test_retitling_is_idempotent():
    report = EnrichReport()
    renames = {"Goa Trip": "Goa/ Trip"}
    photo = _photo("Goa Trip")
    retitle_albums(photo, renames, report)
    retitle_albums(photo, renames, report)
    assert photo.albums == ["Goa/ Trip"]


def test_collision_resolution_does_not_depend_on_the_platform():
    """Replaces a first-draft test whose scenario was unrepresentable:
    index.albums only ever holds directories that HAD a metadata.json, so
    "ignore directories without one" could not be constructed.

    This one is real. sorted() on Path compares case-folded on Windows and raw
    on POSIX, so which of two albums claiming one title wins would differ
    between a contributor's machine and CI. M0 hit this exact bug.
    """
    # "Zulu" vs "beta" is chosen deliberately: raw ASCII puts 'Z' (0x5A)
    # before 'b' (0x62), casefolding puts "beta" before "zulu". A pair like
    # Alpha/beta orders the same either way and would NOT catch the bug.
    #
    # Verified: unfixed, this FAILS on Linux and macOS and PASSES on Windows,
    # because PureWindowsPath already compares case-insensitively. That is the
    # bug - the same code picking different albums on different machines - so
    # do not conclude the test is inert from a green run on Windows. CI covers
    # all three.
    index = SidecarIndex()
    index.albums[Path("/lib/Zulu")] = "Shared"
    index.albums[Path("/lib/beta")] = "Shared"
    report = EnrichReport()
    renames = album_renames(index, report)
    assert renames == {"beta": "Shared"}
    assert report.album_title_collisions == [("Zulu", "Shared")]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_takeout_albums.py -v`
Expected: FAIL — `ImportError: cannot import name 'album_renames' from 'rekindle.enrich.takeout'`

- [ ] **Step 3: Write the implementation**

```python
# src/rekindle/enrich/takeout.py  (append)

def album_renames(index: SidecarIndex, report: EnrichReport) -> dict[str, str]:
    """Folder name -> the album's real title, for titles that actually differ.

    A title identical to its folder is not a rename. A title that would
    collide with another album's name is REPORTED and skipped: merging two
    albums the user kept apart is not a decision this pass gets to make
    (spec section 11).
    """
    taken = {path.name for path in index.albums}
    renames: dict[str, str] = {}
    # Sort by a stable, platform-independent key. `sorted()` on Path objects
    # compares a case-folded form on Windows and a raw one on POSIX, so which
    # album wins a title collision would differ between a contributor's
    # machine and CI. M0 hit and fixed this exact bug in FolderSource.scan.
    def _order(item: tuple[Path, str]) -> tuple[str, str]:
        return str(item[0]).casefold(), str(item[0])

    for path, title in sorted(index.albums.items(), key=_order):
        folder = path.name
        if title == folder:
            continue
        if title in taken or title in renames.values():
            report.album_title_collisions.append((folder, title))
            continue
        renames[folder] = title
    return renames


def retitle_albums(photo: Photo, renames: dict[str, str], report: EnrichReport) -> None:
    """Apply renames wherever the photo lives.

    Not scoped to the album's own directory: 8 album folders in the reference
    export hold sidecars and zero media, so a directory-scoped rule would
    never fire for them.
    """
    updated = [renames.get(album, album) for album in photo.albums]
    if updated != photo.albums:
        # strict=True: the lists are the same length by construction, and
        # bare zip() is `B905` under this project's ruff config.
        report.albums_retitled += sum(
            1 for a, b in zip(photo.albums, updated, strict=True) if a != b
        )
        photo.albums = updated
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_takeout_albums.py -v && uv run ruff check .`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/rekindle/enrich/takeout.py tests/test_takeout_albums.py
git commit -m "feat: album retitling from metadata.json, with collision reporting"
```

---
### Task 10: Derivatives inherit enrichment

Google writes **no sidecar for `-edited` files or `.MP` halves** — 177 and 692
respectively in the reference export, zero matched by any key. Each is a separate
`file_hash` and therefore a separate `Photo` row, so without this the version a
user is most likely to put in a montage is the one with no date and no face tags.

M0 stores the `-edited` link as `Photo.edited_of`. It does **not** persist the
motion-photo link — `FolderSource` only counts `motion_pairs` — so the `.MP`
side is re-derived here from the naming rule M0 documents: the still is the
video's full *name* plus `.jpg` (`PXL_1.MP` pairs with `PXL_1.MP.jpg`), scoped to
one directory.

**Files:**
- Modify: `src/rekindle/enrich/takeout.py`
- Create: `tests/test_takeout_derivatives.py`

**Interfaces:**
- Consumes: `EnrichReport` (Task 6), `Photo`, `PhotoMeta`
- Produces: `propagate_to_derivatives(photos: list[Photo], report: EnrichReport) -> list[Photo]` — returns the photos it changed

- [ ] **Step 1: Write the failing test**

```python
# tests/test_takeout_derivatives.py
from datetime import UTC, datetime
from pathlib import Path

from rekindle.enrich.takeout import EnrichReport, propagate_to_derivatives
from rekindle.models import Gps, MediaType, Photo, PhotoMeta, TzSource

NOW = datetime(2020, 1, 1, tzinfo=UTC)
TAKEN = datetime(2020, 3, 11, 15, 30, tzinfo=UTC)


def _photo(digest: str, path: Path, meta: PhotoMeta, **kw) -> Photo:
    return Photo(
        file_hash=digest,
        paths=[path],
        media_type=kw.pop("media_type", MediaType.IMAGE),
        meta=meta,
        first_seen=NOW,
        last_seen=NOW,
        **kw,
    )


def _enriched() -> PhotoMeta:
    return PhotoMeta(
        taken_at_utc=TAKEN,
        taken_at_local=TAKEN,
        tz_source=TzSource.TAKEOUT,
        people=["Ada"],
        takeout_people=["Ada"],
        gps=Gps(lat=1.0, lon=2.0),
        description="a caption",
        favorite=True,
        archived=True,
    )


def test_an_edited_variant_inherits_from_its_original():
    lib = Path("/lib")
    original = _photo("orig", lib / "IMG.jpg", _enriched(), sidecar_match="exact")
    edited = _photo("ed", lib / "IMG-edited.jpg", PhotoMeta(), edited_of="orig")
    report = EnrichReport()
    changed = propagate_to_derivatives([original, edited], report)
    assert changed == [edited]
    assert edited.meta.taken_at_utc == TAKEN
    assert edited.meta.tz_source is TzSource.TAKEOUT
    assert edited.meta.people == ["Ada"]
    assert edited.meta.takeout_people == ["Ada"]
    assert edited.meta.gps.lat == 1.0
    assert edited.meta.description == "a caption"
    assert edited.meta.favorite is True
    assert edited.meta.archived is True
    assert edited.sidecar_match == "inherited"
    assert report.derivatives_enriched == 1


def test_a_motion_video_inherits_from_its_still():
    """M0 documents the rule: PXL_1.MP pairs with PXL_1.MP.jpg - the video's
    full NAME plus .jpg, not its stem."""
    lib = Path("/lib/Photos from 2011")
    still = _photo("still", lib / "PXL_1.MP.jpg", _enriched(), sidecar_match="exact")
    video = _photo("vid", lib / "PXL_1.MP", PhotoMeta(), media_type=MediaType.VIDEO)
    report = EnrichReport()
    propagate_to_derivatives([still, video], report)
    assert video.meta.taken_at_utc == TAKEN
    assert video.meta.taken_at_local == TAKEN
    assert video.meta.people == ["Ada"]
    assert video.sidecar_match == "inherited"
    assert report.derivatives_enriched == 1


def test_an_inherited_date_never_claims_an_exif_provenance():
    """A .MP video has no EXIF at all. Copying the still's `exif_offset`
    onto it made doctor report a provenance that cannot exist, on 655 files."""
    lib = Path("/lib/Photos from 2011")
    meta = _enriched()
    meta.tz_source = TzSource.EXIF_OFFSET
    still = _photo("still", lib / "PXL_1.MP.jpg", meta, sidecar_match="exact")
    video = _photo("vid", lib / "PXL_1.MP", PhotoMeta(), media_type=MediaType.VIDEO)
    propagate_to_derivatives([still, video], EnrichReport())
    assert video.meta.tz_source is TzSource.TAKEOUT
    assert still.meta.tz_source is TzSource.EXIF_OFFSET  # the donor is untouched
    assert video.meta.taken_at_utc == TAKEN


def test_motion_pairing_is_scoped_to_one_directory():
    still = _photo("still", Path("/lib/A/PXL_1.MP.jpg"), _enriched(), sidecar_match="exact")
    video = _photo("vid", Path("/lib/B/PXL_1.MP"), PhotoMeta(), media_type=MediaType.VIDEO)
    report = EnrichReport()
    assert propagate_to_derivatives([still, video], report) == []
    assert video.meta.taken_at_utc is None


def test_a_derivative_with_its_own_sidecar_is_left_alone():
    lib = Path("/lib")
    original = _photo("orig", lib / "IMG.jpg", _enriched(), sidecar_match="exact")
    own = PhotoMeta(taken_at_utc=datetime(2021, 1, 1, tzinfo=UTC), people=["Grace"])
    edited = _photo("ed", lib / "IMG-edited.jpg", own, edited_of="orig", sidecar_match="exact")
    report = EnrichReport()
    assert propagate_to_derivatives([original, edited], report) == []
    assert edited.meta.people == ["Grace"]


def test_an_unenriched_original_propagates_nothing():
    lib = Path("/lib")
    original = _photo("orig", lib / "IMG.jpg", PhotoMeta())
    edited = _photo("ed", lib / "IMG-edited.jpg", PhotoMeta(), edited_of="orig")
    report = EnrichReport()
    assert propagate_to_derivatives([original, edited], report) == []


def test_propagation_is_idempotent():
    lib = Path("/lib")
    original = _photo("orig", lib / "IMG.jpg", _enriched(), sidecar_match="exact")
    edited = _photo("ed", lib / "IMG-edited.jpg", PhotoMeta(), edited_of="orig")
    report = EnrichReport()
    propagate_to_derivatives([original, edited], report)
    assert propagate_to_derivatives([original, edited], EnrichReport()) == []
    assert edited.meta.people == ["Ada"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_takeout_derivatives.py -v`
Expected: FAIL — `ImportError: cannot import name 'propagate_to_derivatives' from 'rekindle.enrich.takeout'`

- [ ] **Step 3: Write the implementation**

```python
# src/rekindle/enrich/takeout.py  (append)

_ENRICHED_STATES = frozenset({"exact", "inherited"})


def propagate_to_derivatives(photos: list[Photo], report: EnrichReport) -> list[Photo]:
    """Copy enrichment onto photos Google gives no sidecar.

    Two links, both already established by M0's folder source:
      - `-edited` variants, persisted as Photo.edited_of
      - motion-photo halves, NOT persisted (FolderSource only counts them), so
        re-derived here from the naming rule: the still is the video's full
        NAME plus ".jpg" - PXL_1.MP pairs with PXL_1.MP.jpg - scoped to one
        directory, because a .MP in one album must never pair with a
        same-named still in another.
    """
    by_hash = {p.file_hash: p for p in photos}
    by_dir_name: dict[tuple[Path, str], Photo] = {}
    for photo in photos:
        for path in photo.paths:
            by_dir_name.setdefault((path.parent, path.name.casefold()), photo)

    changed: list[Photo] = []
    for photo in photos:
        if photo.sidecar_match in _ENRICHED_STATES:
            continue
        donor: Photo | None = None
        if photo.edited_of:
            donor = by_hash.get(photo.edited_of)
        if donor is None:
            for path in photo.paths:
                if path.suffix.casefold() != ".mp":
                    continue
                donor = by_dir_name.get((path.parent, f"{path.name}.jpg".casefold()))
                if donor is not None:
                    break
        if donor is None or donor.sidecar_match not in _ENRICHED_STATES:
            continue

        src = donor.meta
        dst = photo.meta
        dst.taken_at_utc = src.taken_at_utc
        dst.taken_at_local = src.taken_at_local
        # NOT src.tz_source. This file has no EXIF of its own, so copying
        # `exif_offset` onto 655 .MP videos makes doctor report a provenance
        # that cannot exist. The instant and the wall clock are copied intact -
        # only the claim about where they came from changes, and for this row
        # the answer is the Takeout pass, via its sibling.
        dst.tz_source = TzSource.TAKEOUT
        dst.exif_taken_at_utc = dst.exif_taken_at_utc or src.exif_taken_at_utc
        previous = set(dst.takeout_people)
        kept = [name for name in dst.people if name not in previous]
        dst.people = list(dict.fromkeys([*kept, *src.takeout_people]))
        dst.takeout_people = list(src.takeout_people)
        if dst.gps is None:
            dst.gps = src.gps
        if src.description and (
            dst.description is None or len(src.description) > len(dst.description)
        ):
            dst.description = src.description
        dst.favorite = dst.favorite or src.favorite
        dst.archived = dst.archived or src.archived
        dst.trashed = dst.trashed or src.trashed
        # A fourth state, distinct from "exact": this photo has no sidecar of
        # its own and doctor should not claim it does.
        photo.sidecar_match = "inherited"
        report.derivatives_enriched += 1
        changed.append(photo)
    return changed
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_takeout_derivatives.py -v && uv run ruff check .`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/rekindle/enrich/takeout.py tests/test_takeout_derivatives.py
git commit -m "feat: -edited and .MP derivatives inherit their original's enrichment"
```

---

### Task 11: The TakeoutEnricher, end to end

Wires Tasks 6-10 into one pass and adds the three guards §3 requires. The
`ambiguous` outcome must reach the stored row, so `doctor` can say a photo was
deliberately not enriched rather than merely unmatched.

**Files:**
- Modify: `src/rekindle/enrich/takeout.py`
- Create: `tests/test_takeout_enricher.py`

**Interfaces:**
- Consumes: everything from Tasks 6-10; `PhotoStore.iter_photos`, `update_many`, `get_meta`, `set_meta` (Task 2)
- Produces:
  - `TakeoutEnricher.name: str = "takeout"`
  - `TakeoutEnricher(tz_lookup: TzLookup | None = None)`
  - `TakeoutEnricher.enrich(root: Path, store: PhotoStore) -> EnrichReport`
  - `EmptyIndexError(RuntimeError)`
  - meta keys written: `enriched_at` (ISO 8601 UTC), `enrich_root` (str)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_takeout_enricher.py
import pytest

from rekindle.db import PhotoStore
from rekindle.enrich.takeout import EmptyIndexError, TakeoutEnricher
from rekindle.models import TzSource
from rekindle.sources.folder import FolderSource
from tests.fixtures.takeout import build_takeout


def _indexed(tmp_path):
    root = build_takeout(tmp_path / "Takeout")
    photos, _ = FolderSource().scan(root)
    store = PhotoStore(tmp_path / "data" / "rekindle.sqlite")
    store.upsert_many(photos)
    store.set_meta("index_root", str(root))
    return root, store


def test_enrich_populates_people_and_dates(tmp_path):
    root, store = _indexed(tmp_path)
    report = TakeoutEnricher().enrich(root, store)
    assert report.photos_enriched >= 5
    assert report.people_added >= 3
    by_name = {p.paths[0].name: p for p in store.iter_photos()}
    assert by_name["DSC00107.JPG"].meta.people == ["Grace"]
    assert by_name["DSC00107(1).JPG"].meta.people == ["Ada"]
    assert by_name["IMG_ALBUM.jpg"].meta.people == ["Ada"]
    store.close()


def test_the_accounting_identities_hold_end_to_end(tmp_path):
    root, store = _indexed(tmp_path)
    report = TakeoutEnricher().enrich(root, store)
    assert report.json_files_seen == report.files_accounted
    assert report.sidecars_seen == report.sidecars_accounted
    store.close()


def test_matched_equals_photos_enriched(tmp_path):
    """THE assertion that can fail.

    Both identities above partition sets that `account()` builds itself, so
    neither can see a sidecar parsed and then dropped somewhere downstream.
    This one ties the accounting to what actually reached the database: every
    sidecar counted as `matched` was written onto exactly one photo. Credit
    `matched` with `len(sidecars)` instead of the applied count and this fails
    immediately - by 3,358 on the reference export.
    """
    root, store = _indexed(tmp_path)
    report = TakeoutEnricher().enrich(root, store)
    assert report.matched == report.photos_enriched
    assert report.superseded >= 1  # AMBIG's second candidate
    store.close()


def test_a_distant_disagreeing_candidate_is_reported_as_an_override(tmp_path):
    """AMBIG.jpg has a same-directory sidecar and a disagreeing one in
    Goa Trip. Rule 1 wins by design - but silently, doctor would print
    `ambiguous: 0` and imply nothing was overridden. 942 photos on the
    reference export."""
    root, store = _indexed(tmp_path)
    report = TakeoutEnricher().enrich(root, store)
    assert report.directory_preference_broke_a_tie == 1
    store.close()


def test_derivatives_are_enriched_and_counted(tmp_path):
    root, store = _indexed(tmp_path)
    report = TakeoutEnricher().enrich(root, store)
    assert report.derivatives_enriched >= 2
    by_name = {p.paths[0].name: p for p in store.iter_photos()}
    assert by_name["IMG_EDIT-edited.jpg"].meta.people == ["Grace"]
    assert by_name["IMG_EDIT-edited.jpg"].sidecar_match == "inherited"
    assert by_name["PXL_1.MP"].meta.taken_at_utc is not None
    store.close()


def test_enrich_is_idempotent(tmp_path):
    root, store = _indexed(tmp_path)
    first = TakeoutEnricher().enrich(root, store)
    snapshot = {
        p.file_hash: (
            p.meta.taken_at_utc,
            p.meta.taken_at_local,
            p.meta.tz_source,
            tuple(p.meta.people),
            tuple(p.meta.takeout_people),
            p.meta.exif_taken_at_utc,
            p.metadata_conflict,
            p.sidecar_match,
            tuple(p.albums),
        )
        for p in store.iter_photos()
    }
    second = TakeoutEnricher().enrich(root, store)
    assert {
        p.file_hash: (
            p.meta.taken_at_utc,
            p.meta.taken_at_local,
            p.meta.tz_source,
            tuple(p.meta.people),
            tuple(p.meta.takeout_people),
            p.meta.exif_taken_at_utc,
            p.metadata_conflict,
            p.sidecar_match,
            tuple(p.albums),
        )
        for p in store.iter_photos()
    } == snapshot
    assert second.matched == first.matched
    assert second.orphaned == first.orphaned
    store.close()


def test_enrich_against_an_empty_index_refuses(tmp_path):
    root = build_takeout(tmp_path / "Takeout")
    store = PhotoStore(tmp_path / "data" / "rekindle.sqlite")
    with pytest.raises(EmptyIndexError):
        TakeoutEnricher().enrich(root, store)
    store.close()


def test_enrich_records_when_and_where_it_ran(tmp_path):
    root, store = _indexed(tmp_path)
    TakeoutEnricher().enrich(root, store)
    assert store.get_meta("enrich_root") == str(root)
    assert store.get_meta("enriched_at")
    store.close()


def test_trash_sidecars_are_excluded_by_both_passes(tmp_path):
    """Delete the EXCLUDED_DIRS check in build_index and this fails.

    The first draft asserted `IMG_GONE.jpg` was absent from the store, which
    FolderSource guarantees on its own - that test passed with enrich()'s body
    deleted and said nothing about the enricher. This one pins the enricher's
    own exclusion, and that the excluded sidecar is not miscounted as an
    orphan (which would inflate doctor's INCOMPLETE EXPORT warning on a
    complete library).
    """
    root, store = _indexed(tmp_path)
    report = TakeoutEnricher().enrich(root, store)
    assert report.excluded_dirs == 1
    assert "img_gone.jpg" not in {
        p.paths[0].name.casefold() for p in store.iter_photos()
    }
    store.close()


def test_google_date_replaces_the_mtime_fallback(tmp_path):
    root, store = _indexed(tmp_path)
    TakeoutEnricher().enrich(root, store)
    by_name = {p.paths[0].name: p for p in store.iter_photos()}
    flags = by_name["IMG_FLAGS.jpg"]
    assert flags.meta.tz_source is not TzSource.FILE_MTIME
    assert flags.meta.favorite is True
    assert flags.meta.archived is True
    assert flags.meta.description == "a real caption"
    store.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_takeout_enricher.py -v`
Expected: FAIL — `ImportError: cannot import name 'TakeoutEnricher' from 'rekindle.enrich.takeout'`

- [ ] **Step 3: Write the implementation**

```python
# src/rekindle/enrich/takeout.py  (append)

class EmptyIndexError(RuntimeError):
    """`enrich` ran before `index`. Enriching nothing is not a success."""


class TakeoutEnricher:
    """Second pass over an already-indexed library.

    Not a `Source`, deliberately - see the module docstring. It does honour
    the contract that protocol existed to enforce: every JSON file lands in
    exactly one counter, checked by EnrichReport's two identities.

    Opens no media file. All timing information it needs was stored by M0, so
    a re-run after extracting another archive part costs a JSON walk, not a
    re-hash of 45,900 files.
    """

    name = "takeout"

    def __init__(self, tz_lookup: TzLookup | None = None) -> None:
        self._tz_lookup = tz_lookup

    def enrich(self, root: Path, store: PhotoStore) -> EnrichReport:
        if store.count() == 0:
            raise EmptyIndexError(
                f"The index is empty. Run `rekindle index {root}` before `rekindle enrich`."
            )

        report = EnrichReport()
        index = build_index(root, report)
        clustered = clustered_timestamps(index)
        renames = album_renames(index, report)

        photos = list(store.iter_photos())
        claimed: dict[str, str] = {}
        applied: set[Path] = set()
        touched: dict[str, Photo] = {}

        for photo in photos:
            sidecar, match, tie = resolve(photo, index)
            for path in photo.paths:
                key = path.name.casefold()
                # "ambiguous" must not be downgraded by a later photo that
                # happens to resolve cleanly against the same key.
                if (
                    key in index.by_target
                    and match != "none"
                    and claimed.get(key) != "ambiguous"
                ):
                    claimed[key] = match
            if match == "ambiguous":
                photo.sidecar_match = "ambiguous"
                touched[photo.file_hash] = photo
                continue
            if sidecar is None:
                continue
            apply_sidecar(photo, sidecar, clustered, report, tz_lookup=self._tz_lookup)
            retitle_albums(photo, renames, report)
            applied.add(sidecar.path)
            report.photos_enriched += 1
            if tie:
                report.directory_preference_broke_a_tie += 1
            touched[photo.file_hash] = photo

        for photo in propagate_to_derivatives(photos, report):
            retitle_albums(photo, renames, report)
            touched[photo.file_hash] = photo

        account(index, claimed, applied, report)

        store.update_many(touched.values())
        store.set_meta("enrich_root", str(root))
        store.set_meta("enriched_at", datetime.now(UTC).isoformat())
        return report
```

Add `from rekindle.db import PhotoStore` to the module's imports.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_takeout_enricher.py -v && uv run pytest -v`
Expected: all pass, M0's 122 included

- [ ] **Step 5: Commit**

```bash
git add src/rekindle/enrich/takeout.py tests/test_takeout_enricher.py
git commit -m "feat: TakeoutEnricher end to end, with empty-index and provenance guards"
```

---

### Task 12: The `enrich` verb, and a `doctor` that reads the index

`doctor` currently runs a live `FolderSource().scan()` and never opens the
database, so after a successful enrich it would still print *"No person data
found in XMP sidecars"* and *"the Takeout parser … is not implemented yet"*
forever. The scan path stays the default so M0's 12 doctor tests and 6 CLI tests
keep passing; the index path is new.

**Files:**
- Modify: `src/rekindle/doctor.py`
- Modify: `src/rekindle/cli.py`
- Modify: `README.md`
- Test: `tests/test_doctor.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `EnrichReport`, `TakeoutEnricher`, `EmptyIndexError` (Task 11); `PhotoStore` (Task 2)
- Produces:
  - `doctor.IndexDiagnosis` frozen dataclass
  - `doctor.diagnose_index(store: PhotoStore) -> IndexDiagnosis`
  - `doctor.render_index(diagnosis: IndexDiagnosis, console: Console) -> None`
  - `doctor.render_enrich(report: EnrichReport, console: Console) -> None`
  - CLI `rekindle enrich <root> [--data-dir DIR]`
  - CLI `rekindle doctor <root> [--from-index] [--data-dir DIR]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_doctor.py  (append)
from rekindle.db import PhotoStore
from rekindle.doctor import diagnose_index
from rekindle.enrich.takeout import EnrichReport, TakeoutEnricher
from rekindle.sources.folder import FolderSource
from tests.fixtures.takeout import build_takeout


def test_diagnose_index_reports_people_after_enrichment(tmp_path):
    root = build_takeout(tmp_path / "Takeout")
    photos, _ = FolderSource().scan(root)
    store = PhotoStore(tmp_path / "data" / "rekindle.sqlite")
    store.upsert_many(photos)
    store.set_meta("index_root", str(root))

    before = diagnose_index(store)
    assert before.enriched_at is None
    assert any("has not been enriched" in w for w in before.warnings)

    TakeoutEnricher().enrich(root, store)
    after = diagnose_index(store)
    assert after.with_people > 0
    assert after.enriched_at is not None
    assert not any("has not been enriched" in w for w in after.warnings)
    assert not any("No person data found" in w for w in after.warnings)
    store.close()


def test_diagnose_index_warns_about_a_foreign_enrich_root(tmp_path):
    store = PhotoStore(tmp_path / "data" / "rekindle.sqlite")
    store.set_meta("index_root", str(tmp_path / "A"))
    store.set_meta("enrich_root", str(tmp_path / "B"))
    assert any("different folder" in w for w in diagnose_index(store).warnings)
    store.close()


def test_the_unparsed_sidecar_warning_now_points_at_enrich(tmp_path):
    from rekindle.doctor import diagnose
    from rekindle.models import SourceReport

    d = diagnose(SourceReport(files_seen=100, media_indexed=50, json_sidecars=50, with_people=1))
    hits = [w for w in d.warnings if "rekindle enrich" in w]
    assert len(hits) == 1
    assert "not implemented yet" not in hits[0]
```

```python
# tests/test_cli.py  (append)
from typer.testing import CliRunner

from rekindle.cli import app
from tests.fixtures.takeout import build_takeout

runner = CliRunner()


def test_enrich_before_index_tells_the_user_what_to_do(tmp_path):
    root = build_takeout(tmp_path / "Takeout")
    result = runner.invoke(app, ["enrich", str(root), "--data-dir", str(tmp_path / "data")])
    assert result.exit_code == 2
    assert "rekindle index" in result.stdout


def test_index_then_enrich_reports_people(tmp_path):
    root = build_takeout(tmp_path / "Takeout")
    data = str(tmp_path / "data")
    assert runner.invoke(app, ["index", str(root), "--data-dir", data]).exit_code == 0
    result = runner.invoke(app, ["enrich", str(root), "--data-dir", data])
    assert result.exit_code == 0
    assert "Sidecars seen" in result.stdout


def test_doctor_from_index_reads_the_database(tmp_path):
    root = build_takeout(tmp_path / "Takeout")
    data = str(tmp_path / "data")
    runner.invoke(app, ["index", str(root), "--data-dir", data])
    runner.invoke(app, ["enrich", str(root), "--data-dir", data])
    result = runner.invoke(app, ["doctor", str(root), "--from-index", "--data-dir", data])
    assert result.exit_code == 0
    assert "With people" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_doctor.py tests/test_cli.py -v`
Expected: FAIL — `ImportError: cannot import name 'diagnose_index' from 'rekindle.doctor'`

- [ ] **Step 3: Write the implementation**

In `src/rekindle/doctor.py`, replace the stale warning text and add the index path:

```python
    if report.json_sidecars and total and report.json_sidecars >= total * 0.25:
        warnings.append(
            f"{report.json_sidecars} Google JSON sidecars are present but not yet read. "
            "This looks like a Google Takeout export - run `rekindle enrich <folder>` "
            "after indexing to read the face tags, dates and album titles they carry."
        )
```

```python
# src/rekindle/doctor.py  (append; imports: PhotoStore, EnrichReport)


@dataclass(frozen=True)
class IndexDiagnosis:
    """What the STORED index holds. `diagnose()` above answers a different
    question - what a fresh scan of the filesystem can see - and after a
    successful enrich it would still report zero people forever.
    """

    photos: int = 0
    with_date: int = 0
    with_gps: int = 0
    with_people: int = 0
    enriched: int = 0
    inherited: int = 0
    ambiguous: int = 0
    archived: int = 0
    conflicts: int = 0
    enriched_at: str | None = None
    index_root: str | None = None
    enrich_root: str | None = None
    warnings: list[str] = field(default_factory=list)

    def pct(self, n: int) -> float:
        return round(100.0 * n / self.photos, 1) if self.photos else 0.0


def diagnose_index(store: PhotoStore) -> IndexDiagnosis:
    counts = dict.fromkeys(
        ("photos", "with_date", "with_gps", "with_people", "enriched",
         "inherited", "ambiguous", "archived", "conflicts"),
        0,
    )
    for photo in store.iter_photos():
        counts["photos"] += 1
        meta = photo.meta
        if meta.taken_at_utc and meta.tz_source is not TzSource.FILE_MTIME:
            counts["with_date"] += 1
        if meta.gps:
            counts["with_gps"] += 1
        if meta.people:
            counts["with_people"] += 1
        if photo.sidecar_match == "exact":
            counts["enriched"] += 1
        elif photo.sidecar_match == "inherited":
            counts["inherited"] += 1
        elif photo.sidecar_match == "ambiguous":
            counts["ambiguous"] += 1
        if meta.archived:
            counts["archived"] += 1
        if photo.metadata_conflict:
            counts["conflicts"] += 1

    enriched_at = store.get_meta("enriched_at")
    index_root = store.get_meta("index_root")
    enrich_root = store.get_meta("enrich_root")
    warnings: list[str] = []
    if counts["photos"] and enriched_at is None:
        warnings.append(
            "This index has not been enriched. If it came from Google Takeout, "
            "`rekindle enrich <folder>` adds face tags, capture dates and album titles."
        )
    if index_root and enrich_root and index_root != enrich_root:
        warnings.append(
            f"Enrichment ran against a different folder ({enrich_root}) than the one "
            f"indexed ({index_root}). Photo paths are absolute, so almost nothing "
            "will have matched. Re-run both against the same folder."
        )
    if counts["ambiguous"]:
        warnings.append(
            f"{counts['ambiguous']} photos had two or more sidecars IN THE SAME FOLDER "
            "that disagreed on capture time or people, so they were deliberately not "
            "enriched. This number is small by construction: a disagreeing sidecar in a "
            "DIFFERENT folder is overridden rather than refused, and `rekindle enrich` "
            "reports those separately. Do not read a low count here as no conflicts."
        )
    if counts["archived"]:
        warnings.append(
            f"{counts['archived']} photos are marked archived in Google Photos - the user "
            "deliberately hid them. They must not surface in a montage."
        )
    return IndexDiagnosis(
        **counts,
        enriched_at=enriched_at,
        index_root=index_root,
        enrich_root=enrich_root,
        warnings=warnings,
    )


def render_index(diagnosis: IndexDiagnosis, console: Console) -> None:
    table = Table(title="Index report", show_header=True)
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    table.add_column("Coverage", justify="right")
    d = diagnosis
    table.add_row("Photos in index", str(d.photos), "")
    table.add_row("With capture date", str(d.with_date), f"{d.pct(d.with_date)}%")
    table.add_row("With GPS", str(d.with_gps), f"{d.pct(d.with_gps)}%")
    table.add_row("With people", str(d.with_people), f"{d.pct(d.with_people)}%")
    table.add_row("Enriched from a sidecar", str(d.enriched), f"{d.pct(d.enriched)}%")
    table.add_row("Enriched via a derivative", str(d.inherited), f"{d.pct(d.inherited)}%")
    table.add_row("[yellow]Ambiguous (not enriched)[/yellow]", str(d.ambiguous), "")
    table.add_row("[yellow]Archived in Google Photos[/yellow]", str(d.archived), "")
    table.add_row("[yellow]EXIF/Google date conflicts[/yellow]", str(d.conflicts), "")
    console.print(table)
    console.print(f"\n[dim]Last enriched: {d.enriched_at or 'never'}[/dim]")
    for warning in d.warnings:
        console.print(f"\n[yellow]![/yellow] {warning}")


def render_enrich(report: EnrichReport, console: Console) -> None:
    table = Table(title="Enrichment report", show_header=True)
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    for label, value in (
        ("JSON files seen", report.json_files_seen),
        ("Sidecars seen", report.sidecars_seen),
        ("  matched", report.matched),
        ("  superseded (another candidate won)", report.superseded),
        ("  [yellow]orphaned (no row in the index)[/yellow]", report.orphaned),
        ("  [yellow]ambiguous (refused)[/yellow]", report.ambiguous),
        ("  overridden by directory preference", report.directory_preference_broke_a_tie),
        ("Album metadata", report.album_metadata),
        ("Other JSON", report.other_json),
        ("Excluded (trash/system)", report.excluded_dirs),
        ("[yellow]Unparseable[/yellow]", report.unparseable),
        ("Photos enriched", report.photos_enriched),
        ("Derivatives enriched", report.derivatives_enriched),
        ("People added", report.people_added),
        ("Dates corrected", report.dates_corrected),
        ("GPS added", report.gps_added),
        ("Descriptions added", report.descriptions_added),
        ("Favourites added", report.favourites_added),
        ("Albums retitled", report.albums_retitled),
        ("[yellow]EXIF/Google conflicts[/yellow]", report.conflicts),
        ("Clustered dates suppressed", report.clustered_dates_suppressed),
    ):
        table.add_row(label, str(value))
    console.print(table)

    # Report, never silently drop. If either identity fails, say so loudly -
    # a mismatch here is exactly the class of bug this whole pass was
    # redesigned to prevent.
    if report.json_files_seen != report.files_accounted:
        console.print(
            f"\n[red]ACCOUNTING BUG:[/red] {report.json_files_seen} JSON files seen but "
            f"{report.files_accounted} accounted for. Please file an issue."
        )
    if report.sidecars_seen != report.sidecars_accounted:
        console.print(
            f"\n[red]ACCOUNTING BUG:[/red] {report.sidecars_seen} sidecars seen but "
            f"{report.sidecars_accounted} accounted for. Please file an issue."
        )
    if report.album_title_collisions:
        console.print("\n[yellow]![/yellow] Album titles that would collide, left as folder names:")
        for folder, title in report.album_title_collisions:
            console.print(f"  {folder} -> {title}")
    if report.orphaned:
        console.print(
            f"\n[yellow]![/yellow] {report.orphaned} sidecars name a photo that has no row "
            "in the index. Some belong to archive parts you have not extracted; others are "
            "album copies of photos Takeout did not duplicate. Extract every part into "
            "the SAME folder and re-run if the number is large."
        )
        console.print(
            "  [dim]This deliberately does not equal doctor's own 'Orphan sidecars' row. "
            "That one compares sidecars against FILES ON DISK during a scan; this one "
            "compares them against ROWS IN THE INDEX, which excludes whatever the scan "
            "skipped as non-media or unreadable. On a 45,900-file export the two read "
            "2,444 and 2,565.[/dim]"
        )
    if report.directory_preference_broke_a_tie:
        console.print(
            f"\n[yellow]![/yellow] {report.directory_preference_broke_a_tie} photos had a "
            "sidecar in another album that DISAGREED with the one used; the same-directory "
            "copy was preferred. That is by design, but it is not the same as no conflict "
            "existing - 942 of these on a 45,900-file export, against 2 outright refusals."
        )
```

`doctor.py` needs `from rekindle.models import SourceReport, TzSource`,
`from rekindle.db import PhotoStore` and `from rekindle.enrich.takeout import EnrichReport`.

In `src/rekindle/cli.py`:

```python
@app.command()
def doctor(
    root: Annotated[Path, typer.Argument(help="Folder of photos to inspect.")],
    from_index: Annotated[
        bool, typer.Option("--from-index", help="Report on the stored index instead of rescanning.")
    ] = False,
    data_dir: DataDir = Path("./data"),
) -> None:
    """Report what metadata a library has. Writes nothing."""
    if from_index:
        # PhotoStore CREATES its database, so without this check `doctor
        # --from-index` on a machine that has never indexed silently reports a
        # healthy library of zero photos - and leaves a stray file behind.
        db_path = data_dir / "rekindle.sqlite"
        if not db_path.is_file():
            console.print(
                f"[red]No index at[/red] {db_path}. Run `rekindle index {root}` first."
            )
            raise typer.Exit(code=2)
        with PhotoStore(db_path) as store:
            diagnosis = diagnose_index(store)
        render_index(diagnosis, console)
        # `root` is otherwise unused on this path; warn rather than ignore it.
        if diagnosis.index_root and diagnosis.index_root != str(root):
            console.print(
                f"
[yellow]![/yellow] This index was built from {diagnosis.index_root}, "
                f"not {root}. The report above describes the former."
            )
        return
    _check_root(root)
    _photos, report = FolderSource().scan(root)
    render(diagnose(report), console)


@app.command()
def enrich(
    root: Annotated[Path, typer.Argument(help="Takeout folder whose sidecars to read.")],
    data_dir: DataDir = Path("./data"),
) -> None:
    """Read Google Takeout JSON sidecars into an index that already exists."""
    _check_root(root)
    with PhotoStore(data_dir / "rekindle.sqlite") as store:
        try:
            report = TakeoutEnricher().enrich(root, store)
        except EmptyIndexError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=2) from exc
        indexed_root = store.get_meta("index_root")
    render_enrich(report, console)
    if indexed_root and indexed_root != str(root):
        console.print(
            f"\n[yellow]![/yellow] The index was built from {indexed_root}, not {root}. "
            "Photo paths are absolute, so matches will be few."
        )
```

Have `index` record where it ran, immediately after `upsert_many`:

```python
        inserted, updated = store.upsert_many(photos)
        store.set_meta("index_root", str(root))
        total = store.count()
```

Add the imports `from rekindle.doctor import diagnose, diagnose_index, render, render_enrich, render_index`
and `from rekindle.enrich.takeout import EmptyIndexError, TakeoutEnricher`.

Add to `README.md`, beside the existing `index` and `doctor` lines:

```
rekindle enrich <folder>      # read Google Takeout JSON sidecars into the index
rekindle doctor <folder> --from-index   # report on the stored index
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest -v && uv run ruff check . && uv run ruff format --check .`
Expected: all pass, M0's 122 included

- [ ] **Step 5: Commit**

```bash
git add src/rekindle/doctor.py src/rekindle/cli.py tests/test_doctor.py tests/test_cli.py README.md
git commit -m "feat: rekindle enrich, and a doctor that reads the stored index"
```

---

### Task 13: Conformance against a real export, and docs

**v1's conformance test would not have caught v1's bug.** It asserted "matched
photos exceed a floor", and a floor of 80% passes while 984 sidecars vanish. This
one asserts the accounting identity per sidecar, which cannot pass while anything
is silently dropped.

It runs only when `REKINDLE_TAKEOUT_DIR` is set, so CI never needs anyone's
photos, and a contributor who has an export gets real verification.

**Files:**
- Create: `tests/test_takeout_conformance.py`
- Modify: `docs/known-limitations.md`
- Modify: `CONTRIBUTING.md`

**Interfaces:**
- Consumes: `build_index`, `resolve`, `account`, `EnrichReport`, `Sidecar` (Tasks 4-6)
- Produces: nothing importable

- [ ] **Step 1: Write the failing test**

```python
# tests/test_takeout_conformance.py
"""Structural conformance against a REAL Takeout export.

Skipped unless REKINDLE_TAKEOUT_DIR points at one, so CI never needs anyone's
personal library and no photo data is ever committed.

    REKINDLE_TAKEOUT_DIR="/path/to/Takeout/Google Photos" uv run pytest \
        tests/test_takeout_conformance.py -v

These assertions are the answer to "our fixtures certify our own fiction".
M0's motion-photo bug and v1's title-matching bug were BOTH invisible to a
green suite and both obvious within one run against real data.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rekindle.enrich.takeout import (
    EnrichReport,
    JsonKind,
    account,
    build_index,
    classify_json,
)
from rekindle.sidecars import sidecar_target

_ENV = "REKINDLE_TAKEOUT_DIR"

pytestmark = pytest.mark.skipif(
    not os.environ.get(_ENV), reason=f"set {_ENV} to a real Takeout export to run these"
)


@pytest.fixture(scope="module")
def export() -> Path:
    root = Path(os.environ[_ENV])
    if not root.is_dir():
        pytest.skip(f"{_ENV} is not a directory: {root}")
    return root


@pytest.fixture(scope="module")
def indexed(export: Path):
    report = EnrichReport()
    return build_index(export, report), report


def test_every_json_file_lands_in_exactly_one_bucket(indexed):
    """THE invariant. A floor on the match rate would pass while a thousand
    sidecars vanished; this cannot."""
    _index, report = indexed
    assert report.json_files_seen > 0
    assert report.json_files_seen == report.files_accounted


def test_every_indexed_sidecar_is_matched_superseded_orphaned_or_ambiguous(export, indexed):
    """A partition check, and weak on its own - `account()` cannot fail it,
    because it partitions the very set it is given. It is kept to pin the
    arithmetic; `test_matched_equals_photos_enriched_on_a_real_export` below is
    the one that can actually fail.

    Note it applies exactly ONE sidecar per claimed target, so the rest land in
    `superseded`. Passing `applied=set()` here would put all of them there,
    which is also a valid partition - that is precisely why this test is weak.
    """
    index, report = indexed
    media = {
        p.name.casefold()
        for p in export.rglob("*")
        if p.is_file() and p.suffix.casefold() != ".json"
    }
    claimed = {target: "exact" for target in index.by_target if target in media}
    applied = {index.by_target[target][0].path for target in claimed}
    account(index, claimed, applied, report)
    assert report.sidecars_seen == report.sidecars_accounted
    assert report.matched == len(claimed)


def test_the_counter_relocation_reduces_collisions(export, indexed):
    """The measurement that falsified v1, re-run on whatever export is here.

    NOT a higher raw match count - that is the wrong property, and asserting
    it fails on the reference export by 4. Relocation moves 993 targets; five
    of those name a photo in an archive part the user has not extracted and
    correctly become orphans, one gains, so raw matches fall by 4 while
    correctness rises.

    The property that actually holds is COLLISION REDUCTION: how many sidecars
    beyond the first are claiming one photo. On the reference export, 4,737
    under `title` against 3,772 under the filename - the 965 that `title`
    mis-pairs, which is the 963-mis-pair figure plus the two (1)/(2) groups
    with no plain sibling.
    """
    index, _ = indexed
    name_collisions = sum(len(v) - 1 for v in index.by_target.values())

    by_title: dict[str, int] = {}
    for sidecars in index.by_target.values():
        for sidecar in sidecars:
            key = sidecar.title.casefold()
            by_title[key] = by_title.get(key, 0) + 1
    title_collisions = sum(n - 1 for n in by_title.values())

    assert name_collisions < title_collisions


def test_disagreeing_candidate_groups_are_detected(export, indexed):
    """583 groups on the reference export.

    The first draft asserted `(not _disagree(a, s)) or _disagree(a, s)` here -
    a literal tautology, true for any input including an empty index. If
    `_disagree` ever stopped working this reads 0 and both the refusal and the
    override-reporting machinery are silently dead.
    """
    from rekindle.enrich.takeout import _disagree

    index, _ = indexed
    multi = [group for group in index.by_target.values() if len(group) > 1]
    if not multi:
        pytest.skip("this export has at most one sidecar per photo")
    disagreeing = sum(
        1 for group in multi if any(_disagree(group[0], s) for s in group[1:])
    )
    assert disagreeing > 0


def test_every_sidecar_parses_and_names_a_target(export):
    checked = 0
    for path in export.rglob("*.json"):
        kind, payload = classify_json(path)
        assert kind is not JsonKind.UNPARSEABLE, f"{path} did not parse"
        if kind is JsonKind.PHOTO:
            assert sidecar_target(path.name) != path.name
            assert isinstance(payload.get("title"), str)
            checked += 1
    assert checked > 0


def test_people_names_are_clean(indexed):
    """Measured on 24,248 real sidecars: 40 distinct names, none empty,
    whitespace-only, or colliding on case."""
    index, _ = indexed
    names = {n for sidecars in index.by_target.values() for s in sidecars for n in s.people}
    assert all(n == n.strip() and n for n in names)


@pytest.fixture(scope="module")
def enriched(export: Path, tmp_path_factory) -> EnrichReport:
    """Index and enrich the real export once, into pytest's own tmp dir.

    Never inside the export: this must not write a single byte next to
    anyone's photos.
    """
    store = PhotoStore(tmp_path_factory.mktemp("conformance") / "rekindle.sqlite")
    try:
        photos, _ = FolderSource().scan(export)
        store.upsert_many(photos)
        store.set_meta("index_root", str(export))
        return TakeoutEnricher().enrich(export, store)
    finally:
        store.close()


def test_every_disagreeing_group_is_overridden_or_refused_never_silent(indexed, enriched):
    """No disagreement may be resolved without being counted.

    On the reference export every disagreeing group has its candidates in
    DIFFERENT directories, so rule 3 fires twice and rule 1 overrides 942
    times. Either number may move; what must never happen is a disagreement
    resolved with neither counter incrementing.
    """
    from rekindle.enrich.takeout import _disagree

    index, _ = indexed
    disagreeing = sum(
        1
        for group in index.by_target.values()
        if len(group) > 1 and any(_disagree(group[0], s) for s in group[1:])
    )
    if not disagreeing:
        pytest.skip("this export has no disagreeing sidecar groups")
    assert enriched.directory_preference_broke_a_tie + enriched.ambiguous > 0


def test_matched_equals_photos_enriched_on_a_real_export(enriched):
    """The one assertion here that ties accounting to the database.

    Credit `matched` with `len(sidecars)` rather than the applied count and
    this fails by 3,358.
    """
    assert enriched.matched == enriched.photos_enriched
    assert enriched.sidecars_seen == enriched.sidecars_accounted
    assert enriched.json_files_seen == enriched.files_accounted
```

The `enriched` fixture indexes the whole export, so it is by far the slowest
thing in this module — the scan dominates; the enrich itself is ~6.6s on 45,900
files. It is module-scoped so the cost is paid once. Add to the module imports:
`from rekindle.db import PhotoStore`,
`from rekindle.enrich.takeout import EnrichReport, TakeoutEnricher`, and
`from rekindle.sources.folder import FolderSource`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_takeout_conformance.py -v`
Expected: all SKIPPED (`REKINDLE_TAKEOUT_DIR` unset) — that is the pass condition in CI.
With a real export: `REKINDLE_TAKEOUT_DIR="<path>" uv run pytest tests/test_takeout_conformance.py -v`
must pass; a failure here is a real defect, not a flaky test.

- [ ] **Step 3: Write the documentation**

In `docs/known-limitations.md`, under "Fixed during M0, recorded so they are not
reintroduced", append:

```markdown
- **Matching a Takeout sidecar on its `title` field.** `title` omits the
  disambiguating counter that Google puts in the sidecar's *filename*:
  `DSC00107.JPG.supplemental-metadata(1).json` says `title: "DSC00107.JPG"` but
  belongs to `DSC00107(1).JPG`. Measured across 24,248 real sidecars, the
  property is COLLISION REDUCTION, not a higher raw match count: `title` leaves
  4,737 sidecars beyond the first claiming one photo, the filename leaves 3,772.
  Raw matches actually fall by 4, because five `(N)` sidecars name a photo in an
  un-extracted archive part and correctly become orphans. The design that used
  `title` would have dropped 984 sidecars silently. Match on the filename via
  `rekindle.sidecars.sidecar_target`; `title` is a cross-check.
- **Passing `photoTakenTime` to `timestamps.resolve`.** It is a UTC instant;
  `resolve` expects naive wall-clock time and stamps a zone onto it, so the
  result is `local == utc` and a 21:00 IST photo is relabelled 15:30. Use
  `timestamps.from_takeout`, which derives the offset instead.
- **A `PhotoMeta` field not added to `merge_meta`.** That function builds a new
  record from an explicit field list, so a field left out is destroyed by the
  next `rekindle index`.
- **Crediting `matched` with every sidecar claiming a photo.** A photo with
  three candidates contributes one application and two discards; counting all
  three inflated `matched` by 3,358 on a real export and hid the discards. The
  accounting identity cannot catch this on its own - it partitions a set built
  by the same function - so `matched == photos_enriched` is asserted separately.
- **Reading a low `ambiguous` count as "no conflicts".** Every disagreeing
  sidecar group in the reference export has its candidates in different
  directories, so the same-directory preference resolves 942 of them and the
  refusal branch fires twice. Both numbers are reported for that reason.
```

Add a new section recording spec §10's ordering constraint, which this milestone
delivers half of:

```markdown
## Carried into M2: person data has arrived before the rules that govern it

`rekindle enrich` now writes 40 real people's names into the index. The person
exclusion list that is supposed to govern them (main spec §7.2) is M2 work and
does not exist.

Today's risk is nil: there is no memory engine, so nothing can surface a person
unprompted. **The exclusion list must land before anything auto-triggers.** This
is recorded so the ordering stays deliberate rather than accidental.
```

Correct the stale entry in the Deferred list:

```markdown
- **T4** ~~iter_photos() is implemented but not in the task's Produces list and
  has no consumer yet.~~ It was never implemented. Added in M1 with the Takeout
  enricher as its consumer.
```

Add to the "The one that matters most" section:

```markdown
XMP still has no real-world coverage. Takeout now does, via
`tests/test_takeout_conformance.py` — the same treatment is still owed to XMP.
```

In `CONTRIBUTING.md`, under the testing section:

```markdown
### Testing against a real Takeout export

Two of this project's three worst bugs were invisible to a green test suite and
obvious within one run against real data. If you have a Google Takeout export:

```bash
REKINDLE_TAKEOUT_DIR="/path/to/Takeout/Google Photos" uv run pytest \
    tests/test_takeout_conformance.py -v
```

Those tests skip without the variable, so CI never needs your photos. Never
commit an export, or any file from one.
```

- [ ] **Step 4: Run the full suite and lint**

Run: `uv run pytest -v && uv run ruff check . && uv run ruff format --check .`
Expected: all pass; the conformance module reports as skipped

- [ ] **Step 5: Commit**

```bash
git add tests/test_takeout_conformance.py docs/known-limitations.md CONTRIBUTING.md
git commit -m "test: conformance against a real export, asserting the accounting identity"
```

---

## Definition of done

- [ ] `uv run pytest` passes on Windows, macOS and Linux in CI, on 3.12 and 3.13
- [ ] The no-extras job passes too
- [ ] `uv run ruff check .` and `ruff format --check .` clean
- [ ] All 122 M0 tests still pass. Exactly one M0 assertion changed —
      `_sidecar_target("DSC_0880.JPG.supplemental-metadata(1).json")` now returns
      `"DSC_0880(1).JPG"` — and that assertion was v1's bug written down
- [ ] A database written by M0 opens, migrates, and keeps every row, every path
      and every person. Nobody is told to delete their index
- [ ] `rekindle enrich <folder>` refuses an empty index with an actionable message
- [ ] Running `enrich` twice produces a byte-identical set of rows
- [ ] `rekindle doctor <folder> --from-index` reports people after enrichment,
      and no longer claims the Takeout parser is unimplemented
- [ ] Both accounting identities hold on the fixture tree and on a real export
- [ ] No sidecar is matched by `title`. `title` is recorded as a cross-check only
- [ ] No photo carrying a real EXIF date has its wall clock moved by more than a
      minute, except where `metadata_conflict` is set. (The first draft said "no
      photo's `taken_at_local` moves when only its timezone was in question",
      which is both unmeasurable as written and false for 135 photos whose EXIF
      genuinely disagrees with Google.)
- [ ] Two disagreeing sidecars for one photo cause a refusal or a REPORTED
      directory-preference override, never a silent coin flip
- [ ] `report.matched == report.photos_enriched`, on the fixture tree and on a
      real export. Every sidecar not applied is counted as superseded, orphaned
      or ambiguous — none is parsed and dropped
- [ ] `rekindle doctor --from-index` with no index reports that, and creates
      no database
- [ ] `-edited` and `.MP` derivatives carry their original's people and date
- [ ] No new runtime dependency; no network access anywhere in the suite
- [ ] No personal data committed. `Takeout/` stays git-ignored in every case

## A note on task boundaries

Tasks 1, 3 and 12 modify code M0 already ships and are the ones to review
hardest. Task 1 is the only one that can destroy user data; it is deliberately
first and deliberately small. Task 3 changes a function with existing passing
tests, and the review question there is whether the one changed assertion is
right — everything else in this plan depends on it being right.

Tasks 6 and 8 are the largest. If you prefer smaller units:

- **6a** `EnrichReport` and the two identities
- **6b** `build_index` and its bucketing
- **6c** `resolve` and `account`
- **8a** the time rules (date, cluster suppression, conflict)
- **8b** the field rules (people, gps, description, flags)

`build_takeout()` from Task 5 is the behavioural spec for Tasks 6, 9, 11 and 12 —
changing that tree breaks four tasks at once.
