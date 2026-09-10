# rekindle M0 — Folder Source Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Index a directory of photos into a local SQLite database, reading whatever metadata the files carry, and report honestly on what was found.

**Architecture:** A `FolderSource` walks a directory, identifies each file by a hash of its bytes, sniffs its real type from magic bytes, and reads metadata through a preference ladder (XMP sidecar → EXIF/IPTC → filesystem). Results are normalised into `Photo` records and upserted into SQLite. A `doctor` command reports coverage without modifying anything. No models, no network, no API keys.

**Tech Stack:** Python 3.12+, `uv`, Pillow (EXIF + image open), `defusedxml` (XMP parsing), `typer` + `rich` (CLI), stdlib `sqlite3`, `pytest`, `ruff`.

**Spec:** `docs/superpowers/specs/2026-09-10-rekindle-design.md`

## Global Constraints

- **Python floor is 3.12.** Set `requires-python = ">=3.12"`. Do not use 3.13-only syntax.
- **No network calls anywhere in M0.** No model downloads, no API clients. A test that needs the network is a broken test.
- **No GPU.** Nothing in M0 touches torch.
- **Cross-platform: Windows, macOS, Linux.** Use `pathlib` exclusively. Never hardcode `/` or `\`. Never assume a drive letter. `pathlib` does *not* solve Windows `MAX_PATH` — detect and report long paths, don't crash.
- **Never trust a file extension.** Media type comes from magic bytes.
- **Identity is `file_hash`** — BLAKE2b of *file* bytes, never decoded pixels.
- **Never delete index rows** when a file disappears. Update `last_seen`.
- **Batch database writes.** One transaction per index run, not per photo.
- **Report, never silently drop.** Every skipped file is counted with a reason.
- **All timestamps stored UTC**, ISO 8601 strings in SQLite.
- **MIT licensed, public repo.** No copyleft dependencies in the required set. `pillow-heif` is an optional extra because its wheels are GPL-2.0.

---

### Task 1: Project scaffolding and CI

**Files:**
- Create: `pyproject.toml`
- Create: `src/rekindle/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_smoke.py`
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: nothing
- Produces: installable package `rekindle`, `rekindle.__version__: str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_smoke.py
import rekindle


def test_package_exposes_version():
    assert isinstance(rekindle.__version__, str)
    assert rekindle.__version__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rekindle'`

- [ ] **Step 3: Write pyproject.toml**

```toml
[project]
name = "rekindle"
version = "0.1.0"
description = "Turn your photo library into memories"
readme = "README.md"
license = "MIT"
requires-python = ">=3.12"
dependencies = [
    "pillow>=11.1.0",
    "defusedxml>=0.7.1",
    "typer>=0.15.0",
    "rich>=13.9.0",
]

[project.optional-dependencies]
heic = ["pillow-heif>=1.7.0"]

[project.scripts]
rekindle = "rekindle.cli:app"

[dependency-groups]
dev = ["pytest>=8.3.0", "pytest-cov>=6.0.0", "ruff>=0.8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/rekindle"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "PTH"]
```

- [ ] **Step 4: Write the package init**

```python
# src/rekindle/__init__.py
"""rekindle - turn your photo library into memories."""

__version__ = "0.1.0"
```

Also create an empty `tests/__init__.py`.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv sync && uv run pytest tests/test_smoke.py -v`
Expected: PASS

- [ ] **Step 6: Write the CI workflow**

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
      - run: uv sync --all-extras
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run pytest -v
```

- [ ] **Step 7: Verify lint and format pass**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: no errors. If format fails, run `uv run ruff format .` and re-check.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock src/rekindle/__init__.py tests/__init__.py tests/test_smoke.py .github/workflows/ci.yml
git commit -m "feat: project scaffolding with cross-platform CI"
```

---

### Task 2: Core models

**Files:**
- Create: `src/rekindle/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `MediaType(StrEnum)`: `IMAGE`, `VIDEO`, `UNKNOWN`
  - `TzSource(StrEnum)`: `EXIF_OFFSET`, `EXIF_NAIVE`, `GPS`, `FILE_MTIME`, `NONE`
  - `Gps(lat: float, lon: float, alt: float | None)` — frozen dataclass
  - `FaceRegion(name: str, x: float, y: float, w: float, h: float)` — frozen, normalised centre + size
  - `PhotoMeta` — dataclass, every field optional, `empty()` classmethod
  - `Photo` — dataclass with `file_hash`, `paths`, `media_type`, `meta`, `albums`, `edited_of`, `first_seen`, `last_seen`
  - `SourceReport` — dataclass with counters and `skip(reason)` method

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
from datetime import UTC, datetime

from rekindle.models import (
    FaceRegion,
    Gps,
    MediaType,
    Photo,
    PhotoMeta,
    SourceReport,
    TzSource,
)


def test_photo_meta_empty_has_no_data():
    m = PhotoMeta.empty()
    assert m.taken_at_utc is None
    assert m.gps is None
    assert m.people == []
    assert m.face_regions == []
    assert m.tz_source is TzSource.NONE


def test_photo_meta_lists_are_independent_between_instances():
    a = PhotoMeta.empty()
    b = PhotoMeta.empty()
    a.people.append("Alice")
    assert b.people == []


def test_photo_defaults():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    p = Photo(file_hash="abc", paths=[], media_type=MediaType.IMAGE,
              meta=PhotoMeta.empty(), first_seen=now, last_seen=now)
    assert p.albums == []
    assert p.edited_of is None


def test_face_region_is_normalised():
    r = FaceRegion(name="Alice", x=0.5, y=0.4, w=0.2, h=0.25)
    assert 0.0 <= r.x <= 1.0
    assert r.name == "Alice"


def test_gps_is_frozen():
    import dataclasses
    import pytest

    g = Gps(lat=15.3, lon=74.0, alt=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        g.lat = 0.0


def test_source_report_counts_skips_by_reason():
    r = SourceReport()
    r.skip("not_media")
    r.skip("not_media")
    r.skip("unreadable")
    assert r.skipped == {"not_media": 2, "unreadable": 1}
    assert r.total_skipped == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rekindle.models'`

- [ ] **Step 3: Write the implementation**

```python
# src/rekindle/models.py
"""Core data records. Source-agnostic: every source normalises into these."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path


class MediaType(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    UNKNOWN = "unknown"


class TzSource(StrEnum):
    """How local time was determined. Recorded so doctor can report it."""

    EXIF_OFFSET = "exif_offset"
    EXIF_NAIVE = "exif_naive"
    GPS = "gps"
    FILE_MTIME = "file_mtime"
    NONE = "none"


@dataclass(frozen=True)
class Gps:
    lat: float
    lon: float
    alt: float | None = None


@dataclass(frozen=True)
class FaceRegion:
    """Normalised [0,1] centre and size, per the MWG regions convention."""

    name: str
    x: float
    y: float
    w: float
    h: float


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

    @classmethod
    def empty(cls) -> PhotoMeta:
        return cls()


@dataclass
class Photo:
    file_hash: str
    paths: list[Path]
    media_type: MediaType
    meta: PhotoMeta
    first_seen: datetime
    last_seen: datetime
    albums: list[str] = field(default_factory=list)
    edited_of: str | None = None


@dataclass
class SourceReport:
    """What a source found. Feeds `rekindle doctor`.

    Sources must count everything they skip. Silent drops are a bug.
    """

    files_seen: int = 0
    media_indexed: int = 0
    duplicates_merged: int = 0
    edited_linked: int = 0
    sidecars_ignored: int = 0
    with_date: int = 0
    with_gps: int = 0
    with_people: int = 0
    with_xmp: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    unreadable: list[tuple[Path, str]] = field(default_factory=list)
    long_paths: list[Path] = field(default_factory=list)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    @property
    def total_skipped(self) -> int:
        return sum(self.skipped.values())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/rekindle/models.py tests/test_models.py
git commit -m "feat: core Photo, PhotoMeta and SourceReport records"
```

---

### Task 3: File identity and media type sniffing

**Files:**
- Create: `src/rekindle/identity.py`
- Test: `tests/test_identity.py`

**Interfaces:**
- Consumes: `MediaType` from `rekindle.models`
- Produces:
  - `file_hash(path: Path) -> str` — 32-char hex, BLAKE2b digest_size=16
  - `sniff(path: Path) -> tuple[MediaType, str]` — returns `(media_type, format_name)`; format is e.g. `"jpeg"`, `"heic"`, `"mp4"`, `"unknown"`
  - `is_long_path(path: Path) -> bool`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_identity.py
from pathlib import Path

from rekindle.identity import file_hash, is_long_path, sniff
from rekindle.models import MediaType

JPEG = bytes.fromhex("ffd8ffe000104a46494600010100000100010000") + b"\x00" * 64
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
GIF = b"GIF89a" + b"\x00" * 64
WEBP = b"RIFF" + (100).to_bytes(4, "little") + b"WEBP" + b"\x00" * 64
HEIC = b"\x00\x00\x00\x18ftypheic" + b"\x00" * 64
AVIF = b"\x00\x00\x00\x18ftypavif" + b"\x00" * 64
MP4 = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64
MOV = b"\x00\x00\x00\x14ftypqt  " + b"\x00" * 64
JUNK = b"not a media file at all" + b"\x00" * 64


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


def test_hash_is_stable_and_content_addressed(tmp_path):
    a = _write(tmp_path, "a.jpg", JPEG)
    b = _write(tmp_path, "b.jpg", JPEG)
    c = _write(tmp_path, "c.jpg", PNG)
    assert file_hash(a) == file_hash(b)
    assert file_hash(a) != file_hash(c)
    assert len(file_hash(a)) == 32


def test_hash_reads_in_chunks_without_loading_whole_file(tmp_path):
    big = _write(tmp_path, "big.bin", b"x" * (5 * 1024 * 1024))
    assert len(file_hash(big)) == 32


def test_sniff_identifies_formats_from_magic_bytes(tmp_path):
    cases = [
        ("a.jpg", JPEG, MediaType.IMAGE, "jpeg"),
        ("a.png", PNG, MediaType.IMAGE, "png"),
        ("a.gif", GIF, MediaType.IMAGE, "gif"),
        ("a.webp", WEBP, MediaType.IMAGE, "webp"),
        ("a.heic", HEIC, MediaType.IMAGE, "heic"),
        ("a.avif", AVIF, MediaType.IMAGE, "avif"),
        ("a.mp4", MP4, MediaType.VIDEO, "mp4"),
        ("a.mov", MOV, MediaType.VIDEO, "mov"),
    ]
    for name, data, want_type, want_fmt in cases:
        p = _write(tmp_path, name, data)
        assert sniff(p) == (want_type, want_fmt), name


def test_sniff_ignores_the_extension_entirely(tmp_path):
    """Takeout ships .heic files that are actually JPEG."""
    liar = _write(tmp_path, "actually_jpeg.heic", JPEG)
    assert sniff(liar) == (MediaType.IMAGE, "jpeg")


def test_sniff_returns_unknown_for_junk(tmp_path):
    p = _write(tmp_path, "notes.txt", JUNK)
    assert sniff(p) == (MediaType.UNKNOWN, "unknown")


def test_sniff_handles_empty_file(tmp_path):
    p = _write(tmp_path, "empty.jpg", b"")
    assert sniff(p) == (MediaType.UNKNOWN, "unknown")


def test_long_path_detection():
    assert is_long_path(Path("C:/" + "a" * 300)) is True
    assert is_long_path(Path("short.jpg")) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_identity.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rekindle.identity'`

- [ ] **Step 3: Write the implementation**

```python
# src/rekindle/identity.py
"""File identity and type detection.

Identity is a hash of FILE bytes, never decoded pixels: decoding is expensive
and its output changes between Pillow/libjpeg versions, which would silently
invalidate the whole resume cache.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from rekindle.models import MediaType

_CHUNK = 1 << 20
_MAX_PATH = 260

# ISO-BMFF brands. Anything ftyp-based is decided by its brand, not extension.
_IMAGE_BRANDS = {"heic", "heix", "hevc", "hevx", "mif1", "msf1", "heif"}
_AVIF_BRANDS = {"avif", "avis"}
_VIDEO_BRANDS = {"isom", "iso2", "mp41", "mp42", "dash", "m4v ", "3gp4", "3gp5"}
_QT_BRANDS = {"qt  "}


def file_hash(path: Path, *, chunk_size: int = _CHUNK) -> str:
    """BLAKE2b-128 of the file's bytes, streamed."""
    h = hashlib.blake2b(digest_size=16)
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def sniff(path: Path) -> tuple[MediaType, str]:
    """Identify media type from magic bytes. The extension is never consulted."""
    try:
        with path.open("rb") as f:
            head = f.read(32)
    except OSError:
        return (MediaType.UNKNOWN, "unknown")

    if len(head) < 12:
        return (MediaType.UNKNOWN, "unknown")

    if head.startswith(b"\xff\xd8\xff"):
        return (MediaType.IMAGE, "jpeg")
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return (MediaType.IMAGE, "png")
    if head.startswith((b"GIF87a", b"GIF89a")):
        return (MediaType.IMAGE, "gif")
    if head.startswith(b"BM"):
        return (MediaType.IMAGE, "bmp")
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return (MediaType.IMAGE, "webp")
    if head.startswith((b"II*\x00", b"MM\x00*")):
        return (MediaType.IMAGE, "tiff")

    if head[4:8] == b"ftyp":
        brand = head[8:12].decode("ascii", errors="replace").lower()
        if brand in _AVIF_BRANDS:
            return (MediaType.IMAGE, "avif")
        if brand in _IMAGE_BRANDS:
            return (MediaType.IMAGE, "heic")
        if brand in _QT_BRANDS:
            return (MediaType.VIDEO, "mov")
        if brand in _VIDEO_BRANDS:
            return (MediaType.VIDEO, "mp4")
        return (MediaType.VIDEO, "mp4")

    return (MediaType.UNKNOWN, "unknown")


def is_long_path(path: Path) -> bool:
    """Windows MAX_PATH check. pathlib does not solve this; we report it."""
    return len(str(path)) >= _MAX_PATH
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_identity.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/rekindle/identity.py tests/test_identity.py
git commit -m "feat: content hashing and magic-byte media type detection"
```

---

### Task 4: SQLite store

**Files:**
- Create: `src/rekindle/db.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: `Photo`, `PhotoMeta`, `Gps`, `FaceRegion`, `MediaType`, `TzSource`
- Produces:
  - `PhotoStore(db_path: Path)` — context manager
  - `.upsert_many(photos: Iterable[Photo]) -> tuple[int, int]` returns `(inserted, updated)`
  - `.get(file_hash: str) -> Photo | None`
  - `.count() -> int`
  - `.all_hashes() -> set[str]`
  - `SCHEMA_VERSION: int`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db.py
from datetime import UTC, datetime
from pathlib import Path

from rekindle.db import SCHEMA_VERSION, PhotoStore
from rekindle.models import FaceRegion, Gps, MediaType, Photo, PhotoMeta, TzSource

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
T1 = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def _photo(h="abc123", **kw) -> Photo:
    meta = kw.pop("meta", PhotoMeta.empty())
    return Photo(
        file_hash=h,
        paths=kw.pop("paths", [Path("/photos/a.jpg")]),
        media_type=kw.pop("media_type", MediaType.IMAGE),
        meta=meta,
        first_seen=kw.pop("first_seen", T0),
        last_seen=kw.pop("last_seen", T0),
        albums=kw.pop("albums", []),
        edited_of=kw.pop("edited_of", None),
    )


def test_creates_schema_and_records_version(tmp_path):
    with PhotoStore(tmp_path / "db.sqlite") as s:
        assert s.schema_version() == SCHEMA_VERSION
        assert s.count() == 0


def test_insert_then_read_back_roundtrips_metadata(tmp_path):
    meta = PhotoMeta(
        taken_at_utc=T0,
        taken_at_local=T0,
        tz_source=TzSource.EXIF_OFFSET,
        gps=Gps(lat=15.3, lon=74.08, alt=12.5),
        people=["Alice", "Bob"],
        face_regions=[FaceRegion(name="Alice", x=0.5, y=0.4, w=0.2, h=0.25)],
        keywords=["beach"],
        description="sunset",
        favorite=True,
        camera_make="Canon",
        camera_model="EOS R",
        width=4000,
        height=3000,
    )
    with PhotoStore(tmp_path / "db.sqlite") as s:
        inserted, updated = s.upsert_many([_photo(meta=meta, albums=["Goa Trip"])])
        assert (inserted, updated) == (1, 0)
        got = s.get("abc123")

    assert got is not None
    assert got.meta.gps == Gps(lat=15.3, lon=74.08, alt=12.5)
    assert got.meta.people == ["Alice", "Bob"]
    assert got.meta.face_regions[0].name == "Alice"
    assert got.meta.favorite is True
    assert got.meta.tz_source is TzSource.EXIF_OFFSET
    assert got.albums == ["Goa Trip"]
    assert got.meta.taken_at_utc == T0


def test_upsert_same_hash_updates_last_seen_and_unions_albums(tmp_path):
    with PhotoStore(tmp_path / "db.sqlite") as s:
        s.upsert_many([_photo(albums=["Year 2014"], paths=[Path("/a/x.jpg")])])
        inserted, updated = s.upsert_many(
            [_photo(albums=["Goa Trip"], paths=[Path("/b/x.jpg")],
                    first_seen=T1, last_seen=T1)]
        )
        assert (inserted, updated) == (0, 1)
        got = s.get("abc123")

    assert sorted(got.albums) == ["Goa Trip", "Year 2014"]
    assert sorted(str(p) for p in got.paths) == sorted(["/a/x.jpg", "/b/x.jpg"])
    assert got.first_seen == T0  # earliest wins
    assert got.last_seen == T1   # latest wins


def test_get_returns_none_for_unknown_hash(tmp_path):
    with PhotoStore(tmp_path / "db.sqlite") as s:
        assert s.get("nope") is None


def test_all_hashes_supports_incremental_scan(tmp_path):
    with PhotoStore(tmp_path / "db.sqlite") as s:
        s.upsert_many([_photo("h1"), _photo("h2")])
        assert s.all_hashes() == {"h1", "h2"}


def test_store_persists_across_sessions(tmp_path):
    db = tmp_path / "db.sqlite"
    with PhotoStore(db) as s:
        s.upsert_many([_photo()])
    with PhotoStore(db) as s:
        assert s.count() == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_db.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rekindle.db'`

- [ ] **Step 3: Write the implementation**

```python
# src/rekindle/db.py
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

from rekindle.models import FaceRegion, Gps, MediaType, Photo, PhotoMeta, TzSource

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
    height        INTEGER
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

    def __enter__(self) -> PhotoStore:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
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
        paths = list(dict.fromkeys([*[str(x) for x in old.paths], *[str(x) for x in new.paths]]))
        albums = list(dict.fromkeys([*old.albums, *new.albums]))
        return Photo(
            file_hash=old.file_hash,
            paths=[Path(x) for x in paths],
            media_type=new.media_type,
            meta=new.meta if new.meta.taken_at_utc else old.meta,
            first_seen=min(old.first_seen, new.first_seen),
            last_seen=max(old.last_seen, new.last_seen),
            albums=albums,
            edited_of=new.edited_of or old.edited_of,
        )

    @staticmethod
    def _insert(cur: sqlite3.Cursor, p: Photo) -> None:
        m = p.meta
        cur.execute(
            """INSERT OR REPLACE INTO photos VALUES
               (:file_hash,:media_type,:paths,:albums,:edited_of,:first_seen,:last_seen,
                :taken_at_utc,:taken_at_local,:tz_source,:gps_lat,:gps_lon,:gps_alt,
                :people,:face_regions,:keywords,:description,:favorite,
                :camera_make,:camera_model,:width,:height)""",
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
                    [{"name": r.name, "x": r.x, "y": r.y, "w": r.w, "h": r.h}
                     for r in m.face_regions]
                ),
                "keywords": json.dumps(m.keywords),
                "description": m.description,
                "favorite": int(m.favorite),
                "camera_make": m.camera_make,
                "camera_model": m.camera_model,
                "width": m.width,
                "height": m.height,
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
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_db.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/rekindle/db.py tests/test_db.py
git commit -m "feat: SQLite photo store with merge-on-upsert"
```

---

### Task 5: Synthetic fixture generator

**Files:**
- Create: `tests/fixtures/__init__.py`
- Create: `tests/fixtures/gen.py`
- Test: `tests/test_fixtures.py`

**Interfaces:**
- Consumes: Pillow
- Produces:
  - `make_jpeg(path, *, size=(64,48), color=(120,80,60), taken=None, offset=None, gps=None, make=None, model=None) -> Path`
  - `make_xmp_sidecar(image_path, *, people=(), regions=(), description=None, keywords=()) -> Path`
  - `build_library(root: Path) -> Path` — creates a realistic mixed tree and returns the root

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fixtures.py
from PIL import Image

from tests.fixtures.gen import build_library, make_jpeg, make_xmp_sidecar


def test_make_jpeg_is_a_real_readable_image(tmp_path):
    p = make_jpeg(tmp_path / "a.jpg", size=(32, 24))
    with Image.open(p) as im:
        assert im.size == (32, 24)
        assert im.format == "JPEG"


def test_make_jpeg_embeds_requested_exif(tmp_path):
    from datetime import datetime

    p = make_jpeg(
        tmp_path / "b.jpg",
        taken=datetime(2014, 3, 21, 17, 45, 0),
        offset="+05:30",
        make="Canon",
        model="EOS R",
    )
    with Image.open(p) as im:
        exif = im.getexif()
        ifd = exif.get_ifd(0x8769)
    assert ifd[0x9003] == "2014:03:21 17:45:00"
    assert ifd[0x9011] == "+05:30"
    assert exif[0x010F] == "Canon"


def test_make_xmp_sidecar_writes_parseable_xml(tmp_path):
    img = make_jpeg(tmp_path / "c.jpg")
    side = make_xmp_sidecar(img, people=["Alice"], description="a day out")
    text = side.read_text(encoding="utf-8")
    assert side.name == "c.jpg.xmp"
    assert "Alice" in text
    assert "a day out" in text


def test_build_library_produces_the_expected_shape(tmp_path):
    root = build_library(tmp_path / "lib")
    names = {p.name for p in root.rglob("*") if p.is_file()}

    # year folder and album folder contain the SAME file (duplicate by content)
    assert (root / "Photos from 2014").is_dir()
    assert (root / "Goa Trip").is_dir()
    # an edited variant exists alongside its original
    assert any(n.endswith("-edited.jpg") for n in names)
    # a JSON sidecar exists and must be ignored as media
    assert any(n.endswith(".json") for n in names)
    # an XMP sidecar exists
    assert any(n.endswith(".xmp") for n in names)
    # a file whose extension lies about its content
    assert "actually_jpeg.heic" in names
    # a non-media file
    assert "notes.txt" in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_fixtures.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tests.fixtures.gen'`

- [ ] **Step 3: Write the implementation**

```python
# tests/fixtures/gen.py
"""Synthetic photo libraries for tests.

CI has no photos, so every fixture is generated. The shapes here deliberately
reproduce the messes real libraries contain: duplicates across folders, edited
variants, sidecars, and files whose extensions lie.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from PIL import Image

_XMP_TEMPLATE = """<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:mwg-rs="http://www.metadataworkinggroup.com/schemas/regions/"
    xmlns:stArea="http://ns.adobe.com/xmp/sType/Area#"
    xmlns:stDim="http://ns.adobe.com/xmp/sType/Dimensions#"
    xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/">
{description}
{keywords}
{persons}
{regions}
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>
"""


def make_jpeg(
    path: Path,
    *,
    size: tuple[int, int] = (64, 48),
    color: tuple[int, int, int] = (120, 80, 60),
    taken: datetime | None = None,
    offset: str | None = None,
    gps: tuple[float, float] | None = None,
    make: str | None = None,
    model: str | None = None,
) -> Path:
    """Write a small real JPEG, optionally with EXIF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", size, color)
    exif = im.getexif()
    if make:
        exif[0x010F] = make
    if model:
        exif[0x0110] = model
    if taken or offset:
        sub = {}
        if taken:
            sub[0x9003] = taken.strftime("%Y:%m:%d %H:%M:%S")
        if offset:
            sub[0x9011] = offset
        exif[0x8769] = sub
    if gps:
        lat, lon = gps
        exif[0x8825] = {
            1: "N" if lat >= 0 else "S",
            2: _deg_to_dms(abs(lat)),
            3: "E" if lon >= 0 else "W",
            4: _deg_to_dms(abs(lon)),
        }
    im.save(path, "JPEG", exif=exif)
    return path


def _deg_to_dms(deg: float) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    d = int(deg)
    m_full = (deg - d) * 60
    m = int(m_full)
    s = round((m_full - m) * 60 * 100)
    return ((d, 1), (m, 1), (s, 100))


def make_xmp_sidecar(
    image_path: Path,
    *,
    people: tuple[str, ...] | list[str] = (),
    regions: tuple[tuple[str, float, float, float, float], ...] = (),
    description: str | None = None,
    keywords: tuple[str, ...] | list[str] = (),
) -> Path:
    """Write `<image>.xmp` next to the image."""
    desc = ""
    if description:
        desc = (
            "    <dc:description><rdf:Alt><rdf:li xml:lang='x-default'>"
            f"{description}</rdf:li></rdf:Alt></dc:description>"
        )
    kw = ""
    if keywords:
        items = "".join(f"<rdf:li>{k}</rdf:li>" for k in keywords)
        kw = f"    <dc:subject><rdf:Bag>{items}</rdf:Bag></dc:subject>"
    persons = ""
    if people:
        items = "".join(f"<rdf:li>{p}</rdf:li>" for p in people)
        persons = (
            f"    <Iptc4xmpExt:PersonInImage><rdf:Bag>{items}"
            "</rdf:Bag></Iptc4xmpExt:PersonInImage>"
        )
    regs = ""
    if regions:
        items = "".join(
            "<rdf:li rdf:parseType='Resource'>"
            f"<mwg-rs:Name>{n}</mwg-rs:Name><mwg-rs:Type>Face</mwg-rs:Type>"
            f"<mwg-rs:Area stArea:x='{x}' stArea:y='{y}' "
            f"stArea:w='{w}' stArea:h='{h}' stArea:unit='normalized'/>"
            "</rdf:li>"
            for n, x, y, w, h in regions
        )
        regs = (
            "    <mwg-rs:Regions rdf:parseType='Resource'>"
            f"<mwg-rs:RegionList><rdf:Bag>{items}</rdf:Bag></mwg-rs:RegionList>"
            "</mwg-rs:Regions>"
        )
    out = image_path.with_name(image_path.name + ".xmp")
    out.write_text(
        _XMP_TEMPLATE.format(description=desc, keywords=kw, persons=persons, regions=regs),
        encoding="utf-8",
    )
    return out


def build_library(root: Path) -> Path:
    """A small library containing every mess we intend to handle."""
    root.mkdir(parents=True, exist_ok=True)
    year = root / "Photos from 2014"
    album = root / "Goa Trip"

    # Same bytes in both a year folder and an album folder.
    beach = make_jpeg(
        year / "IMG_0001.jpg",
        color=(10, 120, 200),
        taken=datetime(2014, 3, 21, 17, 45),
        offset="+05:30",
        gps=(15.2993, 74.1240),
        make="Canon",
        model="EOS R",
    )
    (album / "IMG_0001.jpg").parent.mkdir(parents=True, exist_ok=True)
    (album / "IMG_0001.jpg").write_bytes(beach.read_bytes())

    # An edited variant of the same original.
    make_jpeg(year / "IMG_0001-edited.jpg", color=(20, 130, 210),
              taken=datetime(2014, 3, 21, 17, 45))

    # A Google-style JSON sidecar that must be ignored as media.
    (year / "IMG_0001.jpg.json").write_text(
        json.dumps({"photoTakenTime": {"timestamp": "1395423900"}}), encoding="utf-8"
    )

    # A photo with an XMP sidecar carrying people and face regions.
    portrait = make_jpeg(album / "IMG_0002.jpg", color=(200, 160, 140),
                         taken=datetime(2014, 3, 22, 9, 0))
    make_xmp_sidecar(
        portrait,
        people=["Alice", "Bob"],
        regions=(("Alice", 0.4, 0.35, 0.18, 0.24),),
        description="morning on the beach",
        keywords=("beach", "holiday"),
    )

    # A photo with no metadata at all.
    make_jpeg(year / "IMG_0003.jpg", color=(90, 90, 90))

    # An extension that lies: JPEG bytes named .heic
    make_jpeg(year / "actually_jpeg.heic", color=(30, 30, 30))

    # Not media.
    (root / "notes.txt").write_text("just some notes", encoding="utf-8")

    return root
```

Also create an empty `tests/fixtures/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_fixtures.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/ tests/test_fixtures.py
git commit -m "test: synthetic photo library generator"
```

---

### Task 6: EXIF reader

**Files:**
- Create: `src/rekindle/meta/__init__.py`
- Create: `src/rekindle/meta/exif.py`
- Test: `tests/test_exif.py`

**Interfaces:**
- Consumes: `Gps` from `rekindle.models`
- Produces:
  - `ExifData` — frozen dataclass: `taken_naive: datetime | None`, `offset: str | None`, `gps: Gps | None`, `camera_make: str | None`, `camera_model: str | None`, `width: int | None`, `height: int | None`
  - `read_exif(path: Path) -> ExifData` — never raises; returns empty `ExifData` on any failure

- [ ] **Step 1: Write the failing test**

```python
# tests/test_exif.py
from datetime import datetime

from rekindle.meta.exif import ExifData, read_exif
from tests.fixtures.gen import make_jpeg


def test_reads_datetime_offset_and_camera(tmp_path):
    p = make_jpeg(tmp_path / "a.jpg", taken=datetime(2014, 3, 21, 17, 45),
                  offset="+05:30", make="Canon", model="EOS R")
    d = read_exif(p)
    assert d.taken_naive == datetime(2014, 3, 21, 17, 45)
    assert d.offset == "+05:30"
    assert d.camera_make == "Canon"
    assert d.camera_model == "EOS R"


def test_reads_dimensions(tmp_path):
    p = make_jpeg(tmp_path / "b.jpg", size=(120, 90))
    assert read_exif(p).width == 120
    assert read_exif(p).height == 90


def test_reads_gps_and_converts_dms_to_decimal(tmp_path):
    p = make_jpeg(tmp_path / "c.jpg", gps=(15.2993, 74.1240))
    d = read_exif(p)
    assert d.gps is not None
    assert abs(d.gps.lat - 15.2993) < 0.001
    assert abs(d.gps.lon - 74.1240) < 0.001


def test_southern_and_western_hemispheres_are_negative(tmp_path):
    p = make_jpeg(tmp_path / "d.jpg", gps=(-33.8688, -151.2093))
    d = read_exif(p)
    assert d.gps.lat < 0
    assert d.gps.lon < 0


def test_photo_without_exif_returns_empty_not_error(tmp_path):
    p = make_jpeg(tmp_path / "e.jpg")
    d = read_exif(p)
    assert d.taken_naive is None
    assert d.gps is None


def test_unreadable_file_returns_empty_not_error(tmp_path):
    p = tmp_path / "broken.jpg"
    p.write_bytes(b"\xff\xd8\xff" + b"garbage")
    assert read_exif(p) == ExifData()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_exif.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rekindle.meta'`

- [ ] **Step 3: Write the implementation**

```python
# src/rekindle/meta/exif.py
"""EXIF extraction via Pillow. Never raises - a broken file yields empty data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

from rekindle.models import Gps

_EXIF_IFD = 0x8769
_GPS_IFD = 0x8825
_MAKE, _MODEL = 0x010F, 0x0110
_DATETIME_ORIGINAL, _OFFSET_ORIGINAL = 0x9003, 0x9011
_DATETIME_DIGITIZED, _OFFSET_DIGITIZED = 0x9004, 0x9012


@dataclass(frozen=True)
class ExifData:
    taken_naive: datetime | None = None
    offset: str | None = None
    gps: Gps | None = None
    camera_make: str | None = None
    camera_model: str | None = None
    width: int | None = None
    height: int | None = None


def _rational(value: object) -> float:
    if isinstance(value, tuple) and len(value) == 2:
        num, den = value
        return float(num) / float(den) if den else 0.0
    return float(value)  # type: ignore[arg-type]


def _dms_to_decimal(dms: object, ref: object) -> float | None:
    try:
        d, m, s = (_rational(x) for x in dms)  # type: ignore[misc]
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    dec = d + m / 60.0 + s / 3600.0
    if str(ref).upper().strip() in {"S", "W"}:
        dec = -dec
    return dec


def _parse_dt(raw: object) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        return datetime.strptime(raw.strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def read_exif(path: Path) -> ExifData:
    try:
        with Image.open(path) as im:
            width, height = im.size
            exif = im.getexif()
            sub = exif.get_ifd(_EXIF_IFD)
            gps_ifd = exif.get_ifd(_GPS_IFD)
    except Exception:
        return ExifData()

    taken = _parse_dt(sub.get(_DATETIME_ORIGINAL)) or _parse_dt(sub.get(_DATETIME_DIGITIZED))
    offset = sub.get(_OFFSET_ORIGINAL) or sub.get(_OFFSET_DIGITIZED)
    offset = offset.strip() if isinstance(offset, str) else None

    gps = None
    if gps_ifd:
        lat = _dms_to_decimal(gps_ifd.get(2), gps_ifd.get(1))
        lon = _dms_to_decimal(gps_ifd.get(4), gps_ifd.get(3))
        if lat is not None and lon is not None and not (lat == 0.0 and lon == 0.0):
            alt = None
            if gps_ifd.get(6) is not None:
                try:
                    alt = _rational(gps_ifd.get(6))
                except (TypeError, ValueError, ZeroDivisionError):
                    alt = None
            gps = Gps(lat=lat, lon=lon, alt=alt)

    def _clean(v: object) -> str | None:
        return v.strip() or None if isinstance(v, str) else None

    return ExifData(
        taken_naive=taken,
        offset=offset,
        gps=gps,
        camera_make=_clean(exif.get(_MAKE)),
        camera_model=_clean(exif.get(_MODEL)),
        width=width,
        height=height,
    )
```

Also create an empty `src/rekindle/meta/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_exif.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/rekindle/meta/ tests/test_exif.py
git commit -m "feat: EXIF reader with GPS DMS conversion"
```

---

### Task 7: XMP sidecar reader

**Files:**
- Create: `src/rekindle/meta/xmp.py`
- Test: `tests/test_xmp.py`

**Interfaces:**
- Consumes: `FaceRegion` from `rekindle.models`
- Produces:
  - `XmpData` — frozen dataclass: `people: tuple[str, ...]`, `face_regions: tuple[FaceRegion, ...]`, `keywords: tuple[str, ...]`, `description: str | None`
  - `find_sidecar(image_path: Path) -> Path | None` — checks `<name>.xmp` then `<stem>.xmp`
  - `read_xmp(path: Path) -> XmpData` — never raises

- [ ] **Step 1: Write the failing test**

```python
# tests/test_xmp.py
from rekindle.meta.xmp import XmpData, find_sidecar, read_xmp
from tests.fixtures.gen import make_jpeg, make_xmp_sidecar


def test_finds_sidecar_named_image_dot_xmp(tmp_path):
    img = make_jpeg(tmp_path / "a.jpg")
    side = make_xmp_sidecar(img, people=["Alice"])
    assert find_sidecar(img) == side


def test_finds_sidecar_named_stem_dot_xmp(tmp_path):
    img = make_jpeg(tmp_path / "b.jpg")
    alt = tmp_path / "b.xmp"
    alt.write_text("<x/>", encoding="utf-8")
    assert find_sidecar(img) == alt


def test_returns_none_when_no_sidecar(tmp_path):
    img = make_jpeg(tmp_path / "c.jpg")
    assert find_sidecar(img) is None


def test_reads_people_description_and_keywords(tmp_path):
    img = make_jpeg(tmp_path / "d.jpg")
    side = make_xmp_sidecar(img, people=["Alice", "Bob"],
                            description="morning on the beach",
                            keywords=("beach", "holiday"))
    d = read_xmp(side)
    assert sorted(d.people) == ["Alice", "Bob"]
    assert d.description == "morning on the beach"
    assert sorted(d.keywords) == ["beach", "holiday"]


def test_reads_face_regions_with_normalised_coordinates(tmp_path):
    img = make_jpeg(tmp_path / "e.jpg")
    side = make_xmp_sidecar(img, regions=(("Alice", 0.4, 0.35, 0.18, 0.24),))
    d = read_xmp(side)
    assert len(d.face_regions) == 1
    r = d.face_regions[0]
    assert r.name == "Alice"
    assert abs(r.x - 0.4) < 1e-6
    assert abs(r.h - 0.24) < 1e-6


def test_region_names_also_count_as_people(tmp_path):
    """A face region names a person even without PersonInImage."""
    img = make_jpeg(tmp_path / "f.jpg")
    side = make_xmp_sidecar(img, regions=(("Carol", 0.5, 0.5, 0.1, 0.1),))
    assert "Carol" in read_xmp(side).people


def test_malformed_xmp_returns_empty_not_error(tmp_path):
    bad = tmp_path / "bad.xmp"
    bad.write_text("<not-closed>", encoding="utf-8")
    assert read_xmp(bad) == XmpData()


def test_missing_file_returns_empty_not_error(tmp_path):
    assert read_xmp(tmp_path / "nope.xmp") == XmpData()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_xmp.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rekindle.meta.xmp'`

- [ ] **Step 3: Write the implementation**

```python
# src/rekindle/meta/xmp.py
"""XMP sidecar reading.

Sidecars are the richest metadata available to a plain folder: Lightroom,
digiKam and osxphotos all write person names AND face regions here, which
EXIF cannot carry.

Parsed with defusedxml - XMP is arbitrary XML from an untrusted source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from defusedxml import ElementTree as DefusedET

from rekindle.models import FaceRegion

_NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "mwg-rs": "http://www.metadataworkinggroup.com/schemas/regions/",
    "stArea": "http://ns.adobe.com/xmp/sType/Area#",
    "Iptc4xmpExt": "http://iptc.org/std/Iptc4xmpExt/2008-02-29/",
}


@dataclass(frozen=True)
class XmpData:
    people: tuple[str, ...] = ()
    face_regions: tuple[FaceRegion, ...] = ()
    keywords: tuple[str, ...] = ()
    description: str | None = None


def find_sidecar(image_path: Path) -> Path | None:
    """`photo.jpg.xmp` is the common convention; `photo.xmp` is also used."""
    for candidate in (
        image_path.with_name(image_path.name + ".xmp"),
        image_path.with_suffix(".xmp"),
    ):
        if candidate.is_file():
            return candidate
    return None


def _bag_items(root, path: str) -> list[str]:
    out: list[str] = []
    for container in root.iterfind(f".//{path}", _NS):
        for li in container.iterfind(".//rdf:li", _NS):
            text = (li.text or "").strip()
            if text:
                out.append(text)
    return out


def _q(ns: str, name: str) -> str:
    return f"{{{_NS[ns]}}}{name}"


def read_xmp(path: Path) -> XmpData:
    try:
        root = DefusedET.parse(path).getroot()
    except Exception:
        return XmpData()

    people = _bag_items(root, "Iptc4xmpExt:PersonInImage")
    keywords = _bag_items(root, "dc:subject")

    description = None
    for node in root.iterfind(".//dc:description//rdf:li", _NS):
        text = (node.text or "").strip()
        if text:
            description = text
            break

    regions: list[FaceRegion] = []
    for li in root.iterfind(".//mwg-rs:RegionList//rdf:li", _NS):
        type_node = li.find("mwg-rs:Type", _NS)
        if type_node is not None and (type_node.text or "").strip().lower() != "face":
            continue
        name_node = li.find("mwg-rs:Name", _NS)
        area = li.find("mwg-rs:Area", _NS)
        if name_node is None or area is None:
            continue
        name = (name_node.text or "").strip()
        try:
            region = FaceRegion(
                name=name,
                x=float(area.get(_q("stArea", "x"), "0")),
                y=float(area.get(_q("stArea", "y"), "0")),
                w=float(area.get(_q("stArea", "w"), "0")),
                h=float(area.get(_q("stArea", "h"), "0")),
            )
        except (TypeError, ValueError):
            continue
        if name:
            regions.append(region)

    named = list(dict.fromkeys([*people, *(r.name for r in regions)]))
    return XmpData(
        people=tuple(named),
        face_regions=tuple(regions),
        keywords=tuple(keywords),
        description=description,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_xmp.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/rekindle/meta/xmp.py tests/test_xmp.py
git commit -m "feat: XMP sidecar reader with MWG face regions"
```

---

### Task 8: Timestamp resolution

**Files:**
- Create: `src/rekindle/meta/timestamps.py`
- Test: `tests/test_timestamps.py`

**Interfaces:**
- Consumes: `TzSource`, `Gps`
- Produces:
  - `TzLookup = Callable[[Gps], str | None]` type alias
  - `resolve(exif_taken, exif_offset, gps, file_mtime, tz_lookup=None) -> tuple[datetime|None, datetime|None, TzSource]` returning `(utc, local, source)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_timestamps.py
from datetime import UTC, datetime

from rekindle.meta.timestamps import resolve
from rekindle.models import Gps, TzSource

NAIVE = datetime(2014, 3, 21, 17, 45)
MTIME = datetime(2020, 1, 2, 3, 4, tzinfo=UTC)


def test_exif_offset_is_preferred_and_converts_to_utc():
    utc, local, src = resolve(NAIVE, "+05:30", None, MTIME)
    assert src is TzSource.EXIF_OFFSET
    assert local.replace(tzinfo=None) == NAIVE
    assert utc == datetime(2014, 3, 21, 12, 15, tzinfo=UTC)


def test_negative_offset_converts_correctly():
    utc, _, src = resolve(NAIVE, "-08:00", None, MTIME)
    assert src is TzSource.EXIF_OFFSET
    assert utc == datetime(2014, 3, 22, 1, 45, tzinfo=UTC)


def test_gps_lookup_used_when_offset_missing():
    utc, local, src = resolve(NAIVE, None, Gps(15.3, 74.1), MTIME,
                              tz_lookup=lambda _g: "Asia/Kolkata")
    assert src is TzSource.GPS
    assert local.replace(tzinfo=None) == NAIVE
    assert utc == datetime(2014, 3, 21, 12, 15, tzinfo=UTC)


def test_naive_exif_treated_as_utc_when_nothing_else_known():
    utc, local, src = resolve(NAIVE, None, None, MTIME)
    assert src is TzSource.EXIF_NAIVE
    assert utc == NAIVE.replace(tzinfo=UTC)


def test_gps_lookup_failure_falls_back_to_naive():
    _, _, src = resolve(NAIVE, None, Gps(0.1, 0.1), MTIME, tz_lookup=lambda _g: None)
    assert src is TzSource.EXIF_NAIVE


def test_file_mtime_used_when_no_exif_date():
    utc, local, src = resolve(None, None, None, MTIME)
    assert src is TzSource.FILE_MTIME
    assert utc == MTIME


def test_no_information_at_all_yields_none():
    utc, local, src = resolve(None, None, None, None)
    assert (utc, local, src) == (None, None, TzSource.NONE)


def test_malformed_offset_falls_back_to_naive():
    _, _, src = resolve(NAIVE, "banana", None, MTIME)
    assert src is TzSource.EXIF_NAIVE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_timestamps.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rekindle.meta.timestamps'`

- [ ] **Step 3: Write the implementation**

```python
# src/rekindle/meta/timestamps.py
"""Local time resolution.

Preference order, recorded in TzSource so doctor can report it:
    EXIF OffsetTimeOriginal -> GPS timezone -> naive EXIF (assumed UTC)
    -> file mtime -> nothing.

GPS lookup is injected rather than imported: timezonefinder is a large
optional dependency and must never be required.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from rekindle.models import Gps, TzSource

TzLookup = Callable[[Gps], str | None]

_OFFSET_RE = re.compile(r"^([+-])(\d{2}):?(\d{2})$")


def _parse_offset(raw: str | None) -> timezone | None:
    if not raw:
        return None
    m = _OFFSET_RE.match(raw.strip())
    if not m:
        return None
    sign, hours, minutes = m.groups()
    delta = timedelta(hours=int(hours), minutes=int(minutes))
    return timezone(-delta if sign == "-" else delta)


def resolve(
    exif_taken: datetime | None,
    exif_offset: str | None,
    gps: Gps | None,
    file_mtime: datetime | None,
    tz_lookup: TzLookup | None = None,
) -> tuple[datetime | None, datetime | None, TzSource]:
    """Return (utc, local, source)."""
    if exif_taken is not None:
        tz = _parse_offset(exif_offset)
        if tz is not None:
            local = exif_taken.replace(tzinfo=tz)
            return local.astimezone(UTC), local, TzSource.EXIF_OFFSET

        if gps is not None and tz_lookup is not None:
            name = tz_lookup(gps)
            if name:
                try:
                    zone = ZoneInfo(name)
                except (ZoneInfoNotFoundError, ValueError):
                    zone = None
                if zone is not None:
                    local = exif_taken.replace(tzinfo=zone)
                    return local.astimezone(UTC), local, TzSource.GPS

        assumed = exif_taken.replace(tzinfo=UTC)
        return assumed, assumed, TzSource.EXIF_NAIVE

    if file_mtime is not None:
        stamped = file_mtime if file_mtime.tzinfo else file_mtime.replace(tzinfo=UTC)
        return stamped, stamped, TzSource.FILE_MTIME

    return None, None, TzSource.NONE
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_timestamps.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/rekindle/meta/timestamps.py tests/test_timestamps.py
git commit -m "feat: timestamp resolution ladder with injected GPS lookup"
```

---

### Task 9: Folder source

**Files:**
- Create: `src/rekindle/sources/__init__.py`
- Create: `src/rekindle/sources/base.py`
- Create: `src/rekindle/sources/folder.py`
- Test: `tests/test_folder_source.py`

**Interfaces:**
- Consumes: `file_hash`, `sniff`, `is_long_path`, `read_exif`, `read_xmp`, `find_sidecar`, `resolve`, all models
- Produces:
  - `Source` protocol with `name: str` and `scan(root: Path) -> tuple[list[Photo], SourceReport]`
  - `FolderSource(tz_lookup: TzLookup | None = None)`
  - `EDITED_SUFFIXES: tuple[str, ...]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_folder_source.py
from rekindle.models import MediaType, TzSource
from rekindle.sources.folder import FolderSource
from tests.fixtures.gen import build_library, make_jpeg


def _scan(root):
    return FolderSource().scan(root)


def test_indexes_images_and_skips_non_media(tmp_path):
    root = build_library(tmp_path / "lib")
    photos, report = _scan(root)
    assert report.media_indexed == len(photos)
    assert report.skipped.get("not_media", 0) >= 1  # notes.txt
    names = {p.name for photo in photos for p in photo.paths}
    assert "notes.txt" not in names


def test_json_sidecars_are_ignored_but_counted(tmp_path):
    root = build_library(tmp_path / "lib")
    _, report = _scan(root)
    assert report.sidecars_ignored >= 1


def test_duplicate_across_folders_becomes_one_photo_with_both_albums(tmp_path):
    root = build_library(tmp_path / "lib")
    photos, report = _scan(root)
    dupes = [p for p in photos if len(p.paths) > 1]
    assert len(dupes) == 1
    assert sorted(dupes[0].albums) == ["Goa Trip", "Photos from 2014"]
    assert report.duplicates_merged == 1


def test_edited_variant_links_to_its_original(tmp_path):
    root = build_library(tmp_path / "lib")
    photos, report = _scan(root)
    edited = [p for p in photos if any("-edited" in x.name for x in p.paths)]
    assert len(edited) == 1
    assert edited[0].edited_of is not None
    assert report.edited_linked == 1
    originals = {p.file_hash for p in photos}
    assert edited[0].edited_of in originals


def test_extension_lying_file_is_typed_by_content(tmp_path):
    root = build_library(tmp_path / "lib")
    photos, _ = _scan(root)
    liar = [p for p in photos if any(x.name == "actually_jpeg.heic" for x in p.paths)]
    assert liar and liar[0].media_type is MediaType.IMAGE


def test_xmp_metadata_is_read_into_the_photo(tmp_path):
    root = build_library(tmp_path / "lib")
    photos, report = _scan(root)
    with_people = [p for p in photos if p.meta.people]
    assert len(with_people) == 1
    p = with_people[0]
    assert sorted(p.meta.people) == ["Alice", "Bob"]
    assert p.meta.face_regions[0].name == "Alice"
    assert p.meta.description == "morning on the beach"
    assert report.with_people == 1
    assert report.with_xmp == 1


def test_exif_date_and_gps_are_read(tmp_path):
    root = build_library(tmp_path / "lib")
    photos, report = _scan(root)
    dated = [p for p in photos if p.meta.taken_at_utc]
    assert dated
    geo = [p for p in photos if p.meta.gps]
    assert len(geo) == 1
    assert report.with_gps == 1
    assert any(p.meta.tz_source is TzSource.EXIF_OFFSET for p in photos)


def test_album_is_the_containing_folder_name(tmp_path):
    root = tmp_path / "lib"
    make_jpeg(root / "Holiday" / "x.jpg")
    photos, _ = _scan(root)
    assert photos[0].albums == ["Holiday"]


def test_photo_directly_in_root_has_no_album(tmp_path):
    root = tmp_path / "lib"
    make_jpeg(root / "x.jpg")
    photos, _ = _scan(root)
    assert photos[0].albums == []


def test_unreadable_file_is_reported_not_fatal(tmp_path):
    root = tmp_path / "lib"
    root.mkdir(parents=True)
    make_jpeg(root / "good.jpg")
    (root / "truncated.jpg").write_bytes(b"\xff\xd8\xff" + b"\x00" * 20)
    photos, report = _scan(root)
    assert any("good.jpg" in x.name for p in photos for x in p.paths)
    assert report.files_seen >= 2


def test_rescan_is_idempotent(tmp_path):
    root = build_library(tmp_path / "lib")
    first, _ = _scan(root)
    second, _ = _scan(root)
    assert {p.file_hash for p in first} == {p.file_hash for p in second}


def test_empty_directory_yields_empty_report(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    photos, report = _scan(root)
    assert photos == []
    assert report.files_seen == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_folder_source.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rekindle.sources'`

- [ ] **Step 3: Write the base protocol**

```python
# src/rekindle/sources/base.py
"""The Source protocol. Every library format normalises into Photo records."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from rekindle.models import Photo, SourceReport


class Source(Protocol):
    name: str

    def scan(self, root: Path) -> tuple[list[Photo], SourceReport]:
        """Walk a library and return normalised photos plus an honest report.

        Implementations must count every skipped file with a reason. Silently
        dropping input is a bug.
        """
        ...
```

Also create an empty `src/rekindle/sources/__init__.py`.

- [ ] **Step 4: Write the folder source**

```python
# src/rekindle/sources/folder.py
"""Reads a plain directory tree. The only source in v1."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from rekindle.identity import file_hash, is_long_path, sniff
from rekindle.meta.exif import read_exif
from rekindle.meta.timestamps import TzLookup, resolve
from rekindle.meta.xmp import find_sidecar, read_xmp
from rekindle.models import MediaType, Photo, PhotoMeta, SourceReport, TzSource

# Google localises the edited suffix, so this is a list, not a single string.
EDITED_SUFFIXES: tuple[str, ...] = (
    "-edited",
    "-bearbeitet",
    "-modifié",
    "-editado",
    "-modificato",
    "-bewerkt",
    "-redigerad",
)

_SIDECAR_EXTS = {".json", ".xmp", ".aae", ".thm"}


class FolderSource:
    name = "folder"

    def __init__(self, tz_lookup: TzLookup | None = None) -> None:
        self._tz_lookup = tz_lookup

    def scan(self, root: Path) -> tuple[list[Photo], SourceReport]:
        report = SourceReport()
        by_hash: dict[str, Photo] = {}
        stem_index: dict[tuple[Path, str], str] = {}
        edited_pending: list[tuple[str, Path, str]] = []

        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            report.files_seen += 1

            if is_long_path(path):
                report.long_paths.append(path)

            if path.suffix.lower() in _SIDECAR_EXTS:
                report.sidecars_ignored += 1
                continue

            media_type, _fmt = sniff(path)
            if media_type is MediaType.UNKNOWN:
                report.skip("not_media")
                continue

            try:
                digest = file_hash(path)
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            except OSError as exc:
                report.unreadable.append((path, str(exc)))
                report.skip("unreadable")
                continue

            album = path.parent.name if path.parent != root else None

            # Register stem BEFORE the duplicate check. A photo that appears in
            # both a year folder and an album folder is merged on second sight,
            # but its edited variant lives beside one specific copy and must
            # still find it. Skipping this for duplicates silently breaks
            # edit-linking for exactly the Takeout layout we care about.
            base_stem, suffix = _split_edited(path.stem)
            if suffix is None:
                stem_index[(path.parent, path.stem)] = digest
            else:
                edited_pending.append((digest, path.parent, base_stem))

            if digest in by_hash:
                existing = by_hash[digest]
                existing.paths.append(path)
                if album and album not in existing.albums:
                    existing.albums.append(album)
                existing.last_seen = max(existing.last_seen, mtime)
                report.duplicates_merged += 1
                continue

            if find_sidecar(path) is not None:
                report.with_xmp += 1

            meta = self._read_meta(path, media_type, mtime)
            by_hash[digest] = Photo(
                file_hash=digest,
                paths=[path],
                media_type=media_type,
                meta=meta,
                first_seen=mtime,
                last_seen=mtime,
                albums=[album] if album else [],
            )

        for digest, parent, base_stem in edited_pending:
            original = stem_index.get((parent, base_stem))
            if original and original != digest:
                by_hash[digest].edited_of = original
                report.edited_linked += 1

        photos = list(by_hash.values())
        report.media_indexed = len(photos)
        for p in photos:
            # A capture date means a REAL one. Falling back to file mtime is not
            # knowing when the photo was taken, and counting it as coverage would
            # make doctor's low-date warning permanently silent.
            if p.meta.taken_at_utc and p.meta.tz_source is not TzSource.FILE_MTIME:
                report.with_date += 1
            if p.meta.gps:
                report.with_gps += 1
            if p.meta.people:
                report.with_people += 1
        return photos, report

    def _read_meta(self, path: Path, media_type: MediaType, mtime: datetime) -> PhotoMeta:
        exif = read_exif(path) if media_type is MediaType.IMAGE else None
        sidecar = find_sidecar(path)
        xmp = read_xmp(sidecar) if sidecar else None

        utc, local, tz_source = resolve(
            exif.taken_naive if exif else None,
            exif.offset if exif else None,
            exif.gps if exif else None,
            mtime,
            tz_lookup=self._tz_lookup,
        )
        return PhotoMeta(
            taken_at_utc=utc,
            taken_at_local=local,
            tz_source=tz_source,
            gps=exif.gps if exif else None,
            people=list(xmp.people) if xmp else [],
            face_regions=list(xmp.face_regions) if xmp else [],
            keywords=list(xmp.keywords) if xmp else [],
            description=xmp.description if xmp else None,
            camera_make=exif.camera_make if exif else None,
            camera_model=exif.camera_model if exif else None,
            width=exif.width if exif else None,
            height=exif.height if exif else None,
        )


def _split_edited(stem: str) -> tuple[str, str | None]:
    for suffix in EDITED_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)], suffix
    return stem, None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_folder_source.py -v`
Expected: PASS (12 tests)

If `test_edited_variant_links_to_its_original` fails, check that `stem_index` is
populated *before* the duplicate-`continue`, not after.

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -v`
Expected: all tests pass

- [ ] **Step 7: Commit**

```bash
git add src/rekindle/sources/ tests/test_folder_source.py
git commit -m "feat: folder source with dedupe, edited linking and metadata ladder"
```

---

### Task 10: doctor command and CLI

**Files:**
- Create: `src/rekindle/doctor.py`
- Create: `src/rekindle/cli.py`
- Test: `tests/test_doctor.py`
- Test: `tests/test_cli.py`
- Modify: `README.md` (replace the Quick start block)

**Interfaces:**
- Consumes: `FolderSource`, `SourceReport`, `PhotoStore`
- Produces:
  - `Diagnosis` — frozen dataclass with `report: SourceReport`, `warnings: list[str]`, `pct(n) -> float`
  - `diagnose(report: SourceReport) -> Diagnosis`
  - `render(diagnosis: Diagnosis, console) -> None`
  - `app` — typer app with `doctor` and `index` commands

- [ ] **Step 1: Write the failing test**

```python
# tests/test_doctor.py
from rekindle.doctor import diagnose
from rekindle.models import SourceReport


def _report(**kw) -> SourceReport:
    r = SourceReport()
    for k, v in kw.items():
        setattr(r, k, v)
    return r


def test_percentages_are_relative_to_indexed_media():
    d = diagnose(_report(media_indexed=200, with_date=100, with_gps=50, with_people=0))
    assert d.pct(d.report.with_date) == 50.0
    assert d.pct(d.report.with_gps) == 25.0


def test_zero_media_does_not_divide_by_zero():
    d = diagnose(_report(media_indexed=0, with_date=0))
    assert d.pct(0) == 0.0


def test_warns_when_no_people_data():
    d = diagnose(_report(media_indexed=100, with_people=0))
    assert any("person" in w.lower() for w in d.warnings)


def test_warns_when_date_coverage_is_low():
    d = diagnose(_report(media_indexed=100, with_date=10))
    assert any("date" in w.lower() for w in d.warnings)


def test_warns_about_long_paths(tmp_path):
    d = diagnose(_report(media_indexed=10, with_date=10, long_paths=[tmp_path]))
    assert any("long path" in w.lower() for w in d.warnings)


def test_warns_about_unignored_json_sidecars():
    d = diagnose(_report(media_indexed=100, with_date=100, sidecars_ignored=40))
    assert any("takeout" in w.lower() or "sidecar" in w.lower() for w in d.warnings)


def test_healthy_library_has_no_warnings():
    d = diagnose(_report(media_indexed=100, with_date=100, with_gps=80,
                         with_people=50, with_xmp=50))
    assert d.warnings == []
```

```python
# tests/test_cli.py
from typer.testing import CliRunner

from rekindle.cli import app
from tests.fixtures.gen import build_library

runner = CliRunner()


def test_doctor_reports_without_writing_a_database(tmp_path):
    root = build_library(tmp_path / "lib")
    data = tmp_path / "data"
    result = runner.invoke(app, ["doctor", str(root), "--data-dir", str(data)])
    assert result.exit_code == 0
    assert "photos" in result.stdout.lower()
    assert not (data / "rekindle.sqlite").exists()


def test_index_writes_photos_to_the_database(tmp_path):
    root = build_library(tmp_path / "lib")
    data = tmp_path / "data"
    result = runner.invoke(app, ["index", str(root), "--data-dir", str(data)])
    assert result.exit_code == 0
    assert (data / "rekindle.sqlite").exists()

    from rekindle.db import PhotoStore

    with PhotoStore(data / "rekindle.sqlite") as s:
        assert s.count() > 0


def test_index_twice_does_not_duplicate(tmp_path):
    root = build_library(tmp_path / "lib")
    data = tmp_path / "data"
    runner.invoke(app, ["index", str(root), "--data-dir", str(data)])
    from rekindle.db import PhotoStore

    with PhotoStore(data / "rekindle.sqlite") as s:
        first = s.count()
    runner.invoke(app, ["index", str(root), "--data-dir", str(data)])
    with PhotoStore(data / "rekindle.sqlite") as s:
        assert s.count() == first


def test_missing_directory_exits_nonzero(tmp_path):
    result = runner.invoke(app, ["doctor", str(tmp_path / "nope")])
    assert result.exit_code != 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_doctor.py tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rekindle.doctor'`

- [ ] **Step 3: Write doctor**

```python
# src/rekindle/doctor.py
"""Coverage reporting. Tells the user what metadata they actually have
before they spend time indexing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

from rekindle.models import SourceReport

_LOW_DATE_PCT = 50.0


@dataclass(frozen=True)
class Diagnosis:
    report: SourceReport
    warnings: list[str] = field(default_factory=list)

    def pct(self, n: int) -> float:
        total = self.report.media_indexed
        return round(100.0 * n / total, 1) if total else 0.0


def diagnose(report: SourceReport) -> Diagnosis:
    warnings: list[str] = []
    total = report.media_indexed

    if total:
        if 100.0 * report.with_date / total < _LOW_DATE_PCT:
            warnings.append(
                "Low date coverage - most photos have no capture date, so "
                "time-based memories will be unreliable."
            )
        if report.with_people == 0:
            warnings.append(
                "No person data found. Person-based memories and person "
                "exclusions are unavailable. Use date-range and folder "
                "exclusions instead - they always work."
            )
    if report.long_paths:
        warnings.append(
            f"{len(report.long_paths)} long paths found (>=260 chars). On Windows, "
            "enable LongPathsEnabled or some files may be unreadable."
        )
    if report.sidecars_ignored and total and report.sidecars_ignored >= total * 0.25:
        warnings.append(
            f"{report.sidecars_ignored} JSON/XMP sidecars were ignored. This looks "
            "like a Google Takeout export - the Takeout parser that reads those "
            "(face tags, descriptions) is not implemented yet."
        )
    return Diagnosis(report=report, warnings=warnings)


def render(diagnosis: Diagnosis, console: Console) -> None:
    r = diagnosis.report
    table = Table(title="Library report", show_header=True)
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    table.add_column("Coverage", justify="right")

    table.add_row("Files seen", str(r.files_seen), "")
    table.add_row("Photos indexed", str(r.media_indexed), "")
    table.add_row("With capture date", str(r.with_date), f"{diagnosis.pct(r.with_date)}%")
    table.add_row("With GPS", str(r.with_gps), f"{diagnosis.pct(r.with_gps)}%")
    table.add_row("With people", str(r.with_people), f"{diagnosis.pct(r.with_people)}%")
    table.add_row("With XMP sidecar", str(r.with_xmp), f"{diagnosis.pct(r.with_xmp)}%")
    table.add_row("Duplicates merged", str(r.duplicates_merged), "")
    table.add_row("Edited variants linked", str(r.edited_linked), "")
    table.add_row("Sidecars ignored", str(r.sidecars_ignored), "")
    table.add_row("Skipped", str(r.total_skipped), "")
    console.print(table)

    if r.skipped:
        console.print("\n[dim]Skipped by reason:[/dim]")
        for reason, count in sorted(r.skipped.items()):
            console.print(f"  {reason}: {count}")

    for warning in diagnosis.warnings:
        console.print(f"\n[yellow]![/yellow] {warning}")
```

- [ ] **Step 4: Write the CLI**

```python
# src/rekindle/cli.py
"""Command line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from rekindle import __version__
from rekindle.db import PhotoStore
from rekindle.doctor import diagnose, render
from rekindle.sources.folder import FolderSource

app = typer.Typer(help="Turn your photo library into memories.", no_args_is_help=True)
console = Console()

DataDir = Annotated[Path, typer.Option("--data-dir", help="Where rekindle stores its index.")]


def _check_root(root: Path) -> None:
    if not root.is_dir():
        console.print(f"[red]Not a directory:[/red] {root}")
        raise typer.Exit(code=2)


@app.command()
def version() -> None:
    """Print the version."""
    console.print(__version__)


@app.command()
def doctor(
    root: Annotated[Path, typer.Argument(help="Folder of photos to inspect.")],
    data_dir: DataDir = Path("./data"),
) -> None:
    """Report what metadata a library has. Writes nothing."""
    _check_root(root)
    _photos, report = FolderSource().scan(root)
    render(diagnose(report), console)


@app.command()
def index(
    root: Annotated[Path, typer.Argument(help="Folder of photos to index.")],
    data_dir: DataDir = Path("./data"),
) -> None:
    """Scan a folder and store its photos in the local index."""
    _check_root(root)
    photos, report = FolderSource().scan(root)
    with PhotoStore(data_dir / "rekindle.sqlite") as store:
        inserted, updated = store.upsert_many(photos)
        total = store.count()
    render(diagnose(report), console)
    console.print(f"\n[green]Indexed[/green] {inserted} new, {updated} updated. "
                  f"{total} photos in the index.")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_doctor.py tests/test_cli.py -v`
Expected: PASS (11 tests)

- [ ] **Step 6: Verify the CLI works end to end by hand**

```bash
uv run rekindle version
uv run rekindle doctor tests/
```
Expected: a version string, then a rendered table.

- [ ] **Step 7: Update the README Quick start**

Replace the `## Quick start` code block in `README.md` with:

```bash
git clone https://github.com/abhikatoldtrafford/rekindle
cd rekindle
uv sync

uv run rekindle doctor ~/Pictures    # what metadata do you actually have?
uv run rekindle index ~/Pictures     # build the local index
```

Delete the `rekindle serve` line — it does not exist yet.

- [ ] **Step 8: Run the full suite and lint**

Run: `uv run pytest -v && uv run ruff check . && uv run ruff format --check .`
Expected: all pass

- [ ] **Step 9: Commit**

```bash
git add src/rekindle/doctor.py src/rekindle/cli.py tests/test_doctor.py tests/test_cli.py README.md
git commit -m "feat: doctor and index commands"
```

---

## Definition of done for M0

- [ ] `uv run pytest` passes on Windows, macOS and Linux in CI
- [ ] `uv run ruff check .` and `ruff format --check .` clean
- [ ] `rekindle doctor <folder>` reports coverage and writes nothing
- [ ] `rekindle index <folder>` populates SQLite and is idempotent on re-run
- [ ] No network access anywhere in the test suite
- [ ] No torch, numpy, or model dependency in `pyproject.toml`
- [ ] Every skipped file is counted with a reason
