"""A small real library on disk, plus a real index, for the web tests.

Nothing here is a mock. `make_library` writes actual JPEGs and drives
`rekindle.db.PhotoStore` with `rekindle.models.Photo` records - the production
writer - so the fixture cannot invent a column, an encoding or a flag the real
index does not have, and every assertion downstream is about behaviour rather
than about a stub having been called.

The shapes are chosen to exercise the guardrails the UI has to honour:

* **an archived photo** (162 rows in the reference library) - must never be
  visible anywhere in the UI;
* **a photo of an excluded person** - the same, via a different rule;
* **a two-frame burst**, three seconds apart with near-identical dHashes, so
  `dedup.collapse` has something real to collapse and the burst control has
  something real to swap;
* **a photo below the 480 px short-edge floor**, so `composition.compose` has
  a real per-photo rejection to explain.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from PIL import Image

from rekindle.db import PhotoStore
from rekindle.models import MediaType, Photo, PhotoMeta

T0 = datetime(2020, 1, 1, tzinfo=UTC)
ALBUM = "Kashmir"
EXCLUDED_PERSON = "Someone Excluded"

#: Two hashes six bits apart, which is exactly the dedup threshold, so the
#: pair collapses. Everything else is far away.
BURST_A = 0x0F0F0F0F0F0F0F0F
BURST_B = 0x0F0F0F0F0F0F0F3F  # differs in 3 bits -> within DEFAULT_THRESHOLD


def _stable_hash(name: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")


def jpeg(path: Path, size: tuple[int, int] = (960, 720), colour=(120, 90, 60)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path, "JPEG", quality=80)
    return path


def photo(
    file_hash: str,
    path: Path,
    *,
    local: datetime,
    albums=(ALBUM,),
    people=("Abhik Maiti",),
    size=(960, 720),
    sharpness=8.0,
    phash: int | None = None,
    archived: bool = False,
    favorite: bool = False,
    description: str | None = None,
) -> Photo:
    return Photo(
        file_hash=file_hash,
        paths=[path],
        media_type=MediaType.IMAGE,
        meta=PhotoMeta(
            taken_at_utc=local.replace(tzinfo=UTC),
            taken_at_local=local,
            people=list(people),
            width=size[0],
            height=size[1],
            sharpness=sharpness,
            brightness=110.0,
            # NOT `hash()`: string hashing is randomised per process, so a
            # fixture built on it would give a different burst grouping on
            # every run and the dedup assertions would be flaky rather than
            # wrong. A stable digest of the name keeps the library identical
            # across runs, which is what the engine promises anyway.
            phash=phash if phash is not None else _stable_hash(file_hash),
            archived=archived,
            favorite=favorite,
            description=description,
        ),
        first_seen=T0,
        last_seen=T0,
        albums=list(albums),
    )


def make_library(tmp_path: Path, *, days: int = 10) -> tuple[Path, list[Photo]]:
    """Write a library and index it. Returns (data_dir, the Photo records).

    `days` ordinary photos one day apart, and then the four awkward ones the
    guardrail tests need.
    """
    data_dir = tmp_path / "data"
    root = tmp_path / "lib"
    photos: list[Photo] = []

    for i in range(days):
        photos.append(
            photo(
                f"ok{i:02d}",
                jpeg(root / f"p{i:02d}.jpg", colour=(30 + i * 18, 90, 60)),
                local=datetime(2020, 5, 1, 9, 0) + timedelta(days=i),
                favorite=(i == 0),
                description="a description" if i == 1 else None,
            )
        )

    # A burst: two frames three seconds apart with hashes inside the dedup
    # threshold. The first is sharper, so `collapse` keeps it.
    burst_at = datetime(2020, 5, 20, 10, 0)
    photos.append(
        photo(
            "burst_keep",
            jpeg(root / "burst_a.jpg", colour=(200, 40, 40)),
            local=burst_at,
            phash=BURST_A,
            sharpness=9.5,
        )
    )
    photos.append(
        photo(
            "burst_drop",
            jpeg(root / "burst_b.jpg", colour=(201, 41, 41)),
            local=burst_at + timedelta(seconds=3),
            phash=BURST_B,
            sharpness=4.0,
        )
    )

    # Below MIN_SHORT_EDGE (480). A real per-photo composition rejection.
    photos.append(
        photo(
            "tiny",
            jpeg(root / "tiny.jpg", size=(400, 300), colour=(10, 200, 10)),
            local=datetime(2020, 5, 25, 9, 0),
            size=(400, 300),
        )
    )

    # Archived: not configurable, never surfaces.
    photos.append(
        photo(
            "archived",
            jpeg(root / "archived.jpg", colour=(0, 0, 200)),
            local=datetime(2020, 5, 26, 9, 0),
            archived=True,
        )
    )

    # Two photos of a person the user will exclude, on ONE day.
    #
    # Two, not one, deliberately. `prompt.MIN_SEEDS` is 2, so a day holding a
    # single photo can never become a seed day whatever the policy says - a
    # test asserting that an excluded photo's day was refused would pass on a
    # library with no guardrails at all. With two, the day IS a seed day the
    # moment the exclusion stops being applied, and the test can fail.
    for n, minute in enumerate((0, 40)):
        photos.append(
            photo(
                f"excluded{n if n else ''}",
                jpeg(root / f"excluded{n}.jpg", colour=(200, 0, 200 - n * 40)),
                local=datetime(2020, 5, 27, 9, minute),
                people=(EXCLUDED_PERSON,),
            )
        )

    with PhotoStore(data_dir / "rekindle.sqlite") as store:
        store.upsert_many(photos)
    return data_dir, photos


def exclude_person(data_dir: Path, name: str = EXCLUDED_PERSON) -> None:
    """Exclude a person through the real `MemoryState`, not by editing TOML."""
    from rekindle.memory.history import KIND_PERSON, MemoryState

    with PhotoStore(data_dir / "rekindle.sqlite") as store:
        MemoryState(store).dismiss(KIND_PERSON, name, reason="excluded")


def open_library(data_dir: Path):
    """A loaded `Library`, synchronously."""
    from rekindle.web.library import Library

    library = Library(data_dir)
    library.load()
    return library


def make_workshop(tmp_path: Path, data_dir: Path):
    from rekindle.web.api import Workshop

    return Workshop(
        open_library(data_dir),
        out_dir=tmp_path / "memories",
        music_dir=tmp_path / "music",
    )


def drain(events) -> list[dict]:
    return list(events)


def last(events: list[dict], name: str) -> dict:
    for event in reversed(events):
        if event["event"] == name:
            return event["data"]
    raise AssertionError(f"no {name!r} event in {[e['event'] for e in events]}")
