"""Fixtures for the semantic tests: a REAL toy encoder and a REAL index.

TWO RULES FROM THE PROJECT'S DECISION LOG, APPLIED HERE
-------------------------------------------------------
**"A test that mocks the model and asserts the mock was called proves
nothing."** `ToyEncoder` is therefore not a mock. It is a genuine, if weak,
joint image/text embedding function: an image becomes its own downsampled
colour grid, and a colour word becomes the grid that colour would produce. So
a search for "red" over a set containing one red photo has exactly one right
answer, and every test below asserts the ANSWER rather than the call. Break
the ranking, the filtering, the normalisation or the row bookkeeping and these
tests fail - which is the property the decision log says was missing three
times in M1.

**"Fixtures certifying their own fiction."** `make_index` does not hand-write
SQL. It drives `rekindle.db.PhotoStore` with `rekindle.models.Photo` records -
the production writer - so the fixture physically cannot invent a column, a
JSON encoding or a flag that the real index does not have. Values (album names
like "Photos from 2019", empty-list people, the archived flag) are copied from
what was measured on the 19,480-photo reference index.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path

from rekindle.db import PhotoStore
from rekindle.models import MediaType, Photo, PhotoMeta, TzSource

#: 4x4 grid of RGB means. Small enough to write out in a test, large enough
#: that a gradient and a solid colour are not the same vector.
GRID = 4
TOY_DIM = GRID * GRID * 3

#: The colour words ToyEncoder understands, and what they mean.
COLOURS: dict[str, tuple[int, int, int]] = {
    "red": (220, 30, 30),
    "green": (30, 200, 60),
    "blue": (40, 60, 220),
    "yellow": (230, 220, 40),
    "black": (5, 5, 5),
    "white": (250, 250, 250),
}


def _normalise(values: list[float]) -> list[float]:
    norm = sum(v * v for v in values) ** 0.5
    if norm == 0:
        # An all-black image is a legitimate input and must not divide by zero.
        return [1.0 / (len(values) ** 0.5)] * len(values)
    return [v / norm for v in values]


def _grid_of(image) -> list[float]:
    small = image.convert("RGB").resize((GRID, GRID))
    out: list[float] = []
    for y in range(GRID):
        for x in range(GRID):
            out.extend(float(c) for c in small.getpixel((x, y)))
    return out


class ToyEncoder:
    """A real encoder over a real (tiny) feature: the 4x4 colour grid.

    Deterministic, dependency-free, and shared between images and text - which
    is what makes it a usable stand-in for CLIP in a test that asserts search
    results rather than call counts.
    """

    key = "toy"
    dim = TOY_DIM

    def __init__(self) -> None:
        self.image_calls = 0
        self.text_calls = 0

    def encode_images(self, images) -> list[list[float]]:
        self.image_calls += 1
        return [_normalise(_grid_of(im)) for im in images]

    def encode_texts(self, texts) -> list[list[float]]:
        self.text_calls += 1
        out = []
        for text in texts:
            lowered = text.lower()
            found = [rgb for word, rgb in COLOURS.items() if word in lowered]
            if found:
                mean = tuple(sum(c[i] for c in found) / len(found) for i in range(3))
            else:
                # An unknown word is grey: equidistant from every colour, so a
                # test for "no strong match" has a defined answer.
                mean = (128.0, 128.0, 128.0)
            out.append(_normalise(list(mean) * (GRID * GRID)))
        return out


class PreparingToyEncoder(ToyEncoder):
    """`ToyEncoder` split into the CPU half and the device half.

    The same arithmetic as `ToyEncoder`, cut where `TorchEncoder` is cut, so
    the pipelined path in `embed_photos` can be exercised with no model, no
    GPU and no network. It also records WHICH THREAD prepared each batch,
    which is the only way a test can tell that the pool is really being used
    rather than the main thread doing the work and the suite passing anyway.
    """

    def __init__(self, *, prepare_raises: bool = False) -> None:
        super().__init__()
        self.prepare_calls = 0
        self.prepared_calls = 0
        self.prepare_threads: set[int] = set()
        self._prepare_raises = prepare_raises

    def prepare_images(self, images) -> object:
        self.prepare_calls += 1
        self.prepare_threads.add(threading.get_ident())
        if self._prepare_raises:
            raise RuntimeError("prepare_images is broken on purpose")
        if not images:
            return None
        return [_grid_of(im) for im in images]

    def encode_prepared(self, prepared) -> list[list[float]]:
        self.prepared_calls += 1
        if prepared is None:
            return []
        return [_normalise(grid) for grid in prepared]


class HalfPreparedEncoder(ToyEncoder):
    """Only `prepare_images`, no `encode_prepared`. A half-finished migration.

    `embed_photos` must ignore it entirely: preparing without a matching
    finisher would hand raw grids to `encode_images`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.prepare_calls = 0

    def prepare_images(self, images) -> object:
        self.prepare_calls += 1
        return [_grid_of(im) for im in images]


def solid_image(colour: str | tuple[int, int, int], size: int = 64):
    """A plain square. `ToyEncoder` maps it to that colour's unit vector."""
    from PIL import Image

    rgb = COLOURS[colour] if isinstance(colour, str) else colour
    return Image.new("RGB", (size, size), rgb)


def write_photo(root: Path, name: str, colour: str | tuple[int, int, int], size: int = 64) -> Path:
    """A real JPEG on disk, so `embed` exercises a real decode."""
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    solid_image(colour, size).save(path, format="JPEG", quality=95)
    return path


def make_photo(
    file_hash: str,
    path: Path,
    *,
    albums: list[str] | None = None,
    people: list[str] | None = None,
    takeout_people: list[str] | None = None,
    archived: bool = False,
    trashed: bool = False,
    media_type: MediaType = MediaType.IMAGE,
    taken_at: datetime | None = None,
) -> Photo:
    """A `Photo` shaped like a real one, built by the production dataclass.

    Every Takeout row has at least a "Photos from YYYY" album (measured:
    19,480 of 19,480), which is why `albums` defaults to one rather than to the
    empty list a hand-written fixture would reach for.
    """
    when = taken_at or datetime(2019, 6, 1, 12, 0, tzinfo=UTC)
    meta = PhotoMeta(
        taken_at_utc=when,
        taken_at_local=when.replace(tzinfo=None),
        tz_source=TzSource.TAKEOUT,
        people=list(people or []),
        takeout_people=list(takeout_people if takeout_people is not None else (people or [])),
        archived=archived,
        trashed=trashed,
        width=64,
        height=64,
    )
    return Photo(
        file_hash=file_hash,
        paths=[path],
        media_type=media_type,
        meta=meta,
        first_seen=when,
        last_seen=when,
        albums=albums if albums is not None else [f"Photos from {when.year}"],
        source="folder",
        sidecar_match="exact",
    )


def make_index(db_path: Path, photos: list[Photo], index_root: Path | None = None) -> None:
    """Write `photos` through the PRODUCTION store. No hand-written SQL."""
    with PhotoStore(db_path) as store:
        store.upsert_many(photos)
        if index_root is not None:
            store.set_meta("index_root", str(index_root))


def colour_library(tmp_path: Path, per_colour: int = 3) -> tuple[Path, list[Photo]]:
    """A small library of solid-colour JPEGs with a known correct clustering.

    `per_colour` photos of each of red / green / blue, so k=3 has exactly one
    right answer and a broken clustering cannot accidentally look correct.
    """
    root = tmp_path / "library"
    photos = []
    for colour in ("red", "green", "blue"):
        for i in range(per_colour):
            name = f"{colour}_{i}.jpg"
            path = write_photo(root, name, colour)
            photos.append(
                make_photo(f"{colour[0]}{i:02d}" + "0" * 29, path, albums=["Photos from 2019"])
            )
    return root, photos
