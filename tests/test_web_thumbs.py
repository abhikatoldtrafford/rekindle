"""The thumbnail cache.

The orientation test uses `tests/fixtures/oriented.py`, which puts the answer
in the PIXELS rather than in the dimensions - a 400x300 file tagged 90 degrees
decodes to 300x400 whether the transpose ran forwards, backwards or not at
all, and four of the eight tag values do not change the size in the first
place. A grid of two hundred phone photos served on their sides is exactly the
kind of defect a size assertion cannot see.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from rekindle.web.thumbs import DETAIL, GRID, SIZES, ThumbnailCache, ThumbnailError
from tests.fixtures.oriented import UPRIGHT, quadrants, write_oriented
from tests.fixtures.web import make_library, open_library


@pytest.mark.parametrize("orientation", [1, 2, 3, 4, 5, 6, 7, 8])
def test_a_rotated_photo_is_thumbnailed_upright(tmp_path, orientation):
    source = tmp_path / f"o{orientation}.jpg"
    write_oriented(source, orientation, size=(800, 600))
    cache = ThumbnailCache(tmp_path / "cache")

    data = cache._encode(source, GRID)
    with Image.open(io.BytesIO(data)) as thumb:
        assert quadrants(thumb) == UPRIGHT, f"orientation {orientation} was not applied"


def test_the_two_widths_produce_different_sizes(tmp_path):
    data_dir, _ = make_library(tmp_path)
    index = open_library(data_dir).require_index()
    photo = index.get("ok00")
    cache = ThumbnailCache(tmp_path / "cache")

    small = cache.get(photo, index.resolve_path(photo), GRID)
    large = cache.get(photo, index.resolve_path(photo), DETAIL)
    with Image.open(io.BytesIO(small.data)) as image:
        assert max(image.size) == GRID
    with Image.open(io.BytesIO(large.data)) as image:
        assert max(image.size) == DETAIL
    assert small.etag != large.etag


def test_the_second_request_comes_from_disk(tmp_path):
    data_dir, _ = make_library(tmp_path)
    index = open_library(data_dir).require_index()
    photo = index.get("ok01")
    source = index.resolve_path(photo)
    cache = ThumbnailCache(tmp_path / "cache")

    first = cache.get(photo, source, GRID)
    assert first.from_cache is False
    second = cache.get(photo, source, GRID)
    assert second.from_cache is True
    assert second.data == first.data

    # And the cache is keyed by the CONTENT hash, not by the path: the same
    # bytes in two folders are one thumbnail, which is what makes a Takeout
    # export with its duplicates affordable.
    twin = tmp_path / "elsewhere" / "copy.jpg"
    twin.parent.mkdir(parents=True)
    twin.write_bytes(source.read_bytes())
    assert cache.get(photo, twin, GRID).from_cache is True


def test_an_undecodable_file_is_reported_not_raised_as_an_oserror(tmp_path):
    data_dir, _ = make_library(tmp_path)
    index = open_library(data_dir).require_index()
    photo = index.get("ok02")
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"this is not a JPEG")
    cache = ThumbnailCache(tmp_path / "cache")
    with pytest.raises(ThumbnailError, match="could not decode"):
        cache.get(photo, broken, GRID)


def test_a_failure_to_cache_still_serves_the_bytes(tmp_path, monkeypatch):
    """The cache is an optimisation. A read-only disk must not blank the page."""
    data_dir, _ = make_library(tmp_path)
    index = open_library(data_dir).require_index()
    photo = index.get("ok03")
    cache = ThumbnailCache(tmp_path / "cache")

    from rekindle.web import thumbs

    def refuse(*_args, **_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(thumbs.Path, "mkdir", refuse)
    result = cache.get(photo, index.resolve_path(photo), GRID)
    assert result.data[:2] == b"\xff\xd8"
    assert result.from_cache is False


def test_the_shard_keeps_one_directory_from_holding_the_whole_library(tmp_path):
    cache = ThumbnailCache(tmp_path / "cache")
    path = cache.path_for("abcdef0123456789", GRID)
    assert path.parent.name == "ab"
    assert path.parent.parent.name == str(GRID)
    assert path.name.endswith(".jpg")
    assert GRID in SIZES and DETAIL in SIZES


def test_two_threads_writing_the_same_thumbnail_use_different_temp_files(tmp_path, monkeypatch):
    """The atomic write was atomic per PROCESS, and the server is threaded.

    `ThreadingHTTPServer` serves a grid on many threads of one process, and
    this module's docstring calls two simultaneous requests for the same
    thumbnail "the normal case, not the rare one". With the temp name built
    from the pid alone, every one of them wrote the same file: thread A could
    `replace()` bytes that thread B had just truncated, committing a valid but
    TRUNCATED JPEG that nothing detects and `get()` then serves forever.

    Deterministic on purpose. Racing threads and hoping to catch corruption
    would be a flaky test of a narrow window; what the fix has to guarantee is
    simply that no two writers pick the same name.
    """
    import threading

    from rekindle.web.thumbs import _write_atomic

    seen: list[str] = []
    lock = threading.Lock()
    real = Path.write_bytes

    def record(self, data):
        with lock:
            seen.append(self.name)
        return real(self, data)

    monkeypatch.setattr(Path, "write_bytes", record)

    target = tmp_path / "cache" / "abcd.jpg"
    threads = [
        threading.Thread(target=_write_atomic, args=(target, b"x" * (10 + i))) for i in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(seen) == 8
    assert len(set(seen)) == 8, f"two writers shared a temp file: {sorted(seen)}"
    assert target.is_file()
    # Whichever writer won, the file is one whole input and not a blend.
    assert target.read_bytes() in {b"x" * (10 + i) for i in range(8)}
    assert not list(target.parent.glob("*.part")), "a temp file was left behind"
