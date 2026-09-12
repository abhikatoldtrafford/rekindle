"""A thumbnail cache on disk, because 61 GB of originals is not a page load.

Measured on the reference library: the median indexed image is 2976 px on its
short edge and the mean file is 3.3 MB. A 200-thumbnail grid decoded from
originals on every request is tens of seconds of CPU and gigabytes of reads;
the same grid from this cache is a few megabytes of already-encoded JPEG.

Three properties the cache has to have, none of them optional:

* **It is keyed by file_hash and width, not by path.** The same bytes appear
  under several paths in a Takeout export, and a path-keyed cache would
  duplicate every one of them and miss on a folder rename.
* **It never decides what may be seen.** `thumbnail` is handed a Photo that
  the caller already resolved through `MemoryIndex`; it has no way to find a
  photo itself, so it cannot leak one.
* **It writes atomically.** Two browser connections asking for the same
  thumbnail at once is the normal case, not the rare one, and a half-written
  JPEG served to the other one is a broken image with no error anywhere.
"""

from __future__ import annotations

import contextlib
import itertools
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from rekindle.meta.exif import open_upright
from rekindle.models import Photo

#: Distinguishes two writes from ONE thread, which the thread id cannot.
_WRITE_COUNTER = itertools.count()

#: Widths the UI asks for. Restricted to a fixed set so a hostile or buggy
#: query string cannot fill the cache with 4,000 sizes of the same photo.
GRID = 320
DETAIL = 900
SIZES = (GRID, DETAIL)

QUALITY = 82
CACHE_DIRNAME = "thumbs"


class ThumbnailError(RuntimeError):
    """The file is gone or will not decode. Reported, never raised past HTTP."""


@dataclass(frozen=True)
class Thumbnail:
    data: bytes
    etag: str
    from_cache: bool


class ThumbnailCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, file_hash: str, width: int) -> Path:
        # Two-character shard: 18,201 files in one directory is slow to list
        # on NTFS and slow to enumerate on any filesystem a user might back up.
        return self.root / str(width) / file_hash[:2] / f"{file_hash}.jpg"

    def get(self, photo: Photo, source: Path, width: int) -> Thumbnail:
        """The thumbnail bytes for one already-authorised photo.

        `source` is the path `MemoryIndex.resolve_path` returned, so the caller
        has already established that this photo is admissible AND present.
        """
        if width not in SIZES:
            raise ThumbnailError(f"unsupported thumbnail width {width}")
        cached = self.path_for(photo.file_hash, width)
        etag = f'"{photo.file_hash[:24]}-{width}"'
        try:
            return Thumbnail(cached.read_bytes(), etag, from_cache=True)
        except OSError:
            pass
        data = self._encode(source, width)
        _write_atomic(cached, data)
        return Thumbnail(data, etag, from_cache=False)

    def _encode(self, source: Path, width: int) -> bytes:
        import io

        try:
            # `open_upright` applies the EXIF rotation and hints the JPEG
            # decoder, so a portrait phone photo is not delivered on its side
            # and a 4000px original is not fully decoded to make a 320px box.
            image = open_upright(source, draft=(width, width)).convert("RGB")
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise ThumbnailError(f"could not decode {source.name}: {exc}") from exc
        image.thumbnail((width, width), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=QUALITY, optimize=True)
        return buffer.getvalue()


def _write_atomic(path: Path, data: bytes) -> None:
    """Write beside the target and rename over it.

    A rename within one directory is atomic on POSIX and on Windows via
    `os.replace`, so a concurrent reader sees either the old file or the whole
    new one. A failure to cache is not a failure to serve: the bytes are
    already in hand, so every error here is swallowed deliberately.

    THE TEMP NAME CARRIES THE THREAD, not just the process. `ThreadingHTTPServer`
    serves a grid of thumbnails on many threads of ONE process, and this
    module's own docstring says two requests for the same thumbnail at once is
    "the normal case, not the rare one" - so a name built from the pid alone
    was the SAME name for every one of them. Thread A could `replace()` a file
    that thread B had just truncated with `write_bytes`, committing a
    truncated JPEG; B's own `replace` then failed with `OSError` and was
    swallowed by the very design above. The damaged file is a valid prefix of
    the right one, so nothing detects it and `get()` serves it forever - the
    exact outcome the atomic write exists to prevent.

    Pid, thread and a counter rather than a random name: the set of names
    stays small and is reused across runs, so a crash mid-write leaves a
    `.part` file that the next run overwrites instead of accumulating one
    orphan per interrupted request.
    """
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        unique = f"{os.getpid()}.{threading.get_ident()}.{next(_WRITE_COUNTER)}"
        temporary = path.with_name(f"{path.name}.{unique}.part")
        temporary.write_bytes(data)
        temporary.replace(path)
    except OSError:
        # Unique names made the cleanup necessary. With the old pid-only name
        # a failed `replace` left a file the NEXT write would overwrite; now
        # every writer picks its own, so the same failure would leave one
        # orphan per occurrence in a cache directory nothing ever prunes.
        #
        # It is not hypothetical: eight threads replacing the same target on
        # Windows reliably lose at least one to a sharing violation, which is
        # how this line came to be written.
        if temporary is not None:
            with contextlib.suppress(OSError):
                temporary.unlink(missing_ok=True)
        return
