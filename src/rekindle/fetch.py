"""`rekindle music fetch` - the ONE command in this project that uses the network.

Nothing else does. Not `index`, not `enrich`, not `fingerprint`, not `memory`,
not `watch`. The README's promise that the default configuration makes no
network calls at all is unchanged, because this is never reached unless a
person types it.

Why it is a command and not a first run
---------------------------------------

The original brief asked for CC0 tracks to be fetched on first run. That was
declined, and the reason has since proved itself:

    A checksum pinned today is a promise about a third-party host forever.

**freepd.com, the most obvious source for exactly this, shut down permanently
in 2025.** Any release that had pinned URLs there would now fail on first run
for every user, long after anyone remembered why the list existed. An implicit
fetch turns a third party's uptime into your program's correctness.

A command the user runs deliberately has none of that tail. If the item moves,
the command says so, once, to someone who chose to run it - and every other
part of rekindle carries on working.

What the checksums do and do not prove
--------------------------------------

Names and checksums are resolved from the metadata API **at fetch time**,
never pinned in this file and never constructed by pattern. (Guessing a
plausible name returns 404: `Nocturne Op. 9 No. 2.mp3` is not what the item
calls it.) Each download is verified against the SHA-1 the API just gave.

Be clear about what that buys: the metadata and the audio come from the SAME
host, so a checksum match proves the bytes arrived intact, not that the host
is honest. It catches a truncated download, a proxy that mangled the stream,
and a partially written file resumed from - which is every failure mode this
command actually has. It is not a supply-chain guarantee and is not presented
as one. The guarantees that matter here are that the user asked for it, the
licence is checked before a single byte is downloaded, and the source is
printed.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

# The verified item. CC0 1.0, 104 solo-piano MP3s.
DEFAULT_ITEM = "musopen-chopin"

_METADATA_URL = "https://archive.org/metadata/{item}"
_DOWNLOAD_URL = "https://archive.org/download/{item}/{name}"

# The licence the item must declare before anything is downloaded. Checked
# rather than assumed: an item can be relicensed after this file is written,
# and that is precisely the tail a pinned list cannot see coming.
CC0_MARKER = "publicdomain/zero/1.0"

# archive.org's `format` for the derived MP3s. Filtering on format rather than
# on the ".mp3" suffix because the item also holds .m4a originals, .ogg
# derivatives, spectrograms and scans - 744 files in all, of which 104 are the
# ones wanted.
MP3_FORMAT = "VBR MP3"

_TIMEOUT = 60
_CHUNK = 1 << 16

# A caller may substitute the transport. Tests always do: nothing in the test
# suite is allowed to touch the network.
Opener = Callable[[str], object]


def _default_opener(url: str):  # pragma: no cover - exercised only for real
    return urllib.request.urlopen(url, timeout=_TIMEOUT)


@dataclass(frozen=True)
class Track:
    name: str
    sha1: str
    size: int
    title: str = ""
    length: str = ""

    def url(self, item: str) -> str:
        # quote(), because the names contain spaces, commas and accents:
        # "Nocturne Op. 9, No. 2 in E Flat Major.mp3".
        return _DOWNLOAD_URL.format(item=item, name=quote(self.name))

    @property
    def filename(self) -> str:
        """The name to write on disk.

        `Path(name).name` strips any directory component the metadata might
        carry. An item is a third party's namespace, and a file called
        `../../etc/thing.mp3` must land in the destination folder or nowhere.
        """
        return Path(self.name).name


@dataclass(frozen=True)
class Source:
    item: str
    title: str
    licence_url: str

    @property
    def cc0(self) -> bool:
        return CC0_MARKER in self.licence_url

    @property
    def page(self) -> str:
        return f"https://archive.org/details/{self.item}"


class FetchError(RuntimeError):
    """Anything that should stop the command with a message, not a traceback."""


@dataclass
class FetchReport:
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    bytes_written: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def accounted(self) -> int:
        return self.downloaded + self.skipped + self.failed


def list_tracks(
    item: str = DEFAULT_ITEM, *, opener: Opener | None = None
) -> tuple[Source, list[Track]]:
    """Resolve the item's MP3s through the metadata API.

    Sorted by name so that `--count 5` means the same five every time. Two
    runs of the same command must fetch the same files, or "resume" and "skip
    what I already have" both stop meaning anything.
    """
    opener = opener or _default_opener
    url = _METADATA_URL.format(item=quote(item, safe=""))
    try:
        with opener(url) as response:  # type: ignore[attr-defined]
            payload = json.load(response)
    except OSError as exc:
        raise FetchError(f"could not reach {url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise FetchError(f"{url} did not return JSON: {exc}") from exc

    metadata = payload.get("metadata") or {}
    if not metadata:
        # An item that does not exist returns `{}`, not a 404.
        raise FetchError(f"archive.org has no item called {item!r}")
    source = Source(
        item=item,
        title=str(metadata.get("title") or item),
        licence_url=str(metadata.get("licenseurl") or ""),
    )
    tracks = sorted(_mp3s(payload.get("files") or []), key=lambda t: t.name)
    return source, tracks


def _mp3s(files: list[dict]) -> Iterator[Track]:
    for entry in files:
        if entry.get("format") != MP3_FORMAT:
            continue
        name, sha1 = entry.get("name"), entry.get("sha1")
        if not name or not sha1:
            # No checksum means nothing to verify against, so it is not
            # offered at all rather than downloaded unverified.
            continue
        try:
            size = int(entry.get("size") or 0)
        except (TypeError, ValueError):
            size = 0
        yield Track(
            name=str(name),
            sha1=str(sha1).lower(),
            size=size,
            title=str(entry.get("title") or ""),
            length=str(entry.get("length") or ""),
        )


def fetch_tracks(
    tracks: list[Track],
    dest: Path,
    *,
    item: str = DEFAULT_ITEM,
    opener: Opener | None = None,
    on_progress: Callable[[int, int, Track], None] | None = None,
) -> FetchReport:
    """Download each track, verify it, and leave the folder consistent.

    Three properties, each of which is a failure this would otherwise have:

    **Resumable.** A file already present with the right SHA-1 is skipped
    without being re-downloaded. Re-running after an interruption costs the
    remainder, not the whole set.

    **Atomic.** The bytes go to a `.part` file that is renamed only after the
    checksum matches. An interrupted download therefore never leaves a
    truncated MP3 in `music/`, where `resolve_music` would hand it to ffmpeg
    as a memory's soundtrack.

    **It does not stop on the first failure.** One bad file out of 104 is
    reported and the rest are fetched, in the same spirit as the fingerprint
    pass.
    """
    opener = opener or _default_opener
    dest.mkdir(parents=True, exist_ok=True)
    report = FetchReport()
    for index, track in enumerate(tracks, start=1):
        if on_progress:
            on_progress(index, len(tracks), track)
        final = dest / track.filename
        if final.exists() and _sha1_of(final) == track.sha1:
            report.skipped += 1
            continue
        part = final.with_suffix(final.suffix + ".part")
        try:
            digest, written = _download(track.url(item), part, opener)
        except OSError as exc:
            report.failed += 1
            report.errors.append(f"{track.filename}: {exc}")
            part.unlink(missing_ok=True)
            continue
        if digest != track.sha1:
            part.unlink(missing_ok=True)
            report.failed += 1
            report.errors.append(
                f"{track.filename}: checksum mismatch "
                f"(expected {track.sha1[:12]}..., got {digest[:12]}...)"
            )
            continue
        part.replace(final)
        report.downloaded += 1
        report.bytes_written += written
    return report


def _download(url: str, part: Path, opener: Opener) -> tuple[str, int]:
    digest = hashlib.sha1()
    written = 0
    with opener(url) as response, part.open("wb") as handle:  # type: ignore[attr-defined]
        while True:
            chunk = response.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            handle.write(chunk)
            written += len(chunk)
    return digest.hexdigest(), written


def _sha1_of(path: Path) -> str:
    digest = hashlib.sha1()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(_CHUNK):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()
