"""Music: bring your own. Nothing is downloaded, ever.

The brief allowed shipping pinned CC0 URLs with verified SHA-256 checksums,
and only if they could be verified by actually fetching them. Network access
was available and the option was real. It was declined on a different ground:

    A checksum pinned today is a promise about a third-party host forever.

A CC0 host that relicenses, reorganises or simply expires a URL turns
`rekindle memory` into a command that either downloads an unexpected binary or
fails on first run, for every user, long after anyone remembers why the list
exists. Bring-your-own has no such tail, costs the user one file copy, and
keeps the promise the README makes about the default configuration making no
network calls at all.

**Silence is the default and is perfectly good output.** A GIF has no audio
track at all, and a silent MP4 is a normal thing to publish.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# Extensions ffmpeg will accept as an audio input without special handling.
AUDIO_SUFFIXES = frozenset({".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus"})

DEFAULT_DIR = Path("music")

NO_MUSIC_HINT = (
    "No music found. Drop any .mp3/.m4a/.wav into the `music/` folder "
    "(gitignored) and it will be used as the bed, or pass --music PATH. "
    "Silence is the default and rekindle never downloads audio."
)


def available_tracks(folder: Path = DEFAULT_DIR) -> list[Path]:
    """Every usable audio file in `folder`, SORTED.

    Sorted, not arbitrary: `iterdir()` order is filesystem-dependent, and the
    engine's whole promise is that the same library produces the same output
    on every machine. A music-ordering test that could not fail because NTFS
    happens to return sorted entries is already in this project's decision
    log, so the sort is pinned by a test that shuffles the input.
    """
    try:
        if not folder.is_dir():
            return []
        return sorted(
            p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
        )
    except OSError:
        return []


def resolve_music(
    explicit: Path | None = None,
    folder: Path = DEFAULT_DIR,
    memory_id: str | None = None,
) -> Path | None:
    """The audio bed for ONE memory, or None for silence.

    An explicit `--music` path wins outright and is returned even if it is
    unreadable - the caller passed it deliberately, so ffmpeg's own error is
    more useful than this function silently falling back to silence.

    Otherwise a track is chosen from `folder` by the memory's STABLE ID - the
    same `recipe:key` that dismissal and cooldown address it by. This started
    as `candidates[0]`, which was right when the folder held one file and
    became a bug when it held 40: every one of 53 memories opened with the
    same piece, which reads as a defect rather than a choice.

    Deriving the index from the id rather than from position keeps both
    properties that matter. Re-rendering one memory is byte-identical, because
    its id does not change; and two different memories differ, because their
    ids do. A counter or a shuffle would give the first and break the second
    the moment a memory was rendered on its own.

    SHA-256 rather than `hash()`, which is randomised per process for strings
    and would give a different track on every run.

    No mood classification, deliberately: there is no honest metadata signal
    for the mood of a photograph, and inventing one would mean the engine
    asserting something it cannot know - the defect this project keeps
    finding.

    Adding or removing a file in `folder` reshuffles the assignment, because
    the modulus changes. That is accepted: the alternative is recording which
    track each memory was given, which turns a folder of files into state to
    migrate. `--music` pins one when it matters.
    """
    if explicit is not None:
        return explicit
    candidates = available_tracks(folder)
    if not candidates:
        return None
    if memory_id is None:
        # No identity to derive from - a caller rendering something that is
        # not a memory. First track, as before.
        return candidates[0]
    digest = hashlib.sha256(memory_id.encode("utf-8")).digest()
    return candidates[int.from_bytes(digest[:8], "big") % len(candidates)]
