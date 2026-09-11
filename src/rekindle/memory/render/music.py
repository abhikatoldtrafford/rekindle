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

from pathlib import Path

# Extensions ffmpeg will accept as an audio input without special handling.
AUDIO_SUFFIXES = frozenset({".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus"})

DEFAULT_DIR = Path("music")

NO_MUSIC_HINT = (
    "No music found. Drop any .mp3/.m4a/.wav into the `music/` folder "
    "(gitignored) and it will be used as the bed, or pass --music PATH. "
    "Silence is the default and rekindle never downloads audio."
)


def resolve_music(explicit: Path | None = None, folder: Path = DEFAULT_DIR) -> Path | None:
    """The audio bed to use, or None for silence.

    An explicit `--music` path wins and is returned even if it is unreadable -
    the caller passed it deliberately, so ffmpeg's own error is more useful
    than this function silently falling back to silence.

    Otherwise the lexicographically first audio file in `folder`. SORTED, not
    arbitrary: two runs over the same folder must produce the same video, and
    `iterdir()` order is filesystem-dependent.
    """
    if explicit is not None:
        return explicit
    try:
        if not folder.is_dir():
            return None
        candidates = sorted(
            p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
        )
    except OSError:
        return None
    return candidates[0] if candidates else None
