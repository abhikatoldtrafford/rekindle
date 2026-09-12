"""Shared test helpers."""

import re

#: ANSI SGR/CSI escape sequences.
#:
#: `rich` styles an option name in SEGMENTS: `--out` is emitted as
#: `-` ESC `-out` ESC, so the literal substring `--frame-ms` does not exist in
#: coloured output no matter what you do about whitespace. CI has colour and a
#: local `CliRunner` usually does not, which is why an assertion on an option
#: name passes here and fails there.
_ANSI = re.compile("\x1b\\[[0-9;]*[A-Za-z]")


def unwrapped(text: str) -> str:
    """Console text with colour and ALL whitespace removed, for substring
    assertions.

    Two things have to come out before a substring test means anything.

    **Whitespace**, because `rich` wraps to the terminal width and does it two
    different ways: soft-wrapping at spaces, and HARD-BREAKING a token longer
    than the width. Collapsing runs of whitespace to single spaces survives the
    first and fails the second - `--frame-ms` arrives as `--frame\\n-ms` and
    becomes `--frame -ms`, which still does not match. Removing whitespace
    entirely survives both.

    **Colour**, because of the segmenting above.

    Compare BOTH sides with this - the needle as well as the haystack.

    This has now caused CI failures four times, each time passing locally
    first: a macOS tmp path broken mid-segment, a Windows
    `rekindle semantic\\nembed`, `--frame-ms` in a narrow help table, and the
    same option name again once the whitespace was handled but the colour was
    not. The developer's terminal is wide and uncoloured; CI's is neither.
    """
    return "".join(_ANSI.sub("", text).split())
