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
