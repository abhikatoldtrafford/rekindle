"""Google Takeout JSON sidecar enrichment.

Deliberately NOT a `Source`: that protocol returns (list[Photo], SourceReport)
for something producing records from pixels. This updates existing records
addressed by content hash. It does inherit the protocol's real contract -
count everything - which `EnrichReport` enforces in Task 6.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from rekindle.models import Gps
from rekindle.sidecars import is_album_metadata, sidecar_target


class JsonKind(StrEnum):
    PHOTO = "photo"
    ALBUM = "album"
    OTHER = "other"
    UNPARSEABLE = "unparseable"


@dataclass(frozen=True)
class Sidecar:
    path: Path
    # Casefolded filename of the photo this sidecar describes, with Takeout's
    # (N) counter relocated into the stem. THE MATCH KEY.
    target_cf: str
    # Google's own record of the filename. It omits the counter, so it is a
    # cross-check only - never the key. See spec section 0.
    title: str
    taken_at_utc: datetime | None
    people: tuple[str, ...]
    gps: Gps | None
    description: str | None
    favorite: bool
    archived: bool
    trashed: bool

    @property
    def title_disagrees(self) -> bool:
        """True when `title` names something other than the derived target.

        Expected for every `(N)` sidecar. Anything else is worth reporting.
        """
        return self.title.casefold() != self.target_cf


def classify_json(path: Path) -> tuple[JsonKind, dict | None]:
    """Read and classify one .json file. Never raises."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return JsonKind.UNPARSEABLE, None
    if not isinstance(payload, dict):
        # A top-level array is not something we know how to read.
        return JsonKind.UNPARSEABLE, None
    if is_album_metadata(path.name):
        return JsonKind.ALBUM, payload
    # A per-photo sidecar always carries a string `title` (Google's own name
    # for the file) and/or one of these markers. Checking `title`'s TYPE, not
    # just its presence, is what keeps `user-generated-memory-titles.json`
    # (title is a LIST) out of PHOTO; checking the markers too is what lets a
    # sidecar with no `title` at all - or a genuine export we have not seen -
    # still classify correctly. `shared_album_comments.json` has neither and
    # correctly falls through to OTHER.
    has_marker = any(
        k in payload for k in ("photoTakenTime", "creationTime", "geoData", "imageViews")
    )
    if isinstance(payload.get("title"), str) or has_marker:
        return JsonKind.PHOTO, payload
    return JsonKind.OTHER, payload


def _timestamp(block: object) -> datetime | None:
    if not isinstance(block, dict):
        return None
    raw = block.get("timestamp")
    try:
        return datetime.fromtimestamp(int(raw), UTC)
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def _geo(block: object) -> Gps | None:
    """A (0,0) pair is the Gulf of Guinea, not a location. 89.6% of real
    sidecars carry exactly that, and `geoDataExif` is key-ABSENT on them
    rather than zeroed."""
    if not isinstance(block, dict):
        return None
    # latitude and longitude must BOTH be present. Defaulting a missing half
    # to 0.0 would fabricate a wrong-but-plausible coordinate rather than
    # report the absence - not observed in the real export, but the
    # constraint is "never fabricate a value", not "never seen yet".
    # Altitude stays optional: it is genuinely optional in the format.
    if "latitude" not in block or "longitude" not in block:
        return None
    try:
        lat = float(block["latitude"])
        lon = float(block["longitude"])
        alt = float(block.get("altitude", 0.0))
    except (TypeError, ValueError):
        return None
    if abs(lat) < 1e-9 and abs(lon) < 1e-9:
        return None
    return Gps(lat=lat, lon=lon, alt=alt)


def parse_sidecar(path: Path, payload: dict) -> Sidecar:
    """Never raises. A field we cannot read is absent, never fabricated."""
    people: list[str] = []
    raw_people = payload.get("people")
    if isinstance(raw_people, list):
        for entry in raw_people:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if isinstance(name, str) and name.strip() and name not in people:
                people.append(name)

    description = payload.get("description")
    description = description.strip() if isinstance(description, str) else ""

    title = payload.get("title")
    title = title if isinstance(title, str) else ""

    return Sidecar(
        path=path,
        target_cf=sidecar_target(path.name).casefold(),
        title=title,
        taken_at_utc=_timestamp(payload.get("photoTakenTime")),
        people=tuple(people),
        # geoData first: measured on 24,248 sidecars, the two never disagree
        # and geoDataExif never carries data geoData lacks. The fallback costs
        # nothing and covers exports we have not seen.
        gps=_geo(payload.get("geoData")) or _geo(payload.get("geoDataExif")),
        description=description or None,
        # `favorited` is absent when false - it is never written as false.
        favorite=payload.get("favorited") is True,
        archived=payload.get("archived") is True,
        trashed=payload.get("trashed") is True,
    )


def album_title(payload: dict | None) -> str | None:
    """The album's real, pre-sanitisation name, or None.

    Returns None for a null, empty, whitespace-only or LIST-valued title - the
    export root's metadata.json has `title: null` and
    user-generated-memory-titles.json has a list.
    """
    if not isinstance(payload, dict):
        return None
    title = payload.get("title")
    if not isinstance(title, str):
        return None
    return title.strip() or None
