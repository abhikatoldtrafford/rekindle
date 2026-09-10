"""Google Takeout JSON sidecar enrichment.

Deliberately NOT a `Source`: that protocol returns (list[Photo], SourceReport)
for something producing records from pixels. This updates existing records
addressed by content hash. It does inherit the protocol's real contract -
count everything - which `EnrichReport` enforces in Task 6.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from rekindle.models import Gps, Photo
from rekindle.sidecars import is_album_metadata, sidecar_target
from rekindle.sources.folder import EXCLUDED_DIRS


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
    # UnicodeDecodeError is NOT listed separately - it is a ValueError
    # subclass, so it is already covered here. Do not "helpfully" add it
    # back: there is no separate branch for it to isolate.
    except (OSError, ValueError):
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


@dataclass
class EnrichReport:
    """What the enrichment pass found. Two identities, both pinned by a test.

        json_files_seen == sidecars_seen + album_metadata
                           + other_json + excluded_dirs + unparseable
        sidecars_seen   == matched + superseded + orphaned + ambiguous

    The spec's section 7 gives only the second, and without `superseded` in it
    3,358 sidecars on the reference export are parsed and discarded with
    nothing reporting it.

    Neither identity can catch a bug downstream of indexing on its own - both
    partition sets that `account()` builds. The check that CAN fail is
    `matched == photos_enriched`, asserted end to end in Task 11: it ties the
    accounting to what was actually written to the database.
    """

    json_files_seen: int = 0
    sidecars_seen: int = 0
    album_metadata: int = 0
    other_json: int = 0
    excluded_dirs: int = 0
    unparseable: int = 0

    matched: int = 0
    orphaned: int = 0
    ambiguous: int = 0
    # Sidecars that named a photo we DID enrich but were not the one applied -
    # a second or third candidate for the same file. 3,358 of them on the
    # reference export; without this bucket they are discarded unreported.
    superseded: int = 0

    photos_enriched: int = 0
    # Photos where a same-directory candidate was preferred over a DISTANT one
    # that disagreed with it. Spec-conformant (rule 1 outranks rule 3) but
    # 942 on the reference export against an `ambiguous` count of 2, so
    # leaving it unreported makes the hazard look negligible.
    directory_preference_broke_a_tie: int = 0
    derivatives_enriched: int = 0
    people_added: int = 0
    dates_corrected: int = 0
    gps_added: int = 0
    descriptions_added: int = 0
    favourites_added: int = 0
    albums_retitled: int = 0
    conflicts: int = 0
    clustered_dates_suppressed: int = 0
    title_disagreements: int = 0
    album_title_collisions: list[tuple[str, str]] = field(default_factory=list)

    @property
    def files_accounted(self) -> int:
        return (
            self.sidecars_seen
            + self.album_metadata
            + self.other_json
            + self.excluded_dirs
            + self.unparseable
        )

    @property
    def sidecars_accounted(self) -> int:
        return self.matched + self.superseded + self.orphaned + self.ambiguous


@dataclass
class SidecarIndex:
    """Global, keyed on the casefolded target filename.

    v1 keyed on (directory, title). That is wrong twice over: `title` omits
    the (N) counter, and Takeout writes a sidecar into every album a photo
    belongs to WITHOUT duplicating the pixels there - 8 album folders in the
    reference export hold sidecars and zero media.
    """

    by_target: dict[str, list[Sidecar]] = field(default_factory=dict)
    albums: dict[Path, str] = field(default_factory=dict)

    def add(self, sidecar: Sidecar) -> None:
        self.by_target.setdefault(sidecar.target_cf, []).append(sidecar)


def build_index(root: Path, report: EnrichReport) -> SidecarIndex:
    """Walk every .json under `root`, classify it, and bucket it.

    EXCLUDED_DIRS is inherited from FolderSource: parsing Trash/'s sidecars
    and counting them against an index that correctly excludes those photos
    would inflate the orphan count on a perfectly complete library.
    """
    index = SidecarIndex()
    for path in sorted(root.rglob("*.json")):
        report.json_files_seen += 1
        if any(part.casefold() in EXCLUDED_DIRS for part in path.relative_to(root).parts[:-1]):
            report.excluded_dirs += 1
            continue
        kind, payload = classify_json(path)
        if kind is JsonKind.UNPARSEABLE:
            report.unparseable += 1
            continue
        if kind is JsonKind.ALBUM:
            report.album_metadata += 1
            title = album_title(payload)
            if title:
                index.albums[path.parent] = title
            continue
        if kind is JsonKind.OTHER:
            report.other_json += 1
            continue
        report.sidecars_seen += 1
        sidecar = parse_sidecar(path, payload)
        if sidecar.title_disagrees:
            report.title_disagreements += 1
        index.add(sidecar)
    return index


def _disagree(a: Sidecar, b: Sidecar) -> bool:
    """Do two candidates for one photo tell different stories?

    Only the fields that would change the photo's record. Two sidecars
    agreeing on date and people are interchangeable, however many there are.
    """
    return a.taken_at_utc != b.taken_at_utc or set(a.people) != set(b.people)


def resolve(photo: Photo, index: SidecarIndex) -> tuple[Sidecar | None, str, bool]:
    """Find the one sidecar describing `photo`, or refuse.

    Returns (sidecar, outcome, preference_broke_a_tie).

    1. A candidate in the same directory as one of the photo's paths wins.
    2. Otherwise, if the remaining candidates all agree, take the first.
    3. Otherwise REFUSE, and say so. Never let insertion order decide.

    Rule 1 deliberately outranks rule 3, so a DISTANT candidate that disagrees
    is overridden rather than refused. Measured on the reference export that
    happens to 942 photos, and every disagreeing group has its candidates in
    different directories - so rule 3 fires only 2 times. Left silent, doctor
    prints an `ambiguous` count of 2 and the hazard looks negligible when it
    is not. The third return value is what makes the override visible.
    """
    candidates: list[Sidecar] = []
    parents = {p.parent for p in photo.paths}
    for path in photo.paths:
        for sidecar in index.by_target.get(path.name.casefold(), ()):
            if sidecar not in candidates:
                candidates.append(sidecar)
    if not candidates:
        return None, "none", False

    same_dir = [s for s in candidates if s.path.parent in parents]
    if same_dir:
        chosen = same_dir[0]
        if any(_disagree(chosen, s) for s in same_dir[1:]):
            return None, "ambiguous", False
        tie = any(_disagree(chosen, s) for s in candidates if s.path.parent not in parents)
        return chosen, "exact", tie

    if all(not _disagree(candidates[0], s) for s in candidates[1:]):
        return candidates[0], "exact", False
    return None, "ambiguous", False


def account(
    index: SidecarIndex,
    claimed: dict[str, str],
    applied: set[Path],
    report: EnrichReport,
) -> None:
    """Partition every indexed sidecar into exactly one bucket.

    `claimed` maps a casefolded target to the resolution outcome; `applied`
    holds the paths of the sidecars actually WRITTEN onto a photo.

    The `superseded` bucket is the whole point. Crediting `matched` with
    `len(sidecars)` for every claimed target - as the first draft of this plan
    did - inflated it by 3,358 on the reference export, because a target with
    three candidates contributes one application and two discards. Those 3,358
    were parsed and thrown away with nothing reporting it, which is precisely
    v1's failure mode surviving inside the machinery built to prevent it.
    """
    for target, sidecars in index.by_target.items():
        outcome = claimed.get(target)
        if outcome == "ambiguous":
            report.ambiguous += len(sidecars)
        elif outcome == "exact":
            used = sum(1 for s in sidecars if s.path in applied)
            report.matched += used
            report.superseded += len(sidecars) - used
        else:
            report.orphaned += len(sidecars)
