"""Google Takeout JSON sidecar enrichment.

Deliberately NOT a `Source`: that protocol returns (list[Photo], SourceReport)
for something producing records from pixels. This updates existing records
addressed by content hash. It does inherit the protocol's real contract -
count everything - which `EnrichReport` enforces in Task 6.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

from rekindle.meta.timestamps import TzLookup, from_takeout
from rekindle.models import Gps, Photo, PhotoMeta, TzSource
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

    Every field `EnrichReport` can credit as a real change to the record:
    date, people, GPS, description, and favourite. Originally checked only
    date and people; on the reference export, among applied-vs-discarded
    candidate pairs that agreed on those two, 9 disagreed on GPS and 2 on
    favourite - each silently overridden with `tie=False` and no counter,
    which made the docstring's claim false. `archived`/`trashed` are
    deliberately excluded: EnrichReport has no counter that credits either as
    a change, so widening this far would flag disagreements nothing reports.
    """
    return (
        a.taken_at_utc != b.taken_at_utc
        or set(a.people) != set(b.people)
        or a.gps != b.gps
        or a.description != b.description
        or a.favorite != b.favorite
    )


class Resolution(NamedTuple):
    """`resolve()`'s result. A NamedTuple, not a bare 3-tuple, so a caller
    that only cares about the sidecar cannot drop `directory_preference_broke_a_tie`
    without a positional `_` that is visible in review - the exact silence
    that let the override go unreported before. Unpacks like a plain tuple,
    so `sidecar, outcome, tie = resolve(...)` still works unchanged.
    """

    sidecar: Sidecar | None
    outcome: str
    directory_preference_broke_a_tie: bool


def resolve(photo: Photo, index: SidecarIndex) -> Resolution:
    """Find the one sidecar describing `photo`, or refuse.

    1. A candidate in the same directory as one of the photo's paths wins.
    2. Otherwise, if the remaining candidates all agree, take the first.
    3. Otherwise REFUSE, and say so. Never let insertion order decide.

    Rule 1 deliberately outranks rule 3, so a DISTANT candidate that disagrees
    is overridden rather than refused. Measured on the reference export that
    happens to 942 photos, and every disagreeing group has its candidates in
    different directories - so rule 3 fires only 2 times. Left silent, doctor
    prints an `ambiguous` count of 2 and the hazard looks negligible when it
    is not. `Resolution.directory_preference_broke_a_tie` is what makes the
    override visible.
    """
    candidates: list[Sidecar] = []
    parents = {p.parent for p in photo.paths}
    for path in photo.paths:
        for sidecar in index.by_target.get(path.name.casefold(), ()):
            if sidecar not in candidates:
                candidates.append(sidecar)
    if not candidates:
        return Resolution(None, "none", False)

    same_dir = [s for s in candidates if s.path.parent in parents]
    if same_dir:
        chosen = same_dir[0]
        if any(_disagree(chosen, s) for s in same_dir[1:]):
            return Resolution(None, "ambiguous", False)
        tie = any(_disagree(chosen, s) for s in candidates if s.path.parent not in parents)
        return Resolution(chosen, "exact", tie)

    if all(not _disagree(candidates[0], s) for s in candidates[1:]):
        return Resolution(candidates[0], "exact", False)
    return Resolution(None, "ambiguous", False)


def account(
    index: SidecarIndex,
    claimed: dict[str, str],
    applied: set[Path],
    report: EnrichReport,
) -> None:
    """Partition every indexed sidecar into exactly one bucket.

    `claimed` maps a casefolded target to the resolution outcome; `applied`
    holds the paths of the sidecars actually WRITTEN onto a photo. `claimed`
    is keyed by target, not by photo, because `account()` only ever sees
    target-level groups - but a target's outcome must still be trusted as
    "ambiguous" if ANY photo sharing that name was refused, even if another
    photo sharing it resolved exact. 342 filenames are shared by distinct
    photos on the reference export even after content-hash dedup; a caller
    that lets a later `claimed[target] = "exact"` overwrite an earlier
    `"ambiguous"` erases the refusal here with nothing left to catch it -
    the caller MUST make that assignment sticky (see
    `tests/test_takeout_index.py`).

    Both non-orphan branches scan `applied` for the same reason: `matched`
    counts only sidecars that were actually WRITTEN, never merely claimed.
    An outcome of "exact" without a matching entry in `applied` is
    `superseded`, not `matched` - crediting `len(sidecars)` to `matched` for
    every claimed target, as the first draft of this plan did, inflated it
    by 3,358 on the reference export. An outcome of "ambiguous" is symmetric:
    if one of its sidecars WAS applied (to a different photo that shared the
    target name and legitimately resolved exact), that one is `matched`, and
    only the rest count as the refusal - crediting the whole group to
    `ambiguous` regardless of `applied`, as an earlier draft of THIS fix did,
    undercounts `matched` and breaks `matched == len(applied)`.
    """
    for target, sidecars in index.by_target.items():
        outcome = claimed.get(target)
        used = sum(1 for s in sidecars if s.path in applied)
        if outcome == "ambiguous":
            report.matched += used
            report.ambiguous += len(sidecars) - used
        elif outcome == "exact":
            report.matched += used
            report.superseded += len(sidecars) - used
        else:
            report.orphaned += len(sidecars)


# 20 timestamps in the reference export are shared by >=10 sidecars, covering
# 907 records - one value on 618 of them. No 618 photos fire in one second;
# these are Google's day- or year-granular guesses.
CLUSTER_MIN = 10

# After offset normalisation the -05:30 artefact is gone, so a tight bound now
# means genuine disagreement. A day-wide tolerance would hide exactly the
# broken-camera-clock cases (2016 EXIF vs 2020 Google) the flag exists for.
CONFLICT_TOLERANCE = timedelta(minutes=1)


def clustered_timestamps(index: SidecarIndex) -> frozenset[datetime]:
    """photoTakenTime values shared by CLUSTER_MIN or more sidecars.

    These are not 618 photos that fired in the same second - they are
    Google's day- or year-granular guesses, indistinguishable from a real
    timestamp except by how many sidecars repeat it verbatim.
    """
    counts: Counter[datetime] = Counter()
    for sidecars in index.by_target.values():
        for sidecar in sidecars:
            if sidecar.taken_at_utc is not None:
                counts[sidecar.taken_at_utc] += 1
    return frozenset(ts for ts, n in counts.items() if n >= CLUSTER_MIN)


def _has_real_date(meta: PhotoMeta) -> bool:
    """A filesystem mtime - or a prior Takeout guess - is not knowing when a
    photo was taken."""
    return meta.taken_at_utc is not None and meta.tz_source not in (
        TzSource.FILE_MTIME,
        TzSource.NONE,
        TzSource.TAKEOUT,
    )


def apply_sidecar(
    photo: Photo,
    sidecar: Sidecar,
    clustered: frozenset[datetime],
    report: EnrichReport,
    tz_lookup: TzLookup | None = None,
) -> None:
    """Write `sidecar` onto `photo`, in place, per spec section 5.

    Idempotent: running it twice with the same sidecar produces the same
    record. That is why `people` replaces rather than unions, why
    `exif_taken_at_utc` is written once and then left alone, and why
    `from_takeout` is always fed the PRISTINE EXIF instant/source - never the
    already-Takeout-derived state a first run left behind - by reconstructing
    `EXIF_NAIVE` from `exif_taken_at_utc` when `tz_source` now reads TAKEOUT.
    Skipping that reconstruction cannot re-fire rung 3 (`exif_naive -
    google_utc`) on a second run and silently regresses local time to UTC.
    """
    meta = photo.meta

    if sidecar.taken_at_utc is not None:
        suppress = sidecar.taken_at_utc in clustered and _has_real_date(meta)
        if suppress:
            report.clustered_dates_suppressed += 1
        else:
            # Was the EXIF value a naive WALL CLOCK stamped UTC, or a true
            # instant? Only the first needs its offset subtracted before it
            # can be compared. Getting this wrong invents a conflict on every
            # photo that already carried a correct OffsetTimeOriginal - 34%
            # of them.
            was_naive = meta.tz_source is TzSource.EXIF_NAIVE or (
                meta.tz_source is TzSource.TAKEOUT and meta.exif_taken_at_utc is not None
            )
            # Keep the displaced EXIF instant BEFORE overwriting it, and only
            # on the first enrich - on a re-run taken_at_utc is already
            # Google's, and copying that here would erase the real one.
            if meta.exif_taken_at_utc is None and _has_real_date(meta):
                meta.exif_taken_at_utc = meta.taken_at_utc

            utc, local, tz_source = from_takeout(
                google_utc=sidecar.taken_at_utc,
                existing_utc=meta.exif_taken_at_utc or meta.taken_at_utc,
                existing_local=meta.taken_at_local,
                existing_tz_source=(
                    TzSource.EXIF_NAIVE
                    if meta.tz_source is TzSource.TAKEOUT and meta.exif_taken_at_utc
                    else meta.tz_source
                ),
                gps=meta.gps or sidecar.gps,
                tz_lookup=tz_lookup,
            )
            if meta.taken_at_utc != utc:
                report.dates_corrected += 1
            meta.taken_at_utc = utc
            meta.taken_at_local = local
            meta.tz_source = tz_source

            # Instants, after normalisation. from_takeout has already folded
            # a pure timezone difference away, so anything left is real.
            if meta.exif_taken_at_utc is not None:
                normalised = meta.exif_taken_at_utc
                if was_naive and local is not None:
                    offset = local.utcoffset()
                    if offset is not None:
                        normalised = meta.exif_taken_at_utc - offset
                conflict = abs(normalised - utc) > CONFLICT_TOLERANCE
                if conflict and not photo.metadata_conflict:
                    report.conflicts += 1
                photo.metadata_conflict = photo.metadata_conflict or conflict

    # People: REPLACE the previous Takeout contribution, keep everything else.
    # A union can only grow, so a face tag corrected in Google Photos could
    # never be retracted.
    previous = set(meta.takeout_people)
    kept = [name for name in meta.people if name not in previous]
    before = len(meta.people)
    meta.people = list(dict.fromkeys([*kept, *sidecar.people]))
    meta.takeout_people = list(sidecar.people)
    if len(meta.people) > before:
        report.people_added += len(meta.people) - before

    if meta.gps is None and sidecar.gps is not None:
        meta.gps = sidecar.gps
        report.gps_added += 1

    if sidecar.description and (
        meta.description is None or len(sidecar.description) > len(meta.description)
    ):
        meta.description = sidecar.description
        report.descriptions_added += 1

    if sidecar.favorite and not meta.favorite:
        meta.favorite = True
        report.favourites_added += 1

    # Google's own flags. Archived means the user deliberately hid this photo.
    meta.archived = meta.archived or sidecar.archived
    meta.trashed = meta.trashed or sidecar.trashed

    photo.sidecar_match = "exact"


def album_renames(index: SidecarIndex, report: EnrichReport) -> dict[str, str]:
    """Folder name -> the album's real title, for titles that actually differ.

    A title identical to its folder is not a rename. A title that would
    collide with another album's name is REPORTED and skipped: merging two
    albums the user kept apart is not a decision this pass gets to make
    (spec section 11).

    `index.albums` is built by `build_index`, which already excludes a null,
    empty or whitespace-only title - but this function does not trust that as
    its ONLY line of defence. "Never fabricate a value" applies here too: an
    empty title is not a title, so it is skipped rather than ever becoming a
    rename target, even if a future caller populates `SidecarIndex.albums`
    directly (as this module's own platform-collision test does).
    """
    taken = {path.name for path in index.albums}
    renames: dict[str, str] = {}

    # Sort by a stable, platform-independent key. `sorted()` on Path objects
    # compares a case-folded form on Windows and a raw one on POSIX, so which
    # album wins a title collision would differ between a contributor's
    # machine and CI. M0 hit and fixed this exact bug in FolderSource.scan.
    def _order(item: tuple[Path, str]) -> tuple[str, str]:
        return str(item[0]).casefold(), str(item[0])

    for path, title in sorted(index.albums.items(), key=_order):
        folder = path.name
        if not title or title == folder:
            continue
        if title in taken or title in renames.values():
            report.album_title_collisions.append((folder, title))
            continue
        renames[folder] = title
    return renames


def retitle_albums(photo: Photo, renames: dict[str, str], report: EnrichReport) -> None:
    """Apply renames wherever the photo lives.

    Not scoped to the album's own directory: 8 album folders in the reference
    export hold sidecars and zero media, so a directory-scoped rule would
    never fire for them.
    """
    updated = [renames.get(album, album) for album in photo.albums]
    if updated != photo.albums:
        # strict=True: the lists are the same length by construction, and
        # bare zip() is `B905` under this project's ruff config.
        report.albums_retitled += sum(
            1 for a, b in zip(photo.albums, updated, strict=True) if a != b
        )
        photo.albums = updated
