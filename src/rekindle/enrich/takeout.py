"""Google Takeout JSON sidecar enrichment.

Deliberately NOT a `Source`: that protocol returns (list[Photo], SourceReport)
for something producing records from pixels. This updates existing records
addressed by content hash. It does inherit the protocol's real contract -
count everything - which `EnrichReport` enforces in Task 6.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

from rekindle.db import PhotoStore
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
    if "latitude" not in block or "longitude" not in block:
        return None
    try:
        lat = float(block["latitude"])
        lon = float(block["longitude"])
    except (TypeError, ValueError):
        return None
    if abs(lat) < 1e-9 and abs(lon) < 1e-9:
        return None
    # Altitude is genuinely optional, and is treated as optional: ABSENT means
    # None, never 0.0 - sea level is a real reading this library must be able
    # to tell apart from "not recorded". Parsed in its own `try` so a
    # malformed altitude cannot reject a perfectly good lat/lon pair; the same
    # field is handled exactly this way in `meta.exif`.
    alt: float | None = None
    raw_alt = block.get("altitude")
    if raw_alt is not None:
        try:
            alt = float(raw_alt)
        except (TypeError, ValueError):
            alt = None
        else:
            alt = alt if math.isfinite(alt) else None
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
    # Photos refused because a DIFFERENT, content-distinct photo already
    # claimed the identical sidecar - 342 filenames are shared by distinct
    # photos even after content-hash dedup, and resolve() (handed one photo
    # at a time) cannot see that its single-global-candidate match already
    # belongs elsewhere. Silent, this looks exactly like "no misattribution
    # ever happened" rather than "3 were caught and refused" - the same
    # silence `directory_preference_broke_a_tie` exists to prevent.
    cross_photo_collisions: int = 0
    derivatives_enriched: int = 0
    people_added: int = 0
    dates_corrected: int = 0
    gps_added: int = 0
    descriptions_added: int = 0
    favourites_added: int = 0
    albums_retitled: int = 0
    conflicts: int = 0
    # Rows whose EXIF/Google date conflict this run CLEARED, because Google's
    # record no longer disagrees. Retraction is a real change to the record;
    # unreported it would be exactly the silent write `derivatives_enriched`
    # was added to stop.
    conflicts_retracted: int = 0
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

    THE SUFFIX TEST IS CASE-INSENSITIVE, and `rglob("*.json")` is not - on
    POSIX, where the pattern is matched case-sensitively, a `.JSON` sidecar
    was simply invisible here. `sources/folder.py` has always compared
    `suffix.casefold()`, so `doctor` counted such a file as a sidecar and this
    pass did not read it: the two halves of the tool disagreeing about the
    same file, on Linux only, which is the kind of split nobody finds by
    running it on Windows.
    """
    index = SidecarIndex()
    for path in sorted(p for p in root.rglob("*") if p.suffix.casefold() == ".json"):
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
            # `photo.metadata_conflict` is one boolean for two causes (date
            # AND description, see docs/known-limitations.md) - a re-index
            # between two enrich runs can OR a DESCRIPTION conflict onto it
            # via `models.merge_meta` without touching any of the four fields
            # read below. Gating `conflicts`/`conflicts_retracted` on that
            # combined flag credits or blames the DATE row for a change that
            # was never about a date. Re-deriving what THIS check's own
            # verdict was last time - from the same fields it is about to
            # overwrite, snapshotted before that happens - gates both
            # counters on the date cause alone, with no new stored field:
            # `apply_sidecar` is idempotent (see docstring), so replaying its
            # own formula against the untouched prior state reproduces
            # exactly what it last concluded.
            previous_exif = meta.exif_taken_at_utc
            previous_utc = meta.taken_at_utc
            previous_local = meta.taken_at_local
            previous_conflict = False
            if previous_exif is not None and previous_utc is not None:
                previous_normalised = previous_exif
                if was_naive and previous_local is not None:
                    previous_offset = previous_local.utcoffset()
                    if previous_offset is not None:
                        previous_normalised = previous_exif - previous_offset
                previous_conflict = abs(previous_normalised - previous_utc) > CONFLICT_TOLERANCE

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
                if conflict and not previous_conflict:
                    report.conflicts += 1
                elif previous_conflict and not conflict:
                    report.conflicts_retracted += 1
                # ASSIGNED, not OR-ed. `or` is monotonic, so a date the user
                # later corrects in Google Photos stays flagged forever - and
                # a corrected date is an obvious, realistic retraction event,
                # unlike `favorite`/`archived`/`trashed`, which Takeout cannot
                # encode an un-set for. This branch is the only place the
                # EXIF-vs-Google comparison is made, and it only runs once
                # `exif_taken_at_utc` exists, so enrichment takes the flag
                # over exactly when it is in a position to judge it; a photo
                # it never dated keeps whatever the folder source stored.
                # Residual, recorded in docs/known-limitations.md: the flag is
                # one boolean for two causes, so a DESCRIPTION conflict
                # `merge_meta` set on a photo enrich also dates is cleared
                # here along with the date verdict.
                photo.metadata_conflict = conflict

    # People: REPLACE the previous Takeout contribution, keep everything else.
    # A union can only grow, so a face tag corrected in Google Photos could
    # never be retracted.
    previous = set(meta.takeout_people)
    kept = [name for name in meta.people if name not in previous]
    before = set(meta.people)
    meta.people = list(dict.fromkeys([*kept, *sidecar.people]))
    meta.takeout_people = list(sidecar.people)
    # Count the names actually ADDED, not the change in list LENGTH. A length
    # guard (`if len(people) > before`) reports 0 new whenever a retraction
    # and an addition land in the same run: Google drops "Ada", adds "Grace",
    # the list is still one name long and a real new face tag goes uncredited.
    report.people_added += len(set(meta.people) - before)

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


# "exact" and "ambiguous" are EARNED: the photo's own sidecar resolution ran
# to a decision, successful or deliberately refused. Neither is touched here.
# "none" (never resolved) and "inherited" (from a prior propagation) are NOT
# earned, so both stay eligible below - "inherited" must be reprocessed, not
# skipped, or a derivative whose donor is later corrected (a retracted face
# tag, a corrected date) is frozen with stale data forever. Reprocessing is
# safe: same donor, same result, whenever the donor hasn't actually changed.
_EARNED_STATES = frozenset({"exact", "ambiguous"})
# A donor's data is only trustworthy to copy onward once it carries real
# enrichment of its own, earned or (transitively) inherited.
_DONOR_STATES = frozenset({"exact", "inherited"})


def _meta_snapshot(meta: PhotoMeta) -> tuple[object, ...]:
    """Every field this function can touch, as a hashable-free tuple, so a
    reprocessed row can tell whether its donor actually changed anything."""
    return (
        meta.taken_at_utc,
        meta.taken_at_local,
        meta.tz_source,
        meta.exif_taken_at_utc,
        tuple(meta.people),
        tuple(meta.takeout_people),
        meta.gps,
        meta.description,
        meta.favorite,
        meta.archived,
        meta.trashed,
    )


def propagate_to_derivatives(photos: list[Photo], report: EnrichReport) -> list[Photo]:
    """Copy enrichment onto photos Google gives no sidecar.

    Two links, both already established by M0's folder source:
      - `-edited` variants, persisted as Photo.edited_of
      - motion-photo halves, NOT persisted (FolderSource only counts them), so
        re-derived here from the naming rule: the still is the video's full
        NAME plus ".jpg" - PXL_1.MP pairs with PXL_1.MP.jpg - scoped to one
        directory, because a .MP in one album must never pair with a
        same-named still in another. Matched case-insensitively throughout:
        a real export routinely mixes .MP/.mp and .jpg/.JPG, and this exact
        naming rule is where M0's original pairing bug (0 pairs on real data)
        lived.
    """
    by_hash = {p.file_hash: p for p in photos}
    by_dir_name: dict[tuple[Path, str], Photo] = {}
    for photo in photos:
        for path in photo.paths:
            by_dir_name.setdefault((path.parent, path.name.casefold()), photo)

    changed: list[Photo] = []
    for photo in photos:
        if photo.sidecar_match in _EARNED_STATES:
            continue
        donor: Photo | None = None
        via_edited = False
        if photo.edited_of:
            donor = by_hash.get(photo.edited_of)
            via_edited = donor is not None
        if donor is None:
            for path in photo.paths:
                if path.suffix.casefold() != ".mp":
                    continue
                donor = by_dir_name.get((path.parent, f"{path.name}.jpg".casefold()))
                if donor is not None:
                    break
        if donor is None or donor.sidecar_match not in _DONOR_STATES:
            continue

        src = donor.meta
        dst = photo.meta
        before = _meta_snapshot(dst)
        before_date = dst.taken_at_utc
        before_people = set(dst.people)
        before_gps = dst.gps
        before_description = dst.description
        before_favorite = dst.favorite

        # GUARDED, like every other field below it. The donor's date wins when
        # there IS one - that is the point of the pass, since an `-edited`
        # file's own EXIF often carries the moment of the EDIT rather than the
        # moment of the photograph. A donor with no date has nothing to lend.
        #
        # Unconditional, this line DESTROYED data. `apply_sidecar` stamps
        # `sidecar_match = "exact"` on any photograph whose sidecar was found,
        # including one whose sidecar carried no parseable `photoTakenTime`,
        # so such a donor reached here and nulled a derivative holding a
        # perfectly good EXIF date. A dateless photograph is then refused by
        # `ExclusionPolicy.deny_reason` as `no_date` and vanishes from every
        # memory - and the run reported it as "Dates corrected: 1".
        if src.taken_at_utc is not None:
            dst.taken_at_utc = src.taken_at_utc
            dst.taken_at_local = src.taken_at_local
            # NOT src.tz_source. This file has no EXIF of its own, so copying
            # `exif_offset` onto 655 .MP videos makes doctor report a
            # provenance that cannot exist. The instant and the wall clock are
            # copied intact - only the claim about where they came from
            # changes, and for this row the answer is the Takeout pass, via
            # its sibling. Inside the guard with them: a row that inherited no
            # date must not be relabelled as having inherited one.
            dst.tz_source = TzSource.TAKEOUT
        # Same fabrication risk, milder: a .MP video has no EXIF at all, not
        # even a displaced one, so it never inherits `exif_taken_at_utc`. An
        # `-edited` variant IS the same shot re-encoded, so it plausibly
        # shares its original's EXIF lineage - that copy stays conditional
        # only on "don't clobber a value this row already has".
        if via_edited:
            dst.exif_taken_at_utc = dst.exif_taken_at_utc or src.exif_taken_at_utc
        previous = set(dst.takeout_people)
        kept = [name for name in dst.people if name not in previous]
        dst.people = list(dict.fromkeys([*kept, *src.takeout_people]))
        dst.takeout_people = list(src.takeout_people)
        if dst.gps is None:
            dst.gps = src.gps
        if src.description and (
            dst.description is None or len(src.description) > len(dst.description)
        ):
            dst.description = src.description
        dst.favorite = dst.favorite or src.favorite
        dst.archived = dst.archived or src.archived
        dst.trashed = dst.trashed or src.trashed
        # A fourth state, distinct from "exact": this photo has no sidecar of
        # its own and doctor should not claim it does. Written every time
        # this row is (re)processed, including when nothing below changed.
        photo.sidecar_match = "inherited"

        # Credit the SAME counters `apply_sidecar` credits. This path writes
        # the same fields; crediting only `derivatives_enriched` (a per-PHOTO
        # count) made "GPS added" print 38 against 337 rows actually written
        # on the reference export - an 8.9x understatement the user can catch
        # with `doctor --from-index`'s own `with_gps` delta. Every increment
        # is conditional on an observed change, so a re-run that changes
        # nothing credits nothing and the totals stay idempotent.
        if dst.taken_at_utc != before_date:
            report.dates_corrected += 1
        report.people_added += len(set(dst.people) - before_people)
        if dst.gps is not None and before_gps is None:
            report.gps_added += 1
        if dst.description != before_description:
            report.descriptions_added += 1
        if dst.favorite and not before_favorite:
            report.favourites_added += 1

        if _meta_snapshot(dst) != before:
            report.derivatives_enriched += 1
            changed.append(photo)
    return changed


def retitle_albums(photo: Photo, renames: dict[str, str], report: EnrichReport) -> None:
    """Apply renames wherever the photo lives.

    Not scoped to the album's own directory: 8 album folders in the reference
    export hold sidecars and zero media, so a directory-scoped rule would
    never fire for them.
    """
    updated = [renames.get(album, album) for album in photo.albums]
    if updated != photo.albums:
        # Counted BEFORE the dedup below, and positionally: `updated` is the
        # same length as `photo.albums` by construction, so strict=True holds
        # (bare zip() is `B905` under this project's ruff config).
        report.albums_retitled += sum(
            1 for a, b in zip(photo.albums, updated, strict=True) if a != b
        )
        # Deduped, order-preserving. A re-index unions the already-retitled
        # title with the raw folder name the scan re-derives; retitling then
        # maps the folder name onto the title that is ALREADY there, and a
        # plain list comprehension leaves `['Our Coast Trip', 'Our Coast
        # Trip']` stored forever. 0 such rows today because only one
        # index+enrich has ever run against the reference export - the defect
        # is latent until the first re-index, which doctor's own orphan
        # warning tells users to perform.
        photo.albums = list(dict.fromkeys(updated))


_REFUSED = Resolution(None, "ambiguous", False)


def _resolve_without_cross_photo_collisions(
    photos: list[Photo], index: SidecarIndex, report: EnrichReport
) -> list[tuple[Photo, Resolution]]:
    """`resolve()` per photo, then correct for what it cannot see on its own.

    A sidecar describes exactly one real photo. 342 filenames are shared by
    distinct photos even after content-hash dedup (see
    `tests/fixtures/takeout.py`); when only one of those photos has a
    sidecar of its own, `resolve()`'s single-global-candidate fallback (rule
    2) has nothing to disagree with, so it hands that SAME sidecar to every
    other same-named photo too - a real misattribution (a face tag and
    capture time written onto a photo they were never about), not merely an
    accounting gap. First found by running the enricher against the
    reference export: `matched` read 3 lower than a naive per-photo tally of
    "resolve() said exact" (confirmed live: Photo0288.jpg, Photo0361.jpg and
    Photo0365.jpg each exist once under "Photos from 2011", with a sidecar,
    and once under "Photos from 2012", without one).

    `resolve()` cannot see this on its own - it is handed one photo at a
    time. Only here, with every photo's resolution in hand, can the
    collision even be detected. It is broken by preferring whichever photo
    actually lives in the sidecar's own directory UNDER THE SIDECAR'S OWN
    NAME (rule 1, `resolve()`'s reliable branch). Both conditions must hold
    on the SAME path: checking the directory alone is not enough, because a
    multi-path Photo can have one path sharing the sidecar's directory
    (under a different name) and another sharing its name (in a different
    directory) without either path actually being the file the sidecar
    describes. This is not hypothetical - 5 photos in the reference export
    have paths with differing casefolded filenames (e.g.
    `Dida\\IMG_20170927_184011(1).jpg` + `Dida(1)\\IMG_20170927_184011.jpg`).
    With both conditions on one path, at most one candidate can ever
    qualify: two different files cannot share one name in one directory.

    When NO claimant qualifies, there is zero evidence favouring any one of
    them over the others - picking one anyway (even deterministically)
    would fabricate an attribution, which the spec's "never fabricate a
    value" forbids just as much as picking one at random would. Every
    claimant in that case is demoted, not just all-but-one.

    Whoever loses is demoted to "ambiguous", exactly as if `resolve()` had
    refused it directly, and counted in `report.cross_photo_collisions` -
    silent, this looks identical to "no misattribution was ever caught".
    """
    resolved: list[tuple[Photo, Resolution]] = [(photo, resolve(photo, index)) for photo in photos]

    claims: dict[Path, list[int]] = {}
    for i, (_photo, res) in enumerate(resolved):
        if res.sidecar is not None:
            claims.setdefault(res.sidecar.path, []).append(i)

    demoted: set[int] = set()
    for sidecar_path, indices in claims.items():
        if len(indices) < 2:
            continue
        reliable = [
            i
            for i in indices
            if any(
                p.parent == sidecar_path.parent
                and p.name.casefold() == resolved[i][1].sidecar.target_cf
                for p in resolved[i][0].paths
            )
        ]
        # Provably at most one entry can ever reach `reliable` (see
        # docstring) - `min()` here is defensive determinism against a
        # future change to that proof, not a tiebreak this branch should
        # ever actually need. Deterministic on `file_hash` rather than list
        # position: `iter_photos()` gives no `ORDER BY` guarantee, and
        # `INSERT OR REPLACE` churns rowids across runs, so `reliable[0]`
        # would not be reproducible run to run.
        winner = min(reliable, key=lambda i: resolved[i][0].file_hash) if reliable else None
        demoted.update(i for i in indices if i != winner)

    report.cross_photo_collisions += len(demoted)

    return [
        (photo, _REFUSED) if i in demoted else (photo, res)
        for i, (photo, res) in enumerate(resolved)
    ]


class EmptyIndexError(RuntimeError):
    """`enrich` ran before `index`. Enriching nothing is not a success."""


class TakeoutEnricher:
    """Second pass over an already-indexed library.

    Not a `Source`, deliberately - see the module docstring. It does honour
    the contract that protocol existed to enforce: every JSON file lands in
    exactly one counter, checked by EnrichReport's two identities.

    Opens no media file. All timing information it needs was stored by M0, so
    a re-run after extracting another archive part costs a JSON walk, not a
    re-hash of 45,900 files.
    """

    name = "takeout"

    def __init__(self, tz_lookup: TzLookup | None = None) -> None:
        self._tz_lookup = tz_lookup

    def enrich(self, root: Path, store: PhotoStore) -> EnrichReport:
        if store.count() == 0:
            raise EmptyIndexError(
                f"The index is empty. Run `rekindle index {root}` before `rekindle enrich`."
            )

        report = EnrichReport()
        index = build_index(root, report)
        clustered = clustered_timestamps(index)
        renames = album_renames(index, report)

        # Materialised, not streamed: `PhotoStore.iter_photos` yields from a
        # live cursor, and this loop writes back through the same connection
        # via `update_many_with_meta` below - the exact hazard
        # `iter_photos`'s own docstring warns callers off.
        # Sorted, not merely materialised: `iter_photos()` gives no ORDER BY
        # guarantee and `INSERT OR REPLACE` churns rowids, so run-to-run order
        # is arbitrary. Two things below depend on order - the sticky
        # "ambiguous" claim a few lines down (a refusal must be recorded
        # before another photo sharing the target name resolves cleanly) and
        # `propagate_to_derivatives`'s `by_dir_name.setdefault` - so an
        # arbitrary order makes both unreproducible, and makes the mutation
        # that proves the sticky guard alive pass or fail by luck. Keyed on
        # the lexicographically first path, with `file_hash` to break ties,
        # because `paths` is a set-like union with no stable order of its own.
        photos = sorted(
            store.iter_photos(),
            key=lambda p: (min(str(x).casefold() for x in p.paths), p.file_hash),
        )
        claimed: dict[str, str] = {}
        applied: set[Path] = set()
        touched: dict[str, Photo] = {}

        for photo, res in _resolve_without_cross_photo_collisions(photos, index, report):
            sidecar, match, tie = res
            # Keyed on EVERY one of the photo's paths, not just paths[0]:
            # `PhotoStore`'s dedup unions paths by content hash with no
            # guarantee they share a filename, and `resolve()` itself already
            # pooled candidates from every path name into one decision. A
            # target reachable only through a non-first path must still be
            # markable here, or its (possibly applied) sidecar falls through
            # to `orphaned` in `account()` despite being written.
            for path in photo.paths:
                key = path.name.casefold()
                # "ambiguous" must not be downgraded by a later photo that
                # happens to resolve cleanly against the same key.
                if key in index.by_target and match != "none" and claimed.get(key) != "ambiguous":
                    claimed[key] = match
            if match == "ambiguous":
                photo.sidecar_match = "ambiguous"
                touched[photo.file_hash] = photo
                continue
            if sidecar is None:
                continue
            apply_sidecar(photo, sidecar, clustered, report, tz_lookup=self._tz_lookup)
            retitle_albums(photo, renames, report)
            applied.add(sidecar.path)
            report.photos_enriched += 1
            if tie:
                report.directory_preference_broke_a_tie += 1
            touched[photo.file_hash] = photo

        for photo in propagate_to_derivatives(photos, report):
            retitle_albums(photo, renames, report)
            touched[photo.file_hash] = photo

        account(index, claimed, applied, report)

        # One transaction for the WHOLE run, photos and provenance together:
        # a crash between a photo batch commit and a separate `enrich_at`
        # commit would leave rows enriched with no way to tell "ran and
        # found nothing" apart from "never ran" - exactly what guard 3
        # exists to prevent.
        store.update_many_with_meta(
            touched.values(),
            {"enrich_root": str(root), "enriched_at": datetime.now(UTC).isoformat()},
        )
        return report
