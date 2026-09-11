"""Core data records. Source-agnostic: every source normalises into these."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path


class MediaType(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    UNKNOWN = "unknown"


class TzSource(StrEnum):
    """How local time was determined. Recorded so doctor can report it."""

    EXIF_OFFSET = "exif_offset"
    EXIF_NAIVE = "exif_naive"
    GPS = "gps"
    FILE_MTIME = "file_mtime"
    # Google's photoTakenTime supplied the INSTANT. It says nothing about the
    # zone: local time is derived separately (meta.timestamps.from_takeout).
    TAKEOUT = "takeout"
    NONE = "none"


@dataclass(frozen=True)
class Gps:
    lat: float
    lon: float
    alt: float | None = None


@dataclass(frozen=True)
class FaceRegion:
    """Normalised [0,1] centre and size, per the MWG regions convention."""

    name: str
    x: float
    y: float
    w: float
    h: float


@dataclass
class PhotoMeta:
    taken_at_utc: datetime | None = None
    taken_at_local: datetime | None = None
    tz_source: TzSource = TzSource.NONE
    gps: Gps | None = None
    people: list[str] = field(default_factory=list)
    face_regions: list[FaceRegion] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    description: str | None = None
    favorite: bool = False
    camera_make: str | None = None
    camera_model: str | None = None
    width: int | None = None
    height: int | None = None
    # The EXIF instant that Google's photoTakenTime displaced. Without this,
    # `metadata_conflict` is a flag with no payload: the spec promised the
    # record was "flagged rather than silently overwritten" while there was
    # exactly one date field to overwrite.
    exif_taken_at_utc: datetime | None = None
    # Exactly what the last Takeout enrich contributed to `people`. Needed so a
    # second enrich can RETRACT a face tag the user corrected in Google Photos.
    # A flat union can only ever grow.
    takeout_people: list[str] = field(default_factory=list)
    # Google's own flags. Archived means the user deliberately hid this photo;
    # it must never surface in a montage. 162 rows on the reference export.
    archived: bool = False
    # EXPECTED TO BE PERMANENTLY ZERO FOR TAKEOUT. All 12 `trashed: true`
    # sidecars on the reference export live under `Trash/`, which both
    # FolderSource and build_index exclude, so the photo never reaches the
    # store to be flagged. The field exists for sources that expose a
    # soft-delete flag WITHOUT segregating the files - Immich and Apple Photos
    # both do. If a Takeout run ever reports a non-zero count here, Google has
    # changed the export layout and the exclusion needs revisiting.
    trashed: bool = False
    # --- schema v3: perceptual fingerprint, for burst dedup ---
    # 64-bit dHash of the decoded pixels. None means "not computed": either
    # `rekindle fingerprint` has never run, or it ran and failed (see
    # phash_error). Never guess a value - dedup treats unknown as NOT similar.
    phash: int | None = None
    # Relative focus measure; see memory.fingerprint.sharpness for what it is
    # and is not. Comparable WITHIN a burst from one camera, meaningless as an
    # absolute quality score across cameras.
    sharpness: float | None = None
    # Why phash is None, so a re-run does not redo a known failure. "video"
    # (never decoded by design), "unreadable", "undecodable". None means it
    # was simply never attempted - which is what distinguishes "not indexed
    # yet" from "tried and could not", exactly as `enriched_at` does for
    # enrichment.
    phash_error: str | None = None
    # Mean luminance 0-255 of the decoded image. Drives the near-black and
    # blown-out quality gates in memory.composition.
    brightness: float | None = None

    @classmethod
    def empty(cls) -> PhotoMeta:
        return cls()


@dataclass
class Photo:
    file_hash: str
    paths: list[Path]
    media_type: MediaType
    meta: PhotoMeta
    first_seen: datetime
    last_seen: datetime
    albums: list[str] = field(default_factory=list)
    edited_of: str | None = None
    # Which Source produced this record. Load-bearing for later sources
    # (Takeout, Immich, Apple Photos) enriching existing rows in place -
    # and impossible to add later without a migration we don't have.
    source: str = "folder"
    # Set when two sightings of the same bytes disagree on a real capture date
    # or carry different non-empty descriptions.
    metadata_conflict: bool = False
    # "exact"     - a sidecar was resolved for this photo
    # "ambiguous" - candidates disagreed and enrichment was REFUSED
    # "inherited" - this photo had no sidecar of its own; its enrichment was
    #               copied from an -edited original or a motion-photo still
    #               (rekindle.enrich.takeout.propagate_to_derivatives). Never
    #               "exact" - it did not itself match a sidecar.
    # "none"      - no sidecar, or enrich has never run. Distinguish the two by
    #               the `enriched_at` key in the meta table, not by this field.
    sidecar_match: str = "none"


def merge_meta(old: PhotoMeta, new: PhotoMeta) -> tuple[PhotoMeta, bool]:
    """Field-by-field merge per spec 4.2. Returns (merged, conflict).

    Lives here, not in the store, because both the store (re-index) and the
    folder source (the same bytes seen in two folders) need it.

    A wholesale `new if new.taken_at_utc else old` swap would be WRONG: the
    folder source always sets taken_at_utc via its mtime fallback, so `new`
    would always win and every re-index would destroy prior enrichment.
    """

    def _real(m: PhotoMeta) -> bool:
        return m.taken_at_utc is not None and m.tz_source is not TzSource.FILE_MTIME

    def _enriched(m: PhotoMeta) -> bool:
        """Did a Takeout enrich pass write this date?

        Enrichment is Google's OWN record and outranks a folder scan; the
        earliest-wins tiebreak below is between sources of EQUAL authority
        and must never be handed a date correction to overrule (see
        `PhotoStore.update_photo`'s docstring). Without this, `index ->
        enrich -> index` reverts every photo whose camera clock ran EARLY -
        108 of them on the reference export, 100 by more than a day.

        `tz_source is TAKEOUT` alone is NOT sufficient: an enriched row's
        zone may still come from an EXIF offset or GPS (see
        `meta.timestamps.from_takeout`), leaving tz_source EXIF_OFFSET or
        GPS. `exif_taken_at_utc` alone is not sufficient either: only
        `enrich` ever writes it (enrich/takeout.py) but it stays None for a
        photo that had no real date to displace, and THAT photo's tz_source
        IS takeout. Verified by grepping every write site of both fields;
        the union covers all three shapes and nothing else sets either.
        """
        return m.exif_taken_at_utc is not None or m.tz_source is TzSource.TAKEOUT

    def _exif_instant(m: PhotoMeta) -> datetime | None:
        """The EXIF instant `m` still carries: the one enrichment displaced
        when it has one, otherwise its own date."""
        return m.exif_taken_at_utc or m.taken_at_utc

    conflict = False
    if _real(old) and _real(new):
        # An enriched date wins outright rather than entering the tiebreak.
        # The conflict question is then a DIFFERENT one: has the unenriched
        # side's EXIF instant moved away from the one enrichment already
        # arbitrated? Asking the crude question (`old.taken_at_utc !=
        # new.taken_at_utc`, exact, no offset normalisation) instead flags
        # every photo whose local zone is not UTC - 10,062 rows on the
        # reference export, which `doctor` would print as "EXIF/Google date
        # conflicts" until the next enrich recomputed them away.
        if _enriched(old) and not _enriched(new):
            keep = old
            conflict = _exif_instant(old) != new.taken_at_utc
        elif _enriched(new) and not _enriched(old):
            # Unreachable through any call site shipped today: `merge_meta` is
            # called as `(stored, incoming-from-a-Source)` in `db._merge` and
            # as `(existing, new)` in `FolderSource.scan`, so `new` is always
            # folder-scanned and no shipped `Source` can make `_enriched(new)`
            # true. Kept for the symmetry a future enriching `Source` would
            # need - see `tests/test_models.py` for a direct unit test that
            # calls this branch without going through either call site.
            keep = new
            conflict = _exif_instant(new) != old.taken_at_utc
        else:
            conflict = old.taken_at_utc != new.taken_at_utc
            keep = old if old.taken_at_utc <= new.taken_at_utc else new  # earliest wins
    elif _real(old):
        keep = old
    elif _real(new):
        keep = new
    else:
        keep = new

    descs = [d for d in (old.description, new.description) if d]
    if len(descs) == 2 and descs[0] != descs[1]:
        conflict = True

    merged = PhotoMeta(
        taken_at_utc=keep.taken_at_utc,
        taken_at_local=keep.taken_at_local,
        tz_source=keep.tz_source,
        gps=old.gps or new.gps,
        people=list(dict.fromkeys([*old.people, *new.people])),
        face_regions=list(dict.fromkeys([*old.face_regions, *new.face_regions])),
        keywords=list(dict.fromkeys([*old.keywords, *new.keywords])),
        description=max(descs, key=len) if descs else None,
        favorite=old.favorite or new.favorite,
        camera_make=old.camera_make or new.camera_make,
        camera_model=old.camera_model or new.camera_model,
        width=old.width or new.width,
        height=old.height or new.height,
        # A re-index must never destroy enrichment. `new` here is the folder
        # source, which knows none of these, so `old` wins on all four.
        exif_taken_at_utc=old.exif_taken_at_utc or new.exif_taken_at_utc,
        takeout_people=list(old.takeout_people or new.takeout_people),
        archived=old.archived or new.archived,
        trashed=old.trashed or new.trashed,
        # A re-index must never destroy a fingerprint. `new` is a folder scan
        # and never computes one, so a wholesale `new.phash` would silently
        # throw away a six-minute pass on every `rekindle index`. This is the
        # same trap that destroyed enrichment before `exif_taken_at_utc` was
        # added to this list - a field not named here is deleted by the next
        # index run.
        phash=old.phash if old.phash is not None else new.phash,
        sharpness=old.sharpness if old.sharpness is not None else new.sharpness,
        phash_error=old.phash_error or new.phash_error,
        brightness=old.brightness if old.brightness is not None else new.brightness,
    )
    return merged, conflict


@dataclass
class SourceReport:
    """What a source found. Feeds `rekindle doctor`.

    Sources must count everything they skip. Silent drops are a bug.
    """

    files_seen: int = 0
    media_indexed: int = 0
    duplicates_merged: int = 0
    edited_linked: int = 0
    json_sidecars: int = 0
    orphan_sidecars: int = 0
    motion_pairs: int = 0
    excluded_dirs: int = 0
    with_date: int = 0
    with_gps: int = 0
    with_people: int = 0
    with_xmp: int = 0
    # Files that were INDEXED but could not be decoded. Deliberately not a
    # `skipped` reason: they are not skipped, and filing them there made
    # files_seen stop reconciling against the other buckets. Counted per
    # photo (hash), not per path, so a corrupt file present in two folders
    # counts once - `unreadable` below still names both paths.
    undecodable: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    unreadable: list[tuple[Path, str]] = field(default_factory=list)
    long_paths: list[Path] = field(default_factory=list)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    @property
    def total_skipped(self) -> int:
        return sum(self.skipped.values())
