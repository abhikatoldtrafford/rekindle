"""Reads a plain directory tree. The only source in v1."""

from __future__ import annotations

import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

from rekindle.identity import file_hash, is_long_path, sniff
from rekindle.meta.exif import read_exif
from rekindle.meta.timestamps import TzLookup, resolve
from rekindle.meta.xmp import find_sidecar, read_xmp
from rekindle.models import (
    MediaType,
    Photo,
    PhotoMeta,
    SourceReport,
    TzSource,
    merge_meta,
)

# Google localises the edited suffix, so this is a list, not a single string.
# It also emits several derived-image suffixes beyond plain edits.
EDITED_SUFFIXES: tuple[str, ...] = (
    "-edited",
    "-bearbeitet",
    "-modifié",
    "-editado",
    "-modificato",
    "-bewerkt",
    "-redigerad",
    "-effects",
    "-collage",
    "-animation",
    "-mix",
    "-pano",
    "-smile",
)

_SIDECAR_EXTS = {".json", ".xmp", ".aae", ".thm"}

# Never index deleted photos. Verified present in a real Takeout export as
# "Trash"; localised in other locales.
EXCLUDED_DIRS = {
    "trash",
    "bin",
    "papierkorb",
    "corbeille",
    "papelera",
    "cestino",
    "prullenbak",
    "papperskorg",
    ".thumbnails",
    "@eadir",
}


class FolderSource:
    name = "folder"

    def __init__(self, tz_lookup: TzLookup | None = None) -> None:
        self._tz_lookup = tz_lookup

    def scan(self, root: Path) -> tuple[list[Photo], SourceReport]:
        report = SourceReport()
        by_hash: dict[str, Photo] = {}
        stem_index: dict[tuple[Path, str], str] = {}
        edited_pending: list[tuple[str, Path, str]] = []

        media_names: set[str] = set()
        sidecar_stems: list[str] = []
        saw_xmp: set[str] = set()

        # Explicit, platform-stable key. Path.__lt__ case-folds on Windows and
        # does not on POSIX, so bare sorted() visits files in a different order
        # per OS - which decides which copy of a duplicate becomes canonical.
        for path in sorted(
            root.rglob("*"),
            key=lambda p: tuple(part.casefold() for part in p.relative_to(root).parts),
        ):
            if not path.is_file():
                # is_file() swallows a stat() failure and returns False
                # exactly as it would for an ordinary directory - so a path
                # beyond Windows' MAX_PATH is otherwise simply invisible:
                # never counted, never reported. is_dir() is checked too so
                # a normal (accessible) directory - which correctly returns
                # False from is_file() - is not miscounted as a file.
                if not path.is_dir() and is_long_path(path):
                    report.files_seen += 1
                    report.long_paths.append(path)
                    report.skip("long_path_unreadable")
                continue

            report.files_seen += 1

            # Deleted photos must never become memories.
            if any(part.lower() in EXCLUDED_DIRS for part in path.relative_to(root).parts[:-1]):
                report.excluded_dirs += 1
                continue

            if is_long_path(path):
                report.long_paths.append(path)

            if path.suffix.lower() in _SIDECAR_EXTS:
                # .xmp files ARE read (find_sidecar/read_xmp). Counting them as
                # "ignored" made doctor warn that data was being discarded three
                # rows below reporting it as successfully read.
                if path.suffix.lower() == ".xmp":
                    continue
                # Only Google's own .json sidecars feed the orphan check.
                # .aae (Apple edit sidecars) and .thm (video thumbnails) are
                # still skipped as non-media, but _sidecar_target only
                # understands the Google JSON naming convention - feeding an
                # .AAE through it returns the filename unchanged, which then
                # never matches anything in media_names and is miscounted as
                # a permanently orphaned sidecar. An iPhone library emits one
                # .AAE per edited photo, so this would inflate doctor's
                # "missing archive parts" warning on a perfectly complete
                # library.
                if path.suffix.lower() != ".json":
                    continue
                report.json_sidecars += 1
                if path.name != "metadata.json":
                    sidecar_stems.append(_sidecar_target(path.name))
                continue

            media_type, _fmt = sniff(path)
            if media_type is MediaType.UNKNOWN:
                report.skip("not_media")
                continue

            try:
                digest = file_hash(path)
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            except OSError as exc:
                report.unreadable.append((path, str(exc)))
                report.skip("unreadable")
                continue

            media_names.add(path.name)
            album = path.parent.name if path.parent != root else None

            # Register stem BEFORE the duplicate check. A photo that appears in
            # both a year folder and an album folder is merged on second sight,
            # but its edited variant lives beside one specific copy and must
            # still find it. Skipping this for duplicates silently breaks
            # edit-linking for exactly the Takeout layout we care about.
            #
            # RULING 1: both sides of this index must share one Unicode normal
            # form. macOS hands back NFD-decomposed filenames from the
            # filesystem; _split_edited normalises to NFC before matching, so
            # the key registered here must be NFC too, or an edited variant's
            # computed base can never match its original's raw NFD stem.
            base_stem, suffix = _split_edited(path.stem)
            if suffix is None:
                stem_key = (path.parent, unicodedata.normalize("NFC", path.stem))
                # Google's motion photos give a .jpg still and a .MP video the
                # SAME stem. Whichever is visited first would otherwise win
                # unconditionally, so on a different sort order the video's
                # hash could register here and an edited variant of the still
                # would silently link to the wrong original. The still always
                # wins, regardless of visit order.
                if media_type is not MediaType.VIDEO or stem_key not in stem_index:
                    stem_index[stem_key] = digest
            else:
                edited_pending.append((digest, path.parent, base_stem))

            has_xmp = find_sidecar(path) is not None

            if digest in by_hash:
                existing = by_hash[digest]
                existing.paths.append(path)
                if album and album not in existing.albums:
                    existing.albums.append(album)
                existing.first_seen = min(existing.first_seen, mtime)
                existing.last_seen = max(existing.last_seen, mtime)
                # The second copy must still be READ. In Takeout the sidecar
                # frequently sits beside only one of the two copies, so
                # skipping this silently loses the metadata half of dedupe.
                merged, conflict = merge_meta(
                    existing.meta, self._read_meta(path, media_type, mtime, report)
                )
                existing.meta = merged
                existing.metadata_conflict = existing.metadata_conflict or conflict
                if has_xmp and digest not in saw_xmp:
                    saw_xmp.add(digest)
                    report.with_xmp += 1
                report.duplicates_merged += 1
                continue

            if has_xmp:
                saw_xmp.add(digest)
                report.with_xmp += 1

            meta = self._read_meta(path, media_type, mtime, report)
            by_hash[digest] = Photo(
                file_hash=digest,
                paths=[path],
                media_type=media_type,
                meta=meta,
                first_seen=mtime,
                last_seen=mtime,
                albums=[album] if album else [],
                # RULING 2: `source` exists precisely so later sources
                # (Takeout, Immich, Apple Photos) can be told apart. Relying
                # on the dataclass default is not a declaration.
                source=self.name,
            )

        # edited_pending holds one entry per PATH, but only one Photo exists per
        # hash. An edited variant living in both a year folder and an album -
        # entirely ordinary in Takeout - would otherwise be counted twice, and
        # doctor would report more edits linked than photos exist.
        for digest, parent, base_stem in edited_pending:
            photo = by_hash.get(digest)
            if photo is None or photo.edited_of is not None:
                continue
            original = stem_index.get((parent, base_stem))
            if original and original != digest:
                photo.edited_of = original
                report.edited_linked += 1

        # A sidecar with no media file means that photo is in an archive part
        # the user has not extracted. Measured at 44% on a real partial export -
        # by far the most common way an index silently comes out half-empty.
        report.orphan_sidecars = sum(1 for s in sidecar_stems if s not in media_names)

        # Google exports motion photos as a separate .MP video beside the still.
        report.motion_pairs = sum(
            1
            for p in by_hash.values()
            for path in p.paths
            if path.suffix.lower() == ".mp" and f"{path.stem}.jpg" in media_names
        )

        photos = list(by_hash.values())
        report.media_indexed = len(photos)
        for p in photos:
            # A capture date means a REAL one. Falling back to file mtime is not
            # knowing when the photo was taken, and counting it as coverage would
            # make doctor's low-date warning permanently silent.
            if p.meta.taken_at_utc and p.meta.tz_source is not TzSource.FILE_MTIME:
                report.with_date += 1
            if p.meta.gps:
                report.with_gps += 1
            if p.meta.people:
                report.with_people += 1
        return photos, report

    def _read_meta(
        self, path: Path, media_type: MediaType, mtime: datetime, report: SourceReport
    ) -> PhotoMeta:
        exif = read_exif(path) if media_type is MediaType.IMAGE else None
        if exif is not None and not exif.decode_ok:
            # Still index it - the file exists and the user should see it - but
            # never let it pass as healthy.
            report.unreadable.append((path, exif.error or "undecodable"))
            report.skip("undecodable")
        sidecar = find_sidecar(path)
        xmp = read_xmp(sidecar) if sidecar else None

        utc, local, tz_source = resolve(
            exif.taken_naive if exif else None,
            exif.offset if exif else None,
            exif.gps if exif else None,
            mtime,
            tz_lookup=self._tz_lookup,
        )
        return PhotoMeta(
            taken_at_utc=utc,
            taken_at_local=local,
            tz_source=tz_source,
            gps=exif.gps if exif else None,
            people=list(xmp.people) if xmp else [],
            face_regions=list(xmp.face_regions) if xmp else [],
            keywords=list(xmp.keywords) if xmp else [],
            description=xmp.description if xmp else None,
            camera_make=exif.camera_make if exif else None,
            camera_model=exif.camera_model if exif else None,
            width=exif.width if exif else None,
            height=exif.height if exif else None,
        )


def _split_edited(stem: str) -> tuple[str, str | None]:
    """Case- and Unicode-insensitive.

    Google emits -EDITED as well as -edited, and APFS/HFS+ hand back
    decomposed (NFD) filenames - so a French Takeout on macOS would never
    match the NFC '-modifié' in the list above.

    RULING 1: the returned base must be NFC-normalised too, and sliced by the
    NFC-normalised suffix length. Slicing a raw (possibly NFD) stem by the
    length of an NFC suffix is wrong whenever the suffix contains a character
    that decomposes into more than one codepoint (e.g. "-modifié"'s "é" is 1
    codepoint in NFC but 2 in NFD) - the lengths differ, so the slice leaves a
    stray combining character and the base never matches the stem_index key.
    """
    norm = unicodedata.normalize("NFC", stem)
    norm_cf = norm.casefold()
    for suffix in EDITED_SUFFIXES:
        suffix_norm = unicodedata.normalize("NFC", suffix)
        if norm_cf.endswith(suffix_norm.casefold()):
            return norm[: len(norm) - len(suffix_norm)], suffix
    return norm, None


def _sidecar_target(name: str) -> str:
    """Media filename a Takeout sidecar refers to.

    Verified against a real 5,006-sidecar export: the dominant form is
    `IMG_1234.jpg.supplemental-metadata.json`, and when two photos share a
    filename the counter lands INSIDE that suffix -
    `DSC_0880.JPG.supplemental-metadata(1).json` - not after the extension.

    That export contained no truncated suffixes (longest filename was 102
    chars, intact). Truncation is well documented elsewhere though, and the
    `met[a-z]*` wildcard costs nothing, so it stays.
    """
    stem = re.sub(r"\.supplemental-met[a-z]*(\(\d+\))?\.json$", "", name, flags=re.I)
    return re.sub(r"\.json$", "", stem, flags=re.I)
