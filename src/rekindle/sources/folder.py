"""Reads a plain directory tree. The only source in v1."""

from __future__ import annotations

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
from rekindle.sidecars import is_photo_sidecar
from rekindle.sidecars import sidecar_target as _sidecar_target

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

        # Casefolded, because Takeout writes sidecar names from its own record
        # of the filename and that routinely disagrees with the file's own
        # case - a "DSC_0880.JPG.supplemental-metadata.json" beside a
        # "DSC_0880.jpg" is not an orphan.
        media_names: set[str] = set()
        # Motion-photo pairing must additionally be scoped to a single
        # directory: a .MP in one album must never pair with a same-named
        # still in another.
        media_names_by_dir: dict[Path, set[str]] = {}
        sidecar_stems: list[str] = []
        saw_xmp: set[str] = set()
        saw_undecodable: set[str] = set()

        def note_undecodable(digest: str, path: Path, error: str | None) -> None:
            """Report per PATH, count per PHOTO.

            `unreadable` must name every file the user should go and look at,
            so both copies of a corrupt photo belong in it. The COUNT answers
            a different question - "how many of my photos are broken" - and
            the same bytes seen in two folders are one broken photo, not two.
            Counting per path made files_seen stop reconciling against the
            other buckets in doctor's output.
            """
            if error is None:
                return
            report.unreadable.append((path, error))
            if digest not in saw_undecodable:
                saw_undecodable.add(digest)
                report.undecodable += 1

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
            if any(part.casefold() in EXCLUDED_DIRS for part in path.relative_to(root).parts[:-1]):
                report.excluded_dirs += 1
                continue

            if is_long_path(path):
                report.long_paths.append(path)

            if path.suffix.casefold() in _SIDECAR_EXTS:
                # EVERY branch below increments a bucket. `files_seen` has
                # already counted this file, so a `continue` that lands in no
                # bucket is a silent drop by definition - and one that
                # survived ten reviews, because nothing asserted the counters
                # add up. test_every_file_seen_is_accounted_for_by_exactly_
                # one_counter now pins that.
                #
                # .xmp files ARE read (find_sidecar/read_xmp). Counting them
                # under a generic "ignored" reason made doctor warn that data
                # was being discarded three rows below reporting it as
                # successfully read - hence a reason of their own.
                if path.suffix.casefold() == ".xmp":
                    report.skip("sidecar_xmp")
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
                if path.suffix.casefold() != ".json":
                    report.skip("sidecar_other")
                    continue
                report.json_sidecars += 1
                # `is_photo_sidecar`, not a comparison written here. Only a
                # per-photo sidecar can be an orphan, and this file is not the
                # place that decides what one is: two implementations of that
                # rule produced two different orphan counts for one export,
                # which is why `rekindle.sidecars` exists.
                #
                # It excludes album metadata case-INSENSITIVELY, so a
                # `Metadata.json` is not counted here while `build_index`
                # classifies the same file as album metadata (0 live
                # instances; Google writes it lowercase). It also excludes the
                # account-level JSONs Google puts at the root of
                # `Google Photos/` - `shared_album_comments.json` and
                # `user-generated-memory-titles.json` - which named no
                # photograph, matched no media file, and were therefore
                # counted as proof that the export was incomplete. On a
                # genuinely complete export they were the entire count, so
                # `rekindle doctor` told a new user their library was missing
                # photos on the strength of two files about nothing.
                if is_photo_sidecar(path.name):
                    sidecar_stems.append(_sidecar_target(path.name))
                continue

            # Register the NAME before sniffing, and whatever sniff decides. A
            # format we do not recognise is still a file the user has, and it
            # still has its own JSON sidecar: a Canon .CR3 is ISO-BMFF with the
            # brand "crx " and sniffs as UNKNOWN, so recording names only after
            # a successful sniff left "IMG_0100.CR3.supplemental-metadata.json"
            # with nothing to match and drove the INCOMPLETE EXPORT warning on
            # a library that was in fact complete. Same class of bug as the
            # .aae one above; fixing it in one place was not enough.
            media_names.add(path.name.casefold())
            media_names_by_dir.setdefault(path.parent, set()).add(path.name.casefold())

            try:
                media_type, _fmt = sniff(path)
            except OSError as exc:
                # An unopenable file is not a statement about its CONTENT.
                # sniff() used to swallow this into (UNKNOWN, "unknown"), so a
                # permission-denied file was filed as "not_media" - a positive
                # claim never actually checked - and never reached `unreadable`.
                report.unreadable.append((path, str(exc)))
                report.skip("unreadable")
                continue

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
                # `base_stem` already IS normalize("NFC", path.stem) when no
                # edited suffix matched. Recomputing it here was the exact
                # duplication RULING 1 exists to close: two expressions that
                # must stay identical, and nothing forcing them to.
                stem_key = (path.parent, base_stem)
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

            # Found once and reused: find_sidecar stats the filesystem twice
            # per candidate, and calling it again inside _read_meta doubled
            # that for every media file in the library.
            sidecar = find_sidecar(path)
            has_xmp = sidecar is not None

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
                meta, decode_error, xmp_error = self._read_meta(path, media_type, mtime, sidecar)
                if xmp_error:
                    report.xmp_unreadable += 1
                merged, conflict = merge_meta(existing.meta, meta)
                existing.meta = merged
                existing.metadata_conflict = existing.metadata_conflict or conflict
                note_undecodable(digest, path, decode_error)
                if has_xmp and digest not in saw_xmp:
                    saw_xmp.add(digest)
                    report.with_xmp += 1
                report.duplicates_merged += 1
                continue

            if has_xmp:
                saw_xmp.add(digest)
                report.with_xmp += 1

            meta, decode_error, xmp_error = self._read_meta(path, media_type, mtime, sidecar)
            if xmp_error:
                report.xmp_unreadable += 1
            note_undecodable(digest, path, decode_error)
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
        report.orphan_sidecars = sum(1 for s in sidecar_stems if s.casefold() not in media_names)

        # Google exports motion photos as a separate .MP video beside the
        # still, named "<name>.MP.jpg" - i.e. the video's full NAME plus
        # ".jpg", not its stem. Path.stem on "PXL_x.MP" strips only the final
        # ".MP" suffix (giving "PXL_x"), so a still literally named
        # "PXL_x.jpg" almost never exists; matching on `path.stem` finds zero
        # pairs against a real export. Scoped to the same directory and
        # case-insensitive, since real libraries mix .jpg/.JPG.
        report.motion_pairs = sum(
            1
            for p in by_hash.values()
            for path in p.paths
            if path.suffix.casefold() == ".mp"
            and f"{path.name}.jpg".casefold() in media_names_by_dir.get(path.parent, set())
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
        self,
        path: Path,
        media_type: MediaType,
        mtime: datetime,
        sidecar: Path | None,
    ) -> tuple[PhotoMeta, str | None, str]:
        """Returns (meta, decode_error, xmp_error).

        The decode failure is HANDED BACK rather than reported here: only the
        caller knows this path's hash, and the failure has to be counted once
        per photo, not once per path. Reporting it in here also filed an
        indexed file under `skipped`, which it plainly is not. `xmp_error`
        rides along for the same reason and is the empty string when the
        sidecar parsed or there was none.
        """
        exif = read_exif(path) if media_type is MediaType.IMAGE else None
        # Still index it - the file exists and the user should see it - but
        # never let it pass as healthy.
        decode_error = None if exif is None or exif.decode_ok else (exif.error or "undecodable")
        xmp = read_xmp(sidecar) if sidecar else None

        utc, local, tz_source = resolve(
            exif.taken_naive if exif else None,
            exif.offset if exif else None,
            exif.gps if exif else None,
            mtime,
            tz_lookup=self._tz_lookup,
        )
        meta = PhotoMeta(
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
        return meta, decode_error, (xmp.error if xmp else "")


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

    When no suffix matches, the returned base is exactly
    normalize("NFC", stem) - callers registering a stem_index key use it
    rather than recomputing the normalisation.
    """
    norm = unicodedata.normalize("NFC", stem)
    norm_cf = norm.casefold()
    for suffix in EDITED_SUFFIXES:
        suffix_norm = unicodedata.normalize("NFC", suffix)
        if norm_cf.endswith(suffix_norm.casefold()):
            return norm[: len(norm) - len(suffix_norm)], suffix
    return norm, None
