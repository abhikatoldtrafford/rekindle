"""A Takeout export reproducing structure MEASURED on a real 24,292-file
export at `Takeout/Google Photos`. Most of what follows was counted, not
assumed - after two earlier defects in this project came from exactly that
shortcut (M0's `.jpg`+`.MP` motion-photo naming, and this milestone's own
first-draft spec claiming sidecars match on `title`), every shape below was
re-verified against the real export before being written into this file.

MEASURED (counted against all 24,248 real sidecars unless noted):

    (N) collisions with differing people   833 groups, 100 with differing people
    album folders holding zero media       8
    -edited files with no sidecar          177 (0 have one)
    .MP halves with no sidecar             692 (0 have one); .MP.jpg: 729/729 do
    geoData/geoDataExif                    a strict bijection - 2,532 sidecars
                                           have BOTH non-zero, 21,716 have
                                           NEITHER key, 0 have exactly one
    favorited=true                         7 sidecars (0 carry `favorited: false`)
    archived=true                          162 sidecars
    trashed=true                           12 sidecars, all under Trash/
    non-photo JSON at album level          42 metadata.json (4 titles differ
                                           from the folder name, 2 empty),
                                           1 shared_album_comments.json,
                                           1 user-generated-memory-titles.json
                                           (`title` is a list) - 44 total,
                                           zero bare per-photo `.json` sidecars
    per-photo sidecar naming schemes       exactly two: `.supplemental-metadata`
                                           and `.supplemental-metadata(N)` -
                                           no third scheme, no truncated forms

DELIBERATELY INJECTED (not observed in the export; added so parsing/matching
code has something to fail on - do not read these as measured facts):

    broken.json                            malformed JSON, to exercise
                                           `unparseable` classification
    IMG_MISSING.jpg sidecar with no photo  simulates a Takeout part that
                                           wasn't extracted
    two SHARED.jpg photos, one directory   342 real filenames ARE shared by
    resolving exact and one ambiguous      distinct photos even after
                                           content-hash dedup (measured); this
                                           specific pairing - one photo's
                                           resolution overwriting another's
                                           refusal in a target-keyed `claimed`
                                           dict - is the synthetic case that
                                           exercises it deterministically
    root-level metadata.json,              a plausible non-photo JSON at the
    title: null                           export root. An earlier draft of
                                           this file presented this as
                                           MEASURED; scanning all 24,292 files
                                           in the reference export found no
                                           root-level metadata.json at all.
                                           Kept only as a synthetic case for
                                           "non-photo JSON must not be
                                           mistaken for a sidecar" - its shape
                                           (a top-level metadata.json with a
                                           null title) is invented, not seen.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.fixtures.gen import make_jpeg


def _sidecar(path: Path, **overrides: object) -> None:
    payload: dict[str, object] = {
        "title": overrides.pop("title"),
        "description": "",
        "imageViews": "1",
        "creationTime": {"timestamp": "1500000000"},
        "photoTakenTime": {"timestamp": "1400000000"},
        "geoData": {"latitude": 0.0, "longitude": 0.0, "altitude": 0.0},
        "url": "https://photos.google.com/photo/x",
    }
    payload.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_takeout(root: Path) -> Path:
    year = root / "Photos from 2011"
    year.mkdir(parents=True, exist_ok=True)

    # --- the (N) collision -------------------------------------------------
    # Two DIFFERENT photos, two sidecars, one `title`. Matching on `title`
    # pairs both sidecars with DSC00107.JPG and leaves DSC00107(1).JPG bare.
    make_jpeg(year / "DSC00107.JPG", size=(24, 24))
    make_jpeg(year / "DSC00107(1).JPG", size=(25, 25))
    _sidecar(
        year / "DSC00107.JPG.supplemental-metadata.json",
        title="DSC00107.JPG",
        photoTakenTime={"timestamp": "1323826707"},
        people=[{"name": "Grace"}],
    )
    _sidecar(
        year / "DSC00107.JPG.supplemental-metadata(1).json",
        title="DSC00107.JPG",
        photoTakenTime={"timestamp": "1295183562"},
        people=[{"name": "Ada"}],
    )

    # --- an ambiguous pair: same target, disagreeing, no directory tiebreak -
    make_jpeg(year / "AMBIG.jpg", size=(26, 26))
    _sidecar(
        year / "AMBIG.jpg.supplemental-metadata.json",
        title="AMBIG.jpg",
        photoTakenTime={"timestamp": "1000000000"},
        people=[{"name": "Ada"}],
    )
    _sidecar(
        root / "Goa Trip" / "AMBIG.jpg.supplemental-metadata.json",
        title="AMBIG.jpg",
        photoTakenTime={"timestamp": "1200000000"},
        people=[{"name": "Grace"}],
    )

    # --- two DISTINCT photos sharing one target filename --------------------
    # 342 filenames are shared by distinct photos on the reference export
    # even after content-hash dedup. `SHARED.jpg` names two different real
    # files here: one next to a same-directory candidate (resolves exact,
    # with the directory-preference tie against Goa Trip's disagreeing
    # candidate), one in a directory with no candidate of its own (falls
    # back to the two global candidates, which disagree, and is refused).
    # `account()`'s `claimed` dict is keyed by target, not by photo - a
    # caller that lets the second resolution overwrite the first's refusal
    # reproduces "984 sidecars vanished into dict.__setitem__" one level up.
    make_jpeg(year / "SHARED.jpg", size=(35, 35))
    _sidecar(
        year / "SHARED.jpg.supplemental-metadata.json",
        title="SHARED.jpg",
        photoTakenTime={"timestamp": "1100000000"},
        people=[{"name": "Ada"}],
    )
    _sidecar(
        root / "Goa Trip" / "SHARED.jpg.supplemental-metadata.json",
        title="SHARED.jpg",
        photoTakenTime={"timestamp": "1500000000"},
        people=[{"name": "Grace"}],
    )
    kolkata = root / "Kolkata Trip"
    kolkata.mkdir(parents=True, exist_ok=True)
    make_jpeg(kolkata / "SHARED.jpg", size=(37, 37))

    # --- a photo whose only sidecar lives in an album with no media --------
    # 1,204 real photos are in this position; per-directory keying loses them.
    make_jpeg(year / "IMG_ALBUM.jpg", size=(27, 27))
    _sidecar(
        root / "wedding_anniversary" / "IMG_ALBUM.jpg.supplemental-metadata.json",
        title="IMG_ALBUM.jpg",
        photoTakenTime={"timestamp": "1600000000"},
        people=[{"name": "Ada"}],
    )

    # --- derivatives Google never writes a sidecar for ---------------------
    # geoData is left at the `_sidecar` default (zeroed): the real export is a
    # strict bijection between geoData and geoDataExif - 2,532 sidecars have
    # BOTH non-zero, 21,716 have NEITHER key, 0 have exactly one. A zeroed
    # geoData is the case that actually produces a key-absent geoDataExif.
    make_jpeg(year / "IMG_EDIT.jpg", size=(28, 28))
    make_jpeg(year / "IMG_EDIT-edited.jpg", size=(29, 29))
    _sidecar(
        year / "IMG_EDIT.jpg.supplemental-metadata.json",
        title="IMG_EDIT.jpg",
        photoTakenTime={"timestamp": "1400000000"},
        people=[{"name": "Grace"}],
    )
    make_jpeg(year / "PXL_1.MP.jpg", size=(30, 30))
    (year / "PXL_1.MP").write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 32)
    _sidecar(
        year / "PXL_1.MP.jpg.supplemental-metadata.json",
        title="PXL_1.MP.jpg",
        photoTakenTime={"timestamp": "1700000000"},
        people=[{"name": "Ada"}],
        # The other half of the bijection above: when geoData is non-zero,
        # geoDataExif is always present too, and the two never disagree
        # (measured across 2,532 real pairs).
        geoData={"latitude": 22.5, "longitude": 88.3, "altitude": 9.0},
        geoDataExif={"latitude": 22.5, "longitude": 88.3, "altitude": 9.0},
    )

    # --- a derivative that inherits a description AND a favourite ----------
    # Same shape as IMG_EDIT/IMG_EDIT-edited above, but the donor's sidecar
    # also carries `description` and `favorited: true` - `propagate_to_
    # derivatives`'s own `descriptions_added`/`favourites_added` increments
    # (as opposed to `apply_sidecar`'s copies, already exercised by
    # IMG_FLAGS.jpg below) have no fixture data to write without this: no
    # OTHER derivative here carries either field, and neither does the real
    # export - both totals read 142 and 3 before and after the wave with no
    # derivative contribution at all.
    # Sized differently from every other fixture photo here: `make_jpeg`'s
    # default color makes two same-size images BYTE-IDENTICAL with no EXIF to
    # differentiate them, which content-hash dedup then merges into ONE
    # Photo with two names - reusing (28, 28)/(29, 29) here collided this
    # pair with IMG_EDIT/IMG_EDIT-edited and inflated `ambiguous` by 2.
    make_jpeg(year / "IMG_CAPTION.jpg", size=(40, 40))
    make_jpeg(year / "IMG_CAPTION-edited.jpg", size=(41, 41))
    _sidecar(
        year / "IMG_CAPTION.jpg.supplemental-metadata.json",
        title="IMG_CAPTION.jpg",
        photoTakenTime={"timestamp": "1450000000"},
        people=[{"name": "Grace"}],
        description="a captioned original",
        favorited=True,
    )

    # --- an orphan: a sidecar whose photo is in an un-extracted part -------
    _sidecar(year / "IMG_MISSING.jpg.supplemental-metadata.json", title="IMG_MISSING.jpg")

    # --- flags, on the ONLY sidecar naming scheme Google writes -------------
    # `.supplemental-metadata.json` is the correct scheme here too - a bare
    # `.json` per-photo sidecar does not exist anywhere in the real export
    # (measured: 0 of 24,292 files; the only non-`supplemental-metadata` JSON
    # is the 44 album/root-level files handled elsewhere in this tree).
    make_jpeg(year / "IMG_FLAGS.jpg", size=(31, 31))
    _sidecar(
        year / "IMG_FLAGS.jpg.supplemental-metadata.json",
        title="IMG_FLAGS.jpg",
        photoTakenTime={"timestamp": "1450000000"},
        description="a real caption",
        favorited=True,
        archived=True,
    )

    # --- malformed JSON ---------------------------------------------------
    (year / "broken.json").write_text("{not json", encoding="utf-8")

    # --- non-photo JSON ---------------------------------------------------
    (root / "metadata.json").write_text(json.dumps({"title": None}), encoding="utf-8")
    (root / "user-generated-memory-titles.json").write_text(
        json.dumps({"title": ["happy birthday ", "Leh Ladakh"]}), encoding="utf-8"
    )
    goa = root / "Goa Trip"
    goa.mkdir(parents=True, exist_ok=True)
    (goa / "metadata.json").write_text(json.dumps({"title": "Goa/ Trip"}), encoding="utf-8")
    (goa / "shared_album_comments.json").write_text(json.dumps({"comments": []}), encoding="utf-8")

    # --- album title pathologies ------------------------------------------
    for folder, title in (("Untitled", "Untitled"), ("Untitled(1)", "Untitled"), ("No Name", "")):
        (root / folder).mkdir(parents=True, exist_ok=True)
        (root / folder / "metadata.json").write_text(json.dumps({"title": title}), encoding="utf-8")
    make_jpeg(root / "Untitled" / "IMG_U0.jpg", size=(32, 32))
    make_jpeg(root / "Untitled(1)" / "IMG_U1.jpg", size=(33, 33))
    _sidecar(root / "Untitled" / "IMG_U0.jpg.supplemental-metadata.json", title="IMG_U0.jpg")
    _sidecar(root / "Untitled(1)" / "IMG_U1.jpg.supplemental-metadata.json", title="IMG_U1.jpg")

    # --- Trash: never parsed ------------------------------------------------
    # Folder placement alone is enough for FolderSource to exclude this, but
    # all 12 real trashed sidecars also carry `trashed: true` (present only
    # when true, like `favorited` and `archived`) - so the enricher's own
    # JSON-field exclusion path needs data to exercise it too.
    trash = root / "Trash"
    trash.mkdir(parents=True, exist_ok=True)
    make_jpeg(trash / "IMG_GONE.jpg", size=(34, 34))
    _sidecar(trash / "IMG_GONE.jpg.supplemental-metadata.json", title="IMG_GONE.jpg", trashed=True)

    return root
