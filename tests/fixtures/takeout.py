"""A Takeout export reproducing structure MEASURED on a real 45,900-file
export. Nothing here is invented: every pathology below was counted.

    (N) collisions with differing people   833 groups, 100 with differing people
    album folders holding zero media       8
    -edited files with no sidecar          177
    .MP halves with no sidecar             692
    geoDataExif key-absent                 89.6% of sidecars
    favorited absent when false            all but 7 sidecars
    non-photo JSON at album level          metadata.json, shared_album_comments,
                                           user-generated-memory-titles
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
    make_jpeg(year / "IMG_EDIT.jpg", size=(28, 28))
    make_jpeg(year / "IMG_EDIT-edited.jpg", size=(29, 29))
    _sidecar(
        year / "IMG_EDIT.jpg.supplemental-metadata.json",
        title="IMG_EDIT.jpg",
        photoTakenTime={"timestamp": "1400000000"},
        people=[{"name": "Grace"}],
        geoData={"latitude": 22.5, "longitude": 88.3, "altitude": 9.0},
    )
    make_jpeg(year / "PXL_1.MP.jpg", size=(30, 30))
    (year / "PXL_1.MP").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32)
    _sidecar(
        year / "PXL_1.MP.jpg.supplemental-metadata.json",
        title="PXL_1.MP.jpg",
        photoTakenTime={"timestamp": "1700000000"},
        people=[{"name": "Ada"}],
    )

    # --- an orphan: a sidecar whose photo is in an un-extracted part -------
    _sidecar(year / "IMG_MISSING.jpg.supplemental-metadata.json", title="IMG_MISSING.jpg")

    # --- flags, and a bare `.json` sidecar --------------------------------
    make_jpeg(year / "IMG_FLAGS.jpg", size=(31, 31))
    _sidecar(
        year / "IMG_FLAGS.jpg.json",
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

    # --- Trash: never parsed ----------------------------------------------
    trash = root / "Trash"
    trash.mkdir(parents=True, exist_ok=True)
    make_jpeg(trash / "IMG_GONE.jpg", size=(34, 34))
    _sidecar(trash / "IMG_GONE.jpg.supplemental-metadata.json", title="IMG_GONE.jpg")

    return root
