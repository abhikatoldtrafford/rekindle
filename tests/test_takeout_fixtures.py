import json

from tests.fixtures.takeout import build_takeout


def test_the_tree_reproduces_every_measured_pathology(tmp_path):
    """This test verifies NO production code. It is a fixture asserting the
    fixture, and it earns its place only as a drift guard: build_takeout() is
    the behavioural spec for Tasks 6, 9, 11 and 12, so a change to the tree
    that quietly removes a pathology would otherwise weaken four task suites
    at once with all of them still green. Keep it, and do not mistake it for
    coverage."""
    root = build_takeout(tmp_path / "Takeout")
    year = root / "Photos from 2011"

    # Both sidecar filename schemes, and a real (N) collision: two sidecars
    # whose `title` is identical, belonging to two different photos.
    assert (year / "DSC00107.JPG").is_file()
    assert (year / "DSC00107(1).JPG").is_file()
    assert (year / "DSC00107.JPG.supplemental-metadata.json").is_file()
    assert (year / "DSC00107.JPG.supplemental-metadata(1).json").is_file()
    a = json.loads((year / "DSC00107.JPG.supplemental-metadata.json").read_text())
    b = json.loads((year / "DSC00107.JPG.supplemental-metadata(1).json").read_text())
    assert a["title"] == b["title"] == "DSC00107.JPG"
    assert a["photoTakenTime"] != b["photoTakenTime"]
    assert a["people"] != b["people"]

    # An album folder holding sidecars and ZERO media - 8 of these in the
    # reference export, hundreds of sidecars each.
    empty_album = root / "wedding_anniversary"
    assert (empty_album / "IMG_ALBUM.jpg.supplemental-metadata.json").is_file()
    assert not any(p.suffix.lower() == ".jpg" for p in empty_album.iterdir())

    # A derivative with no sidecar of its own, and its original with one.
    assert (year / "IMG_EDIT-edited.jpg").is_file()
    assert not (year / "IMG_EDIT-edited.jpg.supplemental-metadata.json").exists()
    assert (year / "IMG_EDIT.jpg.supplemental-metadata.json").is_file()

    # A motion photo: the .MP video has no sidecar, the .MP.jpg still does.
    assert (year / "PXL_1.MP").is_file()
    assert (year / "PXL_1.MP.jpg").is_file()
    assert not (year / "PXL_1.MP.supplemental-metadata.json").exists()
    assert (year / "PXL_1.MP.jpg.supplemental-metadata.json").is_file()

    # Non-photo JSON that must not be mistaken for a sidecar.
    assert json.loads((root / "metadata.json").read_text())["title"] is None
    assert isinstance(
        json.loads((root / "user-generated-memory-titles.json").read_text())["title"], list
    )
    assert (root / "Goa Trip" / "shared_album_comments.json").is_file()

    # geoDataExif key-ABSENT rather than zeroed, favorited absent when false.
    # geoDataExif is only ever absent when geoData is zeroed too (a strict
    # bijection, measured with 0 exceptions across 24,248 real sidecars) -
    # pin the zeroed geoData here so a regression to a non-zero geoData with
    # no matching geoDataExif (a real defect caught in review) fails loudly.
    plain = json.loads((year / "IMG_EDIT.jpg.supplemental-metadata.json").read_text())
    assert plain["geoData"] == {"latitude": 0.0, "longitude": 0.0, "altitude": 0.0}
    assert "geoDataExif" not in plain
    assert "favorited" not in plain

    # The other half of the geoData/geoDataExif bijection: when geoData is
    # non-zero, geoDataExif is present too and the two agree exactly (a
    # measured fact across all 2,532 real pairs that carry both).
    motion = json.loads((year / "PXL_1.MP.jpg.supplemental-metadata.json").read_text())
    assert motion["geoData"] == motion["geoDataExif"]
    assert motion["geoData"]["latitude"] != 0.0

    # No third sidecar naming scheme: flags/description/favorited/archived
    # live on a `.supplemental-metadata.json` sidecar like everything else,
    # not on a bare `.json` file (which does not exist in the real export).
    assert not (year / "IMG_FLAGS.jpg.json").exists()
    flags = json.loads((year / "IMG_FLAGS.jpg.supplemental-metadata.json").read_text())
    assert flags["favorited"] is True
    assert flags["archived"] is True
    assert flags["description"] == "a real caption"

    # Trash: excluded by FolderSource, and the enricher must exclude it too
    # via the `trashed` field itself (present only when true, like the other
    # boolean flags above).
    trashed_sidecar = json.loads(
        (root / "Trash" / "IMG_GONE.jpg.supplemental-metadata.json").read_text()
    )
    assert trashed_sidecar["trashed"] is True

    # A malformed JSON file, so `unparseable` is exercised.
    assert (year / "broken.json").read_text() == "{not json"


def test_album_titles_cover_differing_empty_and_colliding(tmp_path):
    root = build_takeout(tmp_path / "Takeout")
    assert json.loads((root / "Goa Trip" / "metadata.json").read_text())["title"] == "Goa/ Trip"
    assert json.loads((root / "Untitled" / "metadata.json").read_text())["title"] == "Untitled"
    assert json.loads((root / "Untitled(1)" / "metadata.json").read_text())["title"] == "Untitled"
    assert json.loads((root / "No Name" / "metadata.json").read_text())["title"] == ""
