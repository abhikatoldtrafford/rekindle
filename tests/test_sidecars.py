from rekindle.sidecars import is_album_metadata, sidecar_target


def test_plain_supplemental_metadata_yields_the_filename():
    assert sidecar_target("IMG_1234.jpg.supplemental-metadata.json") == "IMG_1234.jpg"


def test_truncated_supplemental_suffix_still_matches():
    assert sidecar_target("IMG_1234.jpg.supplemental-metad.json") == "IMG_1234.jpg"


def test_bare_json_suffix_yields_the_filename():
    assert sidecar_target("IMG_1234.jpg.json") == "IMG_1234.jpg"


def test_counter_is_relocated_into_the_stem_not_stripped():
    """THE v1 BUG. Google puts the counter in the SIDECAR's name; the photo
    carries it inside the stem. Measured on a real export: 963 sidecars would
    otherwise be paired with a different photo of the same name."""
    assert sidecar_target("DSC00107.JPG.supplemental-metadata(1).json") == "DSC00107(1).JPG"
    assert sidecar_target("Photo0007.jpg.supplemental-metadata(2).json") == "Photo0007(2).jpg"


def test_counter_relocation_preserves_spaces_and_dots_in_the_stem():
    assert (
        sidecar_target("Photo0549 - Copy.jpg.supplemental-metadata(1).json")
        == "Photo0549 - Copy(1).jpg"
    )
    assert sidecar_target("PXL_1.MP.jpg.supplemental-metadata(1).json") == "PXL_1.MP(1).jpg"


def test_extensionless_target_appends_the_counter():
    assert sidecar_target("README.supplemental-metadata(1).json") == "README(1)"


def test_unrecognised_name_is_returned_unchanged():
    assert sidecar_target("IMG_7000.aae") == "IMG_7000.aae"


def test_album_metadata_is_recognised_case_insensitively():
    assert is_album_metadata("metadata.json")
    assert is_album_metadata("Metadata.JSON")
    assert not is_album_metadata("IMG_1234.jpg.supplemental-metadata.json")
