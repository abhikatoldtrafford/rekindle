import pytest

from rekindle.sidecars import is_album_metadata, is_photo_sidecar, sidecar_target


def test_plain_supplemental_metadata_yields_the_filename():
    assert sidecar_target("IMG_1234.jpg.supplemental-metadata.json") == "IMG_1234.jpg"


def test_truncated_supplemental_suffix_still_matches():
    assert sidecar_target("IMG_1234.jpg.supplemental-metad.json") == "IMG_1234.jpg"


def test_bare_json_suffix_yields_the_filename():
    """The no-counter half of the plain-`.json` pair below."""
    assert sidecar_target("IMG_1234.jpg.json") == "IMG_1234.jpg"


def test_bare_json_suffix_with_counter_relocates_it_too():
    """The counter half of the plain-`.json` pair above.

    Measured on a real 24,250-sidecar export: this shape (a counter on a
    plain `.json` sidecar with no `supplemental-metadata` infix) occurs
    zero times - the only two plain `.json` files present are
    `shared_album_comments.json` and `user-generated-memory-titles.json`,
    neither of which carries a counter. So this is currently latent, not
    live.

    It is pinned anyway because the relocation logic runs unconditionally
    here too, and that is a deliberate choice, not an accident: M0's old
    implementation only stripped a trailing `.json` on this path, so it
    would have left the counter glued after the extension
    (`IMG_1234.jpg(1)`). This diverges from that - the counter is
    relocated into the stem (`IMG_1234(1).jpg`) - because that is the more
    sensible target name, consistent with the supplemental-metadata case.
    """
    assert sidecar_target("IMG_1234.jpg(1).json") == "IMG_1234(1).jpg"


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


# --------------------------------------------------------------------------
# truncated markers


@pytest.mark.parametrize(
    "marker",
    [
        "supplemental-metadata",
        "supplemental-metadat",
        "supplemental-met",
        "supplemental-me",
        "supplemental-m",
        "supplemental-",
        "supplemental",
        "supplementa",
        "suppl",
        "sup",
        "su",
        "s",
    ],
)
def test_a_truncated_marker_still_names_the_media_file(marker):
    """Google caps the whole sidecar filename, so a long photo name loses the
    TAIL of the marker.

    The pattern was the literal `supplemental-met` plus `[a-z]*`, under a
    comment claiming the wildcard covered truncation - it covers it only after
    those sixteen characters. Everything shorter fell through to the plain
    `.json` rule, which returns `IMG_1234.jpg.supplemental-me` as the file to
    look for. That cannot exist, so the sidecar becomes a permanent orphan and
    fires the INCOMPLETE EXPORT warning on a complete library.
    """
    assert sidecar_target(f"IMG_1234.jpg.{marker}.json") == "IMG_1234.jpg"


def test_a_truncated_marker_still_relocates_the_counter():
    assert sidecar_target("DSC00107.JPG.supplemental-me(1).json") == "DSC00107(1).JPG"


@pytest.mark.parametrize(
    "name",
    [
        "photo.jpg.settings.json",
        "photo.jpg.supplementalx.json",
        "photo.jpg.supplemental-metadataX.json",
        "photo.jpg.sidecar.json",
    ],
)
def test_a_name_that_only_looks_like_the_marker_is_not_treated_as_one(name):
    """The pattern matches PREFIXES of the word and nothing else, which is why
    it is generated from the literal rather than written as a wildcard. A
    wildcard loose enough to catch `.s` would also swallow `.settings`."""
    assert sidecar_target(name) == name.removesuffix(".json")


# --------------------------------------------------------------------------
# what counts as a per-photo sidecar at all


@pytest.mark.parametrize(
    "name",
    [
        "shared_album_comments.json",
        "user-generated-memory-titles.json",
        "print-subscriptions.json",
        "metadata.json",
        "Metadata.json",
    ],
)
def test_account_level_and_album_json_are_not_photo_sidecars(name):
    """Only a per-photo sidecar can be an ORPHAN, and `doctor` fires its
    loudest warning on the orphan count - "INCOMPLETE EXPORT ... your library
    will be silently missing photos".

    Google writes two account-level JSONs at the root of `Google Photos/`.
    They name no photograph, so they matched no media file and were counted as
    proof the export was incomplete: on a genuinely complete export they were
    the entire count, and the first command a new user runs told them their
    data was missing.
    """
    assert not is_photo_sidecar(name)


@pytest.mark.parametrize(
    "name",
    [
        "PXL_1234.jpg.supplemental-metadata.json",
        "DSC00107.JPG.supplemental-metadata(1).json",
        "IMG_0001.JPG.json",
        "clip.mp4.supplemental-me.json",
    ],
)
def test_a_real_per_photo_sidecar_still_counts(name):
    """The other half: a rule that excluded too much would hide a genuinely
    incomplete export, which is the failure the warning exists for."""
    assert is_photo_sidecar(name)
