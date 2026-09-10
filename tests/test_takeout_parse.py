import json
from datetime import UTC, datetime

from rekindle.enrich.takeout import JsonKind, album_title, classify_json, parse_sidecar


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_photo_sidecar_parses_every_field_we_use(tmp_path):
    p = _write(
        tmp_path / "IMG_1.jpg.supplemental-metadata.json",
        {
            "title": "IMG_1.jpg",
            "description": "  a caption  ",
            "photoTakenTime": {"timestamp": "1774082363"},
            "creationTime": {"timestamp": "1774085809"},
            "geoData": {"latitude": 22.5, "longitude": 88.3, "altitude": 9.0},
            "people": [{"name": "Ada"}, {"name": "Grace"}],
            "favorited": True,
            "archived": True,
        },
    )
    kind, payload = classify_json(p)
    assert kind is JsonKind.PHOTO
    s = parse_sidecar(p, payload)
    assert s.target_cf == "img_1.jpg"
    assert s.title == "IMG_1.jpg"
    assert s.taken_at_utc == datetime.fromtimestamp(1774082363, UTC)
    assert s.people == ("Ada", "Grace")
    assert s.gps.lat == 22.5 and s.gps.lon == 88.3 and s.gps.alt == 9.0
    assert s.description == "a caption"
    assert s.favorite is True
    assert s.archived is True
    assert s.trashed is False


def test_zeroed_geodata_is_absent_not_a_coordinate(tmp_path):
    """(0,0) is the Gulf of Guinea. Measured: 89.6% of real sidecars."""
    p = _write(
        tmp_path / "IMG_2.jpg.supplemental-metadata.json",
        {
            "title": "IMG_2.jpg",
            "photoTakenTime": {"timestamp": "1"},
            "geoData": {"latitude": 0.0, "longitude": 0.0, "altitude": 0.0},
        },
    )
    _, payload = classify_json(p)
    assert parse_sidecar(p, payload).gps is None


def test_geodataexif_is_used_when_geodata_is_zero(tmp_path):
    p = _write(
        tmp_path / "IMG_3.jpg.supplemental-metadata.json",
        {
            "title": "IMG_3.jpg",
            "photoTakenTime": {"timestamp": "1"},
            "geoData": {"latitude": 0.0, "longitude": 0.0, "altitude": 0.0},
            "geoDataExif": {"latitude": 1.5, "longitude": 2.5, "altitude": 0.0},
        },
    )
    _, payload = classify_json(p)
    assert parse_sidecar(p, payload).gps.lat == 1.5


def test_a_partial_geo_block_does_not_fabricate_a_coordinate(tmp_path):
    """Missing longitude must not silently default to 0.0 - that would
    produce a wrong-but-plausible coordinate, not report an absence."""
    p = _write(
        tmp_path / "IMG_7.jpg.supplemental-metadata.json",
        {
            "title": "IMG_7.jpg",
            "photoTakenTime": {"timestamp": "1"},
            "geoData": {"latitude": 5.0},
        },
    )
    _, payload = classify_json(p)
    assert parse_sidecar(p, payload).gps is None


def test_the_counter_is_relocated_when_deriving_the_target(tmp_path):
    p = _write(
        tmp_path / "DSC00107.JPG.supplemental-metadata(1).json",
        {"title": "DSC00107.JPG", "photoTakenTime": {"timestamp": "1"}},
    )
    _, payload = classify_json(p)
    s = parse_sidecar(p, payload)
    assert s.target_cf == "dsc00107(1).jpg"
    assert s.title == "DSC00107.JPG"  # title still disagrees; kept as a cross-check


def test_empty_and_absent_optional_fields(tmp_path):
    p = _write(
        tmp_path / "IMG_4.jpg.supplemental-metadata.json",
        {"title": "IMG_4.jpg", "description": "", "photoTakenTime": {"timestamp": "1"}},
    )
    _, payload = classify_json(p)
    s = parse_sidecar(p, payload)
    assert s.description is None
    assert s.people == ()
    assert s.favorite is False


def test_album_metadata_is_classified_as_album(tmp_path):
    p = _write(tmp_path / "Goa Trip" / "metadata.json", {"title": "Goa/ Trip"})
    kind, payload = classify_json(p)
    assert kind is JsonKind.ALBUM
    assert album_title(payload) == "Goa/ Trip"


def test_album_metadata_with_a_null_or_empty_title_yields_none(tmp_path):
    for payload in ({"title": None}, {"title": ""}, {"title": "   "}, {}):
        p = _write(tmp_path / f"a{id(payload)}" / "metadata.json", payload)
        _, parsed = classify_json(p)
        assert album_title(parsed) is None


def test_a_list_valued_title_is_not_an_album_title(tmp_path):
    """user-generated-memory-titles.json really does carry title: [...]."""
    p = _write(
        tmp_path / "user-generated-memory-titles.json",
        {"title": ["happy birthday ", "Leh Ladakh"]},
    )
    kind, payload = classify_json(p)
    assert kind is JsonKind.OTHER
    assert album_title(payload) is None


def test_shared_album_comments_is_other(tmp_path):
    p = _write(tmp_path / "shared_album_comments.json", {"comments": []})
    assert classify_json(p)[0] is JsonKind.OTHER


def test_malformed_json_is_unparseable_not_an_exception(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text("{not json", encoding="utf-8")
    kind, payload = classify_json(p)
    assert kind is JsonKind.UNPARSEABLE
    assert payload is None


def test_a_json_array_at_the_top_level_is_unparseable(tmp_path):
    p = _write(tmp_path / "arr.json", [1, 2, 3])
    assert classify_json(p)[0] is JsonKind.UNPARSEABLE


def test_a_missing_file_is_unparseable_not_an_exception(tmp_path):
    p = tmp_path / "nope.json"
    kind, payload = classify_json(p)
    assert kind is JsonKind.UNPARSEABLE
    assert payload is None


def test_invalid_utf8_bytes_are_unparseable_not_an_exception(tmp_path):
    p = tmp_path / "badbytes.json"
    p.write_bytes(b"\xff\xfe\x00bad")
    kind, payload = classify_json(p)
    assert kind is JsonKind.UNPARSEABLE
    assert payload is None


def test_a_sidecar_with_no_taken_time_still_parses(tmp_path):
    p = _write(tmp_path / "IMG_5.jpg.json", {"title": "IMG_5.jpg", "people": [{"name": "Ada"}]})
    kind, payload = classify_json(p)
    assert kind is JsonKind.PHOTO
    assert parse_sidecar(p, payload).taken_at_utc is None


def test_people_entries_without_a_usable_name_are_dropped(tmp_path):
    p = _write(
        tmp_path / "IMG_6.jpg.json",
        {
            "title": "IMG_6.jpg",
            "people": [{"name": "Ada"}, {"name": "  "}, {"name": None}, {}, {"name": "Ada"}],
        },
    )
    _, payload = classify_json(p)
    assert parse_sidecar(p, payload).people == ("Ada",)
