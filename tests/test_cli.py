from typer.testing import CliRunner

from rekindle.cli import app
from tests.fixtures.gen import build_library

runner = CliRunner()


def test_doctor_reports_without_writing_a_database(tmp_path, monkeypatch):
    # chdir into tmp_path so a bug that defaults to a *relative* data
    # directory (as `index` does) would land inside tmp_path and get
    # caught below, instead of silently polluting the real cwd.
    monkeypatch.chdir(tmp_path)
    root = build_library(tmp_path / "lib")
    data = tmp_path / "data"
    result = runner.invoke(app, ["doctor", str(root)])
    assert result.exit_code == 0
    assert "photos" in result.stdout.lower()
    assert not data.exists()
    assert not any(tmp_path.rglob("*.sqlite"))


def test_version_flag_works(tmp_path):
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip()


def test_index_writes_photos_to_the_database(tmp_path):
    root = build_library(tmp_path / "lib")
    data = tmp_path / "data"
    result = runner.invoke(app, ["index", str(root), "--data-dir", str(data)])
    assert result.exit_code == 0
    assert (data / "rekindle.sqlite").exists()

    from rekindle.db import PhotoStore

    with PhotoStore(data / "rekindle.sqlite") as s:
        assert s.count() > 0


def test_reindex_does_not_destroy_previously_merged_metadata(tmp_path):
    """The DoD clause "does not destroy previously merged metadata" had no
    coverage at all: `test_index_twice_does_not_duplicate` below asserts only
    a row count, so replacing PhotoStore._merge's merge_meta call with
    `new.meta` left the entire suite green (verified) - the exact defect
    merge_meta exists to prevent.

    IMG_0003.jpg is the fixture's photo with no metadata whatsoever, so every
    field asserted here can only have come from the enrichment. A real
    capture date is included on purpose: the folder source always supplies
    taken_at_utc via its mtime fallback, so a wholesale
    `new if new.taken_at_utc else old` swap would look correct and quietly
    overwrite a known date with a file timestamp on every single re-index.
    """
    from datetime import UTC, datetime

    from rekindle.db import PhotoStore
    from rekindle.models import TzSource

    root = build_library(tmp_path / "lib")
    data = tmp_path / "data"
    db = data / "rekindle.sqlite"
    assert runner.invoke(app, ["index", str(root), "--data-dir", str(data)]).exit_code == 0

    known_date = datetime(2014, 7, 4, 11, 30, tzinfo=UTC)
    with PhotoStore(db) as s:
        plain = next(
            s.get(h)
            for h in s.all_hashes()
            if any(x.name == "IMG_0003.jpg" for x in s.get(h).paths)
        )
        assert plain.meta.people == []
        assert plain.meta.description is None
        assert plain.meta.tz_source is TzSource.FILE_MTIME
        target = plain.file_hash

        # Enrichment of the kind a later source (or the user) contributes.
        plain.meta.people = ["Ravi", "Meera"]
        plain.meta.description = "the afternoon on the terrace"
        plain.meta.keywords = ["terrace"]
        plain.meta.favorite = True
        plain.meta.taken_at_utc = known_date
        plain.meta.taken_at_local = known_date
        plain.meta.tz_source = TzSource.EXIF_OFFSET
        s.upsert_many([plain])

    assert runner.invoke(app, ["index", str(root), "--data-dir", str(data)]).exit_code == 0

    with PhotoStore(db) as s:
        after = s.get(target)
    assert after is not None
    assert after.meta.people == ["Ravi", "Meera"]
    assert after.meta.description == "the afternoon on the terrace"
    assert after.meta.keywords == ["terrace"]
    assert after.meta.favorite is True
    # A known capture date must not be overwritten by the mtime fallback.
    assert after.meta.taken_at_utc == known_date
    assert after.meta.tz_source is TzSource.EXIF_OFFSET


def test_index_twice_does_not_duplicate(tmp_path):
    root = build_library(tmp_path / "lib")
    data = tmp_path / "data"
    first_result = runner.invoke(app, ["index", str(root), "--data-dir", str(data)])
    assert first_result.exit_code == 0
    from rekindle.db import PhotoStore

    with PhotoStore(data / "rekindle.sqlite") as s:
        first = s.count()
    second_result = runner.invoke(app, ["index", str(root), "--data-dir", str(data)])
    assert second_result.exit_code == 0
    with PhotoStore(data / "rekindle.sqlite") as s:
        assert s.count() == first


def test_missing_directory_exits_nonzero(tmp_path):
    result = runner.invoke(app, ["doctor", str(tmp_path / "nope")])
    assert result.exit_code != 0


def test_enrich_before_index_tells_the_user_what_to_do(tmp_path):
    from tests.fixtures.takeout import build_takeout

    root = build_takeout(tmp_path / "Takeout")
    result = runner.invoke(app, ["enrich", str(root), "--data-dir", str(tmp_path / "data")])
    assert result.exit_code == 2
    assert "rekindle index" in result.stdout


def test_index_then_enrich_reports_people(tmp_path):
    from tests.fixtures.takeout import build_takeout

    root = build_takeout(tmp_path / "Takeout")
    data = str(tmp_path / "data")
    assert runner.invoke(app, ["index", str(root), "--data-dir", data]).exit_code == 0
    result = runner.invoke(app, ["enrich", str(root), "--data-dir", data])
    assert result.exit_code == 0
    assert "Sidecars seen" in result.stdout


def test_doctor_from_index_reads_the_database(tmp_path):
    from tests.fixtures.takeout import build_takeout

    root = build_takeout(tmp_path / "Takeout")
    data = str(tmp_path / "data")
    runner.invoke(app, ["index", str(root), "--data-dir", data])
    runner.invoke(app, ["enrich", str(root), "--data-dir", data])
    result = runner.invoke(app, ["doctor", str(root), "--from-index", "--data-dir", data])
    assert result.exit_code == 0
    assert "With people" in result.stdout


def test_doctor_from_index_without_an_index_creates_nothing(tmp_path):
    """M0's standing constraint - `doctor` writes nothing - applies to
    `--from-index` too. `PhotoStore.__init__` creates its file on open, so
    without an explicit existence check up front, running this against a
    machine that has never indexed would both report a fraudulent "0 photos,
    all healthy" and leave a stray database file behind."""
    root = tmp_path / "lib"
    root.mkdir()
    data = tmp_path / "data"
    result = runner.invoke(app, ["doctor", str(root), "--from-index", "--data-dir", str(data)])
    assert result.exit_code == 2
    assert not data.exists()
    assert not any(tmp_path.rglob("*.sqlite"))
