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
