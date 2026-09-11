from pathlib import Path

from typer.testing import CliRunner

from rekindle.cli import app
from rekindle.db import PhotoStore
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


def test_enrich_against_a_foreign_root_warns(tmp_path):
    """Every other CLI test indexes and enriches the SAME root string, so the
    `indexed_root != str(root)` branch in `cli.enrich` never fires without
    this test - confirmed by deleting the block and re-running the full
    suite unchanged (see task-12 review)."""
    from tests.fixtures.takeout import build_takeout

    root_a = build_takeout(tmp_path / "A")
    root_b = tmp_path / "B"
    root_b.mkdir()
    data = str(tmp_path / "data")
    assert runner.invoke(app, ["index", str(root_a), "--data-dir", data]).exit_code == 0
    result = runner.invoke(app, ["enrich", str(root_b), "--data-dir", data])
    assert result.exit_code == 0
    # Rich soft-wraps long lines at whitespace to the terminal width CliRunner
    # reports (~80 cols); the tmp_path fixture's username ("MY PC") itself
    # contains a space, so a literal `str(root_a) in result.stdout` check is
    # one wrap away from a false negative. Collapse whitespace runs first, as
    # `test_render_shows_unreadable_filenames_and_reasons` sidesteps the same
    # hazard by only ever asserting whitespace-free fragments.
    flat = " ".join(result.stdout.split())
    assert "index was built from" in flat
    assert " ".join(str(root_a).split()) in flat


def test_doctor_from_index_against_a_foreign_root_warns(tmp_path):
    """Mirror of the above for `doctor --from-index`: the CLI-argument-vs-
    stored-index_root comparison, not the store-vs-store one already covered
    by test_diagnose_index_warns_about_a_foreign_enrich_root."""
    from tests.fixtures.takeout import build_takeout

    root_a = build_takeout(tmp_path / "A")
    root_b = tmp_path / "B"
    root_b.mkdir()
    data = str(tmp_path / "data")
    assert runner.invoke(app, ["index", str(root_a), "--data-dir", data]).exit_code == 0
    result = runner.invoke(app, ["doctor", str(root_b), "--from-index", "--data-dir", data])
    assert result.exit_code == 0
    # See the comment in test_enrich_against_a_foreign_root_warns: normalise
    # away Rich's soft-wrapping before matching a long path.
    flat = " ".join(result.stdout.split())
    assert "This index was built from" in flat
    assert " ".join(str(root_a).split()) in flat


def test_enrich_before_index_creates_no_stray_database(tmp_path):
    """Mirror of `test_doctor_from_index_without_an_index_creates_nothing`:
    `PhotoStore.__init__` creates its file unconditionally, so without an
    explicit existence check up front, `rekindle enrich` before `rekindle
    index` correctly exits 2 but used to leave a stray, empty
    rekindle.sqlite behind (flagged in task-12 review)."""
    from tests.fixtures.takeout import build_takeout

    root = build_takeout(tmp_path / "Takeout")
    data = tmp_path / "data"
    result = runner.invoke(app, ["enrich", str(root), "--data-dir", str(data)])
    assert result.exit_code == 2
    assert not data.exists()
    assert not any(tmp_path.rglob("*.sqlite"))


def test_enrich_against_an_indexed_but_empty_folder_still_guards(tmp_path):
    """The `db_path.is_file()` check above only catches "index never ran".
    This exercises the other case `EmptyIndexError` guards: `index` DID run,
    but against a folder with no media at all, so the database exists with
    zero rows."""
    empty_root = tmp_path / "Empty"
    empty_root.mkdir()
    data = str(tmp_path / "data")
    assert runner.invoke(app, ["index", str(empty_root), "--data-dir", data]).exit_code == 0
    result = runner.invoke(app, ["enrich", str(empty_root), "--data-dir", data])
    assert result.exit_code == 2
    assert "index is empty" in result.stdout.lower()


def test_a_relative_root_is_stored_and_compared_as_an_absolute_path(tmp_path, monkeypatch):
    """`rekindle index Takeout` used to store relative paths.

    `FolderSource` stores whatever Path it is handed, so `FolderSource().scan(
    Path("Takeout"))` stored `Takeout/Photos from 2019/A.jpg` and
    `is_absolute()` was False - falsifying the comment in `cli.enrich` that
    reads "Photo paths are absolute". The consequence was worse than the
    wrong comment: `rekindle index Takeout` then `rekindle enrich Takeout`
    FROM A DIFFERENT DIRECTORY compared two equal strings ("Takeout" ==
    "Takeout"), fired no foreign-root warning, and matched nothing at all.

    MUTATION (run, not assumed): change `_resolved` to `return root` and this
    fails on `stored.is_absolute()`.
    """
    from tests.fixtures.takeout import build_takeout

    build_takeout(tmp_path / "Takeout")
    data = str(tmp_path / "data")

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["index", "Takeout", "--data-dir", data]).exit_code == 0

    with PhotoStore(Path(data) / "rekindle.sqlite") as store:
        stored = next(iter(store.iter_photos())).paths[0]
        index_root = store.get_meta("index_root")
    assert stored.is_absolute()
    assert Path(index_root).is_absolute()
    assert Path(index_root) == (tmp_path / "Takeout").resolve()

    # Enriching the SAME folder from a DIFFERENT cwd must still match - this
    # is the run that silently matched nothing before.
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    result = runner.invoke(app, ["enrich", str(tmp_path / "Takeout"), "--data-dir", data])
    assert result.exit_code == 0
    flat = " ".join(result.stdout.split())
    assert "The index was built from" not in flat
    assert "Photos enriched | 10" in flat.replace("│", "|")


def test_a_relative_root_still_warns_when_it_resolves_somewhere_else(tmp_path, monkeypatch):
    """The mirror of the above: two DIFFERENT folders that share a relative
    spelling must still fire the foreign-root warning, which the raw string
    compare could never do."""
    from tests.fixtures.takeout import build_takeout

    a = tmp_path / "A"
    a.mkdir()
    build_takeout(a / "Takeout")
    b = tmp_path / "B"
    b.mkdir()
    (b / "Takeout").mkdir()
    data = str(tmp_path / "data")

    monkeypatch.chdir(a)
    assert runner.invoke(app, ["index", "Takeout", "--data-dir", data]).exit_code == 0
    monkeypatch.chdir(b)
    result = runner.invoke(app, ["enrich", "Takeout", "--data-dir", data])
    assert result.exit_code == 0
    assert "index was built from" in " ".join(result.stdout.split())
