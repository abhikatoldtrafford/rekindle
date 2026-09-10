from pathlib import Path

from rich.console import Console

from rekindle.doctor import Diagnosis, diagnose, render
from rekindle.models import SourceReport


def _report(**kw) -> SourceReport:
    r = SourceReport()
    for k, v in kw.items():
        setattr(r, k, v)
    return r


def test_percentages_are_relative_to_indexed_media():
    d = diagnose(_report(media_indexed=200, with_date=100, with_gps=50, with_people=0))
    assert d.pct(d.report.with_date) == 50.0
    assert d.pct(d.report.with_gps) == 25.0


def test_zero_media_does_not_divide_by_zero():
    d = diagnose(_report(media_indexed=0, with_date=0))
    assert d.pct(0) == 0.0


def test_warns_when_no_people_data():
    # with_date=100 so the low-date warning does not fire and stand in for
    # this one - the mirror image of the trap the two tests below describe.
    d = diagnose(_report(media_indexed=100, with_date=100, with_people=0))
    assert any(w.startswith("No person data found") for w in d.warnings)


def test_warns_when_date_coverage_is_low():
    """with_people=1 on purpose: at its default of 0 the "No person data
    found in XMP sidecars... Use DATE-range and folder exclusions" warning
    also fires, and one string containing the word "date" satisfied this
    assertion on its own. Deleting the low-date block entirely left this test
    green (verified). Assert the distinctive phrase, and silence the warning
    that was standing in for it."""
    d = diagnose(_report(media_indexed=100, with_date=10, with_people=1))
    assert any(w.startswith("Low date coverage") for w in d.warnings)


def test_warns_about_long_paths(tmp_path):
    d = diagnose(_report(media_indexed=10, with_date=10, long_paths=[tmp_path]))
    assert any("long path" in w.lower() for w in d.warnings)


def test_warns_about_unignored_json_sidecars():
    """Same trap as above: the person warning names "XMP sidecars", so
    `"sidecar" in w.lower()` was satisfied whether or not this block existed.
    Match the phrase only this warning uses."""
    d = diagnose(_report(media_indexed=100, with_date=100, with_people=1, json_sidecars=40))
    assert any("Google JSON sidecars" in w for w in d.warnings)
    assert any("40" in w for w in d.warnings)


def test_healthy_library_has_no_warnings():
    d = diagnose(
        _report(media_indexed=100, with_date=100, with_gps=80, with_people=50, with_xmp=50)
    )
    assert d.warnings == []


def test_orphan_sidecars_produce_an_incomplete_export_warning():
    """The single most common way a library comes out half-empty."""
    d = diagnose(
        _report(
            media_indexed=2802,
            with_date=2802,
            with_people=1,
            json_sidecars=5013,
            orphan_sidecars=2211,
        )
    )
    hits = [w for w in d.warnings if "INCOMPLETE EXPORT" in w]
    assert len(hits) == 1
    assert "2211" in hits[0]
    assert "44.1%" in hits[0]


def test_no_orphan_warning_when_export_is_complete():
    d = diagnose(
        _report(
            media_indexed=100, with_date=100, with_people=5, json_sidecars=100, orphan_sidecars=0
        )
    )
    assert not any("INCOMPLETE EXPORT" in w for w in d.warnings)


def test_excluded_trash_is_mentioned():
    d = diagnose(_report(media_indexed=100, with_date=100, with_people=1, excluded_dirs=12))
    assert any("trash" in w.lower() for w in d.warnings)


def test_render_shows_unreadable_filenames_and_reasons():
    """DoD: 'unreadable is both populated and rendered.' Nothing previously
    tested the rendering half - a bug that dropped this block silently would
    have passed the whole suite."""
    report = _report(
        media_indexed=5,
        with_date=5,
        unreadable=[
            (Path("truncated.jpg"), "decompression bomb: too many pixels"),
            (Path("broken.heic"), "UnidentifiedImageError: cannot identify image file"),
        ],
    )
    diagnosis = Diagnosis(report=report)
    console = Console(record=True, width=200)
    render(diagnosis, console)
    output = console.export_text()

    assert "truncated.jpg" in output
    assert "decompression bomb: too many pixels" in output
    assert "broken.heic" in output
    assert "UnidentifiedImageError" in output


def test_render_caps_unreadable_list_and_shows_a_tail_count():
    report = _report(
        media_indexed=15,
        with_date=15,
        unreadable=[(Path(f"bad{i}.jpg"), "unreadable") for i in range(15)],
    )
    diagnosis = Diagnosis(report=report)
    console = Console(record=True, width=200)
    render(diagnosis, console)
    output = console.export_text()

    for i in range(10):
        assert f"bad{i}.jpg" in output
    for i in range(10, 15):
        assert f"bad{i}.jpg" not in output
    assert "... and 5 more" in output


def test_diagnose_index_reports_people_after_enrichment(tmp_path):
    from rekindle.db import PhotoStore
    from rekindle.doctor import diagnose_index
    from rekindle.enrich.takeout import TakeoutEnricher
    from rekindle.sources.folder import FolderSource
    from tests.fixtures.takeout import build_takeout

    root = build_takeout(tmp_path / "Takeout")
    photos, _ = FolderSource().scan(root)
    store = PhotoStore(tmp_path / "data" / "rekindle.sqlite")
    store.upsert_many(photos)
    store.set_meta("index_root", str(root))

    before = diagnose_index(store)
    assert before.enriched_at is None
    assert any("has not been enriched" in w for w in before.warnings)

    TakeoutEnricher().enrich(root, store)
    after = diagnose_index(store)
    assert after.with_people > 0
    assert after.enriched_at is not None
    assert not any("has not been enriched" in w for w in after.warnings)
    assert not any("No person data found" in w for w in after.warnings)
    store.close()


def test_diagnose_index_warns_about_a_foreign_enrich_root(tmp_path):
    from rekindle.db import PhotoStore
    from rekindle.doctor import diagnose_index

    store = PhotoStore(tmp_path / "data" / "rekindle.sqlite")
    store.set_meta("index_root", str(tmp_path / "A"))
    store.set_meta("enrich_root", str(tmp_path / "B"))
    assert any("different folder" in w for w in diagnose_index(store).warnings)
    store.close()


def test_the_unparsed_sidecar_warning_now_points_at_enrich(tmp_path):
    from rekindle.doctor import diagnose
    from rekindle.models import SourceReport

    d = diagnose(SourceReport(files_seen=100, media_indexed=50, json_sidecars=50, with_people=1))
    hits = [w for w in d.warnings if "rekindle enrich" in w]
    assert len(hits) == 1
    assert "not implemented yet" not in hits[0]
