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
    d = diagnose(_report(media_indexed=100, with_people=0))
    assert any("person" in w.lower() for w in d.warnings)


def test_warns_when_date_coverage_is_low():
    d = diagnose(_report(media_indexed=100, with_date=10))
    assert any("date" in w.lower() for w in d.warnings)


def test_warns_about_long_paths(tmp_path):
    d = diagnose(_report(media_indexed=10, with_date=10, long_paths=[tmp_path]))
    assert any("long path" in w.lower() for w in d.warnings)


def test_warns_about_unignored_json_sidecars():
    d = diagnose(_report(media_indexed=100, with_date=100, json_sidecars=40))
    assert any("takeout" in w.lower() or "sidecar" in w.lower() for w in d.warnings)


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
