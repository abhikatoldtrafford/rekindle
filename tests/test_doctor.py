import json
from pathlib import Path

from rich.console import Console

from rekindle.db import PhotoStore
from rekindle.doctor import (
    Diagnosis,
    diagnose,
    diagnose_index,
    render,
    render_enrich,
    render_index,
)
from rekindle.enrich.takeout import EnrichReport, TakeoutEnricher
from rekindle.models import SourceReport
from rekindle.sources.folder import FolderSource
from tests.fixtures.gen import make_jpeg
from tests.fixtures.takeout import build_takeout


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
    hits = [w for w in d.warnings if "name a photo that is not in this folder" in w]
    assert len(hits) == 1
    assert "2211" in hits[0]
    assert "44.1%" in hits[0]


def test_no_orphan_warning_when_export_is_complete():
    d = diagnose(
        _report(
            media_indexed=100, with_date=100, with_people=5, json_sidecars=100, orphan_sidecars=0
        )
    )
    assert not any("name a photo that is not in this folder" in w for w in d.warnings)


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


def _both_ambiguity_causes(tmp_path):
    """An export whose `ambiguous` photos come from BOTH causes.

    Cause 1 - `resolve()` refuses: two sidecars name DISAGREE.jpg and tell
    different stories, and neither sits in the photo's own directory, so
    there is nothing to break the tie with.

    Cause 2 - a cross-photo collision: TWIN.jpg names two DIFFERENT photos
    (different bytes, different folders) and there is exactly one sidecar for
    that name, in a third folder. `resolve()`, handed one photo at a time,
    has nothing to disagree with and would give the same sidecar to both;
    the enricher refuses both rather than guessing.
    """
    root = tmp_path / "Takeout"
    make_jpeg(root / "Photos from 2011" / "DISAGREE.jpg", size=(24, 24))
    for album, person, when in (
        ("Album A", "Ada", "1400000000"),
        ("Album B", "Grace", "1500000000"),
        ("Album D", "Hopper", "1600000000"),
    ):
        (root / album).mkdir(parents=True, exist_ok=True)
        (root / album / "DISAGREE.jpg.supplemental-metadata.json").write_text(
            json.dumps(
                {
                    "title": "DISAGREE.jpg",
                    "photoTakenTime": {"timestamp": when},
                    "people": [{"name": person}],
                }
            ),
            encoding="utf-8",
        )

    make_jpeg(root / "Photos from 2012" / "TWIN.jpg", size=(26, 26))
    make_jpeg(root / "Photos from 2013" / "TWIN.jpg", size=(28, 28))
    (root / "Album C").mkdir(parents=True, exist_ok=True)
    (root / "Album C" / "TWIN.jpg.supplemental-metadata.json").write_text(
        json.dumps({"title": "TWIN.jpg", "photoTakenTime": {"timestamp": "1700000000"}}),
        encoding="utf-8",
    )
    return root


def test_the_ambiguous_warning_names_both_causes_and_its_unit(tmp_path):
    """`enrich` printed `ambiguous (refused): 1` and `doctor --from-index`
    printed `Ambiguous (not enriched): 5` for one word, on one export.

    They are different units - SIDECARS against PHOTOS - and the doctor
    warning additionally claimed all of them "had two or more sidecars IN THE
    SAME FOLDER that disagreed on capture time or people". Measured on the
    reference export: `resolve()` refused 2 photos, and the other 3 were
    cross-photo collisions, a different cause entirely that the warning never
    mentioned. So it misdescribed 3 of its 5 photos, and a user running the
    two commands the README puts side by side saw `1` and `5`.

    MUTATION (run, not assumed): change any asserted fragment of the warning
    (e.g. "nothing identified" -> "NOTHING identified") and this fails.
    """
    root = _both_ambiguity_causes(tmp_path)
    photos, _ = FolderSource().scan(root)
    store = PhotoStore(tmp_path / "data" / "rekindle.sqlite")
    store.upsert_many(photos)
    report = TakeoutEnricher().enrich(root, store)
    diagnosis = diagnose_index(store)
    store.close()

    # Both causes really did fire, and the two numbers really do differ - so
    # neither half of the wording below is vacuous.
    assert report.cross_photo_collisions == 2
    assert diagnosis.ambiguous == 3  # DISAGREE.jpg + both TWIN.jpg photos
    assert report.ambiguous == 4  # 3 DISAGREE sidecars + 1 TWIN sidecar
    assert report.ambiguous != diagnosis.ambiguous

    warning = next(w for w in diagnosis.warnings if "not enriched" in w)
    assert "3 PHOTOS" in warning
    # Cause 1, no longer restricted to "IN THE SAME FOLDER".
    assert "two or more sidecars named the photo and disagreed" in warning
    assert "IN THE SAME FOLDER" not in warning
    # Cause 2, which the old text never mentioned at all.
    assert "share the filename also claimed the one sidecar" in warning
    assert "nothing identified" in warning
    # The unit mismatch, stated rather than left for the user to discover.
    assert "counts SIDECARS, not photos" in warning

    console = Console(record=True, width=200)
    render_index(diagnosis, console)
    assert "Ambiguous photos (not enriched)" in console.export_text()


def test_render_enrich_shows_the_title_disagreement_count(tmp_path):
    """`title_disagreements` was counted and never rendered - 993 on the
    reference export, and deleting the whole counting block left all 241
    tests passing (verified by mutation at final review).

    It is the measured gap between a sidecar's own `title` field and the file
    it actually describes: the discovery this milestone was rebuilt around,
    because matching on `title` mis-paired 963 photos.

    MUTATION (run, not assumed): change `report.title_disagreements += 1` to
    `+= 2` in `build_index` and this fails - the rendered row reads 2.
    """
    root = build_takeout(tmp_path / "Takeout")
    photos, _ = FolderSource().scan(root)
    store = PhotoStore(tmp_path / "data" / "rekindle.sqlite")
    store.upsert_many(photos)
    report = TakeoutEnricher().enrich(root, store)
    store.close()
    # DSC00107.JPG.supplemental-metadata(1).json describes DSC00107(1).JPG
    # while its `title` says DSC00107.JPG - the (N) case, which is the bulk
    # of the real export's 993.
    assert report.title_disagreements == 1

    console = Console(record=True, width=200)
    render_enrich(report, console)
    output = console.export_text()
    row = next(ln for ln in output.splitlines() if "Sidecar title disagreed" in ln)
    assert row.split("│")[2].strip() == "1"
    assert "1 sidecars name a file in their `title` field" in " ".join(output.split())


def test_render_enrich_labels_ambiguous_as_a_sidecar_count(tmp_path):
    """The enrich table's row and the index table's row are different units;
    each must say which."""
    report = EnrichReport(ambiguous=7, directory_preference_broke_a_tie=3)
    console = Console(record=True, width=200)
    render_enrich(report, console)
    flat = " ".join(console.export_text().split())
    assert "ambiguous sidecars (refused)" in flat
    assert "'ambiguous sidecars (refused)' count above (7)" in flat


def test_render_enrich_shows_the_conflicts_retracted_row(tmp_path):
    """Residual 6: this repeats exactly the shape Fix 7 existed to close -
    a counter computed and never asserted as rendered. Deleting
    `("EXIF/Google conflicts retracted", report.conflicts_retracted)` from
    `render_enrich` left the suite green, 258 passed.

    MUTATION (run, not assumed): delete that row from `render_enrich` and
    this fails - the row is simply absent from the output.
    """
    report = EnrichReport(conflicts=5, conflicts_retracted=2)
    console = Console(record=True, width=200)
    render_enrich(report, console)
    output = console.export_text()
    row = next(ln for ln in output.splitlines() if "EXIF/Google conflicts retracted" in ln)
    assert row.split("│")[2].strip() == "2"
