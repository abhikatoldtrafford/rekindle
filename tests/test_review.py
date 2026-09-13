"""Contact sheets for the publishing gate's survivors.

The gate ends with a human and the marker file says so. Saying so does not
make it possible: on the reference library the survivors are 519 photographs
scattered across the export, and nobody opens 519 files.

What these tests pin is mostly the ORDER, because the order is the design. A
reviewer who opens one sheet and stops must have seen the worst of it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from rekindle.models import MediaType, Photo, PhotoMeta
from rekindle.review import (
    PER_SHEET,
    Candidate,
    SheetReport,
    candidates,
    suspicion,
    write_sheets,
)

T0 = datetime(2020, 5, 1, 12, 0)


def _p(h, *, people=(), faces=None, media=MediaType.IMAGE) -> Photo:
    return Photo(
        file_hash=h,
        paths=[Path(f"/lib/{h}.jpg")],
        media_type=media,
        meta=PhotoMeta(
            taken_at_utc=T0.replace(tzinfo=UTC),
            taken_at_local=T0,
            people=list(people),
            face_count=faces,
        ),
        first_seen=T0.replace(tzinfo=UTC),
        last_seen=T0.replace(tzinfo=UTC),
    )


# --------------------------------------------------------------------------
# the order


def test_never_examined_comes_first():
    """No detector evidence at all - only the tags vouch for it."""
    rank, reason = suspicion(_p("a", people=["Abhik"], faces=None))
    assert rank == 0
    assert "never examined" in reason


def test_a_tagged_photo_the_detector_found_NOTHING_in_outranks_a_crowd():
    """The case a naive ranking puts last, and the one that matters most.

    Somebody is known to be in this photograph and the detector could not see
    them - so it is demonstrably missing faces in THIS image, and any other
    person in it would have been missed too. A photograph with nine faces and
    nine tags is comparatively well understood.
    """
    blind, _ = suspicion(_p("a", people=["Abhik"], faces=0))
    crowd, _ = suspicion(_p("b", people=["A", "B", "C"], faces=9))
    assert blind < crowd


def test_more_faces_beats_fewer_among_the_rest():
    many, _ = suspicion(_p("a", people=["Abhik"], faces=8))
    few, _ = suspicion(_p("b", people=["Abhik"], faces=1))
    assert many < few


def test_an_untagged_photo_with_no_faces_is_not_flagged_as_blind():
    """The `0 faces` rule needs a TAG to be evidence of a miss. A landscape
    with no tags and no faces is exactly what it looks like."""
    rank, _ = suspicion(_p("a", people=(), faces=0))
    assert rank > 1


def test_the_order_is_stable_across_runs():
    """The sheets are an artefact like any other. A reviewer comparing two
    runs should see a diff only where the library changed, so ties break on
    `file_hash` rather than on dictionary order."""
    photos = [_p(h, people=["Abhik"], faces=2) for h in ("ccc", "aaa", "bbb")]
    assert [c.photo.file_hash for c in candidates(photos)] == ["aaa", "bbb", "ccc"]
    assert [c.photo.file_hash for c in candidates(list(reversed(photos)))] == [
        "aaa",
        "bbb",
        "ccc",
    ]


# --------------------------------------------------------------------------
# what is drawn, and what is only counted


def _resolver(mapping):
    return lambda photo: mapping.get(photo.file_hash)


def test_videos_are_counted_and_left_off_the_grid(tmp_path):
    """They rank as "never examined" - correctly, the detector reads a cached
    still and these have none - and that put 47 undrawable red boxes in the
    first 48 cells of the reference library's sheet 1. Sheet 1 was entirely
    things that cannot be published and sheet 2 was where the real risks
    started, which defeats the whole point of the ordering.
    """
    pytest.importorskip("PIL")
    photos = [_p(f"v{i}", people=["Abhik"], media=MediaType.VIDEO) for i in range(3)]
    photos.append(_p("still", people=["Abhik"], faces=1))

    report = write_sheets(photos, tmp_path, resolve_path=_resolver({}))

    assert report.admitted == 4
    assert report.videos == 3
    assert report.never_examined == 0, "a video is not a photograph awaiting the detector"
    assert report.accounted, (report.admitted, report.drawn, report.unreadable, report.videos)


def test_a_photo_that_will_not_decode_is_marked_not_skipped(tmp_path):
    """It was admitted by the gate, so a reviewer who never sees it has not
    reviewed it. It gets a cell saying so rather than vanishing."""
    pytest.importorskip("PIL")
    report = write_sheets(
        [_p("gone", people=["Abhik"], faces=1)], tmp_path, resolve_path=_resolver({})
    )
    assert report.unreadable == 1
    assert report.drawn == 0
    assert report.sheets, "a sheet is still written, with the cell marked"
    assert report.accounted


def test_nothing_admitted_writes_nothing(tmp_path):
    report = write_sheets([], tmp_path, resolve_path=_resolver({}))
    assert report.sheets == []
    assert report.accounted


def test_the_sheets_are_split_at_the_declared_size(tmp_path):
    pytest.importorskip("PIL")
    photos = [_p(f"h{i:03d}", people=["Abhik"], faces=1) for i in range(PER_SHEET + 1)]
    report = write_sheets(photos, tmp_path, resolve_path=_resolver({}))
    assert len(report.sheets) == 2
    assert all(p.is_file() for p in report.sheets)


def test_a_real_photograph_is_drawn(tmp_path):
    """The one test that proves the grid contains pixels rather than boxes."""
    pytest.importorskip("PIL")
    from tests.fixtures.gen import make_jpeg

    source = make_jpeg(tmp_path / "src" / "a.jpg", color=(10, 200, 40))
    report = write_sheets(
        [_p("a", people=["Abhik"], faces=1)],
        tmp_path / "out",
        resolve_path=_resolver({"a": source}),
    )
    assert report.drawn == 1
    assert report.unreadable == 0

    from PIL import Image

    with Image.open(report.sheets[0]) as sheet:
        colours = {c for _n, c in sheet.convert("RGB").getcolors(maxcolors=100000)}
    greens = [c for c in colours if c[1] > 150 and c[0] < 90 and c[2] < 90]
    assert greens, "the photograph itself is not on the sheet"


def test_the_report_counts_the_two_things_a_reviewer_must_be_told(tmp_path):
    pytest.importorskip("PIL")
    photos = [
        _p("unseen", people=["Abhik"], faces=None),
        _p("blind", people=["Abhik"], faces=0),
        _p("fine", people=["Abhik"], faces=1),
    ]
    report = write_sheets(photos, tmp_path, resolve_path=_resolver({}))
    assert report.never_examined == 1
    assert report.detector_saw_nothing == 1


def test_a_sheet_report_defaults_to_an_empty_list_not_a_shared_one():
    """A mutable default on a dataclass is shared between instances; this one
    is built in `__post_init__` for that reason."""
    first, second = SheetReport(), SheetReport()
    first.sheets.append(Path("x"))
    assert second.sheets == []


def test_candidate_counts_only_real_names():
    """An empty string in `people` is not a person. The tag count feeds the
    gate's comparison, so counting a blank would let a photograph through."""
    assert Candidate(photo=_p("a", people=["", "Abhik"]), reason="", rank=2).tags == 1
