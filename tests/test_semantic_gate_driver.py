"""`gate_photos` - the driver that decides which photos a human may publish.

WHY THIS FILE EXISTS
--------------------
`tests/test_semantic_faces.py` tests `classify`, `_nms`, `_iou`, `Box` and
`precision_recall` exhaustively, and every one of those is worth testing. But
NOTHING tested `gate_photos` itself, and `gate_photos` is where the gate's
policy actually lives:

  * the allow-list pre-filter, which is what stops a photo tagged with a
    stranger from ever reaching the detector;
  * what happens to a file that is missing, or unreadable, or that makes the
    detector raise;
  * whether every considered photo lands in exactly one bucket.

Found by mutation: replacing `if tagged - allowed:` with `if False:` - which
removes the allow-list check entirely and sends photos of known strangers to
the detector to be judged on their pixels - left the whole suite GREEN. That is
the precise failure this milestone is supposed to be incapable of, so it gets
its own file.

`ScriptedDetector` is not a mock of the ONNX session. It is a real detector
over a real, if trivial, feature - the image's own red and blue channels - so
every assertion below is about the VERDICT the gate reached, never about a
call having been made. No model, no weights, no GPU, no network.

Photos go in through `make_index` and come back out through
`PhotoIndexReader`, exactly as `rekindle semantic gate` gets them, so no test
here can invent a row shape the real index does not produce.
"""

from __future__ import annotations

import pytest

from rekindle.semantic.faces import (
    DEFAULT_DETECT_THRESHOLD,
    DEFAULT_GATE_THRESHOLD,
    Box,
    Verdict,
    gate_photos,
)
from rekindle.semantic.photos import PhotoIndexReader, ReadFilter
from tests.fixtures.semantic import make_index, make_photo, write_photo

#: Blue above this makes the scripted detector fail on that image. A property
#: of the picture, not a hook the test reaches into the detector to set.
BLUE_FAILS = 200


class _Spec:
    key = "scripted"


class ScriptedDetector:
    """A real detector over a real feature: how red the image is.

    Redness maps to a detection score, so a test can put a photo of known
    redness in front of the gate and know exactly which verdict is correct.
    Anything the gate does on the way - thresholds, the allow-list
    short-circuit, error handling - shows up in the verdict.

    A very blue image makes it raise, which is how "the detector fell over on
    one photo" is exercised without a fake.
    """

    def __init__(self) -> None:
        self.spec = _Spec()
        self.size = 64
        self.decoded = 0

    def detect(self, image, *, detect_threshold, gate_threshold, **_):
        self.decoded += 1
        r, _g, b = image.convert("RGB").resize((1, 1)).getpixel((0, 0))
        if b > BLUE_FAILS:
            raise RuntimeError("the detector fell over on this image")
        score = r / 255.0
        if score < min(detect_threshold, gate_threshold):
            return (), 0.0
        return (Box(0.0, 0.0, 10.0, 10.0, score),), score


def redness(score: float) -> tuple[int, int, int]:
    """A colour the scripted detector scores at (about) `score`."""
    return (round(score * 255), 0, 0)


EXPLODES = (0, 0, 255)


def gate(tmp_path, specs, **kw):
    """Build an index from `(name, colour, people)` and run the real gate."""
    photos = []
    for name, colour, people in specs:
        if colour is None:  # a row whose file was never written
            path = tmp_path / "lib" / f"{name}.jpg"
        elif colour == "broken":
            path = tmp_path / "lib" / f"{name}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"this is not a JPEG")
        else:
            path = write_photo(tmp_path / "lib", f"{name}.jpg", colour)
        photos.append(make_photo(name.ljust(32, "a"), path, people=list(people)))
    db = tmp_path / "data" / "rekindle.sqlite"
    make_index(db, photos)
    detector = kw.pop("detector", None) or ScriptedDetector()
    with PhotoIndexReader(db) as reader:
        report = gate_photos(list(reader.iter_photos(ReadFilter())), detector, **kw)
    return report, detector


def by_name(report):
    return {d.path.stem: d for d in report.detections}


# --------------------------------------------------------- the allow-list


def test_a_photo_tagged_with_a_stranger_never_reaches_the_detector(tmp_path):
    """The line the mutation removed, and the reason this file exists.

    A photo Google already tagged with somebody outside the allow-list is
    HAS_FACE on the tag alone. It must be blocked WITHOUT being decoded - both
    because the tag is already proof of a face, and because letting the
    detector overrule a known stranger is how a stranger gets published.
    """
    report, detector = gate(
        tmp_path, [("stranger", redness(0.0), ["Someone Else"])], allow_people=["Abhik Maiti"]
    )
    assert report.detections[0].verdict is Verdict.HAS_FACE
    assert report.has_face == 1 and report.eligible == 0
    assert detector.decoded == 0, "a photo of a known stranger was sent to the detector"


def test_an_empty_allow_list_blocks_every_tagged_photo(tmp_path):
    """Default deny. With nobody allowed, any tag at all is a stranger."""
    report, detector = gate(tmp_path, [("tagged", redness(0.0), ["Abhik Maiti"])])
    assert report.detections[0].verdict is Verdict.HAS_FACE
    assert detector.decoded == 0


def test_a_photo_tagged_only_with_allowed_people_still_goes_to_the_detector(tmp_path):
    """Google's tags are recall-poor.

    A photo tagged with the owner may also contain three untagged strangers,
    so an allowed tag buys a look from the detector, never a free pass. Here
    the pixels say "face", and the verdict must follow the pixels.
    """
    report, detector = gate(
        tmp_path, [("owner", redness(0.9), ["Abhik Maiti"])], allow_people=["Abhik Maiti"]
    )
    assert detector.decoded == 1
    assert report.detections[0].verdict is Verdict.HAS_FACE


def test_an_allowed_photo_with_no_face_in_the_pixels_becomes_eligible(tmp_path):
    report, detector = gate(
        tmp_path, [("owner", redness(0.0), ["Abhik Maiti"])], allow_people=["Abhik Maiti"]
    )
    assert detector.decoded == 1
    assert report.detections[0].verdict is Verdict.ELIGIBLE
    assert report.eligible == 1


def test_the_allow_list_ignores_case_but_not_identity(tmp_path):
    report, _ = gate(
        tmp_path,
        [("aa", redness(0.0), ["abhik maiti"]), ("bb", redness(0.0), ["Abhik Maity"])],
        allow_people=["Abhik Maiti"],
    )
    found = by_name(report)
    assert found["aa"].verdict is Verdict.ELIGIBLE, "case alone must not block"
    assert found["bb"].verdict is Verdict.HAS_FACE, "a different name must block"


def test_one_disallowed_tag_among_allowed_ones_blocks(tmp_path):
    """Set difference, not intersection. Any stranger is a stranger."""
    report, detector = gate(
        tmp_path,
        [("mixed", redness(0.0), ["Abhik Maiti", "Someone Else"])],
        allow_people=["Abhik Maiti"],
    )
    assert report.detections[0].verdict is Verdict.HAS_FACE
    assert detector.decoded == 0


# ------------------------------------------------------------ the verdicts


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.0, Verdict.ELIGIBLE),
        (0.10, Verdict.ELIGIBLE),
        (0.30, Verdict.UNCERTAIN),
        (0.44, Verdict.UNCERTAIN),
        (0.90, Verdict.HAS_FACE),
    ],
)
def test_the_gate_maps_detector_confidence_to_the_right_bucket(tmp_path, score, expected):
    """Straddles both thresholds, through the real driver.

    `classify` is tested directly elsewhere; this checks the driver really
    routes the detector's answer through it.
    """
    report, _ = gate(tmp_path, [("s", redness(score), [])])
    assert report.detections[0].verdict is expected


def test_uncertain_is_never_publishable(tmp_path):
    report, _ = gate(tmp_path, [("weak", redness(0.30), [])])
    assert report.uncertain == 1
    assert not report.detections[0].publishable
    assert report.detections[0] in report.review_queue()


def test_a_tighter_gate_threshold_is_passed_through_not_ignored(tmp_path):
    faint = [("faint", redness(0.06), [])]
    loose, _ = gate(tmp_path / "a", faint, gate_threshold=DEFAULT_GATE_THRESHOLD)
    tight, _ = gate(tmp_path / "b", faint, gate_threshold=0.05)
    assert loose.detections[0].verdict is Verdict.ELIGIBLE
    assert tight.detections[0].verdict is Verdict.UNCERTAIN


def test_a_tighter_detect_threshold_is_passed_through_not_ignored(tmp_path):
    mid = [("mid", redness(0.50), [])]
    loose, _ = gate(tmp_path / "a", mid, detect_threshold=DEFAULT_DETECT_THRESHOLD)
    tight, _ = gate(tmp_path / "b", mid, detect_threshold=0.80)
    assert loose.detections[0].verdict is Verdict.HAS_FACE
    assert tight.detections[0].verdict is Verdict.UNCERTAIN


# ------------------------------------------------------------- failure modes


def test_a_missing_file_is_an_error_and_never_eligible(tmp_path):
    """Default deny extends to "I could not look".

    A photo whose file has moved must not become publishable by default. An
    ERROR that fell through to ELIGIBLE would publish a photo nothing ever
    examined - which is the direction that hurts.
    """
    report, detector = gate(tmp_path, [("gone", None, [])])
    assert report.detections[0].verdict is Verdict.ERROR
    assert not report.detections[0].publishable
    assert report.errors == 1 and report.eligible == 0
    assert detector.decoded == 0


def test_an_unreadable_file_is_an_error_and_never_eligible(tmp_path):
    report, _ = gate(tmp_path, [("bad", "broken", [])])
    assert report.detections[0].verdict is Verdict.ERROR
    assert report.errors == 1 and report.eligible == 0
    assert report.detections[0].error, "an error verdict with no reason is not reviewable"


def test_a_detector_that_raises_does_not_end_the_run_and_does_not_publish(tmp_path):
    """One bad photo among 18,000 must not stop the scan - nor sneak through."""
    report, _ = gate(
        tmp_path,
        [("aa", redness(0.0), []), ("bb", EXPLODES, []), ("cc", redness(0.0), [])],
    )
    found = by_name(report)
    assert found["bb"].verdict is Verdict.ERROR
    assert not found["bb"].publishable
    assert found["aa"].verdict is Verdict.ELIGIBLE
    assert found["cc"].verdict is Verdict.ELIGIBLE, "the run stopped at the failure"
    assert report.accounted


# ---------------------------------------------------------------- accounting


def test_every_photo_lands_in_exactly_one_bucket(tmp_path):
    """The accounting identity, over one of each kind at once."""
    report, _ = gate(
        tmp_path,
        [
            ("aclean", redness(0.0), []),
            ("bweak", redness(0.30), []),
            ("cface", redness(0.90), []),
            ("dstrange", redness(0.0), ["Someone Else"]),
            ("eboom", EXPLODES, []),
            ("fbad", "broken", []),
            ("ggone", None, []),
        ],
        allow_people=["Abhik Maiti"],
    )
    assert report.considered == 7
    assert report.eligible == 1
    assert report.uncertain == 1
    assert report.has_face == 2
    assert report.errors == 3
    assert report.accounted
    assert len(report.detections) == 7


def test_progress_is_reported_for_every_photo_including_the_skipped_ones(tmp_path):
    seen: list[tuple[int, int]] = []
    gate(
        tmp_path,
        [("aa", redness(0.0), []), ("bb", redness(0.0), ["Someone Else"]), ("cc", None, [])],
        progress=lambda n, t: seen.append((n, t)),
    )
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_the_review_queue_shows_every_photo_a_human_must_look_at(tmp_path):
    """Including the eligible ones - the detector proposes, the user disposes."""
    report, _ = gate(tmp_path, [("clean", redness(0.0), []), ("weak", redness(0.30), [])])
    queue = report.review_queue()
    assert {d.path.stem for d in queue} == {"clean", "weak"}
    assert queue[0].verdict is Verdict.UNCERTAIN, "uncertain must be looked at first"


def test_nothing_in_the_report_publishes_anything(tmp_path):
    """`publishable` is a proposal, and it is true for ELIGIBLE and nothing else."""
    report, _ = gate(
        tmp_path,
        [
            ("aclean", redness(0.0), []),
            ("bweak", redness(0.30), []),
            ("cface", redness(0.90), []),
            ("dboom", EXPLODES, []),
        ],
    )
    for d in report.detections:
        assert d.publishable == (d.verdict is Verdict.ELIGIBLE)
    assert sum(d.publishable for d in report.detections) == 1


def test_the_defaults_keep_the_gate_threshold_below_the_detect_threshold():
    """The asymmetry is the design.

    A refactor that equalised the two would be a silent loosening: a detection
    between them would stop being held back for review.
    """
    assert DEFAULT_GATE_THRESHOLD < DEFAULT_DETECT_THRESHOLD


def test_the_report_records_the_thresholds_it_actually_used(tmp_path):
    """A count whose threshold cannot be recovered is not a reviewable number."""
    report, _ = gate(tmp_path, [("aa", redness(0.0), [])], detect_threshold=0.7, gate_threshold=0.2)
    assert report.detect_threshold == 0.7
    assert report.gate_threshold == 0.2
    assert report.model_key == "scripted"


def test_an_empty_run_is_accounted_and_publishes_nothing():
    report = gate_photos([], ScriptedDetector())
    assert report.considered == 0 and report.accounted
    assert report.detections == []
    assert report.review_queue() == []


def test_the_image_size_is_recorded_so_boxes_can_be_drawn(tmp_path):
    """Boxes are in original pixels.

    Without the size they cannot be rendered onto the photo, and a gate whose
    output a human cannot eyeball is not a reviewable gate.
    """
    path = write_photo(tmp_path / "lib", "wide.jpg", redness(0.9), size=96)
    db = tmp_path / "data" / "rekindle.sqlite"
    make_index(db, [make_photo("w".ljust(32, "a"), path)])
    with PhotoIndexReader(db) as reader:
        report = gate_photos(list(reader.iter_photos(ReadFilter())), ScriptedDetector())
    assert report.detections[0].image_size == (96, 96)
    assert report.detections[0].face_count == 1


def test_archived_photos_never_reach_the_gate(tmp_path):
    """162 photos on the reference export are archived - deliberately hidden.

    The exclusion lives in `ReadFilter`, but the gate is where it would hurt
    most, so it is asserted here too.
    """
    live = write_photo(tmp_path / "lib", "live.jpg", redness(0.0))
    hidden = write_photo(tmp_path / "lib", "hidden.jpg", redness(0.0))
    db = tmp_path / "data" / "rekindle.sqlite"
    make_index(
        db,
        [
            make_photo("l".ljust(32, "a"), live),
            make_photo("h".ljust(32, "a"), hidden, archived=True),
        ],
    )
    with PhotoIndexReader(db) as reader:
        report = gate_photos(list(reader.iter_photos(ReadFilter())), ScriptedDetector())
    assert {d.path.stem for d in report.detections} == {"live"}
