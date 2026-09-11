"""The real YOLOv11n face detector, on real photos, when both are present.

SKIPPED unless the pinned detector weights are in `./data/models` AND an
index with reachable photos is at `./data/rekindle.sqlite`. CI has neither and
stays green.

WHY A REAL-PHOTO TEST, WHEN `_nms`, `_iou`, `classify` AND `Box` ARE ALREADY
TESTED EXHAUSTIVELY WITHOUT ONE
------------------------------------------------------------------------
Because mutation testing found that deleting the `_nms` CALL from
`FaceDetector.detect` left the whole suite green. `_nms` was thoroughly
tested; its being used was not, and `ScriptedDetector` in the gate-driver
tests replaces the method that would use it.

That gap is not academic. Over 300 random real photos at the shipped gate
threshold, **NMS removed at least one box on 220 of them, and on one photo it
reduced 206 raw boxes to 33.** Without it the gate would report a landscape as
containing two hundred faces, `faces_found` would be meaningless, and the
boxes a human is supposed to eyeball would be unreadable.

A synthetic drawing cannot substitute. It was tried: a hand-drawn face scores
0.0033 on this detector, which is confident-or-silent by design (the audit's
threshold sweep says the same thing). Only real photographs exercise this
path, which is precisely the decision log's point that every genuine bug in
this project was found by running against real data.

These tests assert INVARIANTS, never specific photos or counts. Nothing here
depends on this library in particular, and nothing here is committed from it.
"""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest

from rekindle.semantic.faces import (
    DEFAULT_DETECT_THRESHOLD,
    DEFAULT_GATE_THRESHOLD,
    DEFAULT_NMS_IOU,
    Verdict,
    _iou,
    classify,
    face_model,
    gate_photos,
)

CACHE = Path("data/models").resolve()
INDEX = Path("data/rekindle.sqlite").resolve()
SAMPLE = 40
SEED = 20260911


def _weights_present() -> bool:
    try:
        repo = face_model(None).pin.repo_id.replace("/", "--")
    except Exception:  # noqa: BLE001 - registry shape is checked elsewhere
        return False
    snaps = CACHE / f"models--{repo}" / "snapshots"
    return snaps.is_dir() and any(snaps.iterdir())


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is None
    or importlib.util.find_spec("numpy") is None
    or not _weights_present()
    or not INDEX.is_file(),
    reason="needs onnxruntime, the pinned detector weights and a real index",
)


@pytest.fixture(scope="module")
def detector():
    from rekindle.semantic.faces import load_detector

    return load_detector(CACHE)


@pytest.fixture(scope="module")
def photos():
    """A deterministic sample of real photos whose files are actually there."""
    from rekindle.semantic.photos import PhotoIndexReader, ReadFilter

    with PhotoIndexReader(INDEX) as reader:
        live = list(reader.iter_photos(ReadFilter()))
    rng = random.Random(SEED)
    rng.shuffle(live)
    out = []
    for photo in live:
        if photo.existing_path() is not None:
            out.append(photo)
        if len(out) >= SAMPLE:
            break
    if len(out) < SAMPLE:
        pytest.skip(f"only {len(out)} of {SAMPLE} sampled photos are on disk")
    return out


def _images(photos, detector):
    from PIL import Image as PILImage

    for photo in photos:
        path = photo.existing_path()
        with PILImage.open(path) as im:
            im.draft("RGB", (detector.size * 2, detector.size * 2))
            yield path, im.convert("RGB")


# ------------------------------------------------------------------ NMS runs


def test_no_two_returned_boxes_are_the_same_face(detector, photos):
    """The invariant that proves non-maximum suppression was applied.

    The raw YOLO head emits one candidate per anchor, so a single face
    produces a cluster of near-identical boxes - up to 206 on one photo of
    this library. If the `_nms` call is ever dropped, this assertion fires on
    the first crowded photo.
    """
    checked = 0
    for path, image in _images(photos, detector):
        boxes, _ = detector.detect(
            image,
            detect_threshold=DEFAULT_DETECT_THRESHOLD,
            gate_threshold=DEFAULT_GATE_THRESHOLD,
        )
        for i, a in enumerate(boxes):
            for b in boxes[i + 1 :]:
                assert _iou(a, b) <= DEFAULT_NMS_IOU, (
                    f"{path.name}: two boxes overlap by {_iou(a, b):.2f}; NMS did not run"
                )
        checked += len(boxes)
    assert checked > 0, "no detections at all in the sample; this test proved nothing"


def test_boxes_come_back_sorted_by_confidence(detector, photos):
    """A human reviewing a queue reads the top box first."""
    for _path, image in _images(photos, detector):
        boxes, best = detector.detect(
            image,
            detect_threshold=DEFAULT_DETECT_THRESHOLD,
            gate_threshold=DEFAULT_GATE_THRESHOLD,
        )
        scores = [b.score for b in boxes]
        assert scores == sorted(scores, reverse=True)
        assert best == (scores[0] if scores else 0.0)


# ----------------------------------------------------- the letterbox undo


def test_every_box_lands_inside_the_photo_it_came_from(detector, photos):
    """Boxes are returned in ORIGINAL pixels, after undoing scale and padding.

    Get that arithmetic wrong and the gate still works - the verdict only
    reads scores - while every box drawn for the human lands in the wrong
    place. A reviewer who cannot trust the boxes cannot review, so this is a
    safety property, not cosmetics.
    """
    for path, image in _images(photos, detector):
        width, height = image.size
        boxes, _ = detector.detect(
            image,
            detect_threshold=DEFAULT_DETECT_THRESHOLD,
            gate_threshold=DEFAULT_GATE_THRESHOLD,
        )
        for b in boxes:
            assert 0 <= b.x1 < b.x2 <= width, f"{path.name}: x out of bounds {b}"
            assert 0 <= b.y1 < b.y2 <= height, f"{path.name}: y out of bounds {b}"
            assert b.area > 0


@pytest.fixture(scope="module")
def confident(detector, photos):
    """The first sampled photo with one clear, confident detection.

    One box, not several, so "the box" is unambiguous and a geometry test can
    compare positions rather than sets.
    """
    for path, image in _images(photos, detector):
        boxes, _ = detector.detect(image, detect_threshold=0.60, gate_threshold=0.60)
        if len(boxes) == 1:
            return path, image, boxes[0]
    pytest.skip("no photo in the sample has exactly one confident face")
    return None


def _one_box(detector, image, threshold=0.40):
    boxes, _ = detector.detect(image, detect_threshold=threshold, gate_threshold=threshold)
    return boxes[0] if boxes else None


def test_moving_the_photo_moves_the_box_by_exactly_that_much(detector, confident):
    """PINS THE PADDING UNDO, which a bounds check cannot.

    Boxes come back in the ORIGINAL photo's pixels, which means `detect` has
    to subtract the letterbox padding and divide by the letterbox scale. Get
    either wrong and every box is still inside the image - the code clamps
    them - so it still passes every "is it in bounds" assertion while landing
    on the wrong part of the picture. A reviewer would be shown a box around
    somebody's elbow.

    So: paste the photo at a KNOWN offset on a larger canvas. The face has
    moved by exactly (dx, dy) and the box must move with it.
    """
    from PIL import Image as PILImage

    path, image, box = confident
    w, h = image.size
    dx, dy = w // 4, h // 3
    canvas = PILImage.new("RGB", (w + dx, h + dy), (114, 114, 114))
    canvas.paste(image, (dx, dy))

    moved = _one_box(detector, canvas)
    assert moved is not None, f"{path.name}: the face was lost by being moved"

    tolerance = 0.06 * max(w, h)
    assert abs(moved.x1 - (box.x1 + dx)) < tolerance, (
        f"{path.name}: box moved to x1={moved.x1:.0f}, expected {box.x1 + dx:.0f}"
    )
    assert abs(moved.y1 - (box.y1 + dy)) < tolerance, (
        f"{path.name}: box moved to y1={moved.y1:.0f}, expected {box.y1 + dy:.0f}"
    )


def test_doubling_the_photo_doubles_the_box(detector, confident):
    """PINS THE SCALE UNDO.

    The same face, twice the size, must come back as a box at twice the
    coordinates and twice the width. Dividing by the wrong scale - or not
    dividing at all - leaves the box in letterbox space, where it is still
    inside the image and still completely wrong.
    """
    from PIL import Image as PILImage

    path, image, box = confident
    w, h = image.size
    big = image.resize((w * 2, h * 2), PILImage.BICUBIC)

    doubled = _one_box(detector, big)
    assert doubled is not None, f"{path.name}: the face was lost by being enlarged"

    tolerance = 0.10 * max(w, h) * 2
    assert abs(doubled.x1 - box.x1 * 2) < tolerance, (
        f"{path.name}: x1 {doubled.x1:.0f}, expected about {box.x1 * 2:.0f}"
    )
    assert abs(doubled.width - box.width * 2) < tolerance, (
        f"{path.name}: width {doubled.width:.0f}, expected about {box.width * 2:.0f}"
    )
    assert abs(doubled.y1 - box.y1 * 2) < tolerance, (
        f"{path.name}: y1 {doubled.y1:.0f}, expected about {box.y1 * 2:.0f}"
    )
    assert abs(doubled.height - box.height * 2) < tolerance, (
        f"{path.name}: height {doubled.height:.0f}, expected about {box.height * 2:.0f}"
    )


def test_a_wide_photo_is_letterboxed_not_squashed(detector, confident):
    """The reason `_letterbox` exists rather than a square resize.

    A face squashed by a 16:9 -> 1:1 resize is a face the detector may miss,
    and for a gate a miss is a stranger published. Padding the photo out to a
    much wider frame must keep the face where it was, at the size it was -
    which a square resize, or a scale picked with `max` instead of `min`,
    does not.
    """
    from PIL import Image as PILImage

    path, image, box = confident
    w, h = image.size
    # 4:1, with the photo hard against the left edge. A scale picked with
    # `max` instead of `min` overflows the 640 canvas and PIL crops the
    # overflow from the CENTRE, so a face this far left is cut away entirely
    # and the detector never sees it. 2:1 was not enough to lose it; 4:1 is.
    canvas = PILImage.new("RGB", (w * 4, h), (114, 114, 114))
    canvas.paste(image, (0, 0))

    wide = _one_box(detector, canvas)
    assert wide is not None, f"{path.name}: the face was lost in a wider frame"

    tolerance = 0.10 * max(w, h)
    assert abs(wide.x1 - box.x1) < tolerance, f"{path.name}: x1 {wide.x1:.0f} vs {box.x1:.0f}"
    assert abs(wide.y1 - box.y1) < tolerance, f"{path.name}: y1 {wide.y1:.0f} vs {box.y1:.0f}"
    assert abs(wide.width - box.width) < tolerance, (
        f"{path.name}: the face was resized by widening the frame: "
        f"{wide.width:.0f} vs {box.width:.0f}"
    )
    assert abs(wide.height - box.height) < tolerance, (
        f"{path.name}: the face was squashed by widening the frame: "
        f"{wide.height:.0f} vs {box.height:.0f}"
    )


def test_a_box_always_has_area(detector, photos):
    """A zero-width box is not a face; it is a rendering bug waiting to
    divide by zero in `_iou`."""
    seen = 0
    for _path, image in _images(photos, detector):
        boxes, _ = detector.detect(
            image,
            detect_threshold=DEFAULT_DETECT_THRESHOLD,
            gate_threshold=DEFAULT_GATE_THRESHOLD,
        )
        for b in boxes:
            assert b.width > 0 and b.height > 0 and b.area > 0
            seen += 1
    assert seen > 0, "no boxes at all; this test proved nothing"


# ------------------------------------------------------------- determinism


def test_the_same_photo_twice_gives_the_same_boxes(detector, photos):
    """A gate whose answer changes between runs cannot be reviewed once."""
    for _path, image in _images(photos, detector):
        first = detector.detect(
            image,
            detect_threshold=DEFAULT_DETECT_THRESHOLD,
            gate_threshold=DEFAULT_GATE_THRESHOLD,
        )
        second = detector.detect(
            image,
            detect_threshold=DEFAULT_DETECT_THRESHOLD,
            gate_threshold=DEFAULT_GATE_THRESHOLD,
        )
        assert first == second


def test_the_real_gate_is_deterministic_and_accounted(detector, photos):
    """The whole driver, real detector, real photos, twice."""
    a = gate_photos(photos, detector, workers=1)
    b = gate_photos(photos, detector, workers=4)
    assert a.accounted and b.accounted
    assert a.considered == b.considered == len(photos)
    assert [d.file_hash for d in a.detections] == [d.file_hash for d in b.detections]
    assert [d.verdict for d in a.detections] == [d.verdict for d in b.detections]
    assert [d.boxes for d in a.detections] == [d.boxes for d in b.detections]
    assert (a.eligible, a.has_face, a.uncertain, a.errors) == (
        b.eligible,
        b.has_face,
        b.uncertain,
        b.errors,
    )


def test_the_gate_verdict_matches_the_boxes_it_was_given(detector, photos):
    """`classify` is what the driver must be using, on the real detector's
    real output - not a re-derivation that happens to agree on toy inputs."""
    for _path, image in _images(photos, detector):
        boxes, _ = detector.detect(
            image,
            detect_threshold=DEFAULT_DETECT_THRESHOLD,
            gate_threshold=DEFAULT_GATE_THRESHOLD,
        )
        verdict = classify(
            boxes,
            detect_threshold=DEFAULT_DETECT_THRESHOLD,
            gate_threshold=DEFAULT_GATE_THRESHOLD,
        )
        if verdict is Verdict.ELIGIBLE:
            assert all(b.score < DEFAULT_GATE_THRESHOLD for b in boxes), (
                "an eligible photo had a box at or above the gate threshold"
            )


def test_lowering_the_gate_threshold_never_makes_a_photo_more_eligible(detector, photos):
    """Monotonicity, on real scores.

    The gate is only sound if being MORE suspicious cannot let more through.
    A comparison flipped anywhere in the chain breaks this, and on toy scores
    it is easy to satisfy by accident.
    """
    strict = gate_photos(photos, detector, gate_threshold=0.05, workers=4)
    loose = gate_photos(photos, detector, gate_threshold=0.40, workers=4)
    assert strict.eligible <= loose.eligible
    strict_ok = {d.file_hash for d in strict.detections if d.publishable}
    loose_ok = {d.file_hash for d in loose.detections if d.publishable}
    assert strict_ok <= loose_ok, "a stricter gate proposed a photo the looser one did not"
