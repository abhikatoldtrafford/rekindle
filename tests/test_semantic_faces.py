"""The publishing gate. Its failure mode is a stranger's face on the internet.

`classify` and the box maths are deliberately separable from the ONNX session
so the RULE can be tested exhaustively with no model, no weights and no GPU -
which is the only way CI ever tests the thing that matters most here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rekindle.semantic.faces import (
    DEFAULT_DETECT_THRESHOLD,
    DEFAULT_GATE_THRESHOLD,
    Box,
    Detection,
    GateReport,
    Verdict,
    _iou,
    _nms,
    classify,
    precision_recall,
)


def box(score: float, x1=0.0, y1=0.0, x2=10.0, y2=10.0) -> Box:
    return Box(x1, y1, x2, y2, score)


# ------------------------------------------------------------- the gate rule


def test_no_detections_at_all_is_eligible():
    assert classify([]) is Verdict.ELIGIBLE


def test_a_confident_detection_blocks():
    assert classify([box(0.9)]) is Verdict.HAS_FACE


def test_a_weak_detection_is_uncertain_not_eligible():
    """This is the line that makes the gate default-deny."""
    weak = (DEFAULT_GATE_THRESHOLD + DEFAULT_DETECT_THRESHOLD) / 2
    assert classify([box(weak)]) is Verdict.UNCERTAIN


def test_exactly_at_the_gate_threshold_is_not_eligible():
    assert classify([box(DEFAULT_GATE_THRESHOLD)]) is Verdict.UNCERTAIN


def test_just_below_the_gate_threshold_is_eligible():
    assert classify([box(DEFAULT_GATE_THRESHOLD - 1e-6)]) is Verdict.ELIGIBLE


def test_exactly_at_the_detect_threshold_is_a_face():
    assert classify([box(DEFAULT_DETECT_THRESHOLD)]) is Verdict.HAS_FACE


def test_the_strongest_detection_decides():
    """A confident face among weak noise must not be averaged away."""
    assert classify([box(0.01), box(0.02), box(0.99)]) is Verdict.HAS_FACE


def test_gate_threshold_is_far_below_detect_threshold():
    """The asymmetry IS the design; a regression that equalised them would
    silently turn every uncertain photo into a publishable one."""
    assert DEFAULT_GATE_THRESHOLD < DEFAULT_DETECT_THRESHOLD / 2


def test_thresholds_are_overridable():
    assert classify([box(0.3)], detect_threshold=0.2, gate_threshold=0.1) is Verdict.HAS_FACE
    assert classify([box(0.3)], detect_threshold=0.9, gate_threshold=0.8) is Verdict.ELIGIBLE


# --------------------------------------------------------------- box geometry


def test_iou_of_identical_boxes_is_one():
    assert _iou(box(0.5), box(0.5)) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    assert _iou(box(0.5, 0, 0, 10, 10), box(0.5, 20, 20, 30, 30)) == 0.0


def test_iou_of_half_overlap():
    a = Box(0, 0, 10, 10, 0.9)
    b = Box(5, 0, 15, 10, 0.9)
    assert _iou(a, b) == pytest.approx(50 / 150)


def test_iou_of_zero_area_boxes_does_not_divide_by_zero():
    assert _iou(Box(0, 0, 0, 0, 0.5), Box(0, 0, 0, 0, 0.5)) == 0.0


def test_nms_keeps_the_best_of_an_overlapping_pair():
    kept = _nms([Box(0, 0, 10, 10, 0.6), Box(1, 1, 11, 11, 0.9)], 0.4)
    assert [b.score for b in kept] == [0.9]


def test_nms_keeps_two_separate_faces():
    kept = _nms([Box(0, 0, 10, 10, 0.9), Box(50, 50, 60, 60, 0.8)], 0.4)
    assert len(kept) == 2


def test_nms_on_nothing_returns_nothing():
    assert _nms([], 0.4) == []


def test_box_geometry():
    b = Box(2.0, 4.0, 12.0, 9.0, 0.5)
    assert b.width == 10.0
    assert b.height == 5.0
    assert b.area == 50.0
    assert b.as_tuple() == (2, 4, 12, 9)


# -------------------------------------------------------------- the reporting


def make_report() -> GateReport:
    report = GateReport()
    for verdict, boxes in (
        (Verdict.ELIGIBLE, ()),
        (Verdict.HAS_FACE, (box(0.95),)),
        (Verdict.UNCERTAIN, (box(0.25),)),
        (Verdict.ERROR, ()),
    ):
        report.considered += 1
        report.detections.append(
            Detection(verdict.value, Path(f"/{verdict.value}.jpg"), boxes, verdict)
        )
    report.eligible = report.has_face = report.uncertain = report.errors = 1
    return report


def test_report_accounting_balances():
    assert make_report().accounted


def test_report_accounting_catches_a_dropped_photo():
    """The identity must be capable of failing, or it is decoration."""
    report = make_report()
    report.considered += 1
    assert not report.accounted


def test_review_queue_puts_uncertain_first_and_includes_eligible():
    order = [d.verdict for d in make_report().review_queue()]
    assert order[0] is Verdict.UNCERTAIN
    assert Verdict.ELIGIBLE in order
    assert len(order) == 4


def test_detection_publishable_only_when_eligible():
    for verdict in Verdict:
        det = Detection("h", Path("/x.jpg"), (), verdict)
        assert det.publishable is (verdict is Verdict.ELIGIBLE)


def test_detection_counts_and_top_score():
    det = Detection("h", Path("/x.jpg"), (box(0.3), box(0.8)), Verdict.HAS_FACE)
    assert det.face_count == 2
    assert det.top_score == pytest.approx(0.8)
    assert Detection("h", Path("/x.jpg"), (), Verdict.ELIGIBLE).top_score == 0.0


# ------------------------------------------------------------ the metric itself


def test_precision_recall_on_a_perfect_classifier():
    got = precision_recall([True, False, True], [True, False, True])
    assert got["precision"] == 1.0
    assert got["recall"] == 1.0
    assert got["fp"] == got["fn"] == 0


def test_precision_recall_counts_a_miss_as_a_false_negative():
    """A false negative here is the expensive error: a face that got published."""
    got = precision_recall([False, False], [True, False])
    assert got["fn"] == 1
    assert got["recall"] == 0.0
    assert got["tn"] == 1


def test_precision_recall_counts_a_false_alarm():
    got = precision_recall([True, True], [True, False])
    assert got["fp"] == 1
    assert got["precision"] == pytest.approx(0.5)
    assert got["recall"] == 1.0


def test_precision_recall_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        precision_recall([True], [True, False])
