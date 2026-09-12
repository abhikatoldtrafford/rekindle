"""Face detection, and the publishing gate built on top of it.

WHAT THIS GATE IS FOR
---------------------
Some memories get published to a PUBLIC GitHub repository. The rule is that a
public photo must contain no person other than the owner. Today that is
enforced with Takeout face tags: `people` must be non-empty and a subset of an
allow-list. That default-denies all 8,433 untagged live photos - including
every landscape, which is exactly the material a public demo wants.

This module answers one narrow question - *does this image contain a face, and
how many* - with a purpose-built detector, so that a photo with no face in it
can be a candidate.

WHY NOT CLIP
------------
CLIP zero-shot ("a photo of a person" vs "a photo with no people") was
rejected before this milestone started, and correctly. It is a similarity
score over a whole image: it is unreliable on a face that is small, blurry,
side-on or half out of frame, and those are precisely the cases where the cost
of being wrong is publishing a stranger. A detector that localises faces can
be wrong about *where*, but its miss rate on a visible face is measurable and
an order of magnitude lower.

THE SAFETY MODEL
----------------
1. **Default deny.** `Verdict.ELIGIBLE` is returned ONLY on a confident
   negative: the detector ran, and its most confident detection was below
   `gate_threshold`. Anything else - a detection, a weak detection, an
   unreadable file, a crash - is not eligible.
2. **Two thresholds, and the gate uses the lower one.** `detect_threshold`
   decides what is shown to the user as a box. `gate_threshold` is far lower,
   because the gate's job is to be suspicious: a 0.15-confidence blob that is
   probably a doorknob costs one photo not being published, and being wrong
   the other way costs a stranger's face on the internet.
3. **It proposes; it never publishes.** `review_queue` exists so a human sees
   the count and the boxes. There is no code path in rekindle that turns an
   `ELIGIBLE` verdict into a published file.
4. **Measured, not asserted.** Precision and recall against a hand-checked
   sample from the real library are in docs/audit-m3-semantic.md. If you are
   reading this because you want to trust the gate, read that table first.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from rekindle.config import CATALOGUE, active
from rekindle.semantic.availability import SemanticUnavailable
from rekindle.semantic.registry import FaceModel, face_model

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    from PIL.Image import Image

#: A detection at or above this is shown to the user as a face.
DEFAULT_DETECT_THRESHOLD = CATALOGUE["faces.detect_threshold"].default
#: ANY detection at or above this blocks publication. Deliberately far below
#: `DEFAULT_DETECT_THRESHOLD`: the asymmetry of the two errors is the whole
#: design. Tuned against the hand-checked sample; see the audit document.
DEFAULT_GATE_THRESHOLD = CATALOGUE["faces.gate_threshold"].default
#: IoU above which two boxes are the same face.
DEFAULT_NMS_IOU = 0.4
#: Threads decoding and detecting at once.
#:
#: The face detector is the SLOWEST STAGE IN THIS MILESTONE and it is the one
#: that cannot use the GPU: the installed onnxruntime is the CPU build, whose
#: `get_available_providers()` offers only CPU and Azure, so `prefer_gpu` is
#: dead on this machine. The pinned graph is fixed batch 1 (`[1,3,640,640]`)
#: as well, so batching is not available either without re-exporting it.
#:
#: What IS available is threads. Two runs over 240 real face-tag-free photos,
#: the second through `gate_photos` itself:
#:
#:     workers      1     2     4     6     8
#:     img/s      9.6  18.8  25.3  23.7  16.2   (detect() directly)
#:     img/s     11.7  19.0  26.4  28.5  27.3   (through the gate)
#:
#: The two runs put the peak in different places, so THEY DO NOT AGREE THAT 4
#: BEATS 6 - they agree only that anything from 4 up is roughly 2.4x serial
#: and that the curve is flat there. 4 is chosen as the conservative end of
#: that plateau: it leaves a core for onnxruntime's own intra-op pool, which
#: defaults to all of them, and it does not fall off a 4-core machine.
#:
#: The scores are not merely close at every width - over 240 real photos the
#: verdicts, the boxes and their order were IDENTICAL to the serial run. For
#: a publishing gate that is the bar, and `--workers` exists for anyone whose
#: machine disagrees with this one.
#:
#: A stage split of 31% decode, 29% letterbox, 40% inference is why threads
#: work at all here: only the last of those is onnxruntime's, and Pillow and
#: numpy release the GIL for the other two.
#:
#: At 4 workers the whole library is about 11 minutes instead of 26.
DEFAULT_GATE_WORKERS = 4


class Verdict(StrEnum):
    #: No face found, and the detector was confident about it. Publishable
    #: SUBJECT TO A HUMAN SAYING SO.
    ELIGIBLE = "eligible"
    #: At least one detection above `detect_threshold`.
    HAS_FACE = "has_face"
    #: Something between the two thresholds. Not eligible; goes to review.
    UNCERTAIN = "uncertain"
    #: The file could not be read or the detector failed. Not eligible.
    ERROR = "error"


@dataclass(frozen=True)
class Box:
    """Pixel coordinates in the ORIGINAL image, plus confidence."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_tuple(self) -> tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)


@dataclass(frozen=True)
class Detection:
    file_hash: str
    path: Path
    boxes: tuple[Box, ...]
    verdict: Verdict
    image_size: tuple[int, int] = (0, 0)
    error: str = ""

    @property
    def face_count(self) -> int:
        return len(self.boxes)

    @property
    def top_score(self) -> float:
        return max((b.score for b in self.boxes), default=0.0)

    @property
    def publishable(self) -> bool:
        return self.verdict is Verdict.ELIGIBLE


@dataclass
class GateReport:
    """What the gate decided, in buckets that add up."""

    considered: int = 0
    eligible: int = 0
    has_face: int = 0
    uncertain: int = 0
    errors: int = 0
    faces_found: int = 0
    elapsed_s: float = 0.0
    model_key: str = ""
    detect_threshold: float = DEFAULT_DETECT_THRESHOLD
    gate_threshold: float = DEFAULT_GATE_THRESHOLD
    detections: list[Detection] = field(default_factory=list)

    @property
    def accounted(self) -> bool:
        return self.considered == (self.eligible + self.has_face + self.uncertain + self.errors)

    @property
    def images_per_s(self) -> float:
        return self.considered / self.elapsed_s if self.elapsed_s > 0 else 0.0

    def review_queue(self) -> list[Detection]:
        """Everything a human must look at before anything is published.

        Includes the ELIGIBLE ones. That is the point: the detector proposes,
        the user disposes, and a queue that only showed the rejections would
        make the accepts invisible - which is the direction that hurts.
        """
        order = {
            Verdict.UNCERTAIN: 0,
            Verdict.ERROR: 1,
            Verdict.ELIGIBLE: 2,
            Verdict.HAS_FACE: 3,
        }
        return sorted(self.detections, key=lambda d: (order[d.verdict], -d.top_score, str(d.path)))


class FaceDetector:
    """ONNX face detector. CPU by default; CUDA when onnxruntime offers it."""

    def __init__(self, spec: FaceModel, model_path: Path, *, prefer_gpu: bool = False) -> None:
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as exc:
            raise SemanticUnavailable(f"face detection needs the 'semantic' extra ({exc})") from exc
        if not model_path.is_file():
            raise SemanticUnavailable(
                f"{model_path} is missing. Run `rekindle semantic setup --face {spec.key}` first."
            )
        self.spec = spec
        self._np = np
        available = set(ort.get_available_providers())
        wanted = ["CPUExecutionProvider"]
        if prefer_gpu and "CUDAExecutionProvider" in available:
            wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(str(model_path), opts, providers=wanted)
        self.providers = tuple(self._session.get_providers())
        self.size = spec.input_size

    # -------------------------------------------------------------- inference

    def detect(
        self,
        image: Image,
        *,
        detect_threshold: float | None = None,
        gate_threshold: float | None = None,
        nms_iou: float = DEFAULT_NMS_IOU,
    ) -> tuple[tuple[Box, ...], float]:
        """`(boxes above gate_threshold, best score)` in original pixel coords.

        Boxes are returned down to `gate_threshold`, not `detect_threshold`,
        because the gate needs to see the weak ones. Callers that only want
        confident faces filter on `.score`.

        Both thresholds default to `None`, meaning the ACTIVE configuration
        read now. See `rekindle.config` - a shipped number bound as a default
        argument would be frozen at import and no `rekindle.toml` could reach
        it, which for a PUBLISHING gate is the worst place for that bug.
        """
        cfg = active().faces
        detect_threshold = cfg.detect_threshold if detect_threshold is None else detect_threshold
        gate_threshold = cfg.gate_threshold if gate_threshold is None else gate_threshold
        rgb = image.convert("RGB")
        tensor, scale, pad = self._letterbox(rgb)
        name = self._session.get_inputs()[0].name
        raw = self._session.run(None, {name: tensor})
        floor = min(gate_threshold, detect_threshold)
        if self.spec.family == "yolo":
            boxes = self._decode_yolo(raw, floor)
        else:
            boxes = self._decode_scrfd(raw, floor)
        boxes = _nms(boxes, nms_iou)
        w, h = rgb.size
        out = []
        for b in boxes:
            out.append(
                Box(
                    x1=max(0.0, (b.x1 - pad[0]) / scale),
                    y1=max(0.0, (b.y1 - pad[1]) / scale),
                    x2=min(float(w), (b.x2 - pad[0]) / scale),
                    y2=min(float(h), (b.y2 - pad[1]) / scale),
                    score=b.score,
                )
            )
        out = [b for b in out if b.area > 0]
        out.sort(key=lambda b: -b.score)
        return tuple(out), max((b.score for b in out), default=0.0)

    def _letterbox(self, image: Image) -> tuple[np.ndarray, float, tuple[float, float]]:
        """Resize preserving aspect ratio onto a square canvas.

        A plain square resize distorts faces, and a face detector trained on
        undistorted crops loses real recall on a 16:9 photo - which for a gate
        means missing somebody. Letterboxing costs one paste and keeps the
        aspect ratio exact.
        """
        from PIL import Image as PILImage

        np = self._np
        w, h = image.size
        scale = min(self.size / w, self.size / h)
        new = (max(1, round(w * scale)), max(1, round(h * scale)))
        resized = image.resize(new, PILImage.BILINEAR)
        canvas = PILImage.new("RGB", (self.size, self.size), (114, 114, 114))
        pad = ((self.size - new[0]) / 2, (self.size - new[1]) / 2)
        canvas.paste(resized, (int(pad[0]), int(pad[1])))
        arr = np.asarray(canvas, dtype=np.float32)
        # SCRFD was trained on BGR with (x - 127.5) / 128; YOLO on RGB in [0,1].
        arr = arr / 255.0 if self.spec.family == "yolo" else (arr[:, :, ::-1] - 127.5) / 128.0
        return (
            np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)),
            scale,
            (int(pad[0]), int(pad[1])),
        )

    # --------------------------------------------------------------- decoding

    def _decode_yolo(self, raw: list, threshold: float) -> list[Box]:
        """YOLOv8/v11 single-class head: (1, 5, N) of cx, cy, w, h, score."""
        np = self._np
        pred = raw[0]
        arr = np.asarray(pred)
        if arr.ndim == 3:
            arr = arr[0]
        # (5, N) or (N, 5); the 5 is the channel axis whichever way round.
        if arr.shape[0] != 5 and arr.shape[1] == 5:
            arr = arr.T
        if arr.shape[0] < 5:
            raise ValueError(f"unexpected YOLO output shape {np.asarray(pred).shape}")
        scores = arr[4]
        keep = np.flatnonzero(scores >= threshold)
        out = []
        for i in keep:
            cx, cy, w, h = (float(arr[j][i]) for j in range(4))
            out.append(Box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, float(scores[i])))
        return out

    def _decode_scrfd(self, raw: list, threshold: float) -> list[Box]:
        """SCRFD: per-stride (scores, distances), anchor-centre + distance decode.

        Outputs arrive as three score tensors then three bbox tensors (then
        keypoints, ignored). They are matched to strides BY LENGTH rather than
        by position: (input/stride)^2 * 2 anchors is unique per stride, and
        relying on output order would silently mis-scale every box if the
        export ever reordered them.
        """
        np = self._np
        arrays = [np.asarray(r) for r in raw]
        strides = (8, 16, 32)
        by_len: dict[int, list[np.ndarray]] = {}
        for arr in arrays:
            by_len.setdefault(arr.shape[0], []).append(arr)
        out: list[Box] = []
        for stride in strides:
            count = (self.size // stride) ** 2 * 2
            group = by_len.get(count)
            if not group:
                continue
            scores = next((a for a in group if a.shape[-1] == 1), None)
            deltas = next((a for a in group if a.shape[-1] == 4), None)
            if scores is None or deltas is None:
                continue
            flat = scores.reshape(-1)
            keep = np.flatnonzero(flat >= threshold)
            if not len(keep):
                continue
            centres = _anchor_centres(np, self.size, stride)
            for i in keep:
                cx, cy = centres[i]
                left, top, right, bottom = (float(v) * stride for v in deltas[i])
                out.append(Box(cx - left, cy - top, cx + right, cy + bottom, float(flat[i])))
        return out


def _anchor_centres(np, size: int, stride: int) -> np.ndarray:
    rows = size // stride
    ys, xs = np.mgrid[:rows, :rows]
    centres = np.stack([xs, ys], axis=-1).astype(np.float32) * stride
    # Two anchors share each cell centre; repeat rather than tile, so index i
    # and i+1 are the same location - which is how SCRFD orders them.
    return np.repeat(centres.reshape(-1, 2), 2, axis=0)


def _nms(boxes: Sequence[Box], iou_threshold: float) -> list[Box]:
    """Greedy non-maximum suppression. Pure python: at most a few hundred boxes."""
    ordered = sorted(boxes, key=lambda b: -b.score)
    kept: list[Box] = []
    for box in ordered:
        if all(_iou(box, other) <= iou_threshold for other in kept):
            kept.append(box)
    return kept


def _iou(a: Box, b: Box) -> float:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def classify(
    boxes: Sequence[Box],
    *,
    detect_threshold: float | None = None,
    gate_threshold: float | None = None,
) -> Verdict:
    """The gate's decision rule, isolated so it can be tested without a model.

    Read it as: eligible ONLY when nothing at all reached the suspicious
    threshold. This is the line that makes the gate default-deny, and
    `tests/test_semantic_faces.py` breaks it deliberately to prove the test
    catches it.
    """
    cfg = active().faces
    detect_threshold = cfg.detect_threshold if detect_threshold is None else detect_threshold
    gate_threshold = cfg.gate_threshold if gate_threshold is None else gate_threshold
    best = max((b.score for b in boxes), default=0.0)
    if best >= detect_threshold:
        return Verdict.HAS_FACE
    if best >= gate_threshold:
        return Verdict.UNCERTAIN
    return Verdict.ELIGIBLE


def gate_photos(
    photos: Iterable,
    detector: FaceDetector,
    *,
    detect_threshold: float | None = None,
    gate_threshold: float | None = None,
    allow_people: Sequence[str] = (),
    workers: int = DEFAULT_GATE_WORKERS,
    progress=None,
) -> GateReport:
    """Run the detector over photos and bucket them. Never publishes anything.

    `allow_people` is applied BEFORE the detector and only ever makes a photo
    LESS eligible, never more: a photo tagged with somebody outside the
    allow-list is `HAS_FACE` on the tag alone, without being decoded. A photo
    tagged only with allowed people still goes through the detector, because
    Google's tags are recall-poor - a tagged photo of the owner may also
    contain three untagged strangers.
    """
    import time

    cfg = active().faces
    detect_threshold = cfg.detect_threshold if detect_threshold is None else detect_threshold
    gate_threshold = cfg.gate_threshold if gate_threshold is None else gate_threshold

    allowed = {p.casefold() for p in allow_people}
    # The report records the thresholds ACTUALLY used, resolved above, so a
    # run under a user's `rekindle.toml` says so rather than reprinting the
    # shipped numbers.
    report = GateReport(
        model_key=detector.spec.key,
        detect_threshold=detect_threshold,
        gate_threshold=gate_threshold,
    )
    started = time.perf_counter()
    items = list(photos)
    total = len(items)
    depth = max(1, workers) * 4

    def decide(photo) -> Detection:
        return _examine(
            photo,
            detector,
            allowed,
            detect_threshold=detect_threshold,
            gate_threshold=gate_threshold,
        )

    # Consumed IN ORDER, so `detections` comes out in the order the photos
    # went in and `progress` counts 1, 2, 3 with no gaps. The workers decide;
    # nothing but this thread touches the counters. `report.errors += 1` from
    # four threads is a lost-update race, and the count it would corrupt is
    # the one that says how many photos nobody managed to look at.
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        pending: deque[Future[Detection]] = deque()
        queue = iter(items)

        def submit_more() -> None:
            while len(pending) < depth:
                photo = next(queue, None)
                if photo is None:
                    return
                pending.append(pool.submit(decide, photo))

        submit_more()
        n = 0
        while pending:
            found = pending.popleft().result()
            submit_more()
            n += 1
            report.considered += 1
            report.detections.append(found)
            report.faces_found += sum(1 for b in found.boxes if b.score >= detect_threshold)
            if found.verdict is Verdict.ERROR:
                report.errors += 1
            elif found.verdict is Verdict.ELIGIBLE:
                report.eligible += 1
            elif found.verdict is Verdict.HAS_FACE:
                report.has_face += 1
            else:
                report.uncertain += 1
            if progress:
                progress(n, total)
    report.elapsed_s = time.perf_counter() - started
    return report


def _examine(
    photo,
    detector: FaceDetector,
    allowed: set[str],
    *,
    detect_threshold: float,
    gate_threshold: float,
) -> Detection:
    """One photo's verdict. Runs on a WORKER THREAD and never raises.

    DEFAULT DENY IS IN THIS FUNCTION. Every path that is not "the detector
    ran and saw nothing" returns something other than ELIGIBLE, including
    both of the paths that mean "I could not look at it".
    """
    from rekindle.meta.exif import open_upright

    tagged = {p.casefold() for p in getattr(photo, "people", ())}
    if tagged - allowed:
        # Blocked on the tag alone, WITHOUT being decoded. The tag is already
        # proof of a face, and letting the detector overrule a person Google
        # has already named is how a stranger gets published.
        return Detection(photo.file_hash, photo.path, (), Verdict.HAS_FACE)
    path = photo.existing_path()
    if path is None:
        return Detection(photo.file_hash, photo.path, (), Verdict.ERROR, error="file not found")
    try:
        # UPRIGHT, or the detector is looking at a photo lying on its side.
        # A face detector is not rotation invariant and this is not a cosmetic
        # difference: measured on 200 real orientation-5-8 photos, decoding
        # without the transpose changed 14.5% of gate VERDICTS. 7% showed the
        # detector no face at all where the upright image has one - the gate
        # failing OPEN, which is the direction that publishes a stranger - and
        # one frame scored ten faces sideways against zero upright, which is
        # the gate rejecting a photograph for no reason.
        rgb = open_upright(path, draft=(detector.size * 2, detector.size * 2)).convert("RGB")
        size = rgb.size
        boxes, _ = detector.detect(
            rgb,
            detect_threshold=detect_threshold,
            gate_threshold=gate_threshold,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        return Detection(
            photo.file_hash,
            path,
            (),
            Verdict.ERROR,
            error=f"{type(exc).__name__}: {exc}",
        )
    verdict = classify(boxes, detect_threshold=detect_threshold, gate_threshold=gate_threshold)
    return Detection(photo.file_hash, path, boxes, verdict, image_size=size)


def load_detector(
    cache_dir: Path, key: str | None = None, *, prefer_gpu: bool = False
) -> FaceDetector:
    from rekindle.semantic.setup import snapshot_dir

    spec = face_model(key)
    root = snapshot_dir(spec.pin, cache_dir, offline=True)
    return FaceDetector(spec, root / spec.pin.files[0], prefer_gpu=prefer_gpu)


def precision_recall(predicted: Sequence[bool], actual: Sequence[bool]) -> dict[str, float]:
    """Precision/recall of "this photo contains a face", plus the raw counts.

    Reported on the HAS-FACE class, because that is the class whose misses are
    expensive: a false negative here is a stranger published.
    """
    tp = sum(1 for p, a in zip(predicted, actual, strict=True) if p and a)
    fp = sum(1 for p, a in zip(predicted, actual, strict=True) if p and not a)
    fn = sum(1 for p, a in zip(predicted, actual, strict=True) if not p and a)
    tn = sum(1 for p, a in zip(predicted, actual, strict=True) if not p and not a)
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "n": len(predicted),
    }
