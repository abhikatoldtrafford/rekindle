"""When a file's EXIF orientation tag LIES, and how we prove it from content.

THE DEFECT
----------
`meta.exif.open_upright` applies the EXIF orientation tag, which is correct
and is what every viewer does. Some files in a real Takeout export are
nevertheless stored ALREADY UPRIGHT while still carrying a 90/270-degree tag:
an earlier tool rotated the pixels and left the tag behind. Applying the tag
then turns a correct photograph on its side - the renderer draws it sideways,
and the embedder, the fingerprint pass and the face gate all see it that way.

Nothing in EXIF distinguishes a stale tag from a live one, so the only
evidence available is the picture itself. Faces are strongly orientation
dependent, so the face detector already in `semantic.faces` can be asked
which way up a photograph is.

THE RULE, AND WHY IT IS THIS NARROW
-----------------------------------
This module tests ONE hypothesis per file, and only for files that carry an
axis-swapping tag:

    "this file's orientation tag is stale - the stored pixels are upright."

It can only ever DISABLE a tag the file already carries. It never invents a
rotation for a file whose tag says upright, and it never proposes 180
degrees. That restriction is not caution for its own sake, it is measured.
Hand-checking 36 proposals from an unrestricted four-way detector on the
reference library:

    proposal                            true   false   unsure   precision
    undoes an axis-swapping tag           18       0        1        0.95
    invents a rotation (tag 1 / absent)    2       4        3        0.22
    flips 180 degrees                      0       7        1        0.00

The asymmetry has a mechanism. A stale tag is a real, documented defect with
a known signature, so the prior on it is high; the library contains
essentially no upside-down photographs, so the prior on a 180-degree error is
near zero while the detector's 180-degree false-alarm rate is not. Testing
one hypothesis instead of four also removes the multiple-comparison problem
that generates most of those false alarms.

The earlier, obvious detector - "stored portrait plus a 90/270 tag" - is the
thing this replaces. It flagged 222 files of which a hand-check found 5 of 18
genuinely stale, so it would have broken roughly 13 photographs for every 5
it repaired. See docs/known-limitations.md.

SCORING: CONFIDENCE, NOT COUNT
------------------------------
`evidence` sums the SQUARE of each detection's confidence. Four weak boxes at
0.20 total 0.16 and cannot outvote one strong box at 0.85, which alone is
0.72 - so "more detections" is never by itself better evidence. Squaring
rather than taking the maximum still lets a genuine group photo accumulate
evidence, which a bare maximum throws away: measured at a matched false-alarm
budget on 2,000 real photos, the sum of squares recovered 26% of known
rotations where the maximum recovered 15%.

Ranking by face COUNT first is the specific mistake this replaces. It let
four detections at 0.669 beat two at 0.750.

THE MARGIN
----------
A win must be DECISIVE. `MIN_MARGIN` is an absolute difference in that
evidence. Ninety files were opened and looked at, from the 273 the pass
proposed on the reference library at a margin of 0.20:

    margin band       files   opened and looked at         clear errors
    0.20 .. 0.35         61   30: 21 right, 4 WRONG, 5 unsure     4
    0.35 and up         212   60: 57 right, 0 wrong, 3 unsure     0

0.35 is therefore the threshold: the lowest one at which a hand check - two
independent samples of 30, drawn from the 212 the pass proposes - turned up
NO photograph that the correction would have turned on its side. The three
"unsure" are a macro flower and two pets, where no orientation is objectively
right and the detector had found a weak face in a petal.

The two unambiguous errors below it sat at 0.221 and 0.271, and both were
ordinary upright portraits whose face the detector simply liked slightly
better sideways.

Dropping to 0.20 would repair roughly 43 more files and wrongly rotate
roughly 8. That is still 5 repairs per error - far better than the
dimension-based detector's 5 per 13 - and `--margin` exists for anyone who
wants it. It is not the default because turning a CORRECT photograph sideways
is a regression this tool caused, while leaving a stale-tagged one alone
merely fails to fix what was already broken, and because the 8 is directly
observed while the 43 rests on an extrapolated miss rate.

A RELATIVE margin was tried and rejected on measurement, not taste. Both
unambiguous errors already had a confident face in the tagged decode
(evidence 0.75 and 0.74), so "require the tag-free decode to win by a
FACTOR" looks promising - and does not work: over the hand-labelled band the
false positives' median ratio was 1.40 against the true positives' 1.33, so
the guard would have removed more repairs than errors.

`MIN_FACE` additionally requires the winning decode to contain something the
detector would show a user as a face at all, so that two piles of noise
cannot be compared against each other.

WHAT THIS CANNOT REACH
----------------------
A photograph with no face in it carries no evidence for this method. On the
reference library, of the 2,503 images carrying an axis-swapping tag, 828
(33.1%) have no detection at any rotation. A stale tag among those cannot be
found this way at all, and the pass reports that count rather than implying
it examined them successfully.

NOTHING HERE IMPORTS THE DETECTOR, numpy OR onnxruntime AT MODULE LEVEL.
`decide` is pure arithmetic over scores and is tested without a model; the
overrides registry is plain dict lookups. With no model installed the
override table is empty and `open_upright` behaves exactly as it always has.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

#: EXIF orientation values that exchange the two axes, and are therefore the
#: only ones whose staleness this module can even be asked about. Duplicated
#: from `meta.exif` deliberately: importing it here would make the import
#: cycle `exif -> orientation -> exif`, and `exif` is the one that must win.
SWAPS_AXES = frozenset({5, 6, 7, 8})

#: The winning decode must contain a detection a user would be shown as a
#: face. Equal to `semantic.faces.DEFAULT_DETECT_THRESHOLD`, not imported from
#: it, for the same no-heavy-import reason.
MIN_FACE = 0.45

#: How much more evidence the tag-ignored decode needs before the tag is
#: called stale. See THE MARGIN above - hand-checked, not guessed.
MIN_MARGIN = 0.35

#: The five outcomes, named rather than spelled out at each comparison. A
#: caller that buckets on a reason string and a producer that writes one are
#: the two halves of a bug that no type checker can see.
NO_TAG = "no axis-swapping tag"
NO_FACE = "no face without the tag"
MARGIN_TOO_SMALL = "margin too small"
STALE = "stale tag: upright without it"


def evidence(scores: Sequence[float]) -> float:
    """Sum of squared detection confidences. Higher means "more upright".

    Squared, so one confident face outweighs several weak ones. See SCORING
    in the module docstring for the measurement behind that choice.
    """
    return sum(s * s for s in scores)


@dataclass(frozen=True)
class Verdict:
    """One file's orientation examination, and everything it was based on."""

    #: True only when the file's axis-swapping tag was proved stale.
    ignore_exif: bool
    #: Why, in a phrase, for the report and for the stored evidence.
    reason: str
    #: Evidence with the tag applied - i.e. what ships today.
    e_tag: float = 0.0
    #: Evidence with the tag ignored - the raw stored pixels.
    e_raw: float = 0.0
    #: Best single detection confidence in the tag-ignored decode.
    m_raw: float = 0.0
    #: The file's own orientation tag, as read.
    tag: int | None = None
    #: True when the file could not be examined at all. Distinct from every
    #: "examined and the tag stands" outcome, because an unexamined file must
    #: NOT be recorded as examined - see `run_orientation`.
    error: bool = False

    @property
    def margin(self) -> float:
        return self.e_raw - self.e_tag

    def as_json(self) -> str:
        return json.dumps(
            {
                "reason": self.reason,
                "tag": self.tag,
                "e_tag": round(self.e_tag, 4),
                "e_raw": round(self.e_raw, 4),
                "m_raw": round(self.m_raw, 4),
                "margin": round(self.margin, 4),
            },
            sort_keys=True,
        )


def decide(
    tag: int | None,
    tagged_scores: Sequence[float],
    raw_scores: Sequence[float],
    *,
    min_face: float = MIN_FACE,
    min_margin: float = MIN_MARGIN,
) -> Verdict:
    """The whole decision rule, as pure arithmetic over detection scores.

    Separated from every decode and every model so it can be tested exactly,
    the way `semantic.faces.classify` is. `tests/test_meta_orientation.py`
    breaks each of the three guards below in turn to prove they are load
    bearing.

    DEFAULT IS TO TRUST THE FILE. Every path that is not "an axis-swapping
    tag, a real face without it, and a decisive margin" returns
    `ignore_exif=False`, which leaves `open_upright` doing exactly what it
    does today.
    """
    if tag not in SWAPS_AXES:
        # Not a file this module has an opinion about. A tag of 1, 2, 3, 4 or
        # none at all cannot be "stale" in the sense meant here, and inventing
        # a rotation for one is the 0.22-precision column in the docstring.
        return Verdict(False, NO_TAG, tag=tag)
    e_tag, e_raw = evidence(tagged_scores), evidence(raw_scores)
    m_raw = max(raw_scores, default=0.0)
    base = Verdict(False, "", e_tag=e_tag, e_raw=e_raw, m_raw=m_raw, tag=tag)
    if m_raw < min_face:
        # Either there is no face in this photograph at all - in which case
        # this method simply cannot reach it - or the only "face" the
        # tag-ignored decode found is too weak to argue with the file's own
        # metadata.
        return Verdict(False, NO_FACE, **_fields(base))
    if e_raw - e_tag < min_margin:
        return Verdict(False, MARGIN_TOO_SMALL, **_fields(base))
    return Verdict(True, STALE, **_fields(base))


def _fields(v: Verdict) -> dict[str, object]:
    return {"e_tag": v.e_tag, "e_raw": v.e_raw, "m_raw": v.m_raw, "tag": v.tag}


# --------------------------------------------------------------- the overrides
#
# `open_upright` takes a Path and nothing else - it has no index handle and
# must not grow one, because it is called from four modules in three packages
# and per-call database access at the decode chokepoint is not affordable.
#
# So the corrections live in a process-wide table that the index populates
# when it is OPENED (see `db.PhotoStore.__init__` and
# `semantic.photos.PhotoIndexReader.__init__`). Loading REPLACES the table, so
# opening a second index cannot leave the first one's corrections behind, and
# a process that opens no index - every test that does not ask for this, and
# every CI run - decodes exactly as it did before this module existed.

_ignore_exif: frozenset[str] = frozenset()


def _key(path: Path | str) -> str:
    """The form a path is compared in: resolved, then case-normalised.

    `resolve()` because `rekindle.cli._resolved` puts paths through exactly
    that before they are stored, so anything else would compare a resolved
    path against an unresolved one and silently match nothing. `normcase`
    because on Windows two spellings differing only in case are one file, and
    a caller and the index need not agree on which spelling they use.
    """
    return os.path.normcase(str(Path(path).resolve()))


def load_overrides(paths: Iterable[Path | str]) -> int:
    """Replace the override table. Returns how many files it now covers."""
    global _ignore_exif
    _ignore_exif = frozenset(_key(p) for p in paths)
    return len(_ignore_exif)


def clear_overrides() -> None:
    """Forget every override. The state a fresh process starts in."""
    global _ignore_exif
    _ignore_exif = frozenset()


def override_count() -> int:
    return len(_ignore_exif)


def ignores_exif(path: Path | str) -> bool:
    """Is this file's EXIF orientation tag recorded as stale?

    Hot: called once per decode, on every photo. A frozenset lookup on a
    normalised string, and on the overwhelmingly common empty table it is a
    single hash of a short string against nothing.
    """
    return bool(_ignore_exif) and _key(path) in _ignore_exif


# ------------------------------------------------------------------- the pass

_ORIENTATION_TAG = 0x0112


def examine(path: Path, detector, *, min_face: float = MIN_FACE, min_margin: float = MIN_MARGIN):
    """Run the detector both ways on one file and return its `Verdict`.

    Decodes ONCE and transposes in memory, so the two hypotheses are compared
    on identical pixels rather than on two independent decodes.

    Deliberately does NOT go through `open_upright`. That function consults
    the override table this pass writes, so a second run over an
    already-corrected library would compare the corrected decode against
    itself and "confirm" whatever the first run decided. Re-examination has to
    see the file as it actually is.
    """
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        tag = im.getexif().get(_ORIENTATION_TAG)
        if tag not in SWAPS_AXES:
            # No decode at all, and no detector. 86% of a real library takes
            # this path, which is what makes the pass minutes rather than
            # hours.
            return Verdict(False, NO_TAG, tag=tag)
        # Square, so it needs no axis-order reasoning: both hypotheses want
        # the same number of pixels on the long edge.
        im.draft("RGB", (detector.size * 2, detector.size * 2))
        raw = im.convert("RGB")
        tagged = ImageOps.exif_transpose(raw)
    tagged_boxes, _ = detector.detect(tagged)
    raw_boxes, _ = detector.detect(raw)
    return decide(
        tag,
        [b.score for b in tagged_boxes],
        [b.score for b in raw_boxes],
        min_face=min_face,
        min_margin=min_margin,
    )


@dataclass
class OrientationReport:
    """What the pass looked at and what it decided. Buckets that add up."""

    examined: int = 0
    #: Had no axis-swapping tag, so no opinion was possible or needed.
    no_tag: int = 0
    #: Had one, and the content agreed with it.
    tag_trusted: int = 0
    #: Had one, and the content proved it stale. These get rotated.
    tag_ignored: int = 0
    #: Had one, but no face to reason from - UNREACHABLE by this method.
    no_face: int = 0
    #: Could not be read at all.
    errors: int = 0
    elapsed_s: float = 0.0
    #: `(path, Verdict)` for every file whose tag was called stale. Every
    #: rotation this pass applies is reported, with its evidence: a silent
    #: rotation is indistinguishable from a bug.
    corrected: list = None  # type: ignore[assignment]
    #: Hashes whose pixels changed, and whose fingerprint and embedding are
    #: therefore stale.
    invalidated: list = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.corrected is None:
            self.corrected = []
        if self.invalidated is None:
            self.invalidated = []

    @property
    def accounted(self) -> bool:
        return self.examined == (
            self.no_tag + self.tag_trusted + self.tag_ignored + self.no_face + self.errors
        )

    @property
    def reachable(self) -> int:
        """Files carrying an axis-swapping tag that had a face to argue with."""
        return self.tag_trusted + self.tag_ignored

    @property
    def unreachable(self) -> int:
        """Files carrying one that did NOT. No content evidence exists for
        these, so a stale tag among them cannot be found this way at all."""
        return self.no_face


def run_orientation(
    store,
    detector,
    *,
    workers: int = 4,
    min_margin: float = MIN_MARGIN,
    progress=None,
) -> OrientationReport:
    """Examine every unexamined photo and record the verdicts.

    A one-time, resumable pass, exactly like `rekindle fingerprint`: the store
    resumes on `orient_ignore_exif IS NULL`, so an interrupted run continues
    rather than restarting. It is NOT opportunistic - running a detector at
    two rotations cannot happen at render time.

    Verdicts are consumed IN ORDER on this thread and written in one batch, so
    the result does not depend on which worker finished first.
    """
    import time
    from collections import deque
    from concurrent.futures import Future, ThreadPoolExecutor

    report = OrientationReport()
    started = time.perf_counter()
    photos = list(store.iter_unoriented())
    total = len(photos)
    verdicts: list[tuple[str, bool, str]] = []
    depth = max(1, workers) * 4

    def look(photo):
        # First READABLE path wins, as `memory.fingerprint._row_for` does: the
        # same bytes can sit in two folders and on a partially extracted
        # library one copy may be gone while the other is fine. A verdict is
        # stored against the file HASH and the store installs it for every
        # path that hash has, so which copy was examined does not matter.
        last = Verdict(False, "file not found", error=True)
        for path in photo.paths:
            try:
                return photo, path, examine(path, detector, min_margin=min_margin)
            except (OSError, ValueError, RuntimeError) as exc:
                last = Verdict(False, f"{type(exc).__name__}: {exc}", error=True)
        return photo, None, last

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        pending: deque[Future] = deque()
        queue = iter(photos)

        def submit_more() -> None:
            while len(pending) < depth:
                nxt = next(queue, None)
                if nxt is None:
                    return
                pending.append(pool.submit(look, nxt))

        submit_more()
        done = 0
        while pending:
            photo, path, verdict = pending.popleft().result()
            submit_more()
            done += 1
            report.examined += 1
            if verdict.error:
                report.errors += 1
                # NOT recorded as examined. There is no `orient_error` column
                # saying why, so writing 0 would be indistinguishable from
                # "examined, tag trusted" and the file would never be looked
                # at again. A missing drive is a reason to retry, not a
                # verdict.
            else:
                verdicts.append((photo.file_hash, verdict.ignore_exif, verdict.as_json()))
                if verdict.ignore_exif:
                    report.tag_ignored += 1
                    report.corrected.append((path, verdict))
                    report.invalidated.append(photo.file_hash)
                elif verdict.reason == NO_TAG:
                    report.no_tag += 1
                elif verdict.reason == NO_FACE:
                    report.no_face += 1
                else:
                    report.tag_trusted += 1
            if progress:
                progress(done, total)
    store.set_orientations(verdicts)
    report.elapsed_s = time.perf_counter() - started
    return report
