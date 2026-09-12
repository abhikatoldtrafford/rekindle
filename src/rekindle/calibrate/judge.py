"""Turning yes/no answers about photographs into a number, honestly.

THE METHOD
----------
Every blind-judgement step reduces to the same shape. Each thing shown to the
user has one measured VALUE - a sharpness score, a Hamming distance, a cosine,
a mean luminance - and the user answers one plain question about it with a
yes or a no. The threshold is then the cut that best separates the yeses from
the noes: a one-dimensional decision stump, which is the simplest model that
can be wrong in a way you can see.

This is deliberately not a percentile. The shipped `dedup.cosine = 0.92` could
not have come from a percentile - within 30 seconds the median pair is already
0.917 - and it came from opening six pairs and noticing where the eye changed
its answer. That is exactly what this automates.

WHERE TO ASK NEXT
-----------------
A calibration that shows obvious cases learns nothing: the information is at
the margin. After each answer there is an ambiguous interval - between the
most extreme "reject" and the least extreme "keep" - and the next question is
asked in the middle of it. Ten well-placed questions beat sixty random ones,
and this is the part that makes sixty questions unnecessary.

CONFIDENCE, AND WHY IT IS REPORTED RATHER THAN HIDDEN
----------------------------------------------------
A threshold from four answers, two of them contradictory, must not present
itself like one from thirty. `Derived` therefore carries the number of
judgements, how many of them the cut disagrees with, and how wide the band
was in which the user's answers were mixed. This project's own face gate
carries "rests on 19 positives" in its docs for precisely this reason, and a
calibration that quietly dropped that habit would be a regression.

TIES BREAK TOWARDS THE SAFER SIDE
---------------------------------
Several cuts usually score the same. Which one is chosen is not arbitrary: a
quality gate that is too tight deletes an irreplaceable photograph and one
that is too loose lets a bad frame compete, so ties go to the looser cut. Each
step says which side is safe; nothing here guesses.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The user's answer means "this one is BELOW the line" - blurry, too dark,
#: too small. The threshold is a floor: `value < t` is rejected.
REJECT_BELOW = "reject_below"
#: The user's answer means "this one is ABOVE the line" - too bright, too wide,
#: too alike. The threshold is a ceiling: `value > t` is rejected.
REJECT_ABOVE = "reject_above"

#: Fewer than this and no number is derived at all. Four is not many, and
#: `Derived.confidence` says so out loud; below four the answer is noise
#: wearing a decimal point.
MIN_JUDGEMENTS = 4


@dataclass(frozen=True)
class Judgement:
    """One answer about one thing the user was actually shown.

    `subject` is whatever the step needs to find the photograph again on
    resume - a file hash, or two of them joined. It never leaves the data
    directory.
    """

    value: float
    rejected: bool
    subject: str = ""


@dataclass(frozen=True)
class Derived:
    """A threshold, and everything needed to distrust it."""

    value: float
    judgements: int
    #: Answers the chosen cut disagrees with.
    disagreements: int
    #: The band in which the user answered both ways, as (low, high). Zero
    #: width means their answers separated cleanly.
    ambiguous: tuple[float, float]
    #: How many answers fell on the SMALLER side. Reported because a cut
    #: derived from six noes and one yes sits right next to that one yes, and
    #: "seven judgements, all of which agree" would be a true sentence that
    #: gives entirely the wrong impression. Found by using this on a real
    #: library, where exactly that happened.
    minority: int = 0
    #: Empty when a number was derived. Otherwise the reason none was.
    refusal: str = ""

    @property
    def usable(self) -> bool:
        return not self.refusal

    @property
    def consistency(self) -> float:
        if not self.judgements:
            return 0.0
        return 1.0 - self.disagreements / self.judgements

    def confidence(self) -> str:
        """One plain sentence. No percentages a reader has to interpret."""
        if self.refusal:
            return self.refusal
        n, bad = self.judgements, self.disagreements
        parts = [f"rests on {n} judgement{'s' if n != 1 else ''}"]
        if bad == 0:
            parts.append("all of which agree")
        elif bad == 1:
            parts.append("one of which it disagrees with")
        else:
            parts.append(f"{bad} of which it disagrees with")
        if self.minority <= 1:
            parts.append(
                "but only ONE of them fell on the other side of the line, so the "
                "whole threshold is resting on a single photograph"
            )
        elif self.minority <= 3:
            parts.append(f"only {self.minority} of them on the smaller side of the line")
        low, high = self.ambiguous
        if high > low:
            parts.append(f"your answers were mixed between {low:.4g} and {high:.4g}")
        return "; ".join(parts) + "."

    @property
    def thin(self) -> bool:
        """True when one side of the line has almost no evidence under it.

        A caller that writes a threshold anyway is entitled to; a caller that
        writes it without SAYING so is doing the thing this project keeps
        finding in its own work.
        """
        return self.usable and self.minority <= 2


def derive(
    judgements: list[Judgement],
    *,
    direction: str,
    safer: str = "loose",
) -> Derived:
    """The cut that best separates the answers.

    `safer` says which way to break a tie: "loose" keeps more photographs,
    "tight" keeps fewer. For every quality gate in this project the answer is
    "loose", because a false positive there deletes a photograph and a false
    negative merely lets a bad one compete.
    """
    if len(judgements) < MIN_JUDGEMENTS:
        return Derived(
            value=float("nan"),
            judgements=len(judgements),
            disagreements=0,
            ambiguous=(0.0, 0.0),
            refusal=(
                f"only {len(judgements)} judgement(s) - at least {MIN_JUDGEMENTS} are "
                f"needed before a number means anything. Keeping the current value."
            ),
        )
    rejected = [j.value for j in judgements if j.rejected]
    kept = [j.value for j in judgements if not j.rejected]
    if not rejected:
        return Derived(
            float("nan"),
            len(judgements),
            0,
            (0.0, 0.0),
            refusal=(
                "you accepted every example you were shown, so there is no line to "
                "draw. Keeping the current value."
            ),
        )
    if not kept:
        return Derived(
            float("nan"),
            len(judgements),
            0,
            (0.0, 0.0),
            refusal=(
                "you rejected every example you were shown, so there is no line to "
                "draw. Keeping the current value."
            ),
        )

    values = sorted({j.value for j in judgements})
    # Candidate cuts: every midpoint between adjacent observed values, plus a
    # point outside each end. Only these can change the classification, so
    # nothing is gained by searching a continuum.
    span = values[-1] - values[0]
    pad = (span / len(values)) if span else 1.0
    cuts = (
        [values[0] - pad]
        + [(a + b) / 2.0 for a, b in zip(values, values[1:], strict=False)]
        + [values[-1] + pad]
    )

    def errors(cut: float) -> int:
        wrong = 0
        for j in judgements:
            predicted = j.value < cut if direction == REJECT_BELOW else j.value > cut
            if predicted != j.rejected:
                wrong += 1
        return wrong

    scored = [(errors(c), c) for c in cuts]
    best = min(e for e, _ in scored)
    tied = [c for e, c in scored if e == best]
    # "Loose" means the cut that rejects the FEWEST photographs: the lowest
    # floor, or the highest ceiling.
    if direction == REJECT_BELOW:
        cut = min(tied) if safer == "loose" else max(tied)
    else:
        cut = max(tied) if safer == "loose" else min(tied)

    return Derived(
        value=cut,
        judgements=len(judgements),
        disagreements=best,
        ambiguous=_ambiguous(judgements, direction),
        minority=min(len(rejected), len(kept)),
    )


def _ambiguous(judgements: list[Judgement], direction: str) -> tuple[float, float]:
    """The band in which the user answered both ways.

    For a floor that is [lowest kept, highest rejected] - inverted, so an
    empty band comes back as a zero-width one rather than as a negative
    number a caller might print.
    """
    rejected = [j.value for j in judgements if j.rejected]
    kept = [j.value for j in judgements if not j.rejected]
    if not rejected or not kept:
        return (0.0, 0.0)
    if direction == REJECT_BELOW:
        low, high = min(kept), max(rejected)
    else:
        low, high = min(rejected), max(kept)
    return (low, high) if high > low else (0.0, 0.0)


def next_probe(
    judgements: list[Judgement],
    *,
    direction: str,
    floor: float,
    ceiling: float,
) -> float:
    """The value worth asking about next.

    Bisects the ambiguous interval, which is where the answer actually lives.
    With no answers yet it starts at the middle of the offered range - not at
    the current threshold, because starting there and bisecting outward would
    make the shipped value the prior rather than the thing being tested.
    """
    rejected = [j.value for j in judgements if j.rejected]
    kept = [j.value for j in judgements if not j.rejected]
    if direction == REJECT_BELOW:
        low = max(rejected) if rejected else floor
        high = min(kept) if kept else ceiling
    else:
        low = max(kept) if kept else floor
        high = min(rejected) if rejected else ceiling
    if not (high > low):
        # The answers have crossed - the user contradicted themselves - so
        # there is no interval left to bisect. Go back to the whole range;
        # more evidence in the muddle is exactly what is missing.
        low, high = floor, ceiling
    return (low + high) / 2.0
