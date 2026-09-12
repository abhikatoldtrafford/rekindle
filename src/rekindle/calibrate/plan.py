"""The guided sequence: what gets asked, in what order, and how.

THIS IS NOT A SETTINGS PAGE
---------------------------
It is linear, it has a visible position ("4 of 9"), each step asks one plain
question, and it ends. Someone who has never heard of a perceptual hash gets
through it, because no step ever shows them one - they see two photographs and
answer whether it is the same moment.

The order is not alphabetical and it is not the order of `defaults.toml`. It
runs from the questions everyone can answer to the questions that need a
model, so that a library with no embeddings and no face detector still gets
a complete and useful sitting and simply stops early. `available` is what
decides that, and it decides it from the data rather than from a flag.

THREE MODES, AND THE USER PICKS PER THRESHOLD
----------------------------------------------
* BLIND     - photographs near the current cut, a plain question, no number
              ever shown. The default for every perceptual call, because it is
              the method that produced the shipped values in the first place.
* SLIDER    - a number, with the effect on the library recomputed as it moves.
              For framing decisions, where seeing the set change is the point.
* DEFAULT   - show what rekindle's number does to this library, accept or
              nudge. The two-minute path, and a legitimate answer.

A step declares which modes make sense for it; nothing offers a blind
judgement about an upscale tolerance, because there is no yes/no question
about one photograph that answers it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rekindle.calibrate.judge import REJECT_ABOVE, REJECT_BELOW

MODE_BLIND = "blind"
MODE_SLIDER = "slider"
MODE_DEFAULT = "default"

#: What a step needs before it can run at all.
NEEDS_NOTHING = "nothing"
NEEDS_FINGERPRINTS = "fingerprints"
NEEDS_EMBEDDINGS = "embeddings"
NEEDS_FACES = "faces"

AREA_QUALITY = "perceptual quality"
AREA_SAMENESS = "sameness"
AREA_FRAMING = "framing"
AREA_GATE = "the publishing gate"


@dataclass(frozen=True)
class Step:
    """One threshold, and how to ask about it."""

    setting: str
    area: str
    #: The question, in words a person uses. Shown verbatim.
    question: str
    #: The heading for the step. Not the setting name.
    title: str
    modes: tuple[str, ...]
    needs: str = NEEDS_NOTHING
    #: For a blind step: which side of the cut a YES answer sits on. A yes to
    #: "too blurry?" is a LOW sharpness, so `REJECT_BELOW`; a yes to "same
    #: moment?" about a CLIP cosine is a HIGH one, so `REJECT_ABOVE`. Getting
    #: this backwards derives a threshold that is exactly wrong rather than
    #: merely imprecise, which is why every step states it.
    direction: str = REJECT_BELOW
    #: Which way to break a tie between equally good cuts. "loose" always
    #: means the cut that says YES to fewer things - keeps more photographs
    #: for a quality gate, collapses fewer pairs for dedup. "tight" is only
    #: ever right where saying yes is the SAFE answer, which in this project
    #: is the publishing gate and nothing else.
    safer: str = "loose"
    #: The photo attribute a blind step measures, when it measures one photo.
    attribute: str = ""
    #: What "yes" and "no" are called on the buttons.
    yes: str = "yes"
    no: str = "no"
    #: Extra sentence shown under the question. Never a number.
    note: str = ""

    @property
    def mode(self) -> str:
        return self.modes[0]


#: The sequence. Order matters and is explained in the module docstring.
STEPS: tuple[Step, ...] = (
    Step(
        setting="composition.min_sharpness",
        area=AREA_QUALITY,
        title="Blur",
        question="Too blurry to put in a memory?",
        modes=(MODE_BLIND, MODE_DEFAULT),
        needs=NEEDS_FINGERPRINTS,
        direction=REJECT_BELOW,
        # A gate that is too tight deletes a photograph outright, so a tie
        # goes to the value that keeps more of them.
        safer="loose",
        attribute="sharpness",
        yes="too blurry",
        no="fine",
        note="Say yes only if you would be unhappy to see it in a slideshow.",
    ),
    Step(
        setting="composition.min_brightness",
        area=AREA_QUALITY,
        title="Photos that came out too dark",
        question="Too dark to be worth showing?",
        modes=(MODE_BLIND, MODE_DEFAULT),
        needs=NEEDS_FINGERPRINTS,
        direction=REJECT_BELOW,
        safer="loose",
        attribute="brightness",
        yes="too dark",
        no="fine",
        note="A deliberately dark photograph is not a mistake - say no to those.",
    ),
    Step(
        setting="composition.max_brightness",
        area=AREA_QUALITY,
        title="Photos that came out blown out",
        question="Too washed out to be worth showing?",
        modes=(MODE_BLIND, MODE_DEFAULT),
        needs=NEEDS_FINGERPRINTS,
        direction=REJECT_ABOVE,
        safer="loose",
        attribute="brightness",
        yes="too bright",
        no="fine",
        note="Snow, a beach and a white wall are all legitimately bright.",
    ),
    Step(
        setting="composition.min_short_edge",
        area=AREA_QUALITY,
        title="Small photos",
        question="Too small and soft to show full-screen?",
        modes=(MODE_SLIDER, MODE_DEFAULT),
        needs=NEEDS_NOTHING,
        direction=REJECT_BELOW,
        safer="loose",
        note="On most libraries small means OLD. Raising this quietly deletes your earliest years.",
    ),
    Step(
        setting="dedup.gap_seconds",
        area=AREA_SAMENESS,
        title="How long a burst lasts",
        question="Could these two have been taken as one burst?",
        modes=(MODE_SLIDER, MODE_DEFAULT),
        needs=NEEDS_NOTHING,
        direction=REJECT_ABOVE,
        safer="tight",
        note="This only decides which pairs get compared at all. The pixels decide the rest.",
    ),
    Step(
        setting="dedup.phash_distance",
        area=AREA_SAMENESS,
        title="Near-identical frames",
        question="Same moment, or two different photographs?",
        # The slider is here as well as the blind mode, and it is not
        # decoration: this threshold DELETES, and the slider is the only mode
        # that shows the deletion count across a range before you pick one.
        # Four blind judgements on the reference library derived 21, which
        # would have collapsed 3,264 more photographs than 6 does.
        modes=(MODE_BLIND, MODE_SLIDER, MODE_DEFAULT),
        needs=NEEDS_FINGERPRINTS,
        direction=REJECT_BELOW,
        # A yes here DELETES a photograph, so a tie goes to the cut that says
        # yes to fewer pairs. For a floor that is the lowest cut: "loose".
        safer="loose",
        yes="same moment",
        no="different photos",
        note="Only one of a pair judged the same moment survives into a memory.",
    ),
    Step(
        setting="diversity.lambda_penalty",
        area=AREA_SAMENESS,
        title="Variety against quality",
        question="How hard should a memory work to avoid similar-looking shots?",
        modes=(MODE_SLIDER, MODE_DEFAULT),
        needs=NEEDS_FINGERPRINTS,
        direction=REJECT_ABOVE,
        safer="loose",
    ),
    Step(
        setting="composition.max_aspect",
        area=AREA_FRAMING,
        title="Panoramas",
        question="How wide is too wide to show?",
        modes=(MODE_SLIDER, MODE_DEFAULT),
        needs=NEEDS_NOTHING,
        direction=REJECT_ABOVE,
        safer="loose",
        note="A panorama past this becomes a thin band inside a black frame.",
    ),
    Step(
        setting="composition.square_ratio",
        area=AREA_FRAMING,
        title="Square crops",
        question="How close to square still counts as square?",
        modes=(MODE_SLIDER, MODE_DEFAULT),
        needs=NEEDS_NOTHING,
        direction=REJECT_ABOVE,
        safer="loose",
    ),
    Step(
        setting="composition.upscale_tolerance",
        area=AREA_FRAMING,
        title="Enlarging small photos",
        question="How far may an old photo be stretched before it is padded instead?",
        modes=(MODE_SLIDER, MODE_DEFAULT),
        needs=NEEDS_NOTHING,
        direction=REJECT_ABOVE,
        safer="loose",
    ),
    Step(
        setting="selection.max_shots",
        area=AREA_FRAMING,
        title="How long a memory is",
        question="How many photographs should one memory hold?",
        modes=(MODE_SLIDER, MODE_DEFAULT),
        needs=NEEDS_NOTHING,
        direction=REJECT_ABOVE,
        safer="loose",
        note="At about two and a half seconds a shot, 24 is a minute.",
    ),
    Step(
        setting="dedup.cosine",
        area=AREA_SAMENESS,
        title="The same moment, framed differently",
        question="Same moment, or two different photographs?",
        modes=(MODE_BLIND, MODE_DEFAULT),
        needs=NEEDS_EMBEDDINGS,
        # Unlike the dHash step: a yes here is a HIGH cosine, not a low one.
        direction=REJECT_ABOVE,
        safer="loose",
        yes="same moment",
        no="different photos",
        note=(
            "These pairs look different to the pixels. This asks whether they look the same to you."
        ),
    ),
    Step(
        setting="faces.gate_threshold",
        area=AREA_GATE,
        title="The publishing gate",
        question="Is there a person in this photograph?",
        modes=(MODE_BLIND, MODE_DEFAULT),
        needs=NEEDS_FACES,
        # A yes is a HIGH detector score. And this is the one step in the
        # whole sequence where saying yes is the SAFE answer, so a tie goes
        # to the cut that blocks MORE - "tight" - not to the one that keeps
        # more photographs publishable.
        direction=REJECT_ABOVE,
        safer="tight",
        yes="yes, someone is in it",
        no="no one is in it",
        note=(
            "This gate decides what may be published. It only ever proposes; "
            "the review queue is what actually stands between a photograph and "
            "a public repository."
        ),
    ),
)

BY_SETTING: dict[str, Step] = {s.setting: s for s in STEPS}


@dataclass
class Availability:
    """What this library can actually be asked about."""

    fingerprints: bool = False
    embeddings: bool = False
    faces: bool = False
    #: Why a capability is missing, keyed by `NEEDS_*`, phrased as the sentence
    #: to show. A step that cannot run says so rather than vanishing: silently
    #: dropping steps is how "4 of 9" becomes a lie.
    reasons: dict[str, str] = field(default_factory=dict)

    def allows(self, step: Step) -> bool:
        return {
            NEEDS_NOTHING: True,
            NEEDS_FINGERPRINTS: self.fingerprints,
            NEEDS_EMBEDDINGS: self.embeddings,
            NEEDS_FACES: self.faces,
        }[step.needs]


def available_steps(availability: Availability) -> list[Step]:
    """The steps this library can be asked, in order."""
    return [s for s in STEPS if availability.allows(s)]


def position(steps: list[Step], setting: str) -> tuple[int, int]:
    """(this step's 1-based number, total). The "4 of 9" in the header."""
    for i, step in enumerate(steps, start=1):
        if step.setting == setting:
            return i, len(steps)
    return 0, len(steps)


def next_step(steps: list[Step], done: set[str]) -> Step | None:
    for step in steps:
        if step.setting not in done:
            return step
    return None
