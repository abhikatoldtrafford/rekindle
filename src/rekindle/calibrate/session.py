"""One sitting: the state machine behind both the terminal and the web UI.

Everything with a decision in it lives here and neither front end reimplements
any of it. The CLI asks `next_example`, prints it and calls `answer`; the web
UI does the same over HTTP. That is what makes the flow testable without a
browser, a model, a GPU or a photograph - which is the whole of what CI has.

THE SAFEGUARD ON THE PUBLISHING GATE
-------------------------------------
`propose` refuses, by returning a `Refusal` rather than by raising, to move a
setting in its `loosening` direction. Only `confirm_loosening` clears that,
and it takes the exact sentence back - a caller cannot pass `True` by
accident, and a drag-and-release cannot pass it at all.

That property is the reason it lives here and not in the CLI: the web UI is a
second caller, and a safeguard implemented once per front end is a safeguard
that exists on one of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rekindle import config
from rekindle.calibrate import labels, plan, sampling, state
from rekindle.calibrate.judge import Derived, Judgement, derive, next_probe
from rekindle.calibrate.plan import Availability, Step
from rekindle.models import MediaType, Photo

#: The exact words a caller must send back to widen the publishing gate.
#: Not a boolean: a boolean is what a slider sends.
LOOSEN_PHRASE = "I understand this publishes more photographs of other people"


@dataclass(frozen=True)
class Refusal:
    """A change that will not be made without a separate confirmation."""

    setting: str
    current: float
    proposed: float
    #: What it means, in plain words, with no jargon and no number.
    meaning: str
    #: What the caller must echo back.
    phrase: str = LOOSEN_PHRASE


@dataclass
class StepState:
    """Where one step has got to."""

    step: Step
    judgements: list[Judgement] = field(default_factory=list)
    #: The value this step would write. None until it has one.
    proposed: float | None = None
    mode: str = ""

    @property
    def answered(self) -> int:
        return len(self.judgements)


def _loosening_meaning(setting: str) -> str:
    """Plain words for what weakening this gate does.

    Written per setting rather than generated, because "more photographs
    containing other people will become publishable" is the sentence the
    person has to understand, and a template would produce something about
    thresholds instead.
    """
    if setting == "faces.gate_threshold":
        return (
            "More photographs containing other people will become publishable. "
            "The face gate is what holds back a photograph with somebody in it "
            "before it can reach a public repository; widening it means fewer "
            "photographs are held back, including ones with a stranger in the "
            "background. The review queue is still the thing standing between "
            "them and a public repository - but it will have more to catch, and "
            "it is a human being."
        )
    return "This weakens a safety gate. More will get through than rekindle's default lets through."


class Session:
    """A calibration in progress, over one library."""

    def __init__(
        self,
        data_dir: Path,
        photos: list[Photo],
        *,
        availability: Availability | None = None,
        base: config.Settings | None = None,
        record: state.Calibration | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.photos = photos
        self.availability = availability or detect(photos)
        self.base = base or config.active()
        self.record = record if record is not None else state.load(self.data_dir)
        self.steps = plan.available_steps(self.availability)
        self._states: dict[str, StepState] = {}
        self._pools: dict[str, list] = {}
        #: Settings the user has chosen a new value for, this sitting or a
        #: previous one. Only these are written.
        self.chosen: dict[str, float] = {}
        self._confirmed_loosening: set[str] = set()
        self._replay()

    # ---- lifecycle

    def _replay(self) -> None:
        """Rebuild in-memory step state from the recorded answers.

        This is what makes a sitting resumable: nothing is kept in the process
        that is not also on disk, so stopping is free and there is no "save
        before you quit".
        """
        for step in self.steps:
            answers = self.record.answers_for(step.setting)
            if not answers:
                continue
            st = self._state(step)
            st.judgements = [
                Judgement(value=a.value, rejected=a.rejected, subject=a.subject) for a in answers
            ]
            found = derive(st.judgements, direction=step.direction, safer=step.safer)
            if found.usable:
                st.proposed = _quantise(step.setting, found.value)
                if step.setting in self.record.completed:
                    self.chosen[step.setting] = st.proposed

    def start(self) -> None:
        if not self.record.started_at:
            self.record.start(library_size=len(self.photos))
            self.save()

    def save(self) -> None:
        state.save(self.data_dir, self.record)

    def finish(self) -> None:
        self.record.finish(library_size=len(self.photos))
        self.save()

    def offer(self) -> state.Offer:
        return state.should_offer(self.record, library_size=len(self.photos))

    # ---- progress

    def _state(self, step: Step) -> StepState:
        if step.setting not in self._states:
            self._states[step.setting] = StepState(step=step, mode=step.mode)
        return self._states[step.setting]

    @property
    def done(self) -> set[str]:
        return set(self.record.completed) | set(self.record.skipped)

    def position(self, step: Step) -> tuple[int, int]:
        return plan.position(self.steps, step.setting)

    def next_step(self) -> Step | None:
        return plan.next_step(self.steps, self.done)

    # ---- blind judgement

    def pool(self, step: Step) -> list:
        """Everything this step could ask about, measured, cached per step."""
        if step.setting in self._pools:
            return self._pools[step.setting]
        self._pools[step.setting] = _build_pool(step, self.photos, self.base)
        return self._pools[step.setting]

    def band(self, step: Step) -> tuple[float, float]:
        pool = self.pool(step)
        values = [row[0] for row in pool]
        if not values:
            return (0.0, 1.0)
        low_q, high_q = _band_percentiles(step)
        return sampling.band(values, low_q=low_q, high_q=high_q)

    def next_example(self, step: Step) -> sampling.Example | None:
        """The photograph (or pair) worth asking about next, or None when the
        pool is exhausted."""
        pool = self.pool(step)
        if not pool:
            return None
        st = self._state(step)
        low, high = self.band(step)
        target = next_probe(st.judgements, direction=step.direction, floor=low, ceiling=high)
        used = {j.subject for j in st.judgements}
        if len(pool[0]) == 3:
            return sampling.nearest_pair(pool, target, used=used)
        return sampling.nearest(pool, target, used=used)

    def answer(self, step: Step, example: sampling.Example, said_yes: bool) -> Derived:
        """Record one judgement and re-derive. Persists immediately.

        TWO WRITES, TWO LIFETIMES. `calibration.json` is working state and
        REPLACES an earlier answer about the same photograph, so changing your
        mind changes the threshold. `labels.jsonl` is append-only and keeps
        both, because a judgement that was given is a fact about a moment and
        a later one about the same photograph is a second fact. See
        `calibrate.labels` for why that distinction is worth two files, and
        `docs/decision-log-calibration-labels.md` for what the evidence is
        currently worth.
        """
        st = self._state(step)
        st.judgements = [j for j in st.judgements if j.subject != example.subject]
        st.judgements.append(
            Judgement(value=example.value, rejected=said_yes, subject=example.subject)
        )
        self.record.record(
            state.Answer(
                setting=step.setting,
                value=example.value,
                rejected=said_yes,
                subject=example.subject,
            )
        )
        self.save()
        labels.append(
            self.data_dir,
            setting=step.setting,
            subject=example.subject,
            value=example.value,
            rejected=said_yes,
            question=step.question,
            library_size=len(self.photos),
        )
        found = self.derived(step)
        if found.usable:
            st.proposed = _quantise(step.setting, found.value)
        return found

    def derived(self, step: Step) -> Derived:
        st = self._state(step)
        return derive(st.judgements, direction=step.direction, safer=step.safer)

    def judgements(self, step: Step) -> list[Judgement]:
        return list(self._state(step).judgements)

    # ---- committing a step

    def propose(self, step: Step, value: float) -> Refusal | None:
        """Stage a value for this step, or refuse it.

        Returns `None` when the value is accepted and a `Refusal` when it
        would weaken a safety gate without confirmation. TIGHTENING IS ALWAYS
        FREE: a value that makes a gate stricter never reaches the refusal
        path, so someone hardening their own publishing gate is never asked
        to confirm anything.
        """
        setting = config.CATALOGUE[step.setting]
        value = _quantise(step.setting, value)
        problem = setting.clamp_error(value)
        if problem:
            raise config.ConfigError(problem)
        # Against `self.base` - what this user's config actually says today -
        # and not against the shipped default. The `Refusal` below has always
        # SHOWN the user their current value while the decision above it was
        # made against a different number; see `Setting.loosens`.
        current = self.base.get(step.setting)
        if setting.loosens(value, current) and step.setting not in self._confirmed_loosening:
            return Refusal(
                setting=step.setting,
                current=current,
                proposed=value,
                meaning=_loosening_meaning(step.setting),
            )
        self.chosen[step.setting] = value
        self.record.complete(step.setting)
        self.save()
        return None

    def confirm_loosening(self, step: Step, phrase: str) -> bool:
        """Clear the safeguard for ONE setting, for this sitting only.

        The caller has to echo `LOOSEN_PHRASE` exactly. A boolean argument
        would be satisfiable by a checkbox that a drag-and-release could tick;
        a sentence has to be typed or clicked through a dialog that shows it.
        """
        if phrase.strip() != LOOSEN_PHRASE:
            return False
        self._confirmed_loosening.add(step.setting)
        return True

    def accept_default(self, step: Step) -> None:
        """ "I looked, and rekindle's number is right for me" - a real answer.

        Marks the step done and stages nothing, so the value stays absent from
        `rekindle.toml` and a better default in a later release still reaches
        this user.
        """
        self.chosen.pop(step.setting, None)
        self.record.complete(step.setting)
        self.save()

    def skip(self, step: Step) -> None:
        self.record.skip(step.setting)
        self.save()

    def reset(self, step: Step) -> None:
        self.record.reset_setting(step.setting)
        self.chosen.pop(step.setting, None)
        self._states.pop(step.setting, None)
        self.save()

    # ---- the result

    def settings(self) -> config.Settings:
        """The configuration this sitting would produce."""
        return self.base.with_values(self.chosen)

    def overrides(self) -> dict[str, float]:
        """Exactly what goes in `rekindle.toml` - the differences and nothing
        else, so an upgrade still improves what the user did not choose."""
        return self.settings().changed_from_defaults()


# ---------------------------------------------------------------------------
# what this library can be asked


def detect(photos: list[Photo], *, semantic=None, faces=None) -> Availability:
    """What is available, read off the data rather than off a flag.

    A library indexed but never fingerprinted has no sharpness scores, so the
    blur step cannot run - and saying so, with the command that fixes it, is
    more useful than showing a step that would derive a threshold from
    nothing.
    """
    fingerprinted = any(
        p.meta.sharpness is not None or p.meta.phash is not None
        for p in photos
        if p.media_type is MediaType.IMAGE
    )
    reasons = {}
    if not fingerprinted:
        reasons[plan.NEEDS_FINGERPRINTS] = (
            "This library has no perceptual fingerprints yet, so blur, "
            "brightness and near-duplicate questions cannot be asked. "
            "Run `rekindle fingerprint` first."
        )
    if semantic is None:
        reasons[plan.NEEDS_EMBEDDINGS] = (
            "This library has no embeddings, so the 'same moment, framed "
            "differently' question cannot be asked. "
            "Run `rekindle setup semantic` and `rekindle embed` if you want it."
        )
    if faces is None:
        reasons[plan.NEEDS_FACES] = (
            "The face detector is not available, so the publishing gate "
            "cannot be calibrated. It keeps rekindle's default, which is the "
            "conservative one."
        )
    return Availability(
        fingerprints=fingerprinted,
        embeddings=semantic is not None,
        faces=faces is not None,
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# per-step pools and quantisation


def _band_percentiles(step: Step) -> tuple[float, float]:
    """Which slice of the user's own distribution a step explores.

    Each is chosen so the band contains obvious cases at BOTH ends - the user
    has to be able to answer a confident yes and a confident no somewhere in
    it, or there is no line for bisection to find.
    """
    # The bands below REACH THE EXTREMES on the side the threshold lives on,
    # and that is not cosmetic. A floor of p0.1 was tried first and it hid the
    # eighteen worst photographs in a 18,201-photo library - which is exactly
    # the evidence a blur gate needs, and the part bisection can never reach on
    # its own once the user has said yes once. The other end stops well short
    # of the median: at p5 the photographs are already obviously fine, and
    # questions spent there are questions wasted.
    if step.setting == "composition.max_brightness":
        return 95.0, 100.0
    if step.setting == "composition.min_brightness":
        return 0.0, 5.0
    if step.setting == "composition.min_sharpness":
        return 0.0, 5.0
    if step.setting == "dedup.phash_distance":
        # Within-burst distances. The whole range, because "same moment" and
        # "different photos" are both common inside 30 seconds - on the
        # reference library the median within-run distance is 22.
        return 0.0, 90.0
    if step.setting == "dedup.cosine":
        return 10.0, 100.0
    if step.setting == "faces.gate_threshold":
        return 0.0, 100.0
    return sampling.DEFAULT_LOW_PERCENTILE, sampling.DEFAULT_HIGH_PERCENTILE


def _build_pool(step: Step, photos: list[Photo], base: config.Settings) -> list:
    if step.attribute:
        return sampling.measured(photos, step.attribute)
    if step.setting == "dedup.phash_distance":
        from rekindle.memory.dedup import hamming

        def distance(a: Photo, b: Photo):
            if a.meta.phash is None or b.meta.phash is None:
                return None
            return hamming(a.meta.phash, b.meta.phash)

        return sampling.pairs_within(photos, gap_seconds=base.dedup.gap_seconds, distance=distance)
    return []


#: Settings whose stored type is an integer. A derived cut lands on a
#: midpoint - 6.5 - and writing that into an int setting is a type error three
#: modules away, so it is floored HERE, where the reason is visible: a
#: threshold of "up to 6" and one of "up to 6.5" collapse the same pairs, and
#: the lower of the two collapses fewer.
def _quantise(setting: str, value: float) -> float | int:
    spec = config.CATALOGUE[setting]
    if not spec.is_int:
        return float(value)
    import math

    return int(math.floor(value))
