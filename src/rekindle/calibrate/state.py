"""`calibration.json` - what happened, when, and how far through you are.

WHY THIS IS NOT `rekindle.toml`
-------------------------------
`rekindle.toml` is a file a user may commit to a public repository, so it
holds numbers and nothing else. This file holds the opposite: file hashes of
the photographs someone was shown, a timestamp, a library size. None of that
belongs in a committable file, and all of it is needed to resume a sitting.

So there are two files with two lifetimes. `rekindle.toml` is the RESULT and
is portable. `calibration.json` is the WORKING STATE and lives beside the
index, in the data directory, which is already where every other per-library
artefact lives and is already the directory nobody commits.

FINISHING IS A REAL STATE
-------------------------
`finished_at` is the whole point of the onboarding design. Once it is set,
nothing offers calibration again unprompted - not on the next index, not on
the next `rekindle memory`, not in the UI. `should_offer` is the single place
that decides, so there is one answer rather than one per caller.

Re-entry is first class and is not the same thing as never having calibrated.
A user who calibrated in September against 19,480 photos and now has 24,000
gets told exactly that, and gets to redo everything or jump to one threshold.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

STATE_NAME = "calibration.json"

#: Bumped only if the shape below changes incompatibly. An unrecognised
#: version is treated as "no calibration state", never as an error: a stale
#: working file must not stop someone opening their photographs.
STATE_VERSION = 1

#: How much a library must grow before re-entry is SUGGESTED rather than
#: merely available. A fifth more evidence is enough that the distributions
#: can genuinely have moved - a new camera, a decade imported at once.
#:
#: 0.20 rather than 0.25 because the brief's own worked example is 19,480
#: photographs growing to 24,000, which is 23.2%: a threshold that did not
#: fire on the case the design was written around would be the wrong
#: threshold, and it took writing the test to notice.
REGROWTH_FRACTION = 0.20


@dataclass
class Answer:
    """One judgement, exactly as it was given."""

    setting: str
    value: float
    rejected: bool
    subject: str = ""


@dataclass
class Calibration:
    """The durable record of one calibration, finished or in progress."""

    version: int = STATE_VERSION
    #: ISO 8601, UTC. None while it is still in progress.
    finished_at: str | None = None
    started_at: str | None = None
    #: How many photographs the library held when this was done. The honest
    #: unit for "is this still calibrated?" - a threshold derived against
    #: 19,480 photos is not automatically wrong at 24,000, but the user is
    #: entitled to know the difference.
    library_size: int = 0
    #: Settings the user has worked through, in order. A step in here is done
    #: whether or not it changed anything - "I looked and the default was
    #: right" is an answer.
    completed: list[str] = field(default_factory=list)
    #: Every judgement, so a sitting can be stopped and resumed and so a
    #: threshold can be re-derived without re-asking.
    answers: list[Answer] = field(default_factory=list)
    #: Settings the user explicitly skipped. Distinct from "not reached".
    skipped: list[str] = field(default_factory=list)

    # ---- derived

    @property
    def in_progress(self) -> bool:
        return self.started_at is not None and self.finished_at is None

    @property
    def finished(self) -> bool:
        return self.finished_at is not None

    def answers_for(self, setting: str) -> list[Answer]:
        return [a for a in self.answers if a.setting == setting]

    def record(self, answer: Answer) -> None:
        """Add a judgement, replacing any earlier answer about the same thing.

        Replacing matters on re-entry: someone walking back through a step
        should be able to change their mind about a photograph rather than
        have both answers averaged into a threshold.
        """
        self.answers = [
            a
            for a in self.answers
            if not (a.setting == answer.setting and a.subject and a.subject == answer.subject)
        ]
        self.answers.append(answer)

    def complete(self, setting: str) -> None:
        self.skipped = [s for s in self.skipped if s != setting]
        if setting not in self.completed:
            self.completed.append(setting)

    def skip(self, setting: str) -> None:
        self.completed = [s for s in self.completed if s != setting]
        if setting not in self.skipped:
            self.skipped.append(setting)

    def reset_setting(self, setting: str) -> None:
        """Forget one threshold's judgements, so it can be redone from scratch."""
        self.answers = [a for a in self.answers if a.setting != setting]
        self.completed = [s for s in self.completed if s != setting]
        self.skipped = [s for s in self.skipped if s != setting]

    def start(self, *, library_size: int, now: datetime | None = None) -> None:
        self.started_at = (now or datetime.now(UTC)).isoformat()
        self.library_size = library_size

    def finish(self, *, library_size: int, now: datetime | None = None) -> None:
        self.finished_at = (now or datetime.now(UTC)).isoformat()
        self.library_size = library_size

    # ---- persistence

    def to_json(self) -> dict:
        return {
            "version": STATE_VERSION,
            "finished_at": self.finished_at,
            "started_at": self.started_at,
            "library_size": self.library_size,
            "completed": list(self.completed),
            "skipped": list(self.skipped),
            "answers": [asdict(a) for a in self.answers],
        }

    @classmethod
    def from_json(cls, raw: dict) -> Calibration:
        return cls(
            version=raw.get("version", STATE_VERSION),
            finished_at=raw.get("finished_at"),
            started_at=raw.get("started_at"),
            library_size=raw.get("library_size", 0),
            completed=list(raw.get("completed", [])),
            skipped=list(raw.get("skipped", [])),
            answers=[
                Answer(
                    setting=a["setting"],
                    value=float(a["value"]),
                    rejected=bool(a["rejected"]),
                    subject=a.get("subject", ""),
                )
                for a in raw.get("answers", [])
            ],
        )


def state_path(data_dir: Path) -> Path:
    return Path(data_dir) / STATE_NAME


def load(data_dir: Path) -> Calibration:
    """Read the working state. Absent, unreadable or from a future version all
    mean the same thing to a caller: nothing has been calibrated yet.

    Deliberately NOT fatal, unlike `rekindle.toml`. A malformed override would
    silently run on numbers the user thinks they replaced; a malformed working
    file costs at worst one repeated sitting, and refusing to start over it
    would be the larger harm.
    """
    path = state_path(data_dir)
    if not path.is_file():
        return Calibration()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError, UnicodeDecodeError):
        return Calibration()
    if not isinstance(raw, dict) or raw.get("version") != STATE_VERSION:
        return Calibration()
    try:
        return Calibration.from_json(raw)
    except (KeyError, TypeError, ValueError):
        return Calibration()


def save(data_dir: Path, calibration: Calibration) -> Path:
    """Write the working state after every answer, so a crash costs nothing.

    Written whole and replaced, not appended: a half-written JSON file is
    unreadable, and `load` would then throw away a whole sitting rather than
    the last answer of one.
    """
    path = state_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(
        json.dumps(calibration.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temp.replace(path)
    return path


# ---------------------------------------------------------------------------
# when to offer it


@dataclass(frozen=True)
class Offer:
    """Whether to bring calibration up, and what to say."""

    show: bool
    reason: str
    headline: str = ""


#: The reasons, so callers and tests name them rather than matching prose.
OFFER_FIRST_RUN = "first_run"
OFFER_RESUME = "resume"
OFFER_GREW = "library_grew"
OFFER_DONE = "already_calibrated"
OFFER_NO_LIBRARY = "no_library"


def should_offer(calibration: Calibration, *, library_size: int) -> Offer:
    """The single decision about whether to interrupt someone.

    Called by the CLI after an index and by the web UI on load, so the two
    cannot disagree - and so "never prompt again unaddressed" is one line of
    code rather than a convention.
    """
    if library_size <= 0:
        return Offer(False, OFFER_NO_LIBRARY)

    if calibration.in_progress:
        done = len(calibration.completed) + len(calibration.skipped)
        return Offer(
            True,
            OFFER_RESUME,
            f"You were part-way through calibrating - {done} step"
            f"{'s' if done != 1 else ''} done. Pick up where you left off?",
        )

    if not calibration.finished:
        return Offer(
            True,
            OFFER_FIRST_RUN,
            f"{library_size:,} photographs indexed. rekindle's thresholds were "
            f"derived from somebody else's library; spend a few minutes deriving "
            f"them from yours?",
        )

    # Finished. It never nags - the only thing that reopens the question is a
    # library that has grown enough for the evidence to have changed, and even
    # that is a note, not a prompt.
    grown = library_size - calibration.library_size
    if calibration.library_size and grown > calibration.library_size * REGROWTH_FRACTION:
        return Offer(
            False,
            OFFER_GREW,
            f"You calibrated against {calibration.library_size:,} photographs and "
            f"now have {library_size:,}. `rekindle calibrate` if you want to revisit it.",
        )
    return Offer(False, OFFER_DONE)
