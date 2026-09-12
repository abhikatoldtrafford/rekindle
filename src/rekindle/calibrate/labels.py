"""`labels.jsonl` - every judgement ever given, kept.

WHY THIS EXISTS AND `calibration.json` DOES NOT COVER IT
--------------------------------------------------------
`calibration.json` is WORKING STATE. It holds the answers of the sitting you
are in the middle of, and it is right that it does: it is what makes a sitting
resumable, and `Calibration.record` deliberately REPLACES an earlier answer
about the same photograph so that changing your mind changes the threshold
rather than being averaged into it. `reset_setting` deliberately deletes a
threshold's answers so `--redo` starts clean.

Both of those are correct for working state and both of them destroy evidence.
Recalibrate twice and the first sitting is gone.

This file is the other lifetime: **append-only, one JSON object per line,
never rewritten, never truncated by a reset.** A judgement that was given is a
fact about a moment; a later judgement about the same photograph is a second
fact, not a correction of the first, and a record that cannot hold both cannot
tell you that somebody's standards moved.

WHAT IT IS NOT
--------------
It is not read by anything that decides what you see. Nothing in the memory
engine, the recipes or the configuration loader imports this module. It is
evidence being kept, and `docs/decision-log-calibration-labels.md` is the
measurement of why keeping it is currently all it is worth doing: fitted to
the 17 labels that exist, a learned scorer performs exactly at the
majority-class baseline and would replace 85% of the shots in the reference
library's memories. The honest thing to do with a training set that small is
to let it grow.

**PERSONAL DATA. Never committed, never uploaded.** It holds file hashes of a
specific person's photographs and what they thought of them. It lives in the
data directory, which `.gitignore` excludes as `[Dd]ata/`, for the same reason
`calibration.json` and the caption cache do.

THE FORMAT
----------
One JSON object per line, UTF-8, newline-terminated:

    {"v": 1, "at": "2026-09-12T11:24:43+00:00", "setting": "composition.min_sharpness",
     "subject": "<file hash>", "value": 0.0886, "rejected": true,
     "question": "Too blurry to put in a memory?", "library_size": 19318,
     "rekindle": "0.1.0"}

`question` is stored verbatim rather than as a step id, and that is the field
that makes the log worth keeping. A reworded question is a different question:
"too blurry to use?" and "too blurry to put in a memory?" will not get the
same answers, and a log that recorded only `composition.min_sharpness` would
silently pool them. `library_size` and `rekindle` are there so a label can be
dated against the library and the code that produced it.

A malformed line is SKIPPED, not fatal. This file is evidence, not
configuration; refusing to run over a bad line would be the larger harm, and
a reader that stops at the first bad line loses everything after it too.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

LOG_NAME = "labels.jsonl"

#: Bumped only if the line shape changes incompatibly. A line with an
#: unrecognised version is skipped by `read`, never guessed at.
LABEL_VERSION = 1


@dataclass(frozen=True)
class Label:
    """One judgement, as it was given, with what it was given about."""

    setting: str
    subject: str
    value: float
    rejected: bool
    at: str
    question: str = ""
    library_size: int = 0
    rekindle: str = ""

    def to_line(self) -> str:
        return json.dumps(
            {
                "v": LABEL_VERSION,
                "at": self.at,
                "setting": self.setting,
                "subject": self.subject,
                "value": self.value,
                "rejected": self.rejected,
                "question": self.question,
                "library_size": self.library_size,
                "rekindle": self.rekindle,
            },
            sort_keys=True,
        )


def log_path(data_dir: Path) -> Path:
    return Path(data_dir) / LOG_NAME


def append(
    data_dir: Path,
    *,
    setting: str,
    subject: str,
    value: float,
    rejected: bool,
    question: str = "",
    library_size: int = 0,
    now: datetime | None = None,
) -> Label:
    """Add one judgement. Never rewrites, never reads what is already there.

    Opened in append mode and closed immediately, so an interrupted sitting
    keeps every answer it gave before the interruption - which is the whole
    difference between this and the working state, where a crash mid-write
    would cost the sitting.
    """
    from rekindle import __version__

    label = Label(
        setting=setting,
        subject=subject,
        value=float(value),
        rejected=bool(rejected),
        at=(now or datetime.now(UTC)).isoformat(),
        question=question,
        library_size=int(library_size),
        rekindle=__version__,
    )
    path = log_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(label.to_line() + "\n")
    return label


def read(data_dir: Path) -> Iterator[Label]:
    """Every label ever recorded, oldest first. Absent means none.

    Streamed, and a bad line is skipped rather than raising: see the module
    docstring. Nothing in the running product calls this - it exists so the
    evidence can be looked at, and so the experiment in
    `tests/experiment_calibration_labels.py` has something to read when there
    is more of it than one sitting.
    """
    path = log_path(data_dir)
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict) or raw.get("v") != LABEL_VERSION:
                    continue
                yield Label(
                    setting=str(raw["setting"]),
                    subject=str(raw["subject"]),
                    value=float(raw["value"]),
                    rejected=bool(raw["rejected"]),
                    at=str(raw["at"]),
                    question=str(raw.get("question", "")),
                    library_size=int(raw.get("library_size", 0)),
                    rekindle=str(raw.get("rekindle", "")),
                )
            except (ValueError, TypeError, KeyError):
                continue


def summary(data_dir: Path) -> dict[str, int]:
    """How many labels are on record, per setting. Counts only - never a
    subject, never a value - so it is safe to print."""
    out: dict[str, int] = {}
    for label in read(data_dir):
        out[label.setting] = out.get(label.setting, 0) + 1
    return out
