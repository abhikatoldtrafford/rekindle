"""The calibration flow, over HTTP.

A thin translation layer and nothing else. Every decision - which photograph
to show next, what the answers derive to, how much to trust it, whether the
publishing gate may be widened - lives in `rekindle.calibrate.session`, which
the terminal front end drives through the same functions. Two front ends with
two copies of the safeguard is a safeguard that exists on one of them.

The session is held per server process, keyed by nothing: there is one
library, one user, and one loopback socket. Its DURABLE state is
`calibration.json`, written after every answer, so a browser refresh, a
crash, or a switch from the browser to the terminal all resume the same
sitting.
"""

from __future__ import annotations

import threading
from pathlib import Path

from rekindle import config
from rekindle.calibrate import impact, plan, preview, session, state
from rekindle.calibrate.plan import Step
from rekindle.config.writer import write as write_toml
from rekindle.web.library import Library


class CalibrationRoom:
    """The one calibration session this server has, built on first use.

    Lazy, because the library is loaded on a background thread and a session
    needs its photographs. Locked, because `http.server` is threaded and two
    concurrent answers would race the JSON file.
    """

    def __init__(self, library: Library, *, memories_root: Path) -> None:
        self.library = library
        self.memories_root = Path(memories_root)
        self._lock = threading.Lock()
        self._session: session.Session | None = None
        #: The last example handed out per setting, so `answer` can be posted
        #: with a verdict alone. The browser is never trusted to send back a
        #: value: it would let a hand-crafted POST place a judgement anywhere
        #: on the scale.
        self._pending: dict[str, object] = {}

    def session_for(self) -> session.Session:
        with self._lock:
            if self._session is None:
                index = self.library.require_index()
                photos = index.all()
                self._session = session.Session(
                    self.library.data_dir,
                    photos,
                    availability=session.detect(
                        photos,
                        semantic=self.library.semantic_support(),
                        faces=None,
                    ),
                )
            return self._session

    def reload(self) -> None:
        with self._lock:
            self._session = None
            self._pending.clear()


def _step_or_error(name: str) -> Step:
    from rekindle.web.api import ApiError

    step = plan.BY_SETTING.get(name)
    if step is None:
        raise ApiError(f"No threshold called {name}.", status=404)
    return step


def _setting_json(name: str, current: float) -> dict:
    spec = config.CATALOGUE[name]
    return {
        "name": name,
        "value": current,
        "default": spec.default,
        "unit": spec.unit,
        "what": spec.what,
        "raising": spec.raising,
        "lowering": spec.lowering,
        "measured": spec.measured,
        "minimum": spec.minimum,
        "maximum": spec.maximum,
        "loosening": spec.loosening,
    }


def status(room: CalibrationRoom) -> dict:
    """Where this library stands, and whether to offer calibration at all."""
    sess = room.session_for()
    offer = sess.offer()
    settings = sess.settings()
    return {
        "offer": {"show": offer.show, "reason": offer.reason, "headline": offer.headline},
        "finished_at": sess.record.finished_at,
        "started_at": sess.record.started_at,
        "library_size": len(sess.photos),
        "calibrated_against": sess.record.library_size,
        "total_steps": len(sess.steps),
        "done": sorted(sess.done),
        "unavailable": [
            {"needs": needs, "why": why} for needs, why in sorted(sess.availability.reasons.items())
        ],
        "steps": [
            {
                "setting": step.setting,
                "title": step.title,
                "area": step.area,
                "question": step.question,
                "note": step.note,
                "modes": list(step.modes),
                "yes": step.yes,
                "no": step.no,
                "position": sess.position(step)[0],
                "state": (
                    "done"
                    if step.setting in sess.record.completed
                    else "skipped"
                    if step.setting in sess.record.skipped
                    else "todo"
                ),
                "answered": len(sess.judgements(step)),
                "confidence": (sess.derived(step).confidence() if sess.judgements(step) else ""),
                **_setting_json(step.setting, settings.get(step.setting)),
            }
            for step in sess.steps
        ],
        "overrides": sess.overrides(),
    }


def begin(room: CalibrationRoom) -> dict:
    sess = room.session_for()
    sess.start()
    return status(room)


def example(room: CalibrationRoom, *, setting: str) -> dict:
    """The next photograph (or pair) to ask about.

    Carries thumbnail hashes rather than paths: the page must never be handed
    a filesystem path, which is a username and a drive layout, and the
    existing `/api/thumb/<hash>` route already serves the pixels.
    """
    sess = room.session_for()
    step = _step_or_error(setting)
    found = sess.next_example(step)
    derived = sess.derived(step)
    if found is None:
        room._pending.pop(setting, None)
        return {
            "setting": setting,
            "exhausted": True,
            "answered": len(sess.judgements(step)),
            "derived": None if not derived.usable else derived.value,
            "confidence": derived.confidence(),
        }
    room._pending[setting] = found
    return {
        "setting": setting,
        "exhausted": False,
        # The measured value is DELIBERATELY absent. A blind judgement stops
        # being blind the moment the page can render the number, and a
        # debugging `console.log` is all it would take.
        "hashes": found.subject.split("+"),
        "caption": found.caption,
        "answered": len(sess.judgements(step)),
        "derived": None if not derived.usable else derived.value,
        "confidence": derived.confidence(),
        "thin": derived.thin,
        "question": step.question,
        "yes": step.yes,
        "no": step.no,
    }


def answer(room: CalibrationRoom, *, setting: str, said_yes: bool) -> dict:
    """Record a verdict about the example this server last handed out."""
    from rekindle.web.api import ApiError

    sess = room.session_for()
    step = _step_or_error(setting)
    pending = room._pending.get(setting)
    if pending is None:
        raise ApiError("Ask for an example before answering one.", status=409)
    with room._lock:
        sess.answer(step, pending, said_yes)  # type: ignore[arg-type]
        room._pending.pop(setting, None)
    return example(room, setting=setting)


def consequence(room: CalibrationRoom, *, setting: str, value: float) -> dict:
    """What this number would do to this library, before anything is written."""
    sess = room.session_for()
    _step_or_error(setting)
    if not preview.countable(setting):
        return {
            "setting": setting,
            "countable": False,
            "why": (
                "This one does not decide whether a single photograph is usable, so "
                "there is no 'photos kept' number for it. Its effect shows up in the "
                "memories themselves - the summary on the way out counts those."
            ),
        }
    got = preview.consequence(sess.photos, setting, float(value), settings=sess.base)
    return {
        "setting": setting,
        "countable": True,
        "current": got.current,
        "proposed": got.proposed,
        "kept_now": got.kept_now,
        "kept_then": got.kept_then,
        "gained": got.gained,
        "dropped": got.dropped,
        "sentence": got.sentence(),
        "newly_kept": [p.file_hash for p in got.newly_kept],
        "newly_dropped": [p.file_hash for p in got.newly_dropped],
    }


def choose(room: CalibrationRoom, *, setting: str, value: float | None, accept: bool) -> dict:
    """Stage a value for one step, or accept the default.

    A refusal comes back as a 200 with `refused`, not as an error: it is a
    question the user has to answer, not a failure. The page shows the
    sentence and asks for it back.
    """
    sess = room.session_for()
    step = _step_or_error(setting)
    if accept or value is None:
        sess.accept_default(step)
        return {"setting": setting, "refused": None, **status(room)}
    refusal = sess.propose(step, float(value))
    if refusal is not None:
        return {
            "setting": setting,
            "refused": {
                "current": refusal.current,
                "proposed": refusal.proposed,
                "meaning": refusal.meaning,
                "phrase": refusal.phrase,
            },
        }
    return {"setting": setting, "refused": None, **status(room)}


def confirm(room: CalibrationRoom, *, setting: str, phrase: str) -> dict:
    """Clear the publishing gate's safeguard, for one setting, once.

    The phrase has to match exactly. The page shows it and the person has to
    send it back; there is no boolean anywhere on this path, because a boolean
    is what a drag-and-release sends.
    """
    sess = room.session_for()
    step = _step_or_error(setting)
    return {"setting": setting, "confirmed": sess.confirm_loosening(step, phrase)}


def skip(room: CalibrationRoom, *, setting: str) -> dict:
    sess = room.session_for()
    sess.skip(_step_or_error(setting))
    return status(room)


def reset(room: CalibrationRoom, *, setting: str) -> dict:
    sess = room.session_for()
    sess.reset(_step_or_error(setting))
    return status(room)


def affected(room: CalibrationRoom) -> dict:
    """The exit question. Selection only - nothing is rendered."""
    sess = room.session_for()
    report = impact.analyse(
        room.library.require_index(),
        room.memories_root,
        settings=sess.settings(),
        semantic=room.library.semantic_support(),
    )
    return {
        "sentence": report.sentence(),
        "total": report.total,
        "affected": [
            {
                "memory_id": c.memory_id,
                "title": c.title,
                "verdict": c.verdict,
                "before": c.before,
                "after": c.after,
                "gained": c.gained,
                "lost": c.lost,
            }
            for c in impact.rebuildable(report)
        ],
        "uncheckable": [{"memory_id": c.memory_id, "note": c.note} for c in report.uncheckable],
    }


def finish(room: CalibrationRoom) -> dict:
    """Write `rekindle.toml`, mark the sitting done, and reload the config.

    Writing is never destructive: `writer.write` moves the previous file to
    `rekindle.toml.bak` first, and a user who calibrated badly still has their
    old numbers on disk.
    """
    sess = room.session_for()
    overrides = sess.overrides()
    path = Path(sess.data_dir) / config.CONFIG_NAME
    write_toml(path, overrides)
    config.activate_from(sess.data_dir)
    sess.finish()
    room.reload()
    return {
        "written": str(path.name),
        "overrides": overrides,
        "backup": path.with_suffix(path.suffix + ".bak").name,
    }


def dismiss_offer(room: CalibrationRoom) -> dict:
    """ "Not now, and stop asking."

    Recorded as a finished calibration with nothing overridden, which is
    exactly what it is: the user was asked, looked, and chose the defaults.
    Any other encoding would be a second kind of "done" for `should_offer` to
    disagree with.
    """
    sess = room.session_for()
    sess.finish()
    return {"reason": state.OFFER_DONE}
