"""`labels.jsonl` - the append-only record of every judgement.

The distinction being tested is the whole reason the file exists: working
state REPLACES an answer and a reset DELETES one, and both are correct, and
both destroy evidence. These pin that the log does neither.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rekindle.calibrate import labels


def _read(tmp_path):
    return list(labels.read(tmp_path))


def test_a_label_round_trips(tmp_path):
    labels.append(
        tmp_path,
        setting="composition.min_sharpness",
        subject="abc",
        value=0.14,
        rejected=True,
        question="Too blurry to put in a memory?",
        library_size=19318,
        now=datetime(2026, 9, 12, 11, 24, tzinfo=UTC),
    )
    (one,) = _read(tmp_path)
    assert one.setting == "composition.min_sharpness"
    assert one.subject == "abc"
    assert one.value == pytest.approx(0.14)
    assert one.rejected is True
    assert one.question == "Too blurry to put in a memory?"
    assert one.library_size == 19318
    assert one.at.startswith("2026-09-12T11:24")
    assert one.rekindle


def test_two_answers_about_the_same_photograph_are_both_kept(tmp_path):
    """The point of the file.

    `Calibration.record` replaces the earlier answer, deliberately - someone
    walking back through a step is changing their mind, not adding evidence,
    and a threshold must not average both. The LOG keeps both, because
    "in September they said yes and in March they said no" is a fact about a
    person's standards and there is nowhere else it survives.
    """
    for rejected in (True, False):
        labels.append(
            tmp_path,
            setting="composition.min_sharpness",
            subject="same-photo",
            value=0.14,
            rejected=rejected,
        )
    got = _read(tmp_path)
    assert [x.rejected for x in got] == [True, False]


def test_a_reset_wipes_the_working_state_and_not_the_log(tmp_path):
    """`rekindle calibrate --redo` calls `reset_setting`, which is right and
    which is the exact moment a sitting's evidence used to disappear."""
    from rekindle.calibrate import state

    record = state.Calibration()
    for i in range(3):
        record.record(
            state.Answer(
                setting="composition.min_sharpness",
                value=0.1 + i / 100,
                rejected=False,
                subject=f"h{i}",
            )
        )
        labels.append(
            tmp_path,
            setting="composition.min_sharpness",
            subject=f"h{i}",
            value=0.1 + i / 100,
            rejected=False,
        )
    state.save(tmp_path, record)

    record.reset_setting("composition.min_sharpness")
    state.save(tmp_path, record)

    assert state.load(tmp_path).answers == []
    assert len(_read(tmp_path)) == 3


def test_a_malformed_line_is_skipped_and_the_rest_survive(tmp_path):
    """A reader that stopped at the first bad line would lose everything after
    it, which is the opposite of what an append-only record is for."""
    labels.append(tmp_path, setting="a", subject="1", value=1.0, rejected=False)
    path = labels.log_path(tmp_path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("not json at all\n")
        handle.write(json.dumps(["a", "list", "not", "an", "object"]) + "\n")
        handle.write("\n")
    labels.append(tmp_path, setting="c", subject="2", value=2.0, rejected=True)
    assert [x.setting for x in _read(tmp_path)] == ["a", "c"]


def test_a_line_from_a_future_version_is_skipped_rather_than_guessed_at(tmp_path):
    """Deliberately a SEPARATE test, with a line that is complete and valid in
    every way EXCEPT its version.

    The obvious version of this - a `{"v": 999}` line with no other fields -
    passes whether or not the version is checked, because the missing fields
    raise anyway. That test cannot fail, which is the shape this project keeps
    finding in its own work.
    """
    labels.append(tmp_path, setting="a", subject="1", value=1.0, rejected=False)
    future = {
        "v": labels.LABEL_VERSION + 1,
        "at": "2027-01-01T00:00:00+00:00",
        "setting": "b",
        "subject": "2",
        "value": 2.0,
        "rejected": True,
        "question": "?",
        "library_size": 1,
        "rekindle": "9.9.9",
    }
    with labels.log_path(tmp_path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(future) + "\n")
    assert [x.setting for x in _read(tmp_path)] == ["a"]


def test_no_log_means_no_labels_rather_than_an_error(tmp_path):
    assert _read(tmp_path) == []
    assert labels.summary(tmp_path) == {}


def test_the_summary_counts_and_names_nothing_else(tmp_path):
    """It is the one function here whose output is safe to print, so it must
    not carry a subject or a value out with it."""
    labels.append(tmp_path, setting="a", subject="secret-hash", value=1.0, rejected=False)
    labels.append(tmp_path, setting="a", subject="other", value=2.0, rejected=True)
    labels.append(tmp_path, setting="b", subject="third", value=3.0, rejected=True)
    summary = labels.summary(tmp_path)
    assert summary == {"a": 2, "b": 1}
    assert "secret-hash" not in repr(summary)


def test_the_session_writes_a_label_for_every_answer(tmp_path):
    """The wiring, at the one place a judgement is recorded. Without this the
    module is correct and unreachable."""
    from rekindle.calibrate import plan, session, state
    from rekindle.calibrate.sampling import Example

    step = plan.BY_SETTING["composition.min_sharpness"]
    sess = session.Session(
        tmp_path,
        [],
        availability=plan.Availability(fingerprints=True),
        record=state.Calibration(),
    )
    for i, said_yes in enumerate((True, False, True)):
        sess.answer(
            step, Example(value=0.1 + i / 100, subject=f"h{i}", paths=[Path("x.jpg")]), said_yes
        )

    got = _read(tmp_path)
    assert [x.rejected for x in got] == [True, False, True]
    assert {x.setting for x in got} == {"composition.min_sharpness"}
    # The QUESTION is recorded verbatim, not the step id: a reworded question
    # is a different question and a log that pooled them would be misleading.
    assert all(x.question == step.question for x in got)


def test_the_log_lives_in_the_data_directory_which_is_gitignored(tmp_path):
    """It holds file hashes of somebody's photographs and what they thought of
    them. `.gitignore` excludes `[Dd]ata/`; this pins that the file is put
    there rather than anywhere a `git add .` could reach."""
    labels.append(tmp_path, setting="a", subject="1", value=1.0, rejected=False)
    assert labels.log_path(tmp_path) == Path(tmp_path) / "labels.jsonl"
    assert labels.log_path(tmp_path).is_file()


# --------------------------------------------------------------------------
# the premise the decision log rests on
#
# `docs/decision-log-calibration-labels.md` says three things about the
# WORKING state, and the whole argument for a second file depends on all
# three being true: answers survive a sitting and are replayed, a reset
# deletes them, and a second judgement about the same photograph replaces the
# first. They were true when that was written. These pin them, so the log
# cannot quietly become wrong.


def _session(data_dir, photos, **kw):
    from rekindle.calibrate import plan, session

    return session.Session(
        data_dir, photos, availability=plan.Availability(fingerprints=True), **kw
    )


def _sharp_photos(n=30):
    from datetime import timedelta

    from rekindle.models import MediaType, Photo, PhotoMeta

    t0 = datetime(2025, 8, 4, 10, 0, tzinfo=UTC)
    return [
        Photo(
            file_hash=f"p{i:03d}",
            paths=[Path(f"/lib/p{i}.jpg")],
            media_type=MediaType.IMAGE,
            meta=PhotoMeta(
                taken_at_utc=t0 + timedelta(hours=i),
                taken_at_local=(t0 + timedelta(hours=i)).replace(tzinfo=None),
                sharpness=0.05 + i / 40.0,
                brightness=40.0 + i * 4,
            ),
            first_seen=t0,
            last_seen=t0,
        )
        for i in range(n)
    ]


def test_working_state_replays_a_sitting_rather_than_re_asking_it(tmp_path):
    """The correction the decision log makes to its own brief: the labels are
    NOT thrown away after bisecting. A fresh `Session` over the same data
    directory re-derives the identical threshold from the stored answers."""
    from rekindle.calibrate import plan, state
    from rekindle.calibrate.sampling import Example

    step = plan.BY_SETTING["composition.min_sharpness"]
    photos = _sharp_photos()

    first = _session(tmp_path, photos, record=state.Calibration())
    first.start()
    for i, said_yes in enumerate((True, True, False, False, True, False)):
        first.answer(step, Example(value=0.05 + i / 20.0, subject=f"p{i:03d}", paths=()), said_yes)
    before = first.derived(step)
    assert before.usable and before.judgements == 6

    assert len(state.load(tmp_path).answers) == 6, "the sitting did not reach disk"

    resumed = _session(tmp_path, photos)
    after = resumed.derived(step)
    assert after.judgements == before.judgements, "the answers were not replayed"
    assert after.value == before.value, "the threshold was not re-derived from them"


def test_a_reset_deletes_the_working_answers_which_is_why_the_log_exists(tmp_path):
    from rekindle.calibrate import plan, state
    from rekindle.calibrate.sampling import Example

    step = plan.BY_SETTING["composition.min_sharpness"]
    sess = _session(tmp_path, _sharp_photos(), record=state.Calibration())
    sess.answer(step, Example(value=0.1, subject="p001", paths=()), True)
    assert state.load(tmp_path).answers_for(step.setting)

    sess.reset(step)
    assert state.load(tmp_path).answers_for(step.setting) == []
    # ...and the log still has it. That asymmetry is the whole design.
    assert sum(labels.summary(tmp_path).values()) == 1


def test_a_second_judgement_replaces_the_first_in_working_state(tmp_path):
    from rekindle.calibrate import plan, state
    from rekindle.calibrate.sampling import Example

    step = plan.BY_SETTING["composition.min_sharpness"]
    sess = _session(tmp_path, _sharp_photos(), record=state.Calibration())
    sess.answer(step, Example(value=0.2, subject="same", paths=()), True)
    sess.answer(step, Example(value=0.2, subject="same", paths=()), False)

    kept = state.load(tmp_path).answers_for(step.setting)
    assert len(kept) == 1 and kept[0].rejected is False
    # ...and the log kept both, which is the point of it.
    assert [x.rejected for x in _read(tmp_path)] == [True, False]
