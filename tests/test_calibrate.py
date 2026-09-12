"""Calibration: the derivation, the sequence, the safeguard, the exit question.

Runs with no GPU, no network, no key, no model and no photograph, because that
is what CI has and because a first-run experience that can only be tested on
the author's machine is a first-run experience nobody else's machine is
protected from.

The tests that matter most here are the ones about being WRONG:

* a threshold derived from one dissenting answer must say so;
* a widening of the publishing gate must be refused;
* an exit question that reports "nothing changed" must be reporting on
  memories it actually re-ran.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rekindle import config
from rekindle.calibrate import impact, plan, preview, sampling, session, state
from rekindle.calibrate.judge import (
    MIN_JUDGEMENTS,
    REJECT_ABOVE,
    REJECT_BELOW,
    Judgement,
    derive,
    next_probe,
)
from rekindle.memory.spec import FactSheet, MemorySpec, Shot
from rekindle.models import MediaType, Photo, PhotoMeta

T0 = datetime(2025, 8, 4, 10, 0, tzinfo=UTC)


def _hist(dominant: int = 0) -> str:
    bins = [0] * 64
    bins[dominant % 64] = 255
    return "".join(f"{b:02x}" for b in bins)


def _photo(
    name: str,
    *,
    sharpness: float | None = 0.5,
    brightness: float | None = 120.0,
    phash: int | None = 0,
    at: float = 0.0,
    width: int = 4000,
    height: int = 3000,
    colour: str | None = None,
) -> Photo:
    return Photo(
        file_hash=name,
        paths=[Path(f"/lib/{name}.jpg")],
        media_type=MediaType.IMAGE,
        meta=PhotoMeta(
            taken_at_utc=T0 + timedelta(seconds=at),
            taken_at_local=(T0 + timedelta(seconds=at)).replace(tzinfo=None),
            sharpness=sharpness,
            brightness=brightness,
            phash=phash,
            width=width,
            height=height,
            colour=colour or _hist(0),
        ),
        first_seen=T0,
        last_seen=T0,
    )


def _js(values, direction=REJECT_BELOW, cut=0.5):
    """Judgements that separate cleanly at `cut`."""
    return [
        Judgement(v, (v < cut) if direction == REJECT_BELOW else (v > cut), f"h{i}")
        for i, v in enumerate(values)
    ]


# ---------------------------------------------------------------------------
# derivation


def test_a_clean_split_lands_between_the_two_classes():
    found = derive(_js([0.1, 0.2, 0.8, 0.9]), direction=REJECT_BELOW)
    assert found.usable
    assert 0.2 < found.value <= 0.8
    assert found.disagreements == 0


def test_the_derived_cut_actually_classifies_the_answers_it_was_given():
    """The only property that makes this a derivation rather than a guess."""
    js = _js([0.05, 0.1, 0.3, 0.55, 0.7, 0.95])
    found = derive(js, direction=REJECT_BELOW)
    for j in js:
        assert (j.value < found.value) == j.rejected, j


def test_a_ceiling_threshold_is_derived_the_other_way_round():
    js = [Judgement(v, v > 200.0, f"h{i}") for i, v in enumerate([100.0, 150.0, 220.0, 250.0])]
    found = derive(js, direction=REJECT_ABOVE)
    assert 150.0 <= found.value < 220.0
    for j in js:
        assert (j.value > found.value) == j.rejected


def test_too_few_answers_refuses_rather_than_inventing_a_number():
    found = derive(_js([0.1, 0.9]), direction=REJECT_BELOW)
    assert not found.usable
    assert str(MIN_JUDGEMENTS) in found.refusal


def test_all_one_way_refuses_and_says_which_way():
    up = derive(
        [Judgement(v, False, f"h{i}") for i, v in enumerate([0.1, 0.2, 0.3, 0.4])],
        direction=REJECT_BELOW,
    )
    assert not up.usable and "accepted every" in up.refusal
    down = derive(
        [Judgement(v, True, f"h{i}") for i, v in enumerate([0.1, 0.2, 0.3, 0.4])],
        direction=REJECT_BELOW,
    )
    assert not down.usable and "rejected every" in down.refusal


def test_a_contradiction_is_counted_and_reported_not_smoothed_away():
    js = [
        Judgement(0.1, True, "a"),
        Judgement(0.2, False, "b"),  # the contradiction
        Judgement(0.3, True, "c"),
        Judgement(0.8, False, "d"),
        Judgement(0.9, False, "e"),
    ]
    found = derive(js, direction=REJECT_BELOW)
    assert found.disagreements >= 1
    assert "disagrees" in found.confidence()
    low, high = found.ambiguous
    assert low < high, "the mixed band should be reported"


def test_a_threshold_resting_on_ONE_dissenting_answer_says_so():
    """Found by running this against a real library, where nine noes and one
    yes produced 'rests on ten judgements, all of which agree' - a true
    sentence that gives entirely the wrong impression."""
    js = [Judgement(0.05, True, "a")] + [
        Judgement(v, False, f"h{i}") for i, v in enumerate([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ]
    found = derive(js, direction=REJECT_BELOW)
    assert found.disagreements == 0
    assert found.thin
    assert "single photograph" in found.confidence()


def test_a_balanced_threshold_does_not_carry_the_warning():
    js = _js([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    found = derive(js, direction=REJECT_BELOW)
    assert not found.thin
    assert "single photograph" not in found.confidence()


@pytest.mark.parametrize(
    ("direction", "answers"),
    [
        # Overlapping classes, so more than one cut scores equally and the
        # tie-break is the thing actually being tested. Without the overlap
        # exactly one cut is best and the parameter does nothing - which is
        # how the first version of this test could not fail.
        (REJECT_BELOW, [(0.1, True), (0.4, False), (0.5, True), (0.9, False)]),
        (REJECT_ABOVE, [(0.1, False), (0.5, True), (0.6, False), (0.9, True)]),
    ],
)
def test_a_tie_breaks_towards_whichever_side_the_step_calls_safe(direction, answers):
    """Several cuts score the same. Which one is chosen is a safety decision,
    not an implementation detail, so it is pinned."""
    js = [Judgement(v, r, f"h{i}") for i, (v, r) in enumerate(answers)]
    loose = derive(js, direction=direction, safer="loose")
    tight = derive(js, direction=direction, safer="tight")
    assert loose.disagreements == tight.disagreements, "the setup must produce a genuine tie"
    assert loose.value != tight.value
    if direction == REJECT_BELOW:
        assert loose.value < tight.value, "loose must reject fewer"
    else:
        assert loose.value > tight.value, "loose must reject fewer"


def test_the_probe_bisects_the_interval_the_answers_left_open():
    js = [Judgement(0.2, True, "a"), Judgement(0.6, False, "b")]
    assert next_probe(js, direction=REJECT_BELOW, floor=0.0, ceiling=1.0) == pytest.approx(0.4)


def test_the_probe_starts_in_the_middle_of_the_band_not_at_the_default():
    """Starting at the shipped value and bisecting outward would make it the
    prior, and the point of the exercise is that it has met one library."""
    assert next_probe([], direction=REJECT_BELOW, floor=0.0, ceiling=1.0) == pytest.approx(0.5)


def test_contradictory_answers_reopen_the_whole_band_rather_than_dividing_by_zero():
    js = [Judgement(0.8, True, "a"), Judgement(0.2, False, "b")]
    got = next_probe(js, direction=REJECT_BELOW, floor=0.0, ceiling=1.0)
    assert 0.0 <= got <= 1.0


# ---------------------------------------------------------------------------
# sampling: the questions land at the margin, deterministically


def test_the_example_shown_is_the_one_nearest_the_target():
    photos = [_photo(f"p{i}", sharpness=v) for i, v in enumerate([0.1, 0.3, 0.5, 0.9])]
    pool = sampling.measured(photos, "sharpness")
    got = sampling.nearest(pool, 0.48)
    assert got is not None and got.value == 0.5


def test_a_photograph_is_never_shown_twice():
    photos = [_photo(f"p{i}", sharpness=v) for i, v in enumerate([0.1, 0.3, 0.5, 0.9])]
    pool = sampling.measured(photos, "sharpness")
    first = sampling.nearest(pool, 0.48)
    second = sampling.nearest(pool, 0.48, used={first.subject})
    assert second is not None and second.subject != first.subject


def test_two_runs_of_the_same_sitting_ask_about_the_same_photographs():
    """No random draws anywhere. This is what makes a sitting resumable."""
    photos = [_photo(f"p{i}", sharpness=0.5) for i in range(20)]
    pool = sampling.measured(photos, "sharpness")
    assert sampling.nearest(pool, 0.5).subject == sampling.nearest(pool, 0.5).subject


def test_photos_with_no_measurement_are_dropped_not_defaulted_to_zero():
    """A library that was never fingerprinted has no sharpness scores at all,
    and treating a missing one as zero would derive a threshold from nothing."""
    photos = [_photo("a", sharpness=0.5), _photo("b", sharpness=None)]
    assert [p.file_hash for _, p in sampling.measured(photos, "sharpness")] == ["a"]


def test_the_band_comes_from_the_users_own_distribution():
    values = [float(i) for i in range(101)]
    assert sampling.band(values, low_q=0.0, high_q=100.0) == (0.0, 100.0)
    assert sampling.band(values, low_q=25.0, high_q=75.0) == (25.0, 75.0)


def test_a_degenerate_distribution_still_produces_an_openable_band():
    """Every photo scoring the same. Bisection needs somewhere to go, and a
    zero-width band would ask the same question forever."""
    low, high = sampling.band([0.5] * 50)
    assert high > low


def test_only_consecutive_pairs_inside_the_time_gate_are_offered():
    """Dedup only ever compares a photo to the anchor of a run it is already
    time-adjacent to. Showing arbitrary pairs would ask the user a question
    the code never asks."""
    photos = [_photo("a", at=0), _photo("b", at=10), _photo("c", at=5000)]
    pairs = sampling.pairs_within(photos, gap_seconds=30.0, distance=lambda x, y: 1.0)
    assert [(p[1].file_hash, p[2].file_hash) for p in pairs] == [("a", "b")]


# ---------------------------------------------------------------------------
# the sequence


def test_every_step_names_a_setting_that_actually_exists():
    for step in plan.STEPS:
        assert step.setting in config.CATALOGUE, step.setting


def test_every_step_offers_at_least_one_mode_and_a_question_in_plain_words():
    for step in plan.STEPS:
        assert step.modes
        assert step.question.endswith("?"), step.setting
        assert "threshold" not in step.question.lower()
        assert "hash" not in step.question.lower()


def test_every_blind_step_can_actually_build_a_pool():
    """A blind step with no way to measure anything would show the user
    nothing and derive a number from an empty list."""
    for step in plan.STEPS:
        if plan.MODE_BLIND not in step.modes:
            continue
        assert step.attribute or step.setting in {
            "dedup.phash_distance",
            "dedup.cosine",
            "faces.gate_threshold",
        }, f"{step.setting} is blind but nothing builds it a pool"


def test_every_deleting_step_offers_the_mode_that_shows_the_deletion_count():
    """The slider is the only mode that shows what a dedup threshold costs
    ACROSS a range. Four blind judgements on the reference library derived a
    `phash_distance` of 21; the ladder is what shows that 21 collapses 3,264
    more photographs than 6 does, and it is the only thing that would."""
    for step in plan.STEPS:
        if step.setting in preview.DEDUP:
            assert plan.MODE_SLIDER in step.modes, step.setting


def test_the_only_steps_that_break_a_tie_towards_rejecting_are_the_two_that_should():
    """`faces.gate_threshold` because saying yes is the safe answer there, and
    `dedup.gap_seconds` because a longer gap compares more pairs and therefore
    deletes more photographs. Every other step keeps more."""
    assert {s.setting for s in plan.STEPS if s.safer == "tight"} == {
        "faces.gate_threshold",
        "dedup.gap_seconds",
    }


def test_a_library_with_nothing_measured_gets_a_shorter_sequence_and_a_reason():
    bare = session.detect([_photo("a", sharpness=None, phash=None)])
    assert not bare.fingerprints
    steps = plan.available_steps(bare)
    assert steps, "there is still something to ask about"
    assert all(s.needs != plan.NEEDS_FINGERPRINTS for s in steps)
    assert "rekindle fingerprint" in bare.reasons[plan.NEEDS_FINGERPRINTS]


def test_progress_counts_only_the_steps_this_library_can_be_asked():
    """'4 of 9' has to be true. Counting steps that will never be shown is
    how a progress bar becomes a lie."""
    bare = plan.available_steps(session.detect([_photo("a", sharpness=None, phash=None)]))
    full = plan.available_steps(plan.Availability(fingerprints=True, embeddings=True, faces=True))
    assert len(bare) < len(full)
    assert plan.position(bare, bare[0].setting) == (1, len(bare))


# ---------------------------------------------------------------------------
# state: finishing is real, resuming is first class


def test_a_library_with_no_calibration_is_offered_one(tmp_path):
    got = state.should_offer(state.Calibration(), library_size=19480)
    assert got.show and got.reason == state.OFFER_FIRST_RUN
    assert "19,480" in got.headline


def test_a_finished_calibration_is_never_offered_again(tmp_path):
    done = state.Calibration(finished_at="2026-09-12T00:00:00+00:00", library_size=19480)
    got = state.should_offer(done, library_size=19480)
    assert not got.show
    assert got.reason == state.OFFER_DONE


def test_a_part_finished_sitting_offers_to_resume_and_says_how_far(tmp_path):
    partial = state.Calibration(started_at="2026-09-12T00:00:00+00:00", completed=["a", "b"])
    got = state.should_offer(partial, library_size=100)
    assert got.show and got.reason == state.OFFER_RESUME
    assert "2 steps done" in got.headline


def test_a_much_larger_library_is_a_NOTE_not_a_prompt(tmp_path):
    """It never nags. A grown library earns a sentence, not an interruption."""
    done = state.Calibration(finished_at="2026-09-12T00:00:00+00:00", library_size=19480)
    got = state.should_offer(done, library_size=24000)
    assert not got.show, "a grown library must not reopen the prompt"
    assert got.reason == state.OFFER_GREW
    assert "19,480" in got.headline and "24,000" in got.headline


def test_a_slightly_larger_library_says_nothing_at_all():
    done = state.Calibration(finished_at="2026-09-12T00:00:00+00:00", library_size=19480)
    assert state.should_offer(done, library_size=20000).reason == state.OFFER_DONE


def test_an_empty_library_is_never_offered_calibration():
    assert not state.should_offer(state.Calibration(), library_size=0).show


def test_the_state_survives_a_round_trip(tmp_path):
    record = state.Calibration()
    record.start(library_size=100)
    record.record(state.Answer("composition.min_sharpness", 0.2, True, "abc"))
    record.complete("composition.min_sharpness")
    state.save(tmp_path, record)
    back = state.load(tmp_path)
    assert back.library_size == 100
    assert back.completed == ["composition.min_sharpness"]
    assert back.answers[0].subject == "abc"


def test_answering_the_same_photograph_twice_replaces_rather_than_doubles():
    """On re-entry someone must be able to change their mind about a
    photograph, not have both answers averaged into a threshold."""
    record = state.Calibration()
    record.record(state.Answer("s", 0.2, True, "abc"))
    record.record(state.Answer("s", 0.2, False, "abc"))
    assert len(record.answers_for("s")) == 1
    assert record.answers_for("s")[0].rejected is False


def test_an_unreadable_state_file_means_never_calibrated_not_a_crash(tmp_path):
    """Unlike rekindle.toml, this one is NOT fatal: a malformed working file
    costs one repeated sitting, and refusing to start would be worse."""
    (tmp_path / state.STATE_NAME).write_text("{not json", encoding="utf-8")
    assert state.load(tmp_path).finished_at is None


def test_a_state_file_from_a_future_version_is_ignored_rather_than_misread(tmp_path):
    (tmp_path / state.STATE_NAME).write_text(
        json.dumps({"version": 999, "finished_at": "x"}), encoding="utf-8"
    )
    assert not state.load(tmp_path).finished


def test_the_state_file_is_written_whole_so_a_crash_costs_one_answer(tmp_path):
    record = state.Calibration()
    record.record(state.Answer("s", 0.2, True, "a"))
    state.save(tmp_path, record)
    assert not list(tmp_path.glob("*.tmp")), "the temporary file must not survive"
    json.loads((tmp_path / state.STATE_NAME).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# the session


def _session(tmp_path, photos=None, **kw):
    photos = photos or [
        _photo(f"p{i:02d}", sharpness=i / 20.0, at=i * 3600, phash=i << 8) for i in range(20)
    ]
    return session.Session(
        tmp_path,
        photos,
        availability=plan.Availability(fingerprints=True),
        **kw,
    )


def test_a_sitting_resumes_exactly_where_it_stopped(tmp_path):
    sess = _session(tmp_path)
    step = plan.BY_SETTING["composition.min_sharpness"]
    for said_yes in (True, False, True, False):
        example = sess.next_example(step)
        sess.answer(step, example, said_yes)
    before = [(j.value, j.rejected) for j in sess.judgements(step)]

    resumed = _session(tmp_path)
    after = [(j.value, j.rejected) for j in resumed.judgements(step)]
    assert after == before
    assert resumed.derived(step).value == sess.derived(step).value


def test_the_answered_photographs_are_not_asked_about_again_after_a_resume(tmp_path):
    sess = _session(tmp_path)
    step = plan.BY_SETTING["composition.min_sharpness"]
    seen = set()
    for _ in range(3):
        example = sess.next_example(step)
        seen.add(example.subject)
        sess.answer(step, example, True)
    resumed = _session(tmp_path)
    assert resumed.next_example(step).subject not in seen


def test_accepting_the_default_records_the_step_and_writes_nothing(tmp_path):
    """'I looked and rekindle's number is right' is a real answer, and it must
    leave the value ABSENT from rekindle.toml so a better default in a later
    release still reaches this user."""
    sess = _session(tmp_path)
    step = plan.BY_SETTING["composition.min_sharpness"]
    sess.accept_default(step)
    assert step.setting in sess.record.completed
    assert step.setting not in sess.overrides()


def test_only_what_changed_is_written(tmp_path):
    sess = _session(tmp_path)
    sess.propose(plan.BY_SETTING["composition.min_sharpness"], 0.2)
    assert sess.overrides() == {"composition.min_sharpness": 0.2}


def test_an_out_of_range_proposal_is_refused_by_the_same_rule_the_file_uses(tmp_path):
    sess = _session(tmp_path)
    with pytest.raises(config.ConfigError):
        sess.propose(plan.BY_SETTING["composition.min_sharpness"], 99.0)


def test_an_integer_setting_gets_an_integer_even_from_a_midpoint_cut(tmp_path):
    """A derived cut lands on 6.5, and writing that into an int setting is a
    type error three modules away."""
    photos = [_photo(f"p{i}", at=i * 5, phash=(1 << i) - 1) for i in range(12)]
    sess = session.Session(tmp_path, photos, availability=plan.Availability(fingerprints=True))
    # 8.5, not 6.5: flooring 6.5 gives back the shipped 6, which correctly
    # writes nothing at all and would make this test pass for the wrong reason.
    sess.propose(plan.BY_SETTING["dedup.phash_distance"], 8.5)
    value = sess.overrides()["dedup.phash_distance"]
    assert isinstance(value, int) and value == 8


def test_resetting_a_step_forgets_its_answers_and_its_value(tmp_path):
    sess = _session(tmp_path)
    step = plan.BY_SETTING["composition.min_sharpness"]
    sess.answer(step, sess.next_example(step), True)
    sess.propose(step, 0.3)
    sess.reset(step)
    assert not sess.judgements(step)
    assert step.setting not in sess.overrides()
    assert step.setting not in sess.done


def test_finishing_records_when_and_against_how_many(tmp_path):
    sess = _session(tmp_path)
    sess.start()
    sess.finish()
    back = state.load(tmp_path)
    assert back.finished
    assert back.library_size == 20
    assert not back.in_progress


# ---------------------------------------------------------------------------
# THE FACE GATE SAFEGUARD
#
# This is the part that decides what may reach a public repository.


def test_tightening_the_publishing_gate_is_free(tmp_path):
    """Nobody should be asked to confirm making their own gate stricter."""
    sess = session.Session(
        tmp_path, [_photo("a")], availability=plan.Availability(fingerprints=True, faces=True)
    )
    step = plan.BY_SETTING["faces.gate_threshold"]
    assert sess.propose(step, 0.05) is None
    assert sess.overrides()["faces.gate_threshold"] == 0.05


def test_widening_the_publishing_gate_is_REFUSED(tmp_path):
    sess = session.Session(
        tmp_path, [_photo("a")], availability=plan.Availability(fingerprints=True, faces=True)
    )
    step = plan.BY_SETTING["faces.gate_threshold"]
    refusal = sess.propose(step, 0.40)
    assert refusal is not None
    assert "faces.gate_threshold" not in sess.overrides(), "it must not have been staged"
    assert config.active().faces.gate_threshold == 0.15


def test_the_refusal_says_in_plain_words_what_widening_means(tmp_path):
    sess = session.Session(
        tmp_path, [_photo("a")], availability=plan.Availability(fingerprints=True, faces=True)
    )
    refusal = sess.propose(plan.BY_SETTING["faces.gate_threshold"], 0.40)
    text = refusal.meaning.lower()
    assert "publishable" in text
    assert "other people" in text
    assert "review queue" in text
    assert "threshold" not in text, "the explanation must not be jargon"


def test_a_boolean_cannot_clear_the_safeguard_only_the_exact_sentence_can(tmp_path):
    """A boolean is what a slider sends. A drag-and-release must not be able
    to widen this gate, so the confirmation is a sentence that has to be
    echoed back."""
    sess = session.Session(
        tmp_path, [_photo("a")], availability=plan.Availability(fingerprints=True, faces=True)
    )
    step = plan.BY_SETTING["faces.gate_threshold"]
    for wrong in ("true", "yes", "y", "1", "ok", "I understand", ""):
        assert sess.confirm_loosening(step, wrong) is False
    assert sess.propose(step, 0.40) is not None

    assert sess.confirm_loosening(step, session.LOOSEN_PHRASE) is True
    assert sess.propose(step, 0.40) is None
    assert sess.overrides()["faces.gate_threshold"] == 0.40


def test_confirming_one_setting_does_not_unlock_another(tmp_path):
    """The confirmation is per setting. A blanket unlock would mean widening
    the gate once permits widening anything else that gets marked later."""
    sess = session.Session(
        tmp_path, [_photo("a")], availability=plan.Availability(fingerprints=True, faces=True)
    )
    gate = plan.BY_SETTING["faces.gate_threshold"]
    sess.confirm_loosening(gate, session.LOOSEN_PHRASE)
    assert sess._confirmed_loosening == {"faces.gate_threshold"}


def test_the_confirmation_does_not_survive_into_a_new_sitting(tmp_path):
    """It is cleared with the process. A confirmation given in September must
    not silently authorise a widening in November."""
    sess = session.Session(
        tmp_path, [_photo("a")], availability=plan.Availability(fingerprints=True, faces=True)
    )
    step = plan.BY_SETTING["faces.gate_threshold"]
    sess.confirm_loosening(step, session.LOOSEN_PHRASE)
    sess.propose(step, 0.40)

    fresh = session.Session(
        tmp_path, [_photo("a")], availability=plan.Availability(fingerprints=True, faces=True)
    )
    assert fresh.propose(step, 0.45) is not None, "the confirmation must not persist"


def test_no_other_setting_is_ever_refused(tmp_path):
    """A confirmation dialog on a value with no consequence off this machine
    trains the user to click through the one that matters."""
    sess = session.Session(
        tmp_path,
        [_photo("a")],
        availability=plan.Availability(fingerprints=True, embeddings=True, faces=True),
    )
    for step in plan.STEPS:
        if step.setting == "faces.gate_threshold":
            continue
        setting = config.CATALOGUE[step.setting]
        lo = setting.minimum if setting.minimum is not None else setting.default * 0.5
        hi = setting.maximum if setting.maximum is not None else setting.default * 2
        for value in (max(lo, setting.default * 0.5), min(hi, setting.default * 1.5)):
            assert sess.propose(step, value) is None, step.setting


# ---------------------------------------------------------------------------
# showing the consequence before writing


def test_the_consequence_counts_what_the_real_gate_would_do(tmp_path):
    photos = [_photo(f"p{i}", sharpness=i / 100.0) for i in range(100)]
    got = preview.consequence(photos, "composition.min_sharpness", 0.30)
    assert got.kept_now > got.kept_then, "tightening must keep fewer"
    assert got.dropped == got.kept_now - got.kept_then
    assert got.newly_dropped, "it must be able to show which ones"


def test_loosening_reports_what_it_admits():
    photos = [_photo(f"p{i}", sharpness=i / 100.0) for i in range(100)]
    got = preview.consequence(photos, "composition.min_sharpness", 0.05)
    assert got.gained > 0
    assert "more photograph" in got.sentence()


def test_a_change_that_does_nothing_says_so_rather_than_implying_it_did():
    photos = [_photo(f"p{i}", sharpness=0.9) for i in range(50)]
    got = preview.consequence(photos, "composition.min_sharpness", 0.30)
    assert got.unchanged
    assert "changes nothing at all" in got.sentence()


def test_the_preview_leaves_the_active_configuration_alone():
    before = config.active()
    preview.consequence([_photo("a")], "composition.min_sharpness", 0.9)
    assert config.active() is before


def test_only_the_settings_with_a_real_number_claim_a_countable_consequence():
    """`lambda_penalty` reorders a pick and has no number of its own.
    Inventing one would be worse than having none."""
    assert preview.countable("composition.min_sharpness")
    assert preview.countable("dedup.phash_distance")
    assert not preview.countable("diversity.lambda_penalty")
    assert not preview.countable("selection.max_shots")


# A dedup threshold DELETES photographs, and the preview has to say so.
#
# This exists because of what happened when the flow was run against the real
# reference library: four honest blind judgements derived a `phash_distance`
# of 21, which collapses 5,341 photographs where the shipped 6 collapses
# 2,077. Nothing in the flow would have shown that before writing it.


def _burst(n: int, *, spread: int) -> list[Photo]:
    """`n` photos two seconds apart, `spread` bits apart in dHash."""
    return [_photo(f"b{i}", at=i * 2, phash=(1 << (i * spread)) - 1) for i in range(n)]


def test_a_dedup_threshold_reports_how_many_photographs_it_deletes():
    photos = _burst(8, spread=3)
    got = preview.consequence(photos, "dedup.phash_distance", 40)
    assert got.deletes
    assert got.collapsed_then > got.collapsed_now
    assert "MORE photographs away as duplicates" in got.sentence()
    assert f"{got.collapsed_then - got.collapsed_now:,}" in got.sentence()


def test_loosening_a_dedup_threshold_says_what_comes_back():
    photos = _burst(8, spread=3)
    with config.using({"dedup.phash_distance": 40}):
        got = preview.consequence(photos, "dedup.phash_distance", 1)
    assert "keeps" in got.sentence()
    assert got.collapsed_then < got.collapsed_now


def test_a_dedup_threshold_that_changes_nothing_says_so():
    photos = [_photo(f"p{i}", at=i * 100000, phash=i << 20) for i in range(10)]
    got = preview.consequence(photos, "dedup.phash_distance", 20)
    assert "exactly the same photographs" in got.sentence()


def test_the_deletion_sentence_never_reads_like_a_filter():
    """ "Keeps 3,264 fewer" is a number someone glances at. "Collapses 3,264
    MORE photographs away" is a number that stops them."""
    photos = _burst(8, spread=3)
    sentence = preview.consequence(photos, "dedup.phash_distance", 40).sentence()
    assert "fewer" not in sentence
    assert "collapses" in sentence.lower()


def test_the_dedup_preview_runs_the_real_collapse():
    """Not a reimplementation of it. The two would drift the first time
    anybody touched either."""
    from rekindle.memory.dedup import collapse

    photos = _burst(8, spread=3)
    with config.using({"dedup.phash_distance": 40}):
        kept, _ = collapse(list(photos))
    got = preview.consequence(photos, "dedup.phash_distance", 40)
    assert got.kept_then == len(kept)


# ---------------------------------------------------------------------------
# THE EXIT QUESTION


def _write_spec(root: Path, recipe: str, key: str, hashes: list[str]) -> Path:
    folder = root / f"{recipe}-{key}"
    folder.mkdir(parents=True, exist_ok=True)
    spec = MemorySpec(
        recipe=recipe,
        key=key,
        title=key,
        subtitle="",
        public_safe=False,
        shots=tuple(Shot(h, "", None, False) for h in hashes),
        facts=FactSheet(title=key, recipe=recipe, photo_count=len(hashes)),
    )
    (folder / impact.SPEC_NAME).write_text(spec.dumps(), encoding="utf-8")
    return folder


def _library(tmp_path, photos):
    from rekindle.db import PhotoStore
    from rekindle.memory.index import MemoryIndex

    store = PhotoStore(tmp_path / "db.sqlite")
    store.upsert_many(photos)
    return store, MemoryIndex.open(store)


def _album_photos(n=12, album="Kashmir"):
    out = []
    for i in range(n):
        p = _photo(f"p{i:02d}", at=i * 3600, phash=i << 9, colour=_hist(i))
        out.append(
            Photo(
                file_hash=p.file_hash,
                paths=p.paths,
                media_type=p.media_type,
                meta=p.meta,
                first_seen=p.first_seen,
                last_seen=p.last_seen,
                albums=[album],
            )
        )
    return out


def test_no_memories_on_disk_says_so_rather_than_reporting_zero_changes(tmp_path):
    store, index = _library(tmp_path, _album_photos())
    got = impact.analyse(index, tmp_path / "nothing-here")
    assert got.total == 0
    assert "no built memories" in got.sentence()
    store.close()


def test_a_change_that_affects_nothing_is_reported_as_affecting_nothing(tmp_path):
    """The honest test of whether a threshold change mattered at all, and the
    kind of thing this project has been wrong about before."""
    photos = _album_photos()
    store, index = _library(tmp_path, photos)
    root = tmp_path / "memories"
    from rekindle.memory import engine
    from rekindle.memory.recipes.base import Offer

    built = engine.build(index, Offer(recipe="album_story", key="Kashmir", title="Kashmir"))
    assert built is not None
    _write_spec(root, "album_story", "Kashmir", [s.file_hash for s in built.shots])

    got = impact.analyse(index, root)
    assert got.total == 1
    assert got.affected == []
    assert "affect none" in got.sentence()
    store.close()


def test_a_change_that_removes_photographs_is_counted_and_named(tmp_path):
    photos = _album_photos()
    store, index = _library(tmp_path, photos)
    root = tmp_path / "memories"
    from rekindle.memory import engine
    from rekindle.memory.recipes.base import Offer

    built = engine.build(index, Offer(recipe="album_story", key="Kashmir", title="Kashmir"))
    _write_spec(root, "album_story", "Kashmir", [s.file_hash for s in built.shots])

    got = impact.analyse(
        index, root, settings=config.active().with_values({"selection.max_shots": 4})
    )
    assert len(got.affected) == 1
    change = got.affected[0]
    assert change.verdict == impact.LOST
    assert change.after == 4 and change.before == len(built.shots)
    assert "1 of 1" in got.sentence()
    store.close()


def test_a_memory_that_would_no_longer_be_built_at_all_is_reported_as_such(tmp_path):
    photos = _album_photos()
    store, index = _library(tmp_path, photos)
    root = tmp_path / "memories"
    _write_spec(root, "album_story", "Kashmir", [p.file_hash for p in photos])

    got = impact.analyse(
        index, root, settings=config.active().with_values({"selection.min_shots": 100})
    )
    assert got.affected[0].verdict == impact.GONE
    assert "no longer be built" in got.sentence()
    store.close()


def test_a_memory_whose_recipe_cannot_be_re_run_is_UNCHECKABLE_not_unchanged(tmp_path):
    """Reporting 'no change' for something that was never re-run is exactly
    the silent lie this project keeps finding in its own tests."""
    store, index = _library(tmp_path, _album_photos())
    root = tmp_path / "memories"
    _write_spec(root, "prompt", "durga puja over the years", ["a", "b", "c"])

    got = impact.analyse(index, root)
    assert len(got.uncheckable) == 1
    assert got.uncheckable[0].verdict == impact.UNCHECKABLE
    assert "could not be re-checked" in got.sentence()
    assert "not counted as unchanged" in got.sentence()
    store.close()


def test_an_uncheckable_memory_is_excluded_from_the_denominator(tmp_path):
    """'affect 0 of 2' when one of them was never examined is a false claim
    about the one that was not."""
    photos = _album_photos()
    store, index = _library(tmp_path, photos)
    root = tmp_path / "memories"
    from rekindle.memory import engine
    from rekindle.memory.recipes.base import Offer

    built = engine.build(index, Offer(recipe="album_story", key="Kashmir", title="Kashmir"))
    _write_spec(root, "album_story", "Kashmir", [s.file_hash for s in built.shots])
    _write_spec(root, "prompt", "anything", ["x"])

    got = impact.analyse(index, root)
    assert got.total == 2
    assert "none of your 1 memory" in got.sentence(), "the uncheckable one must not be counted"
    assert "1 more could not be re-checked" in got.sentence()
    store.close()


def test_the_exit_question_never_renders_anything(tmp_path, monkeypatch):
    """Selection is cheap; rendering is expensive. If this ever starts
    decoding frames, the whole feature is the three-hour rebuild it exists to
    avoid."""
    photos = _album_photos()
    store, index = _library(tmp_path, photos)
    root = tmp_path / "memories"
    _write_spec(root, "album_story", "Kashmir", [p.file_hash for p in photos[:6]])

    from PIL import Image

    def explode(*a, **k):  # pragma: no cover - the point is that it is not called
        raise AssertionError("the exit question decoded an image")

    monkeypatch.setattr(Image, "open", explode)
    impact.analyse(index, root)
    store.close()


def test_the_memories_are_listed_in_a_stable_order(tmp_path):
    store, index = _library(tmp_path, _album_photos())
    root = tmp_path / "memories"
    for key in ("Zanzibar", "Kashmir", "Ladakh"):
        _write_spec(root, "album_story", key, ["a"])
    first = [c.memory_id for c in impact.analyse(index, root).changes]
    second = [c.memory_id for c in impact.analyse(index, root).changes]
    assert first == second == sorted(first)
    store.close()


def test_the_biggest_change_is_offered_for_rebuild_first(tmp_path):
    """If someone stops the rebuild half way, the memories that moved most
    should be the ones already redone."""
    small = impact.Change("a", "a", Path("a"), impact.LOST, before=10, after=9, lost=1)
    large = impact.Change("b", "b", Path("b"), impact.LOST, before=10, after=2, lost=8)
    got = impact.Impact(changes=[small, large])
    assert [c.memory_id for c in impact.rebuildable(got)] == ["b", "a"]


def test_the_sentence_reads_like_the_brief_asked_for_it():
    changes = (
        [impact.Change(f"g{i}", "", Path(), impact.GAINED, gained=1) for i in range(11)]
        + [impact.Change(f"l{i}", "", Path(), impact.LOST, lost=1) for i in range(3)]
        + [impact.Change(f"s{i}", "", Path(), impact.SAME) for i in range(175)]
    )
    got = impact.Impact(changes=changes).sentence()
    assert got.startswith("Your changes affect 14 of 189 memories.")
    assert "11 gain photos" in got
    assert "3 lose photos" in got


def test_an_unreadable_spec_is_reported_rather_than_skipped(tmp_path):
    store, index = _library(tmp_path, _album_photos())
    root = tmp_path / "memories"
    folder = root / "broken"
    folder.mkdir(parents=True)
    (folder / impact.SPEC_NAME).write_text("{ not json", encoding="utf-8")
    got = impact.analyse(index, root)
    assert got.uncheckable and "could not be read" in got.uncheckable[0].note
    store.close()


def test_analysing_leaves_the_active_configuration_alone(tmp_path):
    store, index = _library(tmp_path, _album_photos())
    root = tmp_path / "memories"
    _write_spec(root, "album_story", "Kashmir", ["a"])
    before = config.active()
    impact.analyse(index, root, settings=before.with_values({"selection.max_shots": 3}))
    assert config.active() is before
    store.close()
