"""Calibration over HTTP.

Two things are being checked here, and only two, because everything else is
already covered by `test_calibrate.py` against the same session object:

1. The HTTP layer translates faithfully - and in particular does NOT leak the
   measured value of a photograph to the browser, which would stop a blind
   judgement being blind.
2. **The publishing gate's safeguard survives the trip.** A safeguard
   implemented once per front end is a safeguard that exists on one of them,
   so this file drives it through the API the page actually calls.
"""

from __future__ import annotations

import json

import pytest

from rekindle import config
from rekindle.calibrate import session as calsession
from rekindle.calibrate import state
from rekindle.web import api
from rekindle.web import calibrate_api as cal
from tests.fixtures.web import make_library, open_library

SHARPNESS = "composition.min_sharpness"
GATE = "faces.gate_threshold"


@pytest.fixture
def room(tmp_path):
    data_dir, _ = make_library(tmp_path)
    library = open_library(data_dir)
    return cal.CalibrationRoom(library, memories_root=tmp_path / "memories")


# ---------------------------------------------------------------- the status


def test_a_fresh_library_is_offered_calibration(room):
    got = cal.status(room)
    assert got["offer"]["show"] is True
    assert got["offer"]["reason"] == state.OFFER_FIRST_RUN
    assert got["total_steps"] == len(got["steps"])


def test_every_step_carries_what_the_page_has_to_show(room):
    for step in cal.status(room)["steps"]:
        for field in ("title", "question", "unit", "what", "raising", "lowering", "measured"):
            assert step[field], f"{step['setting']} is missing {field}"
        assert step["position"] >= 1
        assert step["state"] in ("todo", "done", "skipped")


def test_a_library_with_no_embeddings_says_why_rather_than_hiding_the_step(room):
    reasons = {u["needs"]: u["why"] for u in cal.status(room)["unavailable"]}
    assert "embeddings" in reasons
    assert "rekindle" in reasons["embeddings"], "the reason must name the command that fixes it"


def test_declining_forever_is_recorded_and_never_asked_again(room):
    cal.dismiss_offer(room)
    assert cal.status(room)["offer"]["show"] is False
    assert cal.status(room)["finished_at"]


# -------------------------------------------------------------- the examples


def test_an_example_carries_thumbnail_hashes_and_no_filesystem_path(room):
    """A path is a username and a drive layout. The page already has a thumb
    route keyed on the hash."""
    got = cal.example(room, setting=SHARPNESS)
    assert got["hashes"]
    blob = json.dumps(got)
    assert "/" not in blob.replace("/api", "") or "\\\\" not in blob
    assert ".jpg" not in blob


def test_the_measured_value_is_NEVER_sent_to_the_browser(room):
    """A blind judgement stops being blind the moment the page can render the
    number, and one debugging console.log is all that would take."""
    got = cal.example(room, setting=SHARPNESS)
    assert "value" not in got
    for key in got:
        assert "sharp" not in key


def test_answering_needs_an_example_to_have_been_handed_out_first(room):
    """The browser never sends back the value, so the server has to be the one
    holding the example. A POST out of order is a refusal, not a judgement
    placed at an arbitrary point on the scale."""
    with pytest.raises(api.ApiError):
        cal.answer(room, setting=SHARPNESS, said_yes=True)


def test_a_verdict_advances_to_the_next_example(room):
    first = cal.example(room, setting=SHARPNESS)
    second = cal.answer(room, setting=SHARPNESS, said_yes=True)
    assert second["answered"] == 1
    assert second["hashes"] != first["hashes"]


def test_the_confidence_line_comes_back_with_every_answer(room):
    cal.example(room, setting=SHARPNESS)
    for said_yes in (True, False, True, False):
        got = cal.answer(room, setting=SHARPNESS, said_yes=said_yes)
    assert got["confidence"]
    assert "judgement" in got["confidence"]


def test_an_unknown_setting_is_a_404_not_a_traceback(room):
    with pytest.raises(api.ApiError) as caught:
        cal.example(room, setting="composition.nonesuch")
    assert caught.value.status == 404


# ------------------------------------------------------- showing consequences


def test_the_consequence_of_a_number_is_available_before_it_is_written(room):
    got = cal.consequence(room, setting=SHARPNESS, value=0.9)
    assert got["countable"] is True
    assert got["sentence"]
    assert got["kept_then"] <= got["kept_now"]
    assert isinstance(got["newly_dropped"], list)


def test_a_setting_with_no_countable_consequence_says_so_rather_than_inventing_one(room):
    got = cal.consequence(room, setting="selection.max_shots", value=8)
    assert got["countable"] is False
    assert got["why"]


# ------------------------------------------------- THE SAFEGUARD, OVER HTTP


def test_tightening_the_publishing_gate_over_http_is_free(room):
    got = cal.choose(room, setting=GATE, value=0.05, accept=False)
    assert got["refused"] is None
    assert got["overrides"][GATE] == 0.05


def test_widening_the_publishing_gate_over_http_is_REFUSED(room):
    got = cal.choose(room, setting=GATE, value=0.40, accept=False)
    assert got["refused"] is not None
    assert "publishable" in got["refused"]["meaning"].lower()
    assert got["refused"]["phrase"] == calsession.LOOSEN_PHRASE
    assert GATE not in cal.status(room)["overrides"]


def test_a_wrong_sentence_does_not_clear_the_safeguard(room):
    cal.choose(room, setting=GATE, value=0.40, accept=False)
    for wrong in ("true", "yes", "1", "I understand", ""):
        assert cal.confirm(room, setting=GATE, phrase=wrong)["confirmed"] is False
    assert cal.choose(room, setting=GATE, value=0.40, accept=False)["refused"] is not None


def test_the_exact_sentence_clears_it_and_only_then(room):
    assert cal.choose(room, setting=GATE, value=0.40, accept=False)["refused"] is not None
    assert cal.confirm(room, setting=GATE, phrase=calsession.LOOSEN_PHRASE)["confirmed"] is True
    got = cal.choose(room, setting=GATE, value=0.40, accept=False)
    assert got["refused"] is None
    assert got["overrides"][GATE] == 0.40


def test_the_page_is_told_which_setting_is_dangerous_before_it_touches_it(room):
    """The slider must be able to warn BEFORE the drag, not only after the
    server refuses.

    Forced to a library WITH a face detector, because on one without it the
    gate step is not in the sequence at all - and a test that passed by the
    step being absent would be a test that could not fail.
    """
    from rekindle.calibrate import plan

    sess = room.session_for()
    sess.availability = plan.Availability(fingerprints=True, embeddings=True, faces=True)
    sess.steps = plan.available_steps(sess.availability)

    steps = cal.status(room)["steps"]
    assert GATE in {s["setting"] for s in steps}, "the gate step must be in the sequence"
    marked = {s["setting"]: s["loosening"] for s in steps if s["loosening"]}
    assert marked == {GATE: "raise"}


def test_a_setting_can_be_changed_even_when_its_step_is_not_in_the_sequence(room):
    """The gate has no step on a library with no detector, but a user with a
    `rekindle.toml` from another machine can still be holding a value for it.
    Refusing to REASON about it would be worse than refusing to ask about it,
    and the safeguard has to apply either way."""
    assert GATE not in {s["setting"] for s in cal.status(room)["steps"]}
    assert cal.choose(room, setting=GATE, value=0.40, accept=False)["refused"] is not None


# --------------------------------------------------------- writing, and out


def test_finishing_writes_only_what_changed_and_keeps_the_old_file(room, tmp_path):
    cal.choose(room, setting=SHARPNESS, value=0.2, accept=False)
    first = cal.finish(room)
    assert first["overrides"] == {SHARPNESS: 0.2}
    path = room.library.data_dir / config.CONFIG_NAME
    assert config.read_override(path) == {SHARPNESS: 0.2}

    room.reload()
    cal.choose(room, setting=SHARPNESS, value=0.3, accept=False)
    cal.finish(room)
    backup = path.with_suffix(path.suffix + ".bak")
    assert config.read_override(backup) == {SHARPNESS: 0.2}, "the previous value must survive"


def test_finishing_makes_the_new_configuration_active(room):
    """Otherwise the exit question would diff against the OLD numbers and
    report that nothing changed."""
    cal.choose(room, setting=SHARPNESS, value=0.25, accept=False)
    cal.finish(room)
    try:
        assert config.active().composition.min_sharpness == 0.25
    finally:
        config.activate(config.defaults())


def test_accepting_the_default_writes_nothing_for_that_setting(room):
    cal.choose(room, setting=SHARPNESS, value=None, accept=True)
    got = cal.status(room)
    assert SHARPNESS not in got["overrides"]
    assert SHARPNESS in got["done"]


def test_the_exit_question_comes_back_as_a_sentence_and_a_list(room):
    got = cal.affected(room)
    assert got["sentence"]
    assert isinstance(got["affected"], list)
    assert isinstance(got["uncheckable"], list)


def test_skipping_and_resetting_move_a_step_between_states(room):
    assert SHARPNESS not in cal.skip(room, setting=SHARPNESS)["overrides"]
    states = {s["setting"]: s["state"] for s in cal.status(room)["steps"]}
    assert states[SHARPNESS] == "skipped"
    states = {s["setting"]: s["state"] for s in cal.reset(room, setting=SHARPNESS)["steps"]}
    assert states[SHARPNESS] == "todo"


# ------------------------------------------------------------ the HTTP shell


def test_every_calibration_route_is_reachable_and_needs_the_token(tmp_path):
    """The routing table, checked against the handler rather than by reading
    it - a route added to `server.py` and misspelt looks exactly like a route
    that was never added."""
    import inspect

    from rekindle.web import server

    source = inspect.getsource(server.Handler._route)
    for path in (
        "/api/calibrate",
        "/api/calibrate/example",
        "/api/calibrate/consequence",
        "/api/calibrate/affected",
        "/api/calibrate/begin",
        "/api/calibrate/answer",
        "/api/calibrate/choose",
        "/api/calibrate/confirm",
        "/api/calibrate/skip",
        "/api/calibrate/reset",
        "/api/calibrate/finish",
        "/api/calibrate/not-now",
    ):
        assert f'"{path}"' in source, f"{path} is not routed"


def test_the_page_and_its_script_carry_the_calibration_flow():
    """The UI is one static HTML file and one script; a handler with no
    element to drive is a route nobody can reach."""
    from rekindle.web.server import ASSETS

    html = (ASSETS / "index.html").read_text(encoding="utf-8")
    js = (ASSETS / "app.js").read_text(encoding="utf-8")
    for element_id in (
        "calibrate-offer",
        "cal-progress",
        "cal-question",
        "cal-yes",
        "cal-no",
        "cal-range",
        "cal-refusal",
        "cal-refusal-input",
        "cal-affected",
    ):
        assert f'id="{element_id}"' in html, f"{element_id} is missing from the page"
        assert element_id in js, f"{element_id} is never driven by the script"


def test_the_script_never_asks_the_server_for_a_measured_value():
    """A blind judgement is blind on the wire, not by convention."""
    from rekindle.web.server import ASSETS

    js = (ASSETS / "app.js").read_text(encoding="utf-8")
    calibration = js[js.index("calibrate\n *") :]
    assert "payload.value" not in calibration
    assert "said_yes" in calibration, "the page sends a verdict, not a number"
