"""What the render tells the page while it is happening.

A render is a blocking POST that can run past a minute, so the browser learns
nothing from the response it is waiting on; it polls this instead. The whole
point of the endpoint is that the number it reports is REAL - counted off the
frame loops - so these tests are about the number, not about the endpoint
existing.

`no_mp4=True` throughout except where the MP4 path is the subject: the full
path is 80 seconds a call on the reference machine, and a test suite nobody
runs catches nothing.
"""

from __future__ import annotations

import pytest

from rekindle.web import api, renderer
from tests.fixtures.web import ALBUM, make_library, make_workshop


def built(tmp_path, data_dir):
    workshop = make_workshop(tmp_path, data_dir)
    events = list(api.build_events(workshop, recipe="album_story", key=ALBUM))
    assert events[-1]["event"] == "ready", events[-1]
    return workshop, events[-1]["data"]["session_id"]


@pytest.fixture
def session(tmp_path):
    data_dir, _ = make_library(tmp_path)
    return built(tmp_path, data_dir)


def record(workshop, monkeypatch) -> list[dict]:
    """Every payload the render publishes, in order.

    Recorded at `Workshop.set_progress` rather than by polling
    `render_progress` on a thread. A poller samples, and on a ten-photograph
    fixture the whole render is over in under a second - the first version of
    this file polled every 10ms and missed both the first phase and the last,
    which made three tests fail for a reason that had nothing to do with the
    code they were about. The wrapper sees every write, and the tests that are
    really about the ENDPOINT read it back through `render_progress` below.
    """
    seen: list[dict] = []
    original = workshop.set_progress

    def spy(session_id: str, payload: dict) -> None:
        seen.append(dict(payload))
        original(session_id, payload)

    monkeypatch.setattr(workshop, "set_progress", spy)
    return seen


# --------------------------------------------------------------- the shape


def test_a_session_that_has_never_rendered_is_not_an_error(session):
    """The page polls as it fires the POST, so the first poll can arrive
    before the render has written anything."""
    workshop, session_id = session
    payload = api.render_progress(workshop, session_id)
    assert payload["running"] is False
    assert payload["fraction"] == 0.0
    assert payload["phase"] == ""


def test_progress_for_a_session_that_is_gone_is_a_404(session):
    workshop, _session_id = session
    with pytest.raises(api.ApiError) as caught:
        api.render_progress(workshop, "no-such-session")
    assert caught.value.status == 404


# ------------------------------------------------------- the number is real


def test_a_render_reports_every_phase_it_runs_in_order(session, monkeypatch):
    workshop, session_id = session
    samples = record(workshop, monkeypatch)
    api.render(workshop, session_id, {"no_mp4": True})

    phases = [s["phase"] for s in samples if s["phase"]]
    seen = list(dict.fromkeys(phases))
    assert seen[0] == renderer.PHASE_SPEC
    assert seen[-1] == renderer.PHASE_DONE
    for phase in (renderer.PHASE_PREVIEW, renderer.PHASE_WEBP, renderer.PHASE_GIF):
        assert phase in seen, f"{phase} never reported; saw {seen}"
    # Skipping the MP4 must not report the phases that did not run.
    for phase in (renderer.PHASE_VIDEO, renderer.PHASE_STILLS, renderer.PHASE_MP4):
        assert phase not in seen, f"{phase} was reported but the MP4 was skipped"


def test_the_fraction_never_goes_backwards_and_ends_at_one(session, monkeypatch):
    """A bar that jumps back is worse than no bar: it says the thing you were
    told was nearly done was not."""
    workshop, session_id = session
    samples = record(workshop, monkeypatch)
    api.render(workshop, session_id, {"no_mp4": True})

    fractions = [s["fraction"] for s in samples if s["phase"]]
    assert fractions, "nothing was sampled; the watcher is not seeing the render"
    assert fractions == sorted(fractions), f"the bar went backwards: {fractions}"
    assert fractions[-1] == 1.0
    assert api.render_progress(workshop, session_id)["running"] is False


def test_the_preview_phase_counts_the_shots_it_is_decoding(session, monkeypatch):
    """`total` is what makes "9 of 16" possible, and it is the difference
    between a real count and a phase name."""
    workshop, session_id = session
    samples = record(workshop, monkeypatch)
    api.render(workshop, session_id, {"no_mp4": True})

    counted = [s for s in samples if s["phase"] == renderer.PHASE_PREVIEW and s["total"]]
    assert counted, "the preview pass reported no count at all"
    assert max(s["done"] for s in counted) == counted[0]["total"]
    shots = len(workshop.get(session_id).draft.order)
    assert counted[0]["total"] == min(shots, api.RenderOptions().preview_frames)


def test_skipping_the_mp4_still_reaches_a_full_bar(session):
    """The weights are shares of the WHOLE render, and the MP4 is nearly half
    of it. Without renormalising over the phases that will actually run, a
    preview-only render stops the bar at 22%."""
    workshop, session_id = session
    before_last = api._phase_fraction(renderer.PHASE_GIF, 0, 0, no_mp4=True)
    assert before_last < 1.0
    assert api._phase_fraction(renderer.PHASE_DONE, 0, 0, no_mp4=True) == 1.0
    # The last phase that runs must start near, not at, the end.
    assert 0.4 < before_last < 1.0, before_last


def test_the_full_path_reports_the_video_phases(session, monkeypatch):
    """The MP4 phases are 78% of a full render between them, so a bar that
    does not know about them holds at 22% for the whole of it."""
    workshop, session_id = session
    samples = record(workshop, monkeypatch)
    api.render(workshop, session_id, {"no_mp4": False})

    seen = {s["phase"] for s in samples}
    assert renderer.PHASE_VIDEO in seen
    if workshop.get(session_id).last_render.mp4 is not None:
        # Only when ffmpeg is actually on this machine; CI has no encoder and
        # `write_mp4` returns "skipped" before it writes a single still.
        assert renderer.PHASE_STILLS in seen
        assert renderer.PHASE_MP4 in seen


def test_the_endpoint_reads_back_what_the_render_published(session, monkeypatch):
    """The tests above watch `set_progress`. This one is the other half: that
    `render_progress` returns what was written, so the endpoint could not be
    deleted with every other test in this file still passing."""
    workshop, session_id = session
    samples = record(workshop, monkeypatch)
    api.render(workshop, session_id, {"no_mp4": True})

    live = api.render_progress(workshop, session_id)
    assert live == samples[-1]
    assert live["phase"] == renderer.PHASE_DONE
    assert live["label"] == api.PHASE_LABELS[renderer.PHASE_DONE]


# --------------------------------------------------------------- when it fails


def test_a_failed_render_stops_saying_it_is_running(session, monkeypatch):
    """Otherwise the page spins forever on a render that died: the POST
    returns an error the page shows, and the poll it started keeps insisting
    something is still happening."""
    workshop, session_id = session

    def explode(*_args, **_kwargs):
        raise RuntimeError("the encoder fell over")

    monkeypatch.setattr(api, "render_spec", explode)
    with pytest.raises(RuntimeError):
        api.render(workshop, session_id, {"no_mp4": True})

    payload = api.render_progress(workshop, session_id)
    assert payload["running"] is False


# ------------------------------------------------------------- the labels


def test_every_phase_the_renderer_can_report_has_a_label():
    """The page prints `label`, so a phase without one prints its constant."""
    phases = {getattr(renderer, name) for name in dir(renderer) if name.startswith("PHASE_")}
    assert phases, "the phase constants moved; this test is checking nothing"
    assert phases <= set(api.PHASE_LABELS), sorted(phases - set(api.PHASE_LABELS))


def test_every_phase_except_done_carries_a_weight():
    """`_phase_fraction` looks each phase up in `PHASE_WEIGHTS`; one that is
    missing silently reports 1.0 and slams the bar to full."""
    phases = {getattr(renderer, name) for name in dir(renderer) if name.startswith("PHASE_")} - {
        renderer.PHASE_DONE
    }
    assert phases <= set(api.PHASE_WEIGHTS), sorted(phases - set(api.PHASE_WEIGHTS))
