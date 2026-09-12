"""The Timeline stage: who is on screen, when, and for how long.

Every assertion here is about seconds and shot indices. No pixels, no encoder,
no audio file - which is the point of the stage existing at all.
"""

from __future__ import annotations

import pytest

from rekindle.memory.render import timeline as tl


def test_a_plan_with_no_shots_is_empty():
    plan = tl.plan(0)
    assert plan.beats == ()
    assert plan.total == 0.0
    # `frames` must never be 0: an encoder handed zero frames writes a file
    # some players refuse and others render as a broken image.
    assert plan.frames == 1


def test_the_first_shot_never_dissolves_in():
    """There is nothing behind it. A fade at zero is a dip up from black."""
    plan = tl.plan(3)
    assert plan.beats[0].start == 0.0
    assert plan.beats[0].fade == 0.0
    assert all(b.fade > 0 for b in plan.beats[1:])


def test_a_dissolve_is_taken_out_of_a_shot_and_not_added_to_the_memory():
    """`--style film` and `--style cuts` produce the same length of video.

    The dissolve overlaps the outgoing shot's hold rather than extending it,
    so turning crossfades on does not add 24 x 0.6 = 14 seconds to a memory,
    and the two styles can be compared side by side. What it costs is time
    ALONE on screen: 1.9 seconds of a 2.5-second hold.
    """
    cuts = tl.plan(25, crossfade=0.0)
    film = tl.plan(25, crossfade=0.6)
    assert cuts.total == pytest.approx(3.5 + 24 * 2.5)
    assert film.total == pytest.approx(cuts.total)
    alone = film.beats[2].duration - 2 * film.beats[2].fade
    assert alone == pytest.approx(2.5 - 0.6)


def test_a_crossfade_longer_than_the_shot_is_clamped():
    """A caller asking for a 2-second dissolve on a 2.5-second shot gets the
    longest one that still leaves the photograph visible, not a memory of
    transitions."""
    plan = tl.plan(4, seconds=2.5, title_seconds=2.5, crossfade=2.0)
    fade = plan.beats[1].fade
    assert 0 < fade <= 2.5 - tl.MIN_CLEAR
    clear = plan.beats[1].duration - 2 * fade
    assert clear >= tl.MIN_CLEAR - 1e-9


def test_two_shots_are_on_screen_during_a_dissolve_and_one_otherwise():
    plan = tl.plan(3, seconds=2.0, title_seconds=2.0, crossfade=0.5)
    cut = plan.beats[1].start + plan.beats[1].fade
    assert len(plan.active_at(cut - 0.25)) == 2
    assert len(plan.active_at(cut + 0.25)) == 1


def test_alpha_ramps_from_nothing_to_everything_across_the_fade():
    plan = tl.plan(2, crossfade=0.6)
    beat = plan.beats[1]
    assert beat.alpha_at(beat.start) == pytest.approx(0.0)
    assert beat.alpha_at(beat.start + 0.3) == pytest.approx(0.5)
    assert beat.alpha_at(beat.start + 0.6) == pytest.approx(1.0)
    assert beat.alpha_at(beat.start + 2.0) == pytest.approx(1.0)
    assert beat.alpha_at(beat.start - 0.1) == 0.0


def test_progress_runs_from_zero_to_one_over_the_whole_beat():
    """Ken Burns reads this, so it has to cover the dissolve too - a pan that
    only started once the shot was fully visible would jump at the handover."""
    beat = tl.Beat(index=1, start=10.0, duration=2.0, fade=0.5)
    assert beat.progress_at(10.0) == 0.0
    assert beat.progress_at(11.0) == pytest.approx(0.5)
    assert beat.progress_at(12.0) == pytest.approx(1.0)
    assert beat.progress_at(99.0) == 1.0


def test_frame_times_cover_the_memory_and_start_at_zero():
    plan = tl.plan(3, fps=25)
    times = plan.times()
    assert times[0] == 0.0
    assert len(times) == plan.frames
    assert times[-1] < plan.total


# --------------------------------------------------------------------------
# beat snapping


def _cuts(plan: tl.Timeline) -> list[float]:
    return [round(b.start + b.fade, 3) for b in plan.beats[1:]]


def test_no_onsets_leaves_the_timeline_exactly_as_it_was():
    """No music, no ffmpeg, an unreadable track: all of them land here, and
    all of them must be the evenly spaced timeline and not a broken one."""
    plan = tl.plan(5)
    assert tl.snap_to(plan, []) is plan


def test_a_cut_moves_onto_a_nearby_onset():
    plan = tl.plan(3, seconds=2.0, title_seconds=2.0, crossfade=0.4)
    before = _cuts(plan)
    onsets = [before[0] + 0.2, before[1] + 0.15]
    after = _cuts(tl.snap_to(plan, onsets))
    assert after == [pytest.approx(onsets[0]), pytest.approx(onsets[1], abs=0.001)]


def test_a_cut_with_no_onset_within_tolerance_does_not_move():
    """Dragging a cut two seconds to reach a beat is worse than not being on
    the beat: the shot lengths visibly stop matching each other."""
    plan = tl.plan(3, seconds=2.0, title_seconds=2.0, crossfade=0.4)
    before = _cuts(plan)
    after = _cuts(tl.snap_to(plan, [before[0] + 5.0]))
    assert after == before


def test_snapping_does_not_accumulate_error():
    """Each boundary is snapped from its OWN ORIGINAL position.

    Chaining each cut off the previously snapped one lets a run of
    same-direction nudges walk the whole timeline. The onset grid here is
    2.1 s while the shots are 2.0 s, so a chained implementation drags every
    cut 0.1 s further than the last and the final cut of a nine-shot memory
    lands nearly a second late. Snapping from the original bounds every error
    by the tolerance.
    """
    plan = tl.plan(9, seconds=2.0, title_seconds=2.0, crossfade=0.4)
    onsets = [i * 2.1 for i in range(50)]
    before = _cuts(plan)
    after = _cuts(tl.snap_to(plan, onsets))
    for was, now in zip(before, after, strict=True):
        assert abs(now - was) <= tl.TOLERANCE + 1e-6, (was, now)


def test_no_two_cuts_are_closer_than_the_minimum_shot():
    """Two onsets 0.2 s apart are a grace note, not a shot change.

    Asserted on the CUT POSITIONS rather than on `Beat.duration`, because the
    duration is separately clamped with a `max(min_duration, ...)` and would
    look right while the timeline underneath it was nonsense.
    """
    plan = tl.plan(6, seconds=2.0, title_seconds=2.0, crossfade=0.4)
    snapped = tl.snap_to(plan, [0.1, 0.2, 0.3, 0.4, 0.5], tolerance=99.0)
    cuts = [0.0] + _cuts(snapped)
    gaps = [b - a for a, b in zip(cuts, cuts[1:], strict=False)]
    assert min(gaps) >= tl.MIN_DURATION - 1e-6, gaps


def test_a_shot_is_never_snapped_below_its_floor():
    """Two onsets 0.2s apart are a grace note, not a shot change."""
    plan = tl.plan(4, seconds=2.0, title_seconds=2.0, crossfade=0.4)
    snapped = tl.snap_to(plan, [0.1, 0.2, 0.3, 0.4], tolerance=99.0)
    for beat in snapped.beats:
        assert beat.duration >= tl.MIN_DURATION


def test_the_first_shot_still_starts_at_zero_after_snapping():
    """Moving it would put a gap of black at the front of the memory."""
    snapped = tl.snap_to(tl.plan(4), [0.7, 3.0, 5.0], tolerance=99.0)
    assert snapped.beats[0].start == 0.0


def test_snapping_a_one_shot_timeline_changes_nothing():
    plan = tl.plan(1)
    assert tl.snap_to(plan, [1.0, 2.0]) is plan


def test_describe_says_whether_there_are_dissolves():
    assert "hard cuts" in tl.describe(tl.plan(3, crossfade=0.0))
    assert "dissolves" in tl.describe(tl.plan(3, crossfade=0.6))


def test_the_styles_are_the_two_the_cli_accepts():
    assert set(tl.STYLES) == {tl.STYLE_FILM, tl.STYLE_CUTS}
