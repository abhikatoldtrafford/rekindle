"""The Timeline stage: who is on screen, when, and for how long.

`CONTRIBUTING.md` has promised this stage since M2:

> *Recipes choose photos and their order. They do not own timing - a separate
> `Timeline` stage does, so beat-synced music doesn't break every recipe.*

Until now nothing owned timing at all. `write_mp4` held every shot for
`DEFAULT_SECONDS` and the title for `TITLE_SECONDS`, hard-coded at the point
of encoding, and `write_gif` had its own second pair of numbers. Anything that
wanted to change WHEN a shot changed - a crossfade, music, a longer beat on a
photograph worth looking at - had to reach into the encoder.

So this module answers one question and nothing else:

    given N shots, when does each one appear, how long does it hold, and
    how long does the previous one take to dissolve into it?

It reads no pixels, imports no encoder, and knows nothing about ffmpeg,
Pillow, Ken Burns or captions. Everything downstream is a pure function of the
`Timeline` it returns, which is what makes beat-sync a change to ONE function
rather than a change to every renderer - and what makes it testable without a
video encoder, an audio file or a photo.

## Beats and onsets

`plan()` lays out even beats. `snap_to()` takes that layout and moves each
shot boundary to the nearest musical onset, when onsets are available and one
is close enough. A boundary with no onset within `TOLERANCE` is left exactly
where it was: pulling a cut two seconds to reach a beat is worse than not
being on the beat, because the shot lengths visibly stop matching each other.

Onset detection itself lives in `render.onsets`, needs ffmpeg for anything
that is not a WAV, and is entirely optional - `plan()` with no onsets is the
shipped default and a complete answer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

#: Motion styles. `film` is crossfades, Ken Burns where the pixels allow it
#: and animated cards; `cuts` is the hard-cut, static-frame output that
#: shipped before this module existed, kept because it encodes several times
#: faster and because someone will prefer it.
STYLE_FILM = "film"
STYLE_CUTS = "cuts"
STYLES = (STYLE_FILM, STYLE_CUTS)

#: How long an ordinary shot holds, INCLUDING the dissolve into it. 2.5s was
#: the previous per-shot duration and is unchanged, so a `cuts` render is the
#: same length it always was.
DEFAULT_SECONDS = 2.5
TITLE_SECONDS = 3.5

#: How long one shot takes to dissolve into the next.
#:
#: 0.6s, chosen by watching. At 0.3s the dissolve reads as a glitch rather
#: than a transition; at 1.2s a 2.5-second shot is dissolving for half its
#: life and the memory feels underwater. The dissolve is subtracted from the
#: outgoing shot's hold rather than added to the total, so turning crossfades
#: on does not make every memory 24 x 0.6 = 14 seconds longer.
DEFAULT_CROSSFADE = 0.6

#: The floor on how long a shot is fully itself, with nothing dissolving over
#: it. Below this a crossfade-heavy timeline stops showing photographs and
#: starts showing transitions, so the crossfade is shortened instead.
MIN_CLEAR = 0.8

#: How far a cut may be moved to land on a musical onset. Half a beat at
#: 100bpm. Beyond this the shot lengths visibly stop matching.
TOLERANCE = 0.30

#: A shot may never be shortened below this by beat-snapping, whatever the
#: music does.
MIN_DURATION = 1.0

FPS = 25


@dataclass(frozen=True)
class Beat:
    """One shot's slot on the timeline. All times in seconds from zero.

    `start` is when this shot begins to appear - the first frame of its
    dissolve, not the first frame on which it is alone. `fade` is how long
    that dissolve lasts, and is 0 for the first shot and for a hard cut.
    """

    index: int
    start: float
    duration: float
    fade: float = 0.0

    @property
    def end(self) -> float:
        return self.start + self.duration

    def alpha_at(self, t: float) -> float:
        """How opaque this shot is at time `t`, in [0, 1].

        1.0 for the whole of its life except the dissolve at its head. The
        shot UNDERNEATH is not faded out - it is covered - which is what makes
        a dissolve a dissolve rather than a dip to black.
        """
        if t < self.start or t >= self.end:
            return 0.0
        if self.fade <= 0.0:
            return 1.0
        return min(1.0, (t - self.start) / self.fade)

    def progress_at(self, t: float) -> float:
        """How far through this shot `t` is, in [0, 1]. For Ken Burns."""
        if self.duration <= 0:  # pragma: no cover - plan() never emits one
            return 0.0
        return min(1.0, max(0.0, (t - self.start) / self.duration))


@dataclass(frozen=True)
class Timeline:
    """The complete schedule. Beats are in order and may overlap by `fade`."""

    beats: tuple[Beat, ...]
    fps: int = FPS

    @property
    def total(self) -> float:
        return max((b.end for b in self.beats), default=0.0)

    @property
    def frames(self) -> int:
        return max(1, round(self.total * self.fps))

    def active_at(self, t: float) -> list[Beat]:
        """Every beat visible at `t`, bottom layer first.

        Two during a dissolve, one otherwise. Returning a list rather than a
        pair is not generality for its own sake: with a long crossfade and a
        short shot three beats can overlap, and a renderer that assumed two
        would silently drop the middle one.
        """
        return [b for b in self.beats if b.start <= t < b.end]

    def times(self) -> list[float]:
        """Every frame's timestamp. The renderer's outer loop."""
        return [i / self.fps for i in range(self.frames)]


def plan(
    count: int,
    *,
    seconds: float = DEFAULT_SECONDS,
    title_seconds: float = TITLE_SECONDS,
    has_title: bool = True,
    crossfade: float = DEFAULT_CROSSFADE,
    fps: int = FPS,
) -> Timeline:
    """`count` shots into an even timeline.

    **The dissolve is taken OUT of each shot's hold, not added to the memory.**
    Shot n+1 starts `crossfade` seconds before shot n's hold ends, so a
    25-beat memory is 63.5 seconds whether it is cut or dissolved, and
    `--style film` against `--style cuts` is a comparison of the same length
    of video. What changes is how much of each shot is ALONE on screen: at a
    2.5-second hold and a 0.6-second dissolve, 1.9 seconds.

    `crossfade` is clamped so that every shot keeps `MIN_CLEAR` seconds alone.
    A caller asking for a 2-second dissolve on a 2.5-second shot gets the
    longest one that leaves the photograph visible, not a memory of
    transitions.
    """
    if count <= 0:
        return Timeline(beats=(), fps=fps)
    fade = max(0.0, min(crossfade, min(seconds, title_seconds) - MIN_CLEAR))
    beats: list[Beat] = []
    cursor = 0.0
    for i in range(count):
        hold = title_seconds if (has_title and i == 0) else seconds
        this_fade = 0.0 if i == 0 else fade
        start = cursor - this_fade
        beats.append(Beat(index=i, start=start, duration=hold + this_fade, fade=this_fade))
        cursor = start + hold + this_fade
    return Timeline(beats=tuple(beats), fps=fps)


def snap_to(
    timeline: Timeline,
    onsets: Sequence[float],
    *,
    tolerance: float = TOLERANCE,
    min_duration: float = MIN_DURATION,
) -> Timeline:
    """Move each cut to the nearest musical onset, where one is close enough.

    The rules, in order, and each exists because the obvious version of this
    function looks fine on paper and produces something unwatchable:

    * **The first shot's start never moves.** It is zero. Moving it would put
      a gap of black at the front of the memory.
    * **A boundary with no onset within `tolerance` stays put.** Music with a
      sparse or rubato opening would otherwise drag one cut a long way and
      make that shot conspicuously longer than every other.
    * **Each boundary is snapped from its OWN ORIGINAL position**, never from
      the previous snapped one. This is the opposite of what the first version
      of this comment claimed, and the claim was the wrong way round: chaining
      each cut off the last one lets a run of same-direction nudges walk the
      whole timeline, so a memory whose music sits 0.1 s late everywhere would
      finish a shot and a half late. Snapping from the original keeps every
      error bounded by `tolerance`.
    * **No shot may fall below `min_duration`.** Two onsets 0.2s apart are a
      grace note, not a shot change. This is the only thing the previous beat
      is consulted for.
    * **The dissolve length is preserved** and re-clamped, so a shot shortened
      by snapping does not end up dissolving for most of its life.

    Returns the timeline unchanged when `onsets` is empty, which is what
    happens with no music, no ffmpeg, or an unreadable track.
    """
    if not onsets or len(timeline.beats) < 2:
        return timeline
    marks = sorted(float(o) for o in onsets)

    out: list[Beat] = [timeline.beats[0]]
    for beat in timeline.beats[1:]:
        previous = out[-1]
        # Where the CUT is: the moment the outgoing shot stops being alone.
        wanted = beat.start + beat.fade
        nearest = _nearest(marks, wanted)
        target = nearest if (nearest is not None and abs(nearest - wanted) <= tolerance) else wanted
        # The previous shot must keep min_duration of life measured from the
        # point it became fully visible.
        floor = previous.start + previous.fade + min_duration
        target = max(target, floor)
        fade = min(beat.fade, max(0.0, (target - (previous.start + previous.fade)) / 2.0))
        start = target - fade
        # The outgoing shot now ends where the incoming one is fully up.
        out[-1] = Beat(
            index=previous.index,
            start=previous.start,
            duration=max(min_duration, target + fade - previous.start),
            fade=previous.fade,
        )
        out.append(
            Beat(
                index=beat.index,
                start=start,
                duration=beat.duration + (beat.start - start),
                fade=fade,
            )
        )
    return Timeline(beats=tuple(out), fps=timeline.fps)


def _nearest(marks: Sequence[float], value: float) -> float | None:
    """The onset closest to `value`. Linear: onset lists are short (a few
    hundred for a three-minute track) and this runs once per shot."""
    if not marks:
        return None
    return min(marks, key=lambda m: abs(m - value))


def describe(timeline: Timeline) -> str:
    """One line for the CLI: how long, how many cuts, how much dissolve."""
    fades = [b.fade for b in timeline.beats if b.fade > 0]
    if not fades:
        return f"{timeline.total:.1f}s, {len(timeline.beats)} shots, hard cuts"
    return (
        f"{timeline.total:.1f}s, {len(timeline.beats)} shots, "
        f"{len(fades)} dissolves of {min(fades):.2f}-{max(fades):.2f}s"
    )
