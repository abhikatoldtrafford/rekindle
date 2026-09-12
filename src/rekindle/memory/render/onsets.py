"""Where the notes are, so a cut can land on one.

The `Timeline` stage owns WHEN a shot changes. This module answers the one
question it needs from the music: at what times does something start?

**This finds note attacks. It is not a beat tracker and does not claim to
be.** There is no tempo estimate here, no bar, no downbeat. On the solo piano
in `music/` there is often no stable tempo to find - a Chopin mazurka is
rubato by construction - so what a cut can honestly land on is the start of a
note, and that is what `timeline.snap_to` moves it to. Measured across all 40
tracks in `music/` (2,285 seconds): 5,325 onsets, 2.33 a second, so the mean
gap is 0.43 s and a cut looking within 0.3 s usually finds one. The effect is
that shot changes stop falling between the notes; it is not the shot changes
marching with a drum.

**No new dependency.** librosa is the obvious answer and it brings numpy,
scipy, numba, soundfile and audioread to a project whose default install is
four packages. The measurement this needs is an energy envelope and a peak
pick, which is about eighty lines, and it runs over a 60-second track in
0.2 seconds. A real beat tracker would be the reason to take that dependency;
this is not.

**No new decoder either.** A `.wav` is read with the standard library's
`wave` module, which is what makes this testable in CI: a test can synthesise
a click track with `wave` and assert the clicks come back. Anything else -
mp3, m4a, flac - is converted to a temporary WAV by ffmpeg, which the MP4 path
already requires. Without ffmpeg, a non-WAV track yields no onsets and the
timeline stays evenly spaced, which is exactly what it did before this module
existed.

Nothing here raises. An unreadable file, a truncated file, a file that is not
audio at all: all of them return no onsets and a sentence saying so.
"""

from __future__ import annotations

import array
import contextlib
import math
import subprocess
import tempfile
import wave
from pathlib import Path

#: Everything is analysed at this rate, mono. Onsets are a question about
#: energy over ~10 ms windows; 11 kHz answers it exactly as well as 44.1 kHz
#: and does it four times faster in pure Python.
RATE = 11025

#: The analysis hop, in seconds. 10 ms is finer than any cut this is used to
#: place - `timeline.TOLERANCE` is 300 ms - so the resolution is free.
HOP_SECONDS = 0.010

# THERE IS NO ADAPTIVE THRESHOLD HERE, AND THERE WAS ONE.
#
# The first version of this module compared each rise against the local mean
# of the flux over 0.8 s, on the standard reasoning that a fixed bar finds
# every note of a loud passage and none of a quiet one. Measured across the
# 40 tracks in `music/` it removed 28 onsets out of 6,076 - 0.5% - and on a
# synthetic tremolo it changed the count from 9 to 9. It was inert.
#
# It was inert because the envelope below is already normalised by the
# TRACK'S OWN median energy, which is the same problem solved once and
# globally rather than approximately and per-window. Deleted rather than kept
# with an apologetic comment: an unused knob is one someone will later tune in
# the belief that it does something.

#: Two onsets closer than this are one event. A piano chord is not four cuts.
#: A ceiling of 4 a second; the shipped configuration measures 2.33, so
#: `MIN_FLUX` is what actually decides the rate and this is the backstop.
MIN_GAP_SECONDS = 0.25

#: How far energy must rise before something has started, in the units of the
#: normalised log envelope. This is THE gate: it is what decides how many
#: onsets a track has.
#:
#: It exists because a test found a defect. A held 440 Hz sine, whose energy
#: is constant to three decimal places, reported eleven onsets at even
#: 0.28-second intervals: a 10 ms window never holds a whole number of cycles,
#: the resulting ripple is the largest thing in a signal where nothing is
#: happening, and any purely relative rule fires on it.
#:
#: The floor is absolute, which is only safe because the envelope is
#: normalised by the track's own median energy first - so this number means
#: the same musical event on a loud recording and a quiet one. See
#: `onset_times`.
#:
#: Swept across the 40 tracks: 0.10 gives 3.05 onsets a second, 0.25 gives
#: 2.33, 0.40 gives 1.59, 0.90 gives 0.27. 0.25 puts the mean gap at 0.43 s
#: against a 0.30 s snapping tolerance, which is where a cut usually finds an
#: onset without the onsets being every note.
MIN_FLUX = 0.25

#: Longer than any memory this produces, and a bound on how much audio is read
#: into memory: 300 s at 11 kHz is 3.3 M samples.
MAX_SECONDS = 300

WAV_SUFFIXES = frozenset({".wav", ".wave"})

NO_FFMPEG = "beat detection needs ffmpeg to read anything but a .wav, and it was not found on PATH"


def detect(path: Path, *, ffmpeg: str | None = None) -> tuple[list[float], str]:
    """Onset times in seconds, and a note when there are none.

    Never raises. `([], reason)` is a supported answer and means the timeline
    stays evenly spaced.
    """
    try:
        samples, rate = read_samples(path, ffmpeg=ffmpeg)
    except OSError as exc:
        return [], f"{path.name} could not be read ({type(exc).__name__})"
    except FfmpegMissing:
        return [], NO_FFMPEG
    except wave.Error as exc:
        return [], f"{path.name} is not audio this can read ({exc})"
    if not samples:
        return [], f"{path.name} holds no audio"
    return onset_times(samples, rate), ""


class FfmpegMissing(RuntimeError):
    """ffmpeg is needed to decode this file and is not on PATH."""


def read_samples(
    path: Path, *, ffmpeg: str | None = None, max_seconds: int = MAX_SECONDS
) -> tuple[array.array, int]:
    """Mono float samples at `RATE`, from a WAV directly or via ffmpeg."""
    if path.suffix.lower() in WAV_SUFFIXES:
        return _read_wav(path, max_seconds=max_seconds)

    from rekindle.memory.render.mp4 import ffmpeg_path

    binary = ffmpeg or ffmpeg_path()
    if binary is None:
        raise FfmpegMissing(NO_FFMPEG)
    with tempfile.TemporaryDirectory(prefix="rekindle-audio-") as tmp:
        target = Path(tmp) / "audio.wav"
        cmd = [
            binary,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-t",
            str(max_seconds),
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            str(RATE),
            "-c:a",
            "pcm_s16le",
            str(target),
        ]
        try:
            done = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise OSError(str(exc)) from exc
        if done.returncode != 0 or not target.is_file():
            raise OSError((done.stderr or "ffmpeg could not decode it").strip()[:200])
        return _read_wav(target, max_seconds=max_seconds)


def _read_wav(path: Path, *, max_seconds: int) -> tuple[array.array, int]:
    """A WAV as mono 16-bit samples, resampled to `RATE` by decimation.

    Decimation rather than a filtered resample: this is an energy envelope,
    the aliasing it introduces is far above the 100 Hz the envelope is read
    at, and a proper polyphase filter would be a hundred lines defending
    against nothing.
    """
    with contextlib.closing(wave.open(str(path), "rb")) as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        if width != 2:
            # 8-bit and 24-bit WAVs exist; converting them here would be a
            # second decoder. ffmpeg is the one that handles the long tail.
            raise wave.Error(f"{width * 8}-bit audio is not read directly; convert it first")
        raw = handle.readframes(min(handle.getnframes(), rate * max_seconds))

    pcm = array.array("h")
    pcm.frombytes(raw[: len(raw) - len(raw) % (2 * channels)])
    if not pcm:
        return array.array("f"), RATE
    if channels > 1:
        pcm = array.array("h", pcm[::channels])

    step = max(1, round(rate / RATE))
    out = array.array("f", (s / 32768.0 for s in pcm[::step]))
    return out, max(1, rate // step)


def onset_times(
    samples: array.array,
    rate: int,
    *,
    hop_seconds: float = HOP_SECONDS,
    min_gap: float = MIN_GAP_SECONDS,
    min_flux: float = MIN_FLUX,
) -> list[float]:
    """Peak-pick a log-energy flux envelope. Returns times in seconds.

    Four steps, and each exists for a reason worth writing down:

    1. **Energy per hop.** Mean square over a 10 ms window.
    2. **Normalised by the track's own median energy, then logged.** This is
       what lets one absolute threshold serve every recording. A note attack
       in a quiet piece and one in a loud piece are the same musical event,
       and neither a raw energy nor a plain log of it treats them alike: a
       recording 16x quieter has 256x less energy and reports nothing.
       Dividing by the median first makes the envelope about THIS piece.
    3. **Positive difference.** Only rises matter. A note ending is not a beat.
    4. **A floor, a local maximum, and a minimum gap.** The floor
       (`MIN_FLUX`) refuses a signal in which nothing is happening. The local
       maximum makes one attack one onset rather than three. The gap makes a
       chord one event.
    """
    hop = max(1, int(rate * hop_seconds))
    if len(samples) < hop * 4:
        return []

    power: list[float] = []
    for start in range(0, len(samples) - hop + 1, hop):
        total = 0.0
        for i in range(start, start + hop):
            value = samples[i]
            total += value * value
        power.append(total / hop)

    # The track's own middle. MEDIAN rather than mean: a mean is dragged up by
    # the loudest few seconds of a piece with a big climax, which is exactly
    # the piece whose quiet opening would then report nothing.
    ordered = sorted(p for p in power if p > 0.0)
    if not ordered:
        return []
    reference = ordered[len(ordered) // 2]
    energy = [math.log1p(p / reference) for p in power]

    flux = [max(0.0, energy[i] - energy[i - 1]) for i in range(1, len(energy))]
    if not flux:
        return []

    times: list[float] = []
    last = -min_gap
    for i, value in enumerate(flux):
        if value < min_flux:
            continue
        # A local maximum, so one attack is one onset and not three.
        if (i > 0 and flux[i - 1] > value) or (i + 1 < len(flux) and flux[i + 1] > value):
            continue
        when = (i + 1) * hop / rate
        if when - last < min_gap:
            continue
        times.append(round(when, 4))
        last = when
    return times
