"""Onset detection, against audio the test synthesises itself.

`wave` is in the standard library, so a click track can be written and read
back with no ffmpeg, no codec and no committed media asset - which is what
makes this run in CI. The assertions are about WHERE the clicks came back,
never about the detector having been called.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pytest

from rekindle.memory.render import onsets

RATE = 22050


def _write_wav(path: Path, samples: list[float], rate: int = RATE, channels: int = 1) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(
            b"".join(struct.pack("<h", max(-32767, min(32767, int(s * 32767)))) for s in samples)
        )
    return path


def _clicks(times: list[float], *, seconds: float = 6.0, rate: int = RATE) -> list[float]:
    """A quiet tone with a short loud burst at each time. A note attack."""
    out = [0.02 * math.sin(2 * math.pi * 220 * i / rate) for i in range(int(seconds * rate))]
    for when in times:
        start = int(when * rate)
        for i in range(start, min(len(out), start + int(0.05 * rate))):
            decay = 1.0 - (i - start) / (0.05 * rate)
            out[i] = 0.8 * decay * math.sin(2 * math.pi * 880 * i / rate)
    return out


def test_clicks_come_back_at_the_times_they_were_written(tmp_path):
    wanted = [0.5, 1.5, 2.5, 3.5, 4.5]
    path = _write_wav(tmp_path / "click.wav", _clicks(wanted))
    found, note = onsets.detect(path)
    assert note == ""
    for when in wanted:
        assert any(abs(f - when) < 0.05 for f in found), (when, found)


def test_silence_produces_no_onsets(tmp_path):
    path = _write_wav(tmp_path / "quiet.wav", [0.0] * (RATE * 3))
    found, _ = onsets.detect(path)
    assert found == []


def test_a_steady_tone_is_not_a_stream_of_onsets(tmp_path):
    """The whole point of the positive-difference step. A held note is one
    event at its start, not 300 events at 100 a second."""
    tone = [0.4 * math.sin(2 * math.pi * 440 * i / RATE) for i in range(RATE * 3)]
    path = _write_wav(tmp_path / "tone.wav", tone)
    found, _ = onsets.detect(path)
    assert len(found) <= 2, found


def test_a_note_ending_is_not_an_onset(tmp_path):
    """Only RISES matter. Taking the absolute difference instead of the
    positive one reports the end of every note as well as its start, which
    doubles the onsets and puts half of them in the silence after a phrase.
    """
    rate = RATE
    out = [0.0] * (rate * 4)
    for i in range(int(1.0 * rate), int(2.5 * rate)):
        out[i] = 0.7 * math.sin(2 * math.pi * 440 * i / rate)
    path = _write_wav(tmp_path / "burst.wav", out)
    found, _ = onsets.detect(path)
    assert any(abs(f - 1.0) < 0.08 for f in found), found
    assert not any(2.3 < f < 2.8 for f in found), f"the note ENDING was reported: {found}"


def test_one_absolute_floor_works_at_both_ends_of_the_dynamic_range(tmp_path):
    """Why the envelope is LOG energy and not linear.

    `MIN_FLUX` is one number for every track. On a linear envelope, "energy
    rose by 0.25" means a different thing at an amplitude of 0.05 than at 0.8
    - one floor cannot serve both, and the quiet recording gets no onsets at
    all. On a log envelope it means "energy rose by about 30%", which is the
    same musical event at any volume.

    The same clicks, recorded 64x quieter, must give the same onsets. 64 and
    not 16: at 16 an un-normalised log envelope still clears the floor, so
    the weaker version of this test passed with the normalisation deleted.
    """
    loud = _clicks([1.0, 2.0, 3.0], seconds=4.0)
    quiet = [s / 64.0 for s in loud]
    found_loud, _ = onsets.detect(_write_wav(tmp_path / "loud.wav", loud))
    found_quiet, _ = onsets.detect(_write_wav(tmp_path / "quiet16.wav", quiet))
    for when in (1.0, 2.0, 3.0):
        assert any(abs(f - when) < 0.08 for f in found_loud), (when, found_loud)
        assert any(abs(f - when) < 0.08 for f in found_quiet), (when, found_quiet)


def test_two_clicks_closer_than_the_minimum_gap_are_one_event(tmp_path):
    """A piano chord is not four cuts."""
    path = _write_wav(tmp_path / "chord.wav", _clicks([1.0, 1.08, 1.15]))
    found, _ = onsets.detect(path)
    near = [f for f in found if 0.8 < f < 1.5]
    assert len(near) == 1, near


def test_a_small_wobble_in_a_loud_passage_is_not_an_onset(tmp_path):
    """Why the normalised envelope is LOGGED and not left linear.

    `MIN_FLUX` is an absolute number, so on a linear envelope it means "energy
    rose by 0.25 times the median" - which a 6% wobble in a passage running at
    100x the median clears comfortably, every cycle. On a log envelope it
    means "energy rose by about 30%", which a 6% wobble never does, at any
    level.

    The loud passage has to be the MINORITY of the track, so that the median
    lands in the quiet part and the loud part really is at 100x it. With the
    loud part in the majority the median lands inside it, the ratio is 1.1,
    and this test passes with the log deleted - which is what the first
    version of it did.
    """
    rate = RATE
    out = [0.05 * math.sin(2 * math.pi * 220 * i / rate) for i in range(rate * 10)]
    for i in range(int(7.0 * rate), rate * 10):
        wobble = 1.0 + 0.03 * math.sin(2 * math.pi * 5 * i / rate)
        out[i] = 0.5 * wobble * math.sin(2 * math.pi * 440 * i / rate)
    path = _write_wav(tmp_path / "wobble.wav", out)
    found, _ = onsets.detect(path)
    during = [f for f in found if 7.3 < f < 9.9]
    assert len(during) <= 2, f"{len(during)} onsets from one wobbling note: {during[:12]}"


def test_a_quiet_passage_after_a_very_loud_one_is_still_read(tmp_path):
    """Why the reference is the MEDIAN energy and not the mean.

    Two seconds of a very loud chord, then eight seconds of quiet playing.
    The mean energy is dominated by the loud two seconds, so normalising by
    it pushes the whole quiet passage under the floor and the memory stops
    being synced two seconds in. The median is where the piece actually sits.
    """
    rate = RATE
    out = [0.001 * math.sin(2 * math.pi * 220 * i / rate) for i in range(rate * 10)]
    for i in range(0, int(2.0 * rate)):
        out[i] = 0.95 * math.sin(2 * math.pi * 110 * i / rate)
    for when in (4.0, 6.0, 8.0):
        start = int(when * rate)
        for i in range(start, start + int(0.05 * rate)):
            decay = 1.0 - (i - start) / (0.05 * rate)
            out[i] = 0.02 * decay * math.sin(2 * math.pi * 880 * i / rate)
    path = _write_wav(tmp_path / "climax.wav", out)
    found, _ = onsets.detect(path)
    for when in (4.0, 6.0, 8.0):
        assert any(abs(f - when) < 0.08 for f in found), (when, found)


def test_a_quiet_click_and_a_loud_one_are_both_found(tmp_path):
    """A piece with a loud half and a quiet half is synced in both.

    The envelope is normalised by the track's MEDIAN energy, so the quiet
    half's attacks are measured against the piece rather than against the
    loudest thing in it. Dividing by the mean instead lets a climax swallow
    the opening.
    """
    rate = RATE
    out = _clicks([1.0, 2.0], seconds=6.0)
    # Halve everything after 3 seconds, then put clicks in the quiet part.
    for i in range(int(3 * rate), len(out)):
        out[i] *= 0.1
    for when in (4.0, 5.0):
        start = int(when * rate)
        for i in range(start, min(len(out), start + int(0.05 * rate))):
            decay = 1.0 - (i - start) / (0.05 * rate)
            out[i] = 0.08 * decay * math.sin(2 * math.pi * 880 * i / rate)
    path = _write_wav(tmp_path / "dynamic.wav", out)
    found, _ = onsets.detect(path)
    for when in (1.0, 2.0, 4.0, 5.0):
        assert any(abs(f - when) < 0.08 for f in found), (when, found)


def test_a_stereo_file_is_read(tmp_path):
    mono = _clicks([1.0, 2.0])
    interleaved = [s for value in mono for s in (value, value)]
    path = _write_wav(tmp_path / "stereo.wav", interleaved, channels=2)
    found, note = onsets.detect(path)
    assert note == ""
    assert any(abs(f - 1.0) < 0.08 for f in found)


def test_a_file_that_is_not_audio_is_a_reason_and_not_a_crash(tmp_path):
    path = tmp_path / "notaudio.wav"
    path.write_bytes(b"this is not a RIFF header at all")
    found, note = onsets.detect(path)
    assert found == []
    assert note


def test_a_missing_file_is_a_reason_and_not_a_crash(tmp_path):
    found, note = onsets.detect(tmp_path / "absent.wav")
    assert found == []
    assert note


def test_an_eight_bit_wav_is_declined_rather_than_misread(tmp_path):
    """Reading it as 16-bit would produce noise and a wall of false onsets."""
    with wave.open(str(tmp_path / "eight.wav"), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(1)
        handle.setframerate(RATE)
        handle.writeframes(bytes(RATE))
    found, note = onsets.detect(tmp_path / "eight.wav")
    assert found == []
    assert "8-bit" in note


def test_a_non_wav_without_ffmpeg_says_so(tmp_path, monkeypatch):
    """The degradation the whole feature rests on: no ffmpeg means no beat
    sync and an evenly spaced timeline, not a crash and not a wrong answer."""
    monkeypatch.setattr("rekindle.memory.render.mp4.ffmpeg_path", lambda: None)
    path = tmp_path / "track.mp3"
    path.write_bytes(b"\x00" * 100)
    found, note = onsets.detect(path)
    assert found == []
    assert note == onsets.NO_FFMPEG


def test_a_very_short_file_yields_nothing_rather_than_raising(tmp_path):
    path = _write_wav(tmp_path / "tiny.wav", [0.1] * 10)
    assert onsets.detect(path)[0] == []


def test_onsets_are_in_order_and_within_the_track(tmp_path):
    path = _write_wav(tmp_path / "c.wav", _clicks([0.5, 1.5, 2.5], seconds=4.0))
    found, _ = onsets.detect(path)
    assert found == sorted(found)
    assert all(0 <= f <= 4.0 for f in found)


@pytest.mark.parametrize("suffix", [".wav", ".WAV", ".wave", ".Wav"])
def test_wav_is_recognised_whatever_its_case(tmp_path, suffix, monkeypatch):
    """A track named `.WAV` must be read directly and not sent to ffmpeg.

    ffmpeg is removed for the duration, or this passes on any machine that
    has one - which is every machine that renders an MP4, and is why the
    first version of this test could not fail.
    """
    monkeypatch.setattr("rekindle.memory.render.mp4.ffmpeg_path", lambda: None)
    path = _write_wav(tmp_path / f"c{suffix}", _clicks([1.0]))
    found, note = onsets.detect(path)
    assert note == ""
    assert found
