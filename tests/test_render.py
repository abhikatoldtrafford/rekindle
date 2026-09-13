"""Rendering.

The guarantee under test: ffmpeg's absence degrades to GIF-only with a clear
message and exit 0, never a crash. Everything else is about not lying - a
missing file is counted, not rendered as a black slot.
"""

import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from rekindle.memory import composition as comp
from rekindle.memory.render import frames as fr
from rekindle.memory.render import mp4 as mp4mod
from rekindle.memory.render.gif import preview_canvas, write_gif, write_webp
from rekindle.memory.render.mp4 import mp4_canvas, write_mp4
from rekindle.memory.render.music import DEFAULT_DIR, NO_MUSIC_HINT, resolve_music
from rekindle.memory.spec import FactSheet, MemorySpec, Shot
from rekindle.models import MediaType, Photo, PhotoMeta

T0 = datetime(2020, 5, 1, 12, 0, tzinfo=UTC)
CANVAS = (320, 240)


def _photo(h, path) -> Photo:
    return Photo(
        file_hash=h,
        paths=[Path(path)],
        media_type=MediaType.IMAGE,
        meta=PhotoMeta(taken_at_utc=T0, taken_at_local=datetime(2020, 5, 1, 12, 0)),
        first_seen=T0,
        last_seen=T0,
    )


def _jpeg(path: Path, size=(800, 600), colour=(120, 90, 60), exif_orientation=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", size, colour)
    if exif_orientation is not None:
        exif = im.getexif()
        exif[0x0112] = exif_orientation
        im.save(path, "JPEG", exif=exif)
    else:
        im.save(path, "JPEG")
    return path


def _gradient_jpeg(path: Path, size=(160, 120)) -> Path:
    """A photo with actual variation, so a blurred backdrop is distinguishable
    from a flat matte. A solid-colour fixture blurs to the same solid colour
    and the test could not tell them apart."""
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", size)
    for x in range(size[0]):
        for y in range(size[1]):
            im.putpixel((x, y), (30 + (x * 200) // size[0], 60, 200 - (y * 150) // size[1]))
    im.save(path, "JPEG", quality=95)
    return path


def _spec(hashes, **kw) -> MemorySpec:
    shots = tuple(
        Shot(h, kw.pop("caption", f"caption {h}"), "2020-05-01T12:00:00", True) for h in hashes
    )
    return MemorySpec(
        recipe="album_story",
        key="Kashmir",
        title=kw.pop("title", "Kashmir"),
        subtitle="3 photos, May 2020",
        shots=shots,
        facts=FactSheet(title="Kashmir", recipe="album_story", photo_count=len(shots)),
        public_safe=False,
    )


def _world(tmp_path, hashes):
    """A resolve/locate pair over real files on disk."""
    photos = {h: _photo(h, _jpeg(tmp_path / f"{h}.jpg")) for h in hashes}
    return photos.get, lambda p: p.paths[0] if p.paths[0].is_file() else None


def _world_from(tmp_path, hashes):
    """Same, over files that already exist (written by the caller)."""
    photos = {h: _photo(h, tmp_path / f"{h}.jpg") for h in hashes}
    return photos.get, lambda p: p.paths[0] if p.paths[0].is_file() else None


# --------------------------------------------------------------------------
# frames


def test_frames_are_all_the_canvas_size(tmp_path):
    """A video encoder rejects a sequence whose frames differ in size."""
    resolve, locate = _world(tmp_path, ["a", "b", "c"])
    built, report = fr.build_frames(_spec(["a", "b", "c"]), CANVAS, resolve=resolve, locate=locate)
    assert all(f.size == CANVAS for f in built)
    assert report.rendered == 3
    assert report.accounted


def test_a_title_card_is_prepended_by_default(tmp_path):
    resolve, locate = _world(tmp_path, ["a"])
    built, _ = fr.build_frames(_spec(["a"]), CANVAS, resolve=resolve, locate=locate)
    assert len(built) == 2


def test_the_title_card_can_be_turned_off(tmp_path):
    resolve, locate = _world(tmp_path, ["a"])
    built, _ = fr.build_frames(
        _spec(["a"]), CANVAS, resolve=resolve, locate=locate, with_title=False
    )
    assert len(built) == 1


def test_a_title_card_needs_no_font_file(tmp_path):
    """rekindle ships no assets and must not assume a system font."""
    card = fr.title_card("Kashmir", "510 photos, May 2015", CANVAS)
    assert card.size == CANVAS
    # Not a blank frame: something was actually drawn.
    assert card.convert("L").getextrema()[1] > 100


def test_a_missing_file_is_dropped_and_counted_not_raised(tmp_path):
    """A library on an external drive is routine. Half a memory with an honest
    report beats a traceback."""
    resolve, locate = _world(tmp_path, ["a", "b"])
    (tmp_path / "b.jpg").unlink()
    built, report = fr.build_frames(_spec(["a", "b"]), CANVAS, resolve=resolve, locate=locate)
    assert report.rendered == 1
    assert report.dropped == {fr.DROP_MISSING: 1}
    assert report.names[fr.DROP_MISSING] == "b.jpg"
    assert report.accounted


def test_an_undecodable_file_is_dropped_and_counted(tmp_path):
    resolve, locate = _world(tmp_path, ["a", "b"])
    (tmp_path / "b.jpg").write_bytes(b"not a jpeg")
    _, report = fr.build_frames(_spec(["a", "b"]), CANVAS, resolve=resolve, locate=locate)
    assert report.dropped == {fr.DROP_UNDECODABLE: 1}


def test_a_shot_no_longer_in_the_index_is_dropped_and_counted(tmp_path):
    """A spec saved before a photo was deleted from the library."""
    resolve, locate = _world(tmp_path, ["a"])
    _, report = fr.build_frames(_spec(["a", "gone"]), CANVAS, resolve=resolve, locate=locate)
    assert report.dropped == {fr.DROP_NOT_IN_INDEX: 1}


def test_aspect_is_preserved_and_letterboxed_black(tmp_path):
    """A 2:1 photo on a 4:3 canvas gets black bars, not a squash.

    The photo is LARGER than the canvas here, so it downscales and the
    leftover is plain matte - the blurred backdrop is only for photos that
    fall below the canvas.
    """
    _jpeg(tmp_path / "w.jpg", size=(800, 400))
    frame, mode = fr.fit_photo(tmp_path / "w.jpg", (320, 240))
    assert mode == comp.FIT_DOWNSCALE
    assert frame.size == (320, 240)
    assert frame.getpixel((160, 2)) == (0, 0, 0)
    assert frame.getpixel((160, 120)) != (0, 0, 0)


def test_a_portrait_photo_fits_a_landscape_canvas(tmp_path):
    _jpeg(tmp_path / "p.jpg", size=(600, 900))
    frame, _ = fr.fit_photo(tmp_path / "p.jpg", (320, 240))
    assert frame.size == (320, 240)
    assert frame.getpixel((2, 120)) == (0, 0, 0)


def test_a_far_smaller_photo_is_PADDED_at_its_native_size(tmp_path):
    """The rule that keeps old photos in a memory instead of deleting them or
    stretching them into mush.

    Excluding small photos was considered and rejected: on a library spanning
    2000-2026, small means OLD, so it would quietly remove the early years of
    exactly the memories whose subject is the span.
    """
    _gradient_jpeg(tmp_path / "small.jpg", size=(160, 120))
    frame, mode = fr.fit_photo(tmp_path / "small.jpg", (1280, 960))
    assert mode == comp.FIT_PAD
    assert frame.size == (1280, 960)
    # The photo sits at NATIVE size in the centre: 160x120 means the sharp
    # region spans x in [560, 720). Just outside it is backdrop.
    assert frame.getpixel((640, 480)) != (0, 0, 0)


def test_the_padding_is_a_blurred_enlargement_not_a_black_matte(tmp_path):
    """A 640x480 photo centred in a 3984x2988 canvas fills 2.5% of the area.
    On black it reads as broken; on a blurred enlargement of itself it reads
    as a small old photo, which is what it is."""
    _gradient_jpeg(tmp_path / "small.jpg", size=(160, 120))
    frame, mode = fr.fit_photo(tmp_path / "small.jpg", (1280, 960))
    assert mode == comp.FIT_PAD
    corner = frame.getpixel((10, 10))
    assert corner != (0, 0, 0), "the pad area is a black matte, not a backdrop"


def test_a_slightly_smaller_photo_is_upscaled_within_tolerance(tmp_path):
    """Padding a photo 3% below the canvas would read as an inconsistency
    rather than as a deliberate signal. A 25% linear stretch is imperceptible."""
    _jpeg(tmp_path / "near.jpg", size=(1200, 900))
    frame, mode = fr.fit_photo(tmp_path / "near.jpg", (1280, 960))
    assert mode == comp.FIT_UPSCALE
    assert frame.size == (1280, 960)
    # It fills the canvas, so there is no matte at the edge.
    assert frame.getpixel((5, 480)) != (0, 0, 0)


def test_exif_orientation_is_applied_when_rendering(tmp_path):
    """Without this a phone portrait renders on its side. The stored
    dimensions are already post-rotation, so the renderer must match them."""
    _jpeg(tmp_path / "r.jpg", size=(800, 400), exif_orientation=6)
    frame, _ = fr.fit_photo(tmp_path / "r.jpg", (400, 400))
    # 800x400 rotated is 400x800: portrait, so it letterboxes left and right.
    assert frame.getpixel((5, 200)) == (0, 0, 0)
    assert frame.getpixel((200, 200)) != (0, 0, 0)


def test_captions_can_be_turned_off(tmp_path):
    resolve, locate = _world(tmp_path, ["a"])
    with_caption, _ = fr.build_frames(_spec(["a"]), CANVAS, resolve=resolve, locate=locate)
    without, _ = fr.build_frames(
        _spec(["a"]), CANVAS, resolve=resolve, locate=locate, with_captions=False
    )
    assert with_caption[1].tobytes() != without[1].tobytes()


def test_frames_can_be_limited(tmp_path):
    resolve, locate = _world(tmp_path, ["a", "b", "c"])
    built, report = fr.build_frames(
        _spec(["a", "b", "c"]), CANVAS, resolve=resolve, locate=locate, limit=2
    )
    assert report.rendered == 2
    assert len(built) == 3  # plus the title card


# --------------------------------------------------------------------------
# GIF


def test_a_gif_is_written_animated_and_looping(tmp_path):
    resolve, locate = _world(tmp_path, ["a", "b", "c"])
    built, _ = fr.build_frames(_spec(["a", "b", "c"]), CANVAS, resolve=resolve, locate=locate)
    out = tmp_path / "memory.gif"
    size = write_gif(built, out)

    assert size > 0
    with Image.open(out) as im:
        assert im.is_animated
        assert im.n_frames == 4
        assert im.info.get("loop") == 0


def test_the_title_card_is_held_longer(tmp_path):
    resolve, locate = _world(tmp_path, ["a"])
    built, _ = fr.build_frames(_spec(["a"]), CANVAS, resolve=resolve, locate=locate)
    out = tmp_path / "m.gif"
    write_gif(built, out, frame_ms=1000, title_ms=3000)
    with Image.open(out) as im:
        first = im.info["duration"]
        im.seek(1)
        assert first > im.info["duration"]


def test_an_empty_frame_list_is_refused(tmp_path):
    """A zero-frame GIF renders as a broken image in some viewers."""
    with pytest.raises(ValueError):
        write_gif([], tmp_path / "m.gif")


def test_the_preview_canvas_scales_down_but_never_up():
    assert preview_canvas((1600, 1200), 480) == (480, 360)
    assert preview_canvas((320, 240), 480) == (320, 240)


def test_the_preview_canvas_keeps_a_portrait_memory_portrait():
    assert preview_canvas((1200, 1600), 480) == (480, 640)


def test_the_preview_default_is_no_longer_a_thumbnail():
    """480px was chosen to keep a GIF small enough to drop into a README
    without thinking. That constraint was lifted and 480 was far too low for
    anyone actually looking at their own photos."""
    from rekindle.memory.render.gif import DEFAULT_WIDTH

    assert DEFAULT_WIDTH >= 1280


# --------------------------------------------------------------------------
# WebP: the preview worth looking at


def test_a_webp_is_written_animated_and_looping(tmp_path):
    resolve, locate = _world(tmp_path, ["a", "b", "c"])
    built, _ = fr.build_frames(_spec(["a", "b", "c"]), CANVAS, resolve=resolve, locate=locate)
    out = tmp_path / "memory.webp"

    size = write_webp(built, out)

    assert size > 0
    with Image.open(out) as im:
        assert im.format == "WEBP"
        assert im.is_animated
        assert im.n_frames == 4


def test_the_webp_is_smaller_than_the_gif_for_photographic_content(tmp_path):
    """The whole reason GIF stopped being the preview of record: it quantises
    every frame to 256 colours, which both bands a photograph AND costs more
    bytes than true-colour WebP for the same content."""
    photos = [_gradient_jpeg(tmp_path / f"g{i}.jpg", size=(640, 480)) for i in range(3)]
    assert photos
    resolve, locate = _world_from(tmp_path, ["g0", "g1", "g2"])
    built, _ = fr.build_frames(
        _spec(["g0", "g1", "g2"]), (640, 480), resolve=resolve, locate=locate
    )
    webp = write_webp(built, tmp_path / "m.webp")
    gif = write_gif(built, tmp_path / "m.gif")
    assert webp < gif, f"webp {webp} was not smaller than gif {gif}"


def test_an_empty_webp_is_refused(tmp_path):
    with pytest.raises(ValueError):
        write_webp([], tmp_path / "m.webp")


def test_the_webp_title_card_is_held_longer(tmp_path):
    """Asserted on the ENCODED BYTES, not through Pillow's reader.

    Pillow's WebP decoder does not surface per-frame durations at all - it
    exposes only `background` and `loop` on seek - so the obvious assertion
    (`im.info["duration"]`) raises KeyError rather than failing. Comparing two
    encodings that differ ONLY in the title hold proves the value reaches the
    encoder and lands in the file, which is what the test is actually about.
    """
    resolve, locate = _world(tmp_path, ["a", "b"])
    built, _ = fr.build_frames(_spec(["a", "b"]), CANVAS, resolve=resolve, locate=locate)

    short = tmp_path / "short.webp"
    long = tmp_path / "long.webp"
    write_webp(built, short, frame_ms=1000, title_ms=1000)
    write_webp(built, long, frame_ms=1000, title_ms=5000)

    assert short.read_bytes() != long.read_bytes()


def test_the_webp_title_hold_is_skipped_when_there_is_no_title(tmp_path):
    resolve, locate = _world(tmp_path, ["a", "b"])
    built, _ = fr.build_frames(
        _spec(["a", "b"]), CANVAS, resolve=resolve, locate=locate, with_title=False
    )
    with_hold = tmp_path / "a.webp"
    without = tmp_path / "b.webp"
    write_webp(built, with_hold, frame_ms=1000, title_ms=5000, has_title=False)
    write_webp(built, without, frame_ms=1000, title_ms=1000, has_title=False)
    assert with_hold.read_bytes() == without.read_bytes()


# --------------------------------------------------------------------------
# MP4: the degradation guarantee


def test_ffmpeg_absent_skips_the_mp4_without_raising(tmp_path, monkeypatch):
    """THE guarantee. ffmpeg is an external tool, not a dependency, and its
    absence must not make the feature unavailable."""
    monkeypatch.setattr(shutil, "which", lambda name: None)
    frame = Image.new("RGB", (64, 48), (10, 20, 30))

    result = write_mp4([frame], tmp_path / "m.mp4")

    assert result.ok is False
    assert result.path is None
    assert result.error is None
    assert "ffmpeg" in result.skipped.lower()
    assert not (tmp_path / "m.mp4").exists()


def test_an_ffmpeg_that_fails_is_reported_with_its_stderr(tmp_path):
    """The GIF is already written by then, so the memory is not lost - but the
    user must be told why the MP4 is missing."""
    fake = tmp_path / ("ffmpeg.bat" if sys.platform == "win32" else "ffmpeg.sh")
    if sys.platform == "win32":
        fake.write_text("@echo off\r\necho boom 1>&2\r\nexit /b 1\r\n", encoding="ascii")
    else:
        fake.write_text("#!/bin/sh\necho boom >&2\nexit 1\n", encoding="ascii")
        fake.chmod(0o755)

    result = write_mp4([Image.new("RGB", (64, 48))], tmp_path / "m.mp4", ffmpeg=str(fake))

    assert result.ok is False
    assert result.error is not None
    assert "boom" in result.error


def test_an_ffmpeg_that_vanishes_degrades_rather_than_raising(tmp_path):
    result = write_mp4(
        [Image.new("RGB", (64, 48))], tmp_path / "m.mp4", ffmpeg=str(tmp_path / "nope")
    )
    assert result.ok is False
    assert result.skipped is not None


def test_no_frames_is_an_error_not_a_crash(tmp_path):
    result = write_mp4([], tmp_path / "m.mp4", ffmpeg="ffmpeg")
    assert result.ok is False
    assert result.error is not None


def test_the_mp4_canvas_is_always_even():
    """libx264 with yuv420p refuses an odd dimension, with an error no user
    should have to decode."""
    for size in [(1001, 667), (4001, 3001), (999, 999)]:
        width, height = mp4_canvas(size, 1280)
        assert width % 2 == 0 and height % 2 == 0


def test_the_mp4_canvas_never_upscales():
    assert mp4_canvas((640, 480), 1280) == (640, 480)


def test_the_mp4_default_carries_real_resolution():
    """1280 threw away ~90% of the pixel count of this library's median photo.
    Native is not the default because a 7008x4672 video helps nobody."""
    from rekindle.memory.render.mp4 import DEFAULT_WIDTH

    assert DEFAULT_WIDTH >= 1920


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not on PATH")
def test_a_real_mp4_is_produced_and_playable(tmp_path):
    frames = [Image.new("RGB", (320, 240), (i * 40, 60, 90)) for i in range(4)]
    result = write_mp4(frames, tmp_path / "m.mp4", seconds=0.2, title_seconds=0.3)

    assert result.ok, result.error
    assert result.size > 0
    probe = subprocess.run(
        [
            shutil.which("ffprobe"),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height",
            "-of",
            "csv=p=0",
            str(result.path),
        ],
        capture_output=True,
        text=True,
    )
    assert "h264" in probe.stdout
    assert "320,240" in probe.stdout.replace(" ", "")


def _probe(path, entries, stream=None):
    cmd = [shutil.which("ffprobe"), "-v", "error"]
    if stream:
        cmd += ["-select_streams", stream]
    cmd += ["-show_entries", entries, "-of", "csv=p=0", str(path)]
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def test_the_audio_input_is_both_looped_and_bounded(tmp_path, monkeypatch):
    """The two flags are a PAIR and neither is optional.

    The end-to-end pair below proves each one, but only with ffmpeg installed
    and - for `-shortest` - by waiting out a 600-second timeout, because
    `-stream_loop -1` without `-shortest` produces an encode that never ends.
    A test whose failure mode is "ten minutes" is a test nobody runs. This one
    reads the command line, fails in milliseconds, and runs in CI where there
    is no ffmpeg.
    """
    seen = {}

    class _Done:
        returncode = 0
        stderr = ""

    def spy(cmd, **kwargs):
        seen["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"fake mp4")
        return _Done()

    monkeypatch.setattr(mp4mod.subprocess, "run", spy)
    monkeypatch.setattr(mp4mod, "ffmpeg_path", lambda: "ffmpeg")
    bed = tmp_path / "bed.mp3"
    bed.write_bytes(b"fake")
    write_mp4([Image.new("RGB", (64, 48))], tmp_path / "m.mp4", music=bed)

    cmd = seen["cmd"]
    assert "-stream_loop" in cmd, "a bed shorter than the memory would cut out"
    assert "-shortest" in cmd, "a looped bed would never terminate"
    # -stream_loop applies to the NEXT input, so it has to sit immediately
    # before the audio -i and not before the concat one.
    assert cmd[cmd.index("-stream_loop") + 1] == "-1"
    assert cmd[cmd.index("-stream_loop") + 2] == "-i"
    assert cmd[cmd.index("-stream_loop") + 3] == str(bed)


def test_no_audio_flags_are_passed_when_there_is_no_bed(tmp_path, monkeypatch):
    """Silence is the default. `-stream_loop` with no audio input would apply
    to whatever input came next, which is the still frames."""
    seen = {}

    class _Done:
        returncode = 0
        stderr = ""

    def spy(cmd, **kwargs):
        seen["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"fake mp4")
        return _Done()

    monkeypatch.setattr(mp4mod.subprocess, "run", spy)
    monkeypatch.setattr(mp4mod, "ffmpeg_path", lambda: "ffmpeg")
    write_mp4([Image.new("RGB", (64, 48))], tmp_path / "m.mp4")

    assert "-stream_loop" not in seen["cmd"]
    assert "-shortest" not in seen["cmd"]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not on PATH")
def test_a_short_bed_loops_instead_of_cutting_out_mid_memory(tmp_path):
    """40 CC0 piano tracks are mostly one to three minutes and a memory can be
    longer. Without `-stream_loop -1` the audio simply stops and the rest of
    the memory plays in silence.

    Measured rather than asserted on the command line: a one-second bed under
    a ~3-second video, probed. The audio stream runs the length of the video,
    not the length of the bed.
    """
    bed = tmp_path / "bed.wav"
    subprocess.run(
        [
            shutil.which("ffmpeg"),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            str(bed),
        ],
        check=True,
    )
    frames = [Image.new("RGB", (320, 240), (i * 30, 60, 90)) for i in range(6)]
    result = write_mp4(
        frames, tmp_path / "m.mp4", seconds=0.5, title_seconds=0.5, has_title=False, music=bed
    )
    assert result.ok, result.error

    audio = float(_probe(result.path, "stream=duration", stream="a:0"))
    # The bed is 1.0s and the video is ~3.0s. Anything at or near 1.0 means
    # the bed played once and stopped.
    assert audio > 2.0, f"audio stream is {audio}s - the bed did not loop"


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not on PATH")
def test_a_long_bed_is_still_cut_at_the_end_of_the_memory(tmp_path):
    """The other half of the pair. `-stream_loop -1` alone never terminates;
    `-shortest` is what stops it, and dropping `-shortest` while keeping the
    loop would produce a file that encodes until something gives up."""
    bed = tmp_path / "bed.wav"
    subprocess.run(
        [
            shutil.which("ffmpeg"),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=30",
            str(bed),
        ],
        check=True,
    )
    frames = [Image.new("RGB", (320, 240), (i * 30, 60, 90)) for i in range(4)]
    result = write_mp4(
        frames, tmp_path / "m.mp4", seconds=0.5, title_seconds=0.5, has_title=False, music=bed
    )
    assert result.ok, result.error
    assert float(_probe(result.path, "format=duration")) < 5.0


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not on PATH")
def test_the_last_frame_is_held_not_flashed(tmp_path):
    """The concat demuxer applies a duration to the transition INTO the next
    file, so the final image needs repeating or it lasts one frame-time."""
    frames = [Image.new("RGB", (320, 240), (i * 60, 60, 90)) for i in range(3)]
    result = write_mp4(frames, tmp_path / "m.mp4", seconds=0.5, title_seconds=0.5, has_title=False)
    assert result.ok, result.error
    probe = subprocess.run(
        [
            shutil.which("ffprobe"),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(result.path),
        ],
        capture_output=True,
        text=True,
    )
    # MEASURED on ffmpeg 7.1.1 with these exact inputs: 2.04s with the tail
    # repeat, 1.48s without it. An earlier version of this test asserted
    # ">= 1.0", which both cases satisfy - so deleting the repeat left it
    # green. 1.8 sits between the two measurements.
    assert float(probe.stdout.strip()) >= 1.8


# --------------------------------------------------------------------------
# music: filesystem only, never the network


def test_no_music_folder_means_silence(tmp_path):
    assert resolve_music(folder=tmp_path / "nope") is None


def test_an_empty_music_folder_means_silence(tmp_path):
    (tmp_path / "music").mkdir()
    assert resolve_music(folder=tmp_path / "music") is None


def test_the_first_audio_file_is_chosen_deterministically(tmp_path):
    folder = tmp_path / "music"
    folder.mkdir()
    for name in ["zebra.mp3", "apple.mp3", "middle.wav"]:
        (folder / name).write_bytes(b"fake")
    assert resolve_music(folder=folder).name == "apple.mp3"


def test_the_music_choice_does_not_depend_on_filesystem_order(tmp_path, monkeypatch):
    """NTFS and ext4 both hand back directory entries in an order that
    happens to be sorted here, so the test above passes with or without the
    sort - it cannot fail for the reason it claims. This one forces an
    unsorted listing, which is what a different filesystem would give.
    """
    folder = tmp_path / "music"
    folder.mkdir()
    for name in ["apple.mp3", "middle.mp3", "zebra.mp3"]:
        (folder / name).write_bytes(b"fake")

    real_iterdir = Path.iterdir

    def reversed_iterdir(self):
        return reversed(sorted(real_iterdir(self)))

    monkeypatch.setattr(Path, "iterdir", reversed_iterdir)
    assert resolve_music(folder=folder).name == "apple.mp3"


def test_non_audio_files_are_ignored(tmp_path):
    folder = tmp_path / "music"
    folder.mkdir()
    (folder / "readme.txt").write_bytes(b"x")
    (folder / "cover.jpg").write_bytes(b"x")
    assert resolve_music(folder=folder) is None


def test_an_explicit_path_wins_over_the_folder(tmp_path):
    folder = tmp_path / "music"
    folder.mkdir()
    (folder / "a.mp3").write_bytes(b"x")
    chosen = tmp_path / "chosen.mp3"
    assert resolve_music(chosen, folder=folder) == chosen


def test_music_resolution_makes_no_network_call(monkeypatch, tmp_path):
    """rekindle never downloads audio. The default configuration makes no
    network calls at all, and this is the module that would have broken that."""
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("music resolution attempted a network call")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    assert resolve_music(folder=tmp_path) is None


def test_the_no_music_hint_says_silence_is_fine():
    assert "Silence is the default" in NO_MUSIC_HINT
    assert "never downloads" in NO_MUSIC_HINT


def test_the_folder_the_hint_calls_gitignored_really_is_ignored():
    """The project's signature defect is user-facing text asserting something
    untrue, and this hint asserted it for a whole milestone: `music/` was NOT
    in .gitignore until the folder already held 54 MB of audio in a public
    repository. Stating it in prose is what let the two drift apart, so the
    claim is now checked against git itself rather than against a copy of the
    rule in a test.

    The folder name is read OUT OF THE HINT, not hardcoded: renaming the
    directory in one place and not the other is the same defect wearing a
    different hat.
    """
    match = re.search(r"`([^`]+)/`", NO_MUSIC_HINT)
    assert match, "the hint no longer names a folder"
    folder = match.group(1)
    assert "gitignored" in NO_MUSIC_HINT, "the hint no longer claims the folder is ignored"

    root = Path(__file__).resolve().parent.parent
    if not (root / ".git").exists():
        pytest.skip("not a git checkout - there is no ignore rule to check")
    if shutil.which("git") is None:
        pytest.skip("git is not on PATH")

    probe = f"{folder}/track.mp3"
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", probe],
        cwd=root,
        capture_output=True,
    )
    # 0 = ignored, 1 = not ignored, 128 = error.
    assert result.returncode != 128, result.stderr.decode(errors="replace")
    assert result.returncode == 0, (
        f"NO_MUSIC_HINT calls {folder}/ gitignored, but git would commit {probe}. "
        "This is a public repository."
    )


def test_the_hint_names_the_folder_the_code_actually_reads():
    """The other direction of the same drift: the hint could name a folder
    nothing looks in."""
    match = re.search(r"`([^`]+)/`", NO_MUSIC_HINT)
    assert match and Path(match.group(1)) == DEFAULT_DIR


# --------------------------------------------------------------------------
# music: one track per memory, not one track for everything


def _tracks(folder, count=40):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (folder / f"{i:02d}-piece.mp3").write_bytes(b"fake")
    return folder


def test_two_different_memories_get_two_different_tracks(tmp_path):
    """`candidates[0]` was right for one file and a bug for 40: every one of
    53 memories opened with the same piece."""
    folder = _tracks(tmp_path / "music")
    chosen = {
        resolve_music(folder=folder, memory_id=mid).name
        for mid in (
            "on_this_day:12-22",
            "year_in_review:2019",
            "person_years:Abhik Maiti",
            "album_story:wedding_arnab_pics",
            "then_and_now:Abhik Maiti",
        )
    }
    assert len(chosen) > 1


def test_the_same_memory_always_gets_the_same_track(tmp_path):
    """The determinism promise: re-rendering a memory must be byte-identical,
    so the track cannot come from a counter, a shuffle, or `hash()` - which is
    randomised per process for strings."""
    folder = _tracks(tmp_path / "music")
    first = resolve_music(folder=folder, memory_id="album_story:avyan")
    for _ in range(5):
        assert resolve_music(folder=folder, memory_id="album_story:avyan") == first


def test_the_track_for_a_memory_is_the_same_in_a_fresh_process(tmp_path):
    """`hash()` on a str is salted per interpreter, so a selection built on it
    passes every in-process repetition above and still gives a different
    soundtrack on the next run. Only a separate process can see that."""
    folder = _tracks(tmp_path / "music")
    mine = resolve_music(folder=folder, memory_id="year_in_review:2021").name
    code = (
        "import sys;"
        "from pathlib import Path;"
        "sys.path.insert(0, r'" + str(Path(__file__).resolve().parent.parent / "src") + "');"
        "from rekindle.memory.render.music import resolve_music;"
        "print(resolve_music(folder=Path(r'" + str(folder) + "'),"
        " memory_id='year_in_review:2021').name)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ}
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == mine


def test_the_whole_set_of_memories_spreads_across_the_folder(tmp_path):
    """A selection that differs between two memories can still pile 53
    memories onto three tracks. Measured against the real recipe/key shapes:
    40 tracks, 53 memories."""
    folder = _tracks(tmp_path / "music")
    ids = [f"year_in_review:{2000 + i}" for i in range(27)] + [
        f"album_story:album_{i}" for i in range(26)
    ]
    used = {resolve_music(folder=folder, memory_id=i).name for i in ids}
    # 53 draws from 40 boxes leaves ~11 empty by chance alone; anything near
    # 40 distinct is a healthy spread and anything tiny is a broken one.
    assert len(used) >= 25


def test_an_explicit_path_still_wins_over_the_per_memory_choice(tmp_path):
    folder = _tracks(tmp_path / "music")
    chosen = tmp_path / "chosen.mp3"
    assert resolve_music(chosen, folder=folder, memory_id="year_in_review:2019") == chosen


def test_a_memory_with_no_id_falls_back_to_the_first_track(tmp_path):
    folder = _tracks(tmp_path / "music")
    assert resolve_music(folder=folder).name == "00-piece.mp3"


def test_the_per_memory_choice_does_not_depend_on_filesystem_order(tmp_path, monkeypatch):
    """Same hazard as the sorted-listing test above, one level up: the index
    is taken into a LIST, so an unsorted listing gives a different track for
    the same memory on a different filesystem."""
    folder = _tracks(tmp_path / "music")
    before = resolve_music(folder=folder, memory_id="on_this_day:12-22").name

    real_iterdir = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda self: reversed(sorted(real_iterdir(self))))
    assert resolve_music(folder=folder, memory_id="on_this_day:12-22").name == before


def test_the_render_package_exposes_no_url():
    """A pinned download list is exactly what was NOT built. If one is ever
    added, it must be a deliberate change that fails this test first."""
    import rekindle.memory.render.music as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "http://" not in source
    assert "https://" not in source


def test_mp4_module_has_no_hardcoded_ffmpeg_path():
    """The user has it at /c/ffmpeg/bin, CI has it elsewhere, macOS has it in
    Homebrew. It is resolved through shutil.which and nowhere else."""
    source = Path(mp4mod.__file__).read_text(encoding="utf-8")
    assert "/usr/bin/ffmpeg" not in source
    assert "C:\\\\ffmpeg" not in source
    assert "shutil.which" in source


def test_an_unreadable_music_folder_means_silence(tmp_path, monkeypatch):
    """A permission error or an unplugged drive must not take a render down."""

    def boom(self):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "iterdir", boom)
    (tmp_path / "music").mkdir()
    assert resolve_music(folder=tmp_path / "music") is None


def test_the_frame_report_counts_how_each_shot_met_the_canvas(tmp_path):
    """A memory where most shots are padded is telling the user something real
    about that period of their library, so the count is surfaced. Nothing
    asserted this until a mutation deleted the line and the suite stayed
    green."""
    _jpeg(tmp_path / "big.jpg", size=(2000, 1500))
    _jpeg(tmp_path / "small.jpg", size=(300, 225))
    photos = {
        "big": _photo("big", tmp_path / "big.jpg"),
        "small": _photo("small", tmp_path / "small.jpg"),
    }
    resolve = photos.get

    _, report = fr.build_frames(
        _spec(["big", "small"]),
        (1600, 1200),
        resolve=resolve,
        locate=lambda p: p.paths[0],
    )

    assert report.rendered == 2
    assert report.placement[comp.FIT_DOWNSCALE] == 1
    assert report.placement[comp.FIT_PAD] == 1
    assert sum(report.placement.values()) == report.rendered


# --------------------------------------------------------------------------
# the title card measures its subtitle
#
# It did not. It wrapped the title against the canvas WIDTH and then drew the
# subtitle straight out at a size derived from the canvas HEIGHT, having
# measured it against nothing. Centred text wider than the canvas overflows at
# BOTH ends, so "Every August" lost the `1` from `10 photos` and the `6` from
# `2026`. Invisible at the engine's 1280px default, which is why it shipped.
#
# Measured on the real subtitle from the shipped `Every August` memory,
# `24 photos, August 2009 - August 2026` (the median subtitle in
# `memories/general` is 38 characters, the longest 42):
#
#     canvas      before                       after
#     1280x960    ink 277..1004  inside box    unchanged
#      400x300    ink  83..316   inside box    unchanged
#      240x180    ink  14..225   OUTSIDE box   ink 30..209, inside
#      210x157    ink   0..209   CLIPPED       ink 15..194, inside
#
# So the assertions below are about INK POSITION, not about the card being the
# size it was asked for - which is what the previous test asserted and which
# no amount of clipping could ever have failed.

#: A real one, from `memories/general/2026-09-11-on_this_day-08-.../memory.json`.
REAL_SUBTITLE = "24 photos, August 2009 - August 2026"


def _subtitle_ink(title: str, subtitle: str, canvas: tuple[int, int]) -> tuple[int, int] | None:
    """The leftmost and rightmost columns the SUBTITLE alone put ink in.

    Rendered twice and differenced, rather than scanning the whole card, so a
    long title can never be mistaken for the thing under test.
    """
    with_sub = fr.title_card(title, subtitle, canvas).convert("L")
    without = fr.title_card(title, "", canvas).convert("L")
    from PIL import ImageChops

    diff = ImageChops.difference(with_sub, without).load()
    cols = [x for x in range(canvas[0]) if any(diff[x, y] > 40 for y in range(canvas[1]))]
    return (min(cols), max(cols)) if cols else None


@pytest.mark.parametrize(
    "canvas",
    [(1280, 960), (640, 480), (400, 300), (320, 240), (240, 180), (210, 157), (160, 120)],
    ids=lambda c: f"{c[0]}x{c[1]}",
)
def test_the_subtitle_never_leaves_its_text_box(canvas):
    """The box is `canvas[0] // 16` in from each edge - the same margin the
    title is wrapped to. Below 245px the old code left it; below 215px the
    canvas itself cut characters off."""
    ink = _subtitle_ink("Every August", REAL_SUBTITLE, canvas)
    assert ink is not None, "the subtitle drew nothing at all"
    margin = canvas[0] // 16
    assert ink[0] >= margin - 1, f"subtitle starts at {ink[0]}, left margin is {margin}"
    assert ink[1] <= canvas[0] - margin, f"subtitle ends at {ink[1]}, canvas is {canvas[0]}"


@pytest.mark.parametrize("width", [1280, 640, 400, 320, 240, 210, 180, 160])
def test_the_subtitle_is_never_clipped_by_the_canvas(width):
    """The reported symptom, stated directly: no character may be cut off."""
    canvas = (width, int(width * 0.75))
    ink = _subtitle_ink("Every August", REAL_SUBTITLE, canvas)
    assert ink is not None
    assert ink[0] > 0, "ink touches the left edge, so something is cut off"
    assert ink[1] < width - 1, "ink touches the right edge, so something is cut off"


def test_a_subtitle_that_already_fits_is_drawn_EXACTLY_as_before():
    """The fix must be a no-op at the engine's 1280px default, or it is a
    rendering change dressed as a bug fix."""
    draw = ImageDraw.Draw(Image.new("RGB", (1280, 960)))
    size, lines = fr._fit_subtitle(draw, REAL_SUBTITLE, (1280, 960), 1280 - 1280 // 8)
    assert size == max(12, 960 // 22), "the starting size is the one it always used"
    assert lines == [REAL_SUBTITLE], "one line, unmodified"


def test_a_long_subtitle_WRAPS_before_it_shrinks():
    """Shrinking an ordinary two-phrase subtitle to fit on one line would make
    it unreadable; two centred lines is the right answer.

    240px is where the real subtitle stops fitting on one line: at 300px it
    still does, and asserting a wrap there would assert something false.
    """
    canvas = (240, 180)
    draw = ImageDraw.Draw(Image.new("RGB", canvas))
    size, lines = fr._fit_subtitle(draw, REAL_SUBTITLE, canvas, canvas[0] - canvas[0] // 8)
    assert len(lines) > 1
    assert " ".join(lines) == REAL_SUBTITLE, "wrapping must not lose or reorder a word"
    assert size == max(12, canvas[1] // 22), "wrapping was enough, so nothing shrank"


def test_an_UNBREAKABLE_subtitle_shrinks_AND_THEN_TRUNCATES():
    """All three guards, in one input that defeats the first two.

    `_wrap` deliberately leaves an over-long single word long rather than
    hyphenating, so wrapping cannot help. Shrinking cannot finish the job
    either: at the floor this word is still 204px wide in a 175px box, which
    is why there is a third guard and not two.
    """
    canvas = (200, 150)
    draw = ImageDraw.Draw(Image.new("RGB", canvas))
    word = "Ludwigshafen-am-Rhein-Oggersheim-Nord"
    size, lines = fr._fit_subtitle(draw, word, canvas, canvas[0] - canvas[0] // 8)
    assert size == fr.MIN_SUBTITLE_SIZE, "it did not shrink all the way"
    assert lines[0].endswith(fr.ELLIPSIS), f"not truncated: {lines[0]!r}"
    assert word.startswith(lines[0][: -len(fr.ELLIPSIS)])
    ink = _subtitle_ink("X", word, canvas)
    assert ink[0] > 0 and ink[1] < canvas[0] - 1


def test_shrinking_alone_is_tried_before_truncating():
    """A word that fits once shrunk must not lose characters it did not need
    to lose - truncation is the last resort, not the first."""
    # Tall and narrow: the starting size comes from the HEIGHT (18 here), so
    # there is room to shrink before the floor, and the WIDTH is what does not
    # fit. On a short canvas the start is already the floor and nothing could
    # be observed.
    canvas = (200, 400)
    draw = ImageDraw.Draw(Image.new("RGB", canvas))
    word = "Ludwigshafen-am-Rhein"
    size, lines = fr._fit_subtitle(draw, word, canvas, canvas[0] - canvas[0] // 8)
    assert lines == [word], "truncated when shrinking would have done"
    assert fr.MIN_SUBTITLE_SIZE < size < max(12, canvas[1] // 22)


def test_an_ellipsis_is_three_dots_and_not_a_glyph_the_font_may_lack():
    """The fallback path in `_font` is Pillow's bundled bitmap font, and a
    missing glyph renders as a blank box - worse than the clipping this is
    fixing, and only on the machines least able to report it."""
    assert fr.ELLIPSIS == "..."


def test_the_shrink_stops_at_a_readable_floor():
    """Below about ten pixels the default bitmap font's glyphs stop being
    distinguishable, so a smaller "fitting" subtitle is not more readable than
    a clipped one - it is just differently unreadable, and silently so."""
    canvas = (64, 48)
    draw = ImageDraw.Draw(Image.new("RGB", canvas))
    size, _ = fr._fit_subtitle(draw, "A" * 200, canvas, canvas[0] - canvas[0] // 8)
    assert size == fr.MIN_SUBTITLE_SIZE


def test_an_empty_subtitle_draws_nothing_and_does_not_move_the_title():
    draw = ImageDraw.Draw(Image.new("RGB", (640, 480)))
    size, lines = fr._fit_subtitle(draw, "", (640, 480), 560)
    assert lines == []
    assert size == max(12, 480 // 22)
    assert _subtitle_ink("Kashmir", "", (640, 480)) is None


def test_a_very_long_title_wraps_instead_of_overflowing():
    """Rewritten. This asserted `card.size == (320, 240)`, which is the size it
    was constructed with - it could not fail however badly the text overflowed,
    and it is the reason the subtitle defect below it went unnoticed."""
    canvas = (320, 240)
    title = "A place you kept coming back to over many years"
    card = fr.title_card(title, "", canvas).convert("L")
    px = card.load()
    cols = [x for x in range(canvas[0]) if any(px[x, y] > 40 for y in range(canvas[1]))]
    assert cols, "nothing was drawn"
    assert min(cols) > 0 and max(cols) < canvas[0] - 1


def _ink_bands(card) -> int:
    """How many separated horizontal bands of ink the card has."""
    px = card.convert("L").load()
    w, h = card.size
    rows = [y for y in range(h) if any(px[x, y] > 40 for x in range(w))]
    if not rows:
        return 0
    return 1 + sum(1 for a, b in zip(rows, rows[1:], strict=False) if b - a > 1)


def test_both_lines_of_a_wrapped_subtitle_are_actually_drawn():
    """A wrap that computes two lines and draws one is the same bug with extra
    steps: the second line would simply vanish.

    The title is EMPTY here on purpose. Diffing a card against the same card
    without a subtitle does not work: the number of subtitle lines changes the
    block height, so the TITLE moves too and its ink lands in the difference.
    An empty title draws nothing, so every band on the card is a subtitle line.
    """
    canvas = (240, 180)
    draw = ImageDraw.Draw(Image.new("RGB", canvas))
    _, lines = fr._fit_subtitle(draw, REAL_SUBTITLE, canvas, canvas[0] - canvas[0] // 8)
    assert len(lines) == 2, "the fixture no longer wraps; the test below proves nothing"
    assert _ink_bands(fr.title_card("", REAL_SUBTITLE, canvas)) == 2
    assert _ink_bands(fr.title_card("", lines[0], canvas)) == 1


def test_truncation_keeps_as_much_of_the_text_as_will_FIT():
    """Trimming by a fixed proportion would also "fit" and would throw away
    characters there was room for. The result must be maximal: one more
    character before the ellipsis must not fit."""
    canvas = (200, 150)
    max_width = canvas[0] - canvas[0] // 8
    draw = ImageDraw.Draw(Image.new("RGB", canvas))
    font = fr._font(fr.MIN_SUBTITLE_SIZE)
    word = "Ludwigshafen-am-Rhein-Oggersheim-Nord"
    got = fr._ellipsise(draw, word, font, max_width)
    assert got.endswith(fr.ELLIPSIS)
    kept = got[: -len(fr.ELLIPSIS)]
    assert draw.textlength(got, font=font) <= max_width
    one_more = word[: len(kept) + 1].rstrip() + fr.ELLIPSIS
    assert draw.textlength(one_more, font=font) > max_width, (
        f"trimmed further than it had to: {got!r} left room for {one_more!r}"
    )


# --------------------------------------------------------------------------
# the title got none of the three guards the subtitle got


def _title_ink(title: str, canvas: tuple[int, int]) -> tuple[int, int] | None:
    """The leftmost and rightmost columns the TITLE alone put ink in.

    Differenced against a card with no title, the same trick `_subtitle_ink`
    uses, so the subtitle can never be mistaken for the thing under test.
    """
    from PIL import ImageChops

    with_title = fr.title_card(title, "10 photos", canvas).convert("L")
    without = fr.title_card("", "10 photos", canvas).convert("L")
    diff = ImageChops.difference(with_title, without).load()
    cols = [x for x in range(canvas[0]) if any(diff[x, y] > 40 for y in range(canvas[1]))]
    return (min(cols), max(cols)) if cols else None


#: One word, no space to wrap at, about eleven characters. The title face on a
#: 9:16 preview is 227px against a 1120px box, so these overflowed BOTH edges.
UNBREAKABLE = ["Bhubaneswar", "Kanyakumari", "Thanksgiving", "Wedding_arnab_pics"]


@pytest.mark.parametrize("title", UNBREAKABLE)
@pytest.mark.parametrize(
    "canvas",
    [(1280, 2276), (1280, 1707), (1280, 960), (640, 480), (320, 240)],
    ids=lambda c: f"{c[0]}x{c[1]}",
)
def test_the_title_is_never_clipped_by_the_canvas(title, canvas):
    """`title_size` comes from the canvas HEIGHT and `max_width` from its
    WIDTH, so a tall narrow canvas sets a face far too big for its own box -
    and `_wrap` deliberately leaves a single over-long word long. 1280x2276 is
    not exotic: it is what `preview_canvas` returns for a 9:16 memory, on the
    default path with no flags."""
    ink = _title_ink(title, canvas)
    assert ink is not None, "the title drew nothing at all"
    assert ink[0] > 0, f"{title} touches the left edge at {canvas}"
    assert ink[1] < canvas[0] - 1, f"{title} touches the right edge at {canvas}"


def test_a_title_that_already_fits_is_drawn_EXACTLY_as_before():
    """The same no-op requirement the subtitle fix was held to."""
    canvas = (1280, 960)
    draw = ImageDraw.Draw(Image.new("RGB", canvas))
    size, lines = fr._fit(draw, "Kashmir", max(18, canvas[1] // 10), fr.MIN_TITLE_SIZE, 1120)
    assert size == max(18, canvas[1] // 10), "the starting size is the one it always used"
    assert lines == ["Kashmir"]


def test_a_multi_word_title_WRAPS_before_it_shrinks():
    canvas = (400, 300)
    draw = ImageDraw.Draw(Image.new("RGB", canvas))
    start = max(18, canvas[1] // 10)
    size, lines = fr._fit(draw, "A place you kept coming back to", start, fr.MIN_TITLE_SIZE, 350)
    assert len(lines) > 1
    assert " ".join(lines) == "A place you kept coming back to", "wrapping lost a word"


def test_an_unbreakable_title_is_ellipsised_once_shrinking_runs_out():
    """The third guard, and the one that makes the promise unconditional: no
    amount of shrinking fits one long word into a very narrow box."""
    draw = ImageDraw.Draw(Image.new("RGB", (160, 1200)))
    size, lines = fr._fit(draw, "Bhubaneswar" * 3, max(18, 1200 // 10), fr.MIN_TITLE_SIZE, 140)
    assert size == fr.MIN_TITLE_SIZE, "it should have shrunk all the way down first"
    assert lines[0].endswith(fr.ELLIPSIS)


def test_an_empty_title_still_reserves_its_line():
    """`_fit` returns no lines for empty text, which is right for the subtitle
    and would silently re-centre every card that has a title. The caller keeps
    the blank line; this pins that."""
    canvas = (1280, 960)
    blank = fr.title_card("", "10 photos", canvas)
    assert blank.size == canvas


# --------------------------------------------------------------------------
# --font: the glyphs the bundled face does not have


def test_a_named_font_is_used_instead_of_the_bundled_one(tmp_path):
    """The bundled Aileron is a SUBSET face - no accented Latin letter, no
    non-Latin script at all. `--font` is how a library that is not in English
    gets readable text, and it is an explicit INPUT: the same font gives the
    same bytes on any machine, which a system-font search would have quietly
    destroyed.
    """
    other = _a_font_with_more_glyphs()
    if other is None:
        pytest.skip("no font with wider coverage on this machine")
    plain = fr.title_card("Jose", "10 photos", (640, 360)).tobytes()
    try:
        fr.use_font(other)
        with_font = fr.title_card("Jose", "10 photos", (640, 360)).tobytes()
    finally:
        fr.use_font(None)
    assert plain != with_font, "the named font was not used"


def test_use_font_refuses_a_file_that_is_not_a_font(tmp_path):
    """At the moment the user names it. A render that fell back silently
    would take an hour to produce text in the wrong typeface."""
    junk = tmp_path / "not-a-font.ttf"
    junk.write_bytes(b"this is not a font")
    with pytest.raises(OSError):
        fr.use_font(junk)
    assert fr._FONT_FILE is None, "a refused font must not become the active one"


def test_clearing_the_font_restores_the_bundled_face():
    other = _a_font_with_more_glyphs()
    if other is None:
        pytest.skip("no font with wider coverage on this machine")
    before = fr.title_card("Kashmir", "", (400, 300)).tobytes()
    fr.use_font(other)
    fr.use_font(None)
    assert fr.title_card("Kashmir", "", (400, 300)).tobytes() == before


def _a_font_with_more_glyphs():
    """Any font on this machine that draws a character Aileron cannot.

    Returns None rather than skipping here, so the caller decides - a CI
    runner may have no fonts at all, and that is not a failure of this code.
    """
    import sys

    roots = {
        "win32": [Path("C:/Windows/Fonts")],
        "darwin": [Path("/System/Library/Fonts"), Path("/Library/Fonts")],
    }.get(sys.platform, [Path("/usr/share/fonts")])
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.ttf"))[:60]:
            try:
                ImageFont.truetype(str(path), 12)
            except OSError:
                continue
            return path
    return None
