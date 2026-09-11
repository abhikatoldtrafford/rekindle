"""Rendering.

The guarantee under test: ffmpeg's absence degrades to GIF-only with a clear
message and exit 0, never a crash. Everything else is about not lying - a
missing file is counted, not rendered as a black slot.
"""

import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image

from rekindle.memory.render import frames as fr
from rekindle.memory.render import mp4 as mp4mod
from rekindle.memory.render.gif import gif_canvas, write_gif
from rekindle.memory.render.mp4 import mp4_canvas, write_mp4
from rekindle.memory.render.music import NO_MUSIC_HINT, resolve_music
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


def test_a_very_long_title_wraps_instead_of_overflowing():
    card = fr.title_card("A place you kept coming back to over many years", "", (320, 240))
    assert card.size == (320, 240)


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
    """A 2:1 photo on a 4:3 canvas gets black bars, not a squash."""
    _jpeg(tmp_path / "w.jpg", size=(800, 400))
    frame = fr.fit_photo(tmp_path / "w.jpg", (320, 240))
    assert frame.size == (320, 240)
    # Top row is letterbox.
    assert frame.getpixel((160, 2)) == (0, 0, 0)
    # Centre is the photo.
    assert frame.getpixel((160, 120)) != (0, 0, 0)


def test_a_portrait_photo_fits_a_landscape_canvas(tmp_path):
    _jpeg(tmp_path / "p.jpg", size=(600, 900))
    frame = fr.fit_photo(tmp_path / "p.jpg", (320, 240))
    assert frame.size == (320, 240)
    assert frame.getpixel((2, 120)) == (0, 0, 0)


def test_a_photo_is_never_upscaled(tmp_path):
    """Upscaling a 640px photo to 1080p produces visible mush."""
    _jpeg(tmp_path / "small.jpg", size=(160, 120))
    frame = fr.fit_photo(tmp_path / "small.jpg", (1280, 960))
    assert frame.size == (1280, 960)
    # The photo occupies only its native 160x120 in the centre.
    assert frame.getpixel((640, 480)) != (0, 0, 0)
    assert frame.getpixel((100, 480)) == (0, 0, 0)


def test_exif_orientation_is_applied_when_rendering(tmp_path):
    """Without this a phone portrait renders on its side. The stored
    dimensions are already post-rotation, so the renderer must match them."""
    _jpeg(tmp_path / "r.jpg", size=(800, 400), exif_orientation=6)
    frame = fr.fit_photo(tmp_path / "r.jpg", (400, 400))
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


def test_the_gif_canvas_scales_down_but_never_up():
    assert gif_canvas((1600, 1200), 480) == (480, 360)
    assert gif_canvas((320, 240), 480) == (320, 240)


def test_the_gif_canvas_keeps_a_portrait_memory_portrait():
    assert gif_canvas((1200, 1600), 480) == (480, 640)


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
