"""MP4 output via ffmpeg, when ffmpeg is present.

ffmpeg is an external TOOL, not a Python dependency. It is found through
`shutil.which` and never hardcoded - the user has it at /c/ffmpeg/bin, CI has
it somewhere else, and a macOS user has it in Homebrew.

**Its absence is not an error.** `rekindle memory` produces the GIF, says
plainly that it skipped the MP4 and why, and exits 0. A crash here would make
the whole feature unavailable to anyone who has not installed a video encoder.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

FFMPEG_MISSING = (
    "ffmpeg was not found on PATH, so only the GIF was written. "
    "Install ffmpeg and re-run to get the full-resolution MP4."
)

# 2560 (1440p class), not 1280. The MP4 is the full memory, and this library's
# median photo is ~3984px wide - rendering that into 1280 throws away 90% of
# the pixel count for no reason. Native resolution is not the default because
# a 7008x4672 video is useful to nobody and takes minutes to encode; a flag
# raises the bound for anyone who wants it. The canvas is still never
# UPSCALED past what the photos support.
DEFAULT_WIDTH = 2560
DEFAULT_SECONDS = 2.5
TITLE_SECONDS = 3.5
# How much of ffmpeg's stderr to show when it fails. Enough to name the real
# cause, short enough not to bury it.
STDERR_TAIL = 1600


@dataclass(frozen=True)
class Mp4Result:
    path: Path | None
    size: int = 0
    skipped: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.path is not None


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def mp4_canvas(canvas: tuple[int, int], width: int = DEFAULT_WIDTH) -> tuple[int, int]:
    """Scale to the target width, never up, and force EVEN dimensions.

    libx264 with yuv420p cannot encode an odd width or height - it fails with
    "width not divisible by 2", which is a baffling error to hand a user whose
    only mistake was owning a 4001px photo.
    """
    source_w, source_h = canvas
    scale = min(width / source_w, 1.0)
    out_w = max(2, round(source_w * scale))
    out_h = max(2, round(source_h * scale))
    return (out_w - (out_w % 2), out_h - (out_h % 2))


def write_mp4(
    frames: list[Image.Image],
    path: Path,
    *,
    seconds: float = DEFAULT_SECONDS,
    title_seconds: float = TITLE_SECONDS,
    has_title: bool = True,
    music: Path | None = None,
    ffmpeg: str | None = None,
) -> Mp4Result:
    """Encode frames to H.264. Never raises; reports instead.

    Uses the concat demuxer with per-image durations rather than an `xfade`
    filter chain. Crossfades across 24 inputs need a 24-deep filtergraph whose
    offsets have to be computed by hand and which breaks differently on every
    ffmpeg build; hard cuts are robust everywhere and their absence is
    documented rather than half-built.
    """
    binary = ffmpeg or ffmpeg_path()
    if binary is None:
        return Mp4Result(path=None, skipped=FFMPEG_MISSING)
    if not frames:
        return Mp4Result(path=None, error="no frames to encode")

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = mp4_canvas(frames[0].size, frames[0].size[0])

    with tempfile.TemporaryDirectory(prefix="rekindle-mp4-") as tmp:
        work = Path(tmp)
        listing = []
        for i, frame in enumerate(frames):
            still = work / f"{i:04d}.png"
            frame.convert("RGB").resize(canvas, Image.Resampling.LANCZOS).save(still)
            hold = title_seconds if (has_title and i == 0) else seconds
            # The concat demuxer needs POSIX-style forward slashes even on
            # Windows, and quotes around the path for names containing spaces.
            listing.append(f"file '{still.as_posix()}'\nduration {hold}")
        # The last image needs repeating: the concat demuxer applies a
        # duration to the transition INTO the next file, so without this the
        # final frame flashes by in one frame-time.
        listing.append(f"file '{(work / f'{len(frames) - 1:04d}.png').as_posix()}'")
        manifest = work / "frames.txt"
        manifest.write_text("\n".join(listing) + "\n", encoding="utf-8")

        cmd = [binary, "-y", "-hide_banner", "-loglevel", "error"]
        cmd += ["-f", "concat", "-safe", "0", "-i", str(manifest)]
        if music is not None:
            # -stream_loop -1 so a 40-second track does not leave the last
            # half of a 90-second memory in silence. Looping rather than
            # picking a long enough track on purpose: reading a duration needs
            # ffprobe, which would make WHICH track a memory gets depend on
            # whether ffmpeg is installed - and the same library must produce
            # the same output on every machine. -shortest then cuts the loop
            # at the end of the video. Both flags are needed: -stream_loop
            # alone never terminates.
            #
            # -stream_loop must precede its own -i; after it, it would apply
            # to the next input instead.
            cmd += ["-stream_loop", "-1"]
            # The audio is re-encoded to AAC because an arbitrary
            # user-supplied file may be anything.
            cmd += ["-i", str(music), "-c:a", "aac", "-b:a", "160k", "-shortest"]
        cmd += [
            "-vf",
            # fps must be set explicitly: a concat of stills has no inherent
            # frame rate and some players refuse the result.
            "fps=25,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            str(path),
        ]

        try:
            done = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except FileNotFoundError:
            # `which` found it and then it vanished, or PATH lied.
            return Mp4Result(path=None, skipped=FFMPEG_MISSING)
        except subprocess.TimeoutExpired:
            return Mp4Result(path=None, error="ffmpeg timed out after 600s")

    if done.returncode != 0:
        # The GIF is already written by the time this runs, so the memory is
        # not lost - the caller reports the failure and keeps going.
        return Mp4Result(path=None, error=(done.stderr or "")[-STDERR_TAIL:].strip())
    if not path.is_file():
        return Mp4Result(path=None, error="ffmpeg reported success but wrote no file")
    return Mp4Result(path=path, size=path.stat().st_size)
