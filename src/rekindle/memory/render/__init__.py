"""Rendering a MemorySpec into something you can watch.

A spec is the durable artefact; renderers are pure functions of it plus the
photo bytes. GIF is always produced (Pillow only, no new dependency, works in
CI, embeds in a README). MP4 is produced when ffmpeg is on PATH, and its
absence degrades to GIF-only with a clear message - never a crash.
"""

from rekindle.memory.render.frames import FrameReport, build_frames, title_card
from rekindle.memory.render.gif import write_gif
from rekindle.memory.render.mp4 import FFMPEG_MISSING, ffmpeg_path, write_mp4
from rekindle.memory.render.music import resolve_music

__all__ = [
    "FrameReport",
    "build_frames",
    "title_card",
    "write_gif",
    "write_mp4",
    "ffmpeg_path",
    "FFMPEG_MISSING",
    "resolve_music",
]
