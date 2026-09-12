"""Spec -> files on disk, with the result returned rather than printed.

`memory/cli.py:_render_one` already does this and writes straight to a rich
Console. Both callers here - the `rekindle render` verb and the browser - need
the outcome as DATA (sizes, drop counts, output paths) rather than as coloured
text, and one of them has no terminal at all. So this is the same four calls in
the same order with the console taken out, not a second renderer: frames come
from `render.frames.build_frames`, the animations from `render.gif`, the video
from `render.mp4`, and the canvas from `composition.canvas_for`.

The one behaviour it adds is that resolution goes through `MemoryIndex`, so a
spec naming a photo the policy now refuses renders WITHOUT it and says so.
That is what makes a memory.json safe to keep: excluding a person tomorrow
removes them from every spec already written, without editing any of them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from rekindle.memory.composition import FIT_PAD, canvas_for
from rekindle.memory.history import memory_id
from rekindle.memory.index import MemoryIndex
from rekindle.memory.render.frames import DROP_NOT_IN_INDEX, build_frames
from rekindle.memory.render.gif import (
    DEFAULT_FRAME_MS,
    DEFAULT_MAX_FRAMES,
    TITLE_MS,
    preview_canvas,
    write_gif,
    write_webp,
)
from rekindle.memory.render.gif import DEFAULT_WIDTH as PREVIEW_WIDTH
from rekindle.memory.render.mp4 import DEFAULT_WIDTH as MP4_WIDTH
from rekindle.memory.render.mp4 import mp4_canvas, write_mp4
from rekindle.memory.render.music import resolve_music
from rekindle.memory.spec import MemorySpec

#: The fallback canvas when not one shot resolved to a photo with dimensions.
#: Same value `memory/cli.py` uses, for the same reason: something has to be
#: chosen and a 4:3 SD frame is the least surprising.
FALLBACK_CANVAS = (1280, 960)

#: The phases `render_spec` reports through `progress`, in order. Named
#: constants because the browser draws a bar from them and a typo in a phase
#: name is a bar that silently never moves.
PHASE_SPEC = "spec"
PHASE_PREVIEW = "preview"
PHASE_WEBP = "webp"
PHASE_GIF = "gif"
PHASE_VIDEO = "video"
PHASE_STILLS = "stills"
PHASE_MP4 = "mp4"
PHASE_DONE = "done"

SPEC_NAME = "memory.json"
WEBP_NAME = "memory.webp"
GIF_NAME = "memory.gif"
MP4_NAME = "memory.mp4"


@dataclass(frozen=True)
class RenderOptions:
    frame_ms: int = DEFAULT_FRAME_MS
    title_ms: int = TITLE_MS
    preview_frames: int = DEFAULT_MAX_FRAMES
    preview_width: int = PREVIEW_WIDTH
    mp4_width: int = MP4_WIDTH
    music: Path | None = None
    no_mp4: bool = False
    write_spec: bool = True


@dataclass
class RenderResult:
    folder: Path
    canvas: tuple[int, int] = (0, 0)
    preview_size: tuple[int, int] = (0, 0)
    webp: Path | None = None
    gif: Path | None = None
    mp4: Path | None = None
    webp_bytes: int = 0
    gif_bytes: int = 0
    mp4_bytes: int = 0
    #: How many shots the SPEC names. Not how many are in the preview: the
    #: WebP and GIF stop at `preview_frames` (16 by default) while the MP4
    #: carries every shot, and reporting the preview's count as the memory's
    #: made a 24-shot memory print as "16 of 16 shots".
    shots: int = 0
    preview_rendered: int = 0
    video_rendered: int = 0
    padded: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    examples: dict[str, str] = field(default_factory=dict)
    mp4_skipped: str = ""
    mp4_error: str = ""
    music_used: Path | None = None

    @property
    def ok(self) -> bool:
        return self.preview_rendered > 0

    @property
    def withheld(self) -> int:
        """Shots the spec names that the guardrails no longer admit."""
        return self.dropped.get(DROP_NOT_IN_INDEX, 0)

    def to_json(self) -> dict:
        return {
            "folder": str(self.folder),
            "canvas": list(self.canvas),
            "preview_size": list(self.preview_size),
            "webp": self.webp.name if self.webp else None,
            "gif": self.gif.name if self.gif else None,
            "mp4": self.mp4.name if self.mp4 else None,
            "webp_bytes": self.webp_bytes,
            "gif_bytes": self.gif_bytes,
            "mp4_bytes": self.mp4_bytes,
            "shots": self.shots,
            "preview_rendered": self.preview_rendered,
            "video_rendered": self.video_rendered,
            "padded": self.padded,
            "dropped": dict(self.dropped),
            "examples": dict(self.examples),
            "withheld": self.withheld,
            "mp4_skipped": self.mp4_skipped,
            "mp4_error": self.mp4_error,
            "music": self.music_used.name if self.music_used else None,
        }


#: `(phase, done, total)`. `total` is 0 for a phase with nothing to count.
Progress = Callable[[str, int, int], None]


def _counting(resolve, report: Progress, phase: str, total: int):
    """Wrap `index.get` so the frame loop counts itself.

    `build_frames` calls `resolve` exactly once per shot, at the top of its
    loop, so wrapping it gives a real count of shots reached without
    `frames.py` growing a progress parameter it has no other use for. The
    count therefore runs one shot AHEAD of the decode, which is the right
    direction to be wrong in: a bar that reaches the end a moment early beats
    one that sits at 95% through the slowest frame.
    """
    done = {"n": 0}

    def wrapped(file_hash: str):
        done["n"] += 1
        report(phase, done["n"], total)
        return resolve(file_hash)

    return wrapped


def render_spec(
    spec: MemorySpec,
    index: MemoryIndex,
    folder: Path,
    options: RenderOptions | None = None,
    progress: Progress | None = None,
) -> RenderResult:
    """Write memory.json, memory.webp, memory.gif and (if ffmpeg) memory.mp4.

    `folder` is taken as given rather than derived from today's date: the
    caller decides, because `rekindle render` must be able to rewrite the
    folder a memory already lives in. A date-stamped name that changed on
    every re-render would mean the reproduce command never reproduces the
    thing it was printed beside.
    """
    options = options or RenderOptions()
    report: Progress = progress or (lambda phase, done, total: None)
    folder.mkdir(parents=True, exist_ok=True)
    result = RenderResult(folder=folder, shots=len(spec.shots))

    report(PHASE_SPEC, 0, 0)
    if options.write_spec:
        (folder / SPEC_NAME).write_text(spec.dumps(), encoding="utf-8")

    photos = [p for p in (index.get(s.file_hash) for s in spec.shots) if p is not None]
    canvas = canvas_for(photos) or FALLBACK_CANVAS
    result.canvas = canvas

    preview_size = preview_canvas(canvas, options.preview_width or PREVIEW_WIDTH)
    result.preview_size = preview_size
    previewed = min(len(spec.shots), options.preview_frames or len(spec.shots))
    frames, frame_report = build_frames(
        spec,
        preview_size,
        resolve=_counting(index.get, report, PHASE_PREVIEW, previewed),
        locate=index.resolve_path,
        limit=options.preview_frames,
    )
    result.preview_rendered = frame_report.rendered
    result.dropped = dict(frame_report.dropped)
    result.examples = dict(frame_report.names)
    result.padded = frame_report.placement.get(FIT_PAD, 0)
    if frame_report.rendered == 0:
        report(PHASE_DONE, 0, 0)
        return result

    report(PHASE_WEBP, 0, 0)
    result.webp_bytes = write_webp(
        frames, folder / WEBP_NAME, frame_ms=options.frame_ms, title_ms=options.title_ms
    )
    result.webp = folder / WEBP_NAME
    report(PHASE_GIF, 0, 0)
    result.gif_bytes = write_gif(
        frames, folder / GIF_NAME, frame_ms=options.frame_ms, title_ms=options.title_ms
    )
    result.gif = folder / GIF_NAME

    if options.no_mp4:
        report(PHASE_DONE, 0, 0)
        return result

    video_size = mp4_canvas(canvas, options.mp4_width or MP4_WIDTH)
    mp4_frames, mp4_report = build_frames(
        spec,
        video_size,
        resolve=_counting(index.get, report, PHASE_VIDEO, len(spec.shots)),
        locate=index.resolve_path,
    )
    result.video_rendered = mp4_report.rendered
    bed = resolve_music(options.music, memory_id=memory_id(spec.recipe, spec.key))
    result.music_used = bed
    outcome = write_mp4(
        mp4_frames,
        folder / MP4_NAME,
        # The writers take milliseconds and ffmpeg takes seconds; converting
        # here rather than storing two numbers is what keeps the GIF, the WebP
        # and the MP4 showing each photo for the same length of time.
        seconds=options.frame_ms / 1000.0,
        title_seconds=options.title_ms / 1000.0,
        music=bed,
        on_still=lambda done, total: report(
            PHASE_STILLS if done < total else PHASE_MP4, done, total
        ),
    )
    if outcome.ok and outcome.path is not None:
        result.mp4 = outcome.path
        result.mp4_bytes = outcome.size
    result.mp4_skipped = outcome.skipped or ""
    result.mp4_error = outcome.error or ""
    # The full-resolution pass sees every shot, not just the preview's first
    # sixteen, so its drop counts are the complete ones when it ran.
    if mp4_report.total_dropped >= frame_report.total_dropped:
        result.dropped = dict(mp4_report.dropped)
        result.examples = dict(mp4_report.names)
    report(PHASE_DONE, 0, 0)
    return result
