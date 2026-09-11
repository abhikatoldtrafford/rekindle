"""GIF output. Pillow only, always available, deliberately a teaser.

No new dependency, works in CI with no ffmpeg, and embeds in a GitHub README -
which is what the user will actually publish. It is NOT the full memory: the
frame count and the width are both capped well below the MP4's, so the file
stays small enough to sit in a README without anyone having to think about it.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

# A README-embeddable teaser. 480px wide and 12 frames keeps a typical memory
# comfortably inside a couple of megabytes; the render reports the size it
# actually produced rather than promising one, because GIF size depends
# entirely on how much the frames differ.
DEFAULT_WIDTH = 480
DEFAULT_MAX_FRAMES = 12
DEFAULT_FRAME_MS = 1400
# The title card earns a longer beat - it is text, and 1.4s is not enough to
# read a title and a subtitle.
TITLE_MS = 2200


def gif_canvas(canvas: tuple[int, int], width: int = DEFAULT_WIDTH) -> tuple[int, int]:
    """Scale a canvas down to the GIF width, never up.

    Preserves the aspect ratio of the canvas the composition stage chose, so a
    portrait memory produces a portrait GIF rather than a letterboxed
    landscape one.
    """
    source_w, source_h = canvas
    if source_w <= width:
        return canvas
    scale = width / source_w
    return (width, max(1, round(source_h * scale)))


def write_gif(
    frames: list[Image.Image],
    path: Path,
    *,
    frame_ms: int = DEFAULT_FRAME_MS,
    title_ms: int = TITLE_MS,
    has_title: bool = True,
) -> int:
    """Write an animated, looping GIF. Returns its size in bytes.

    Raises ValueError on an empty frame list rather than writing a zero-frame
    GIF, which some viewers render as a broken image and others refuse.
    """
    if not frames:
        raise ValueError("cannot write a GIF with no frames")
    path.parent.mkdir(parents=True, exist_ok=True)

    durations = [frame_ms] * len(frames)
    if has_title and durations:
        durations[0] = title_ms

    # Palettise with a shared adaptive palette. Letting Pillow quantise each
    # frame independently makes the background shimmer between frames, which
    # on a montage of photos is very visible.
    first, *rest = [f.convert("RGB") for f in frames]
    first.save(
        path,
        save_all=True,
        append_images=rest,
        duration=durations,
        # 0 means loop forever. Pillow treats a MISSING loop key as "play
        # once", which is not what a teaser wants.
        loop=0,
        optimize=True,
    )
    return path.stat().st_size
