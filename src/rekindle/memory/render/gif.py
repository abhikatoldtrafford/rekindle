"""Animated preview output: GIF and WebP.

**GIF is no longer the preview of record.** Its ceiling is structural, not a
matter of resolution: the format allows 256 colours PER FRAME, so photographic
content bands visibly however large the image is, and a photo animation at
1080p runs to tens of megabytes. Raising the resolution alone does not fix how
a GIF looks.

So `write_webp` is the default preview - true colour, dramatically smaller for
the same content, and rendered inline by GitHub markdown. GIF stays available
because it is the most universally embeddable animation there is, and is
written alongside.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

# The preview width. This was 480 to keep a GIF small enough to drop into a
# README without thinking; that constraint has been lifted, and 480 was far
# too low for anyone actually looking at their photos. 1280 is a real preview
# of a 4000px photo rather than a thumbnail of one.
#
# Exposed as `--preview-width` because the right answer genuinely differs: a
# README still wants something small, and someone reviewing their own memories
# wants it large.
DEFAULT_WIDTH = 1280
DEFAULT_MAX_FRAMES = 16
DEFAULT_FRAME_MS = 1400
# The title card earns a longer beat - it is text, and 1.4s is not enough to
# read a title and a subtitle.
TITLE_MS = 2200


def preview_canvas(canvas: tuple[int, int], width: int = DEFAULT_WIDTH) -> tuple[int, int]:
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


def write_webp(
    frames: list[Image.Image],
    path: Path,
    *,
    frame_ms: int = DEFAULT_FRAME_MS,
    title_ms: int = TITLE_MS,
    has_title: bool = True,
    quality: int = 82,
) -> int:
    """Animated WebP: true colour, and the preview that should be looked at.

    GIF quantises every frame to 256 colours, which bands a photograph no
    matter how many pixels it has. WebP carries full colour and compresses
    photographic content properly, so the same memory is both better looking
    and several times smaller. GitHub renders it inline in markdown.

    `method=4` is Pillow's middle encoding effort - noticeably smaller output
    than the default without the very slow settings above it.
    """
    if not frames:
        raise ValueError("cannot write a WebP with no frames")
    path.parent.mkdir(parents=True, exist_ok=True)

    durations = [frame_ms] * len(frames)
    if has_title and durations:
        durations[0] = title_ms

    first, *rest = [f.convert("RGB") for f in frames]
    first.save(
        path,
        save_all=True,
        append_images=rest,
        duration=durations,
        loop=0,
        quality=quality,
        method=4,
    )
    return path.stat().st_size
