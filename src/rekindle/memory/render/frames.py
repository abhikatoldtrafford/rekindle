"""Decoding shots into canvas-sized frames. Shared by both backends.

The canvas comes from the photos themselves (`composition.canvas_for`), so
nothing is ever upscaled, and every frame is the same size so a video encoder
will accept the sequence.

A shot whose file has gone missing is DROPPED AND COUNTED, never rendered as a
black slot and never raised. A library is routinely on an external drive, and
half a rendered memory with an honest report beats a traceback.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from rekindle.memory.composition import FIT_PAD, plan_placement
from rekindle.memory.spec import MemorySpec
from rekindle.meta.exif import open_upright
from rekindle.models import Photo

BACKGROUND = (0, 0, 0)
TEXT = (245, 245, 245)
DIM = (170, 170, 170)

DROP_MISSING = "file_missing"
DROP_UNDECODABLE = "undecodable"
DROP_NOT_IN_INDEX = "not_in_index"


@dataclass
class FrameReport:
    requested: int = 0
    rendered: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=dict)
    # How each rendered shot met the canvas: downscale, upscale or pad. A high
    # pad count is not a fault - it says the photos of that period are
    # genuinely smaller than the rest of the memory.
    placement: dict[str, int] = field(default_factory=dict)

    def drop(self, reason: str, label: str = "") -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1
        if label and reason not in self.names:
            self.names[reason] = label

    @property
    def total_dropped(self) -> int:
        return sum(self.dropped.values())

    @property
    def accounted(self) -> bool:
        return self.requested == self.rendered + self.total_dropped


def _font(size: int) -> ImageFont.FreeTypeFont:
    """A scalable font with no font FILE and no system font assumed.

    Pillow >= 10.1 lets `load_default` scale its bundled Aileron face, which
    is the only way to get readable text in a wheel that ships no assets. The
    fallback is Pillow's 11px bitmap font - ugly, but a caption is better than
    a crash on an old Pillow.
    """
    try:
        return ImageFont.load_default(size=size)
    except (TypeError, AttributeError):  # pragma: no cover - very old Pillow
        return ImageFont.load_default()


# How much of the canvas a blurred backdrop is enlarged by before blurring,
# and how hard. Enlarging past the canvas first means the blur never reveals
# an edge, and a radius proportional to the canvas keeps the effect identical
# at every output size.
_BACKDROP_ZOOM = 1.08
_BLUR_DIVISOR = 22


def _backdrop(photo: Image.Image, canvas: tuple[int, int]) -> Image.Image:
    """A blurred, cover-cropped enlargement of the photo, filling the canvas.

    Chosen over a flat matte deliberately. A 640x480 photo centred in a
    3984x2988 canvas occupies 2.5% of the area; on black it reads as broken,
    while on a blurred enlargement of itself it reads as a small old photo -
    which is exactly what it is. It is also what every slideshow tool does
    with a mixed-era library, so it is the familiar signal rather than a novel
    one.

    The backdrop IS an upscale of the photo, but a deliberately blurred one,
    so the "never upscale into mush" rule is not violated - nothing is
    presented as detail that the photo does not have.
    """
    target_w = int(canvas[0] * _BACKDROP_ZOOM)
    target_h = int(canvas[1] * _BACKDROP_ZOOM)
    scale = max(target_w / photo.width, target_h / photo.height)
    enlarged = photo.resize(
        (max(1, round(photo.width * scale)), max(1, round(photo.height * scale))),
        Image.Resampling.BILINEAR,
    )
    blurred = enlarged.filter(ImageFilter.GaussianBlur(radius=max(2, canvas[0] // _BLUR_DIVISOR)))
    left = (blurred.width - canvas[0]) // 2
    top = (blurred.height - canvas[1]) // 2
    return blurred.crop((left, top, left + canvas[0], top + canvas[1]))


def fit_photo(path: Path, canvas: tuple[int, int]) -> tuple[Image.Image, str]:
    """One photo placed on the canvas. Returns (frame, placement mode).

    Raises OSError on a bad file. The mode is reported so the CLI can tell a
    user how many shots were padded - a memory where most shots are padded is
    telling them something real about that period of their library.
    """
    # ORIENTATION FIRST, always. A phone stores a portrait photo as landscape
    # pixels plus a tag; skipping this renders it on its side. `open_upright`
    # also hints the JPEG decoder with the canvas - but never below it, since
    # draft() scales in powers of two and asking for a small size on a large
    # canvas would throw away the resolution this whole change exists to keep.
    rgb = open_upright(path, draft=canvas).convert("RGB")
    target, mode = plan_placement(rgb.size, canvas)
    resized = rgb.resize(target, Image.Resampling.LANCZOS)
    backdrop = _backdrop(rgb, canvas) if mode == FIT_PAD else None

    frame = backdrop if backdrop is not None else Image.new("RGB", canvas, BACKGROUND)
    frame.paste(resized, ((canvas[0] - target[0]) // 2, (canvas[1] - target[1]) // 2))
    return frame, mode


def caption_frame(frame: Image.Image, text: str) -> Image.Image:
    """Burn a caption into the bottom-left of a frame.

    Drawn twice - once offset in black, once in white - because a caption over
    a bright sky is invisible otherwise and Pillow has no text stroke that
    works reliably across versions.
    """
    if not text:
        return frame
    size = max(14, frame.height // 22)
    draw = ImageDraw.Draw(frame)
    font = _font(size)
    x, y = size // 2, frame.height - size - size // 2
    draw.text((x + 1, y + 1), text, font=font, fill=BACKGROUND)
    draw.text((x, y), text, font=font, fill=TEXT)
    return frame


def title_card(title: str, subtitle: str, canvas: tuple[int, int]) -> Image.Image:
    """The opening frame. Text only - no photo, so nothing can leak into it."""
    frame = Image.new("RGB", canvas, BACKGROUND)
    draw = ImageDraw.Draw(frame)
    title_size = max(18, canvas[1] // 10)
    sub_size = max(12, canvas[1] // 22)

    title_font = _font(title_size)
    sub_font = _font(sub_size)
    lines = _wrap(draw, title, title_font, canvas[0] - canvas[0] // 8)

    block = len(lines) * (title_size + 6) + (sub_size + 10 if subtitle else 0)
    y = max(0, (canvas[1] - block) // 2)
    for line in lines:
        width = draw.textlength(line, font=title_font)
        draw.text(((canvas[0] - width) / 2, y), line, font=title_font, fill=TEXT)
        y += title_size + 6
    if subtitle:
        width = draw.textlength(subtitle, font=sub_font)
        draw.text(((canvas[0] - width) / 2, y + 4), subtitle, font=sub_font, fill=DIM)
    return frame


def _wrap(draw, text: str, font, max_width: int) -> list[str]:
    """Greedy word wrap. A single word longer than the line is left long
    rather than broken mid-word - album titles are short and a hyphenated
    break reads worse than a slight overflow."""
    words = text.split()
    if not words:
        return [""]
    lines, current = [], words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def build_frames(
    spec: MemorySpec,
    canvas: tuple[int, int],
    *,
    resolve: Callable[[str], Photo | None],
    locate: Callable[[Photo], Path | None],
    with_captions: bool = True,
    with_title: bool = True,
    limit: int | None = None,
) -> tuple[list[Image.Image], FrameReport]:
    """Decode a spec's shots into frames.

    `resolve` and `locate` are injected rather than a `MemoryIndex` being
    passed in, so the renderer can be tested without a database and so it
    cannot accidentally reach a photo the index would have blocked - it can
    only ask about hashes the spec already contains.
    """
    report = FrameReport()
    frames: list[Image.Image] = []
    if with_title:
        frames.append(title_card(spec.title, spec.subtitle, canvas))

    shots = spec.shots if limit is None else spec.shots[:limit]
    for shot in shots:
        report.requested += 1
        photo = resolve(shot.file_hash)
        if photo is None:
            report.drop(DROP_NOT_IN_INDEX, shot.file_hash[:12])
            continue
        path = locate(photo)
        if path is None:
            report.drop(DROP_MISSING, photo.paths[0].name if photo.paths else "")
            continue
        try:
            frame, mode = fit_photo(path, canvas)
        except (OSError, ValueError, Image.DecompressionBombError):
            report.drop(DROP_UNDECODABLE, path.name)
            continue
        if with_captions:
            frame = caption_frame(frame, shot.caption)
        frames.append(frame)
        report.rendered += 1
        report.placement[mode] = report.placement.get(mode, 0) + 1
    return frames, report
