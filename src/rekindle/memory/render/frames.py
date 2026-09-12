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

from rekindle.memory.captions import renderable
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


def backdrop_at(photo: Image.Image, size: tuple[int, int]) -> Image.Image:
    """The blurred backdrop rendered at an ARBITRARY size, not just the canvas.

    `render.motion` drifts the backdrop behind a sub-canvas photograph, which
    needs it rendered larger than the canvas so there is something to drift
    through. Exposed rather than duplicated: the blur radius and the
    enlargement are the reason a small old photograph reads as a small old
    photograph and not as a broken one, and two copies of that would drift.
    """
    return _backdrop(photo, size)


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
    # The last line of defence, not the fix. `memory.llm` folds a model's
    # typography before the caption is ever cached; this catches a caption
    # that reached the renderer by some other road - an older cache row, a
    # hand-edited spec - because the failure is silent: FreeType draws a
    # filled rectangle for a missing glyph rather than raising.
    text = renderable(text)
    size = max(14, frame.height // 22)
    draw = ImageDraw.Draw(frame)
    font = _font(size)
    x, y = size // 2, frame.height - size - size // 2
    draw.text((x + 1, y + 1), text, font=font, fill=BACKGROUND)
    draw.text((x, y), text, font=font, fill=TEXT)
    return frame


def title_card(title: str, subtitle: str, canvas: tuple[int, int]) -> Image.Image:
    """The opening frame. Text only - no photo, so nothing can leak into it.

    THE SUBTITLE IS MEASURED, WHICH IT WAS NOT. This function wrapped the
    title against the canvas WIDTH and then drew the subtitle straight out at
    a size derived from the canvas HEIGHT, having measured it against nothing.
    Centred text that is wider than the canvas overflows at BOTH ends, so on a
    narrow preview the memory "Every August" lost the `1` from `10 photos` and
    the `6` from `2026` - one character off each edge. Invisible at the
    engine's 1280px default, which is why it shipped; found by the gallery
    agent rendering at a smaller size, which worked around it in its own
    script rather than here.

    Three guards now, applied in that order, each doing what the one before
    it cannot:

      1. WRAP against the same width the title is wrapped against, so a long
         subtitle becomes two centred lines. Not enough on its own: `_wrap`
         deliberately leaves an over-long single word long rather than
         hyphenating it.
      2. SHRINK, down to `MIN_SUBTITLE_SIZE`, when an unbreakable run still
         will not fit. Not enough on its own either - it would drive an
         ordinary two-phrase subtitle to unreadable rather than putting it on
         two lines - and not enough even after wrapping: at the floor, a
         37-character unbreakable word is still 204px wide in a 175px box.
      3. TRUNCATE with an ellipsis, as a last resort at the floor.

    The third guard is what makes the promise UNCONDITIONAL. Without it the
    honest statement would be "the subtitle does not overflow unless it is
    pathological", and a rule with an exception nobody can enumerate is how
    this defect shipped in the first place.

    All three are no-ops when the subtitle already fits, so nothing about the
    default 1280px canvas changes.
    """
    # Folded before anything is MEASURED, not just before it is drawn: the
    # fold changes the width of the string, and wrapping the unfolded one
    # would lay the card out for characters that never appear on it.
    title = renderable(title)
    subtitle = renderable(subtitle)
    frame = Image.new("RGB", canvas, BACKGROUND)
    draw = ImageDraw.Draw(frame)
    max_width = canvas[0] - canvas[0] // 8

    # `or [""]`: an empty title keeps its blank line, so the block height and
    # therefore the vertical centring are unchanged for every memory that
    # was already fitting.
    title_size, lines = _fit(draw, title, max(18, canvas[1] // 10), MIN_TITLE_SIZE, max_width)
    lines = lines or [""]
    title_font = _font(title_size)
    sub_size, sub_lines = _fit_subtitle(draw, subtitle, canvas, max_width)
    sub_font = _font(sub_size)

    block = len(lines) * (title_size + 6) + (
        len(sub_lines) * (sub_size + 4) + 6 if sub_lines else 0
    )
    y = max(0, (canvas[1] - block) // 2)
    for line in lines:
        width = draw.textlength(line, font=title_font)
        draw.text(((canvas[0] - width) / 2, y), line, font=title_font, fill=TEXT)
        y += title_size + 6
    if sub_lines:
        y += 4
        for line in sub_lines:
            width = draw.textlength(line, font=sub_font)
            draw.text(((canvas[0] - width) / 2, y), line, font=sub_font, fill=DIM)
            y += sub_size + 4
    return frame


#: The subtitle will not be shrunk below this to make it fit. Below about ten
#: pixels the glyphs of the default bitmap font stop being distinguishable, so
#: a smaller "fitting" subtitle is not more readable than a clipped one - it
#: is just differently unreadable, and silently so.
MIN_SUBTITLE_SIZE = 10


#: The same floor for the title. A title is the larger of the two to begin
#: with, so it reaches this only on a canvas narrow enough that nothing was
#: going to look good; below it, ellipsising is the honest outcome.
MIN_TITLE_SIZE = 14


def _fit(draw, text: str, size: int, min_size: int, max_width: int):
    """`(size, lines)` for text that fits inside `max_width`.

    Three guards, applied in that order, each doing what the one before it
    cannot: WRAP on whitespace, SHRINK the face, then ELLIPSISE. The third is
    what makes the promise unconditional - `_wrap` deliberately leaves a
    single over-long word long, and no amount of shrinking helps once the
    floor is reached.

    Shared by the title and the subtitle. It was written for the subtitle
    alone, and the title - drawn at a size derived from the canvas HEIGHT
    against a box derived from its WIDTH - got none of the three. On a 9:16
    preview (1280x2276, the default for a portrait memory) the title face is
    227 px against a 1120 px box, so any single word of about eleven
    characters overflowed BOTH edges, the text being centred: `Bhubaneswar`,
    `Kanyakumari` and `Thanksgiving` all clipped. This library happens not to
    trigger it - its one long album name, `Wedding_arnab_pics`, is
    landscape-majority and lands on 1280x853 - which is exactly the kind of
    luck that should not be the reason a bug is absent.

    Returns `(size, [])` for empty text so a caller can distinguish "nothing
    to draw" from "one blank line".
    """
    if not text:
        return size, []
    while True:
        font = _font(size)
        lines = _wrap(draw, text, font, max_width)
        if max(draw.textlength(line, font=font) for line in lines) <= max_width:
            return size, lines
        if size <= min_size:
            return size, [_ellipsise(draw, line, font, max_width) for line in lines]
        size -= 1


def _fit_subtitle(draw, subtitle: str, canvas: tuple[int, int], max_width: int):
    """`(size, lines)` for a subtitle that fits inside `max_width`.

    The starting size is the one this function has always used - derived from
    the canvas HEIGHT - so an unclipped subtitle is drawn exactly as before.
    """
    return _fit(draw, subtitle, max(12, canvas[1] // 22), MIN_SUBTITLE_SIZE, max_width)


#: What a truncated line ends with. Three dots rather than U+2026 because the
#: fallback path in `_font` is Pillow's bundled bitmap font, and a glyph it
#: lacks renders as a blank box - a worse outcome than the clipping this is
#: fixing, and one that only appears on the machines least able to report it.
ELLIPSIS = "..."


def _ellipsise(draw, text: str, font, max_width: int) -> str:
    """`text`, trimmed until it fits, with `ELLIPSIS` on the end.

    Character by character rather than by proportion: the bundled face is
    proportional, so `len` is not width, and a proportional estimate would
    over-trim a line of narrow characters and under-trim a line of wide ones.
    A subtitle is a few dozen characters, so the loop is not worth optimising.
    """
    if draw.textlength(text, font=font) <= max_width:
        return text
    for end in range(len(text) - 1, 0, -1):
        candidate = text[:end].rstrip() + ELLIPSIS
        if draw.textlength(candidate, font=font) <= max_width:
            return candidate
    # A canvas too narrow for one character plus an ellipsis. Drawing nothing
    # is the only thing left that does not overflow.
    return ""


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
