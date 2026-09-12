"""Ken Burns, crossfades and cards that arrive rather than appear.

Everything here is a pure function of a `Timeline` plus decoded pixels, and
none of it needs ffmpeg: `frames_for` yields PIL images and the encoder's only
job is to take them. That is deliberate. CI has no video encoder, so a
filtergraph would be untestable, and this project's own history says an
untestable renderer is one nobody looks at until it is wrong.

## Ken Burns against the canvas rule, which is the hard part

`composition.canvas_for` makes the canvas the MEDIAN size of the selected set.
Photographs above it downscale; photographs below it are drawn at NATIVE size
on a blurred backdrop and are never upscaled, because upscaling is exactly the
mush that rule exists to prevent. A pan that zoomed into a padded sub-canvas
photograph would undo that in the most visible way available.

**So the motion is defined as a crop window in SOURCE pixels, and the window
may never be smaller than the canvas.** That is the whole rule, and it makes
the guarantee arithmetic rather than a promise:

    the widest window is the largest canvas-shaped rectangle that fits
    inside the photograph:      cw_max = min(w, h * W / H)
    the tightest window is the canvas itself:            cw_tight = W
    so the available zoom is                    z = cw_max / W

A photograph BELOW the canvas has `cw_max < W`, so `z < 1`, so there is no
motion to plan - the padded case excludes itself without a special branch. A
photograph with lots of surplus gets the zoom capped at `MAX_ZOOM` anyway,
because a 2x zoom is a zoom effect and not Ken Burns. `test_motion.py` asserts
the invariant directly: over every frame of every plan, no crop window is ever
narrower or shorter than the canvas.

The second constraint is about the PHOTOGRAPH rather than its pixels. Filling
a 16:9 canvas from a 4:3 photograph crops a quarter of its height, and
cropping a quarter off someone's photograph to make it move is a bad trade -
the composition stage went to some trouble to show the whole frame. So a shot
whose cover crop would lose more than `MAX_CROP` of either axis holds still.

**And then the sub-canvas shots get their motion back a different way.** That
was not the original design; it came from rendering `scenery:sea` and looking
at it. Fourteen of its twenty-four shots could not pan, most of each of those
frames is blurred backdrop, and the result is a slideshow with a dissolve on
it. So for a sub-canvas shot the BACKDROP drifts - it is already an admitted
blurred upscale, so moving it asserts nothing new - while the photograph is
pasted over it at a fixed position and exactly native size, pixel for pixel
what the still render would show. See `BACKDROP_ZOOM`.

What is left holding completely still is the aspect-ratio outlier, which reads
as a deliberate beat among moving shots and is still crossfading at both ends.

## Restraint

The animated card is where "dynamic and memorable" turns into a 2004
screensaver, so: text fades up and rises by 2% of the frame height, over
0.8 seconds, once. Nothing spins, nothing bounces, nothing flies in from the
side, and nothing moves after it has arrived. The caption does the same at a
fifth of the distance. That is the entire vocabulary.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw

from rekindle.memory.render import frames as static
from rekindle.memory.render.timeline import Beat, Timeline
from rekindle.memory.spec import MemorySpec
from rekindle.models import Photo

#: The most a shot may zoom over its life. 1.09 was chosen by watching: at
#: 1.03 the movement is invisible at 2.5 seconds and reads as an encoding
#: wobble, and at 1.25 the frame is visibly travelling and the photograph
#: stops being the subject. Slow is the point - a Ken Burns move you notice
#: is one that has gone wrong.
MAX_ZOOM = 1.09

#: Below this there is not enough surplus for the movement to be worth the
#: cost, and the shot holds.
MIN_ZOOM = 1.02

#: The most of a photograph's width or height that filling the canvas may cut
#: off. Beyond this the photograph holds still rather than being cropped in
#: order to move.
#:
#: 0.15 is a judgement and not a measurement: it is roughly the crop you get
#: putting a 3:2 photograph on a 4:3 canvas, and losing a sixth of a
#: photograph to make it move is about the limit of what the composition stage
#: - which went to some trouble to show the whole frame - should tolerate.
#: What WAS measured is the consequence: see `BACKDROP_ZOOM` for the counts
#: across the reference library's scenery memories.
MAX_CROP = 0.15

#: How far the blurred backdrop drifts behind a photograph that is holding
#: still, and THE ANSWER TO THE COLLISION THE BRIEF NAMES.
#:
#: A photograph below the canvas is drawn at native size on a blurred,
#: enlarged copy of itself, and it may not be zoomed - that is the whole point
#: of the rule. Measured on `scenery:sea`, which spans 2009 to 2025: 10 of 24
#: shots could pan, 5 were below the canvas and 9 would have been cropped too
#: hard, so **14 of 24 shots were completely motionless**, and most of each of
#: those frames is backdrop. Rendered and watched, that is not restraint, it
#: is a slideshow with a dissolve on it.
#:
#: So the BACKDROP drifts and the photograph does not move by one pixel. The
#: backdrop is already an admitted upscale - `frames._backdrop` blurs it
#: precisely so that nothing is presented as detail the photograph does not
#: have - so moving it asserts nothing new, while the photograph stays exactly
#: where the composition stage put it, at exactly native size.
#:
#: 1.05, half the photograph's own zoom range: it is parallax behind a still
#: subject, and if you can tell how fast it is moving it is moving too fast.
BACKDROP_ZOOM = 1.05

#: How long a caption takes to arrive, and how far it rises doing it.
CAPTION_FADE = 0.5
CAPTION_RISE = 0.004

#: The same, for the title card. Longer and further, because it is the only
#: thing on screen and it is the first thing anyone sees.
TITLE_FADE = 0.8
TITLE_RISE = 0.02

#: Anchor pairs for the pan, as (start, end) positions of the crop window
#: within its own slack, in [0, 1] per axis.
#:
#: A fixed table rather than random offsets: the move must be reproducible
#: from the photograph alone, and a table is also the cheapest way to
#: guarantee every move is a gentle straight drift. Six entries, chosen so
#: that consecutive shots in a memory rarely repeat one.
_MOVES = (
    ((0.30, 0.50), (0.70, 0.50)),  # left to right
    ((0.70, 0.50), (0.30, 0.50)),  # right to left
    ((0.50, 0.30), (0.50, 0.70)),  # up to down
    ((0.50, 0.70), (0.50, 0.30)),  # down to up
    ((0.35, 0.35), (0.65, 0.65)),  # diagonal
    ((0.65, 0.35), (0.35, 0.65)),  # the other diagonal
)


@dataclass(frozen=True)
class KenBurns:
    """A crop window travelling across a PLATE, which is a scaled source.

    `plate` is the source resized so that the TIGHTEST window is exactly the
    canvas. Every frame is then a crop plus a small downscale, which is about
    ten times cheaper than cropping the full-resolution source, and the
    tightest window is a 1:1 copy rather than an upscale.
    """

    plate: tuple[int, int]
    canvas: tuple[int, int]
    zoom: float
    start: tuple[float, float]
    end: tuple[float, float]
    #: True when the window starts wide and closes in.
    zoom_in: bool

    def window_at(self, progress: float) -> tuple[int, int, int, int]:
        """The crop box in plate coordinates at `progress` in [0, 1].

        Never smaller than the canvas in either axis, and never outside the
        plate. Both are asserted by `test_motion.py` over every frame of every
        plan, because they are the two ways this could quietly ruin a photo.
        """
        t = min(1.0, max(0.0, progress))
        wide, tight = self.zoom, 1.0
        scale = (wide + (tight - wide) * t) if self.zoom_in else (tight + (wide - tight) * t)
        width = self.canvas[0] * scale
        height = self.canvas[1] * scale
        # Clamp before rounding. `plan_motion` and `plan_backdrop` both size
        # the plate so this can only ever bite by half a pixel, which rounding
        # absorbs - a mutation run confirmed that deleting these two lines
        # fails nothing on any plan those two produce. It is kept because
        # `window_at` is public and `KenBurns` is constructible directly, and
        # a window wider than its plate pans off the edge into a black stripe
        # down the side of someone's photograph. `test_motion.py` builds that
        # case explicitly rather than pretending the planners can produce it.
        width = min(width, self.plate[0])
        height = min(height, self.plate[1])
        ax = self.start[0] + (self.end[0] - self.start[0]) * t
        ay = self.start[1] + (self.end[1] - self.start[1]) * t
        left = int(round(ax * (self.plate[0] - width)))
        top = int(round(ay * (self.plate[1] - height)))
        return (left, top, left + int(round(width)), top + int(round(height)))


def plan_motion(
    source: tuple[int, int],
    canvas: tuple[int, int],
    seed: str,
    *,
    max_zoom: float = MAX_ZOOM,
    min_zoom: float = MIN_ZOOM,
    max_crop: float = MAX_CROP,
) -> KenBurns | None:
    """A move for one photograph, or None when it should hold still.

    None for three reasons, and all three are correct outcomes:

    * the photograph is smaller than the canvas, so any cover crop would
      upscale it - this is the padded case, and it excludes itself;
    * filling the canvas would crop more than `max_crop` of the photograph;
    * there is less than `min_zoom` of surplus to travel through.

    `seed` picks the direction. The file hash is what the caller passes, so
    the same photograph moves the same way in every memory it appears in and
    on every machine - the engine's byte-for-byte promise reaches the video.
    """
    w, h = source
    cw, ch = canvas
    if w <= 0 or h <= 0 or cw <= 0 or ch <= 0:
        return None
    # The largest canvas-shaped rectangle that fits inside the photograph.
    wide_w = min(w, h * cw / ch)
    wide_h = wide_w * ch / cw
    # How much of the photograph that rectangle shows. One axis is always 1.0.
    if min(wide_w / w, wide_h / h) < 1.0 - max_crop:
        return None
    zoom = wide_w / cw
    if zoom < min_zoom:
        return None
    zoom = min(zoom, max_zoom)

    start, end, zoom_in = _move_for(seed)
    # The plate: the source scaled so the TIGHTEST window is exactly the
    # canvas. `wide_w / zoom` is that window in source pixels.
    factor = cw / (wide_w / zoom)
    plate = (max(cw, round(w * factor)), max(ch, round(h * factor)))
    return KenBurns(
        plate=plate,
        canvas=canvas,
        zoom=zoom,
        start=start,
        end=end,
        zoom_in=zoom_in,
    )


def plan_backdrop(canvas: tuple[int, int], seed: str, *, zoom: float = BACKDROP_ZOOM) -> KenBurns:
    """A drift for the blurred backdrop behind a photograph that holds still.

    The plate is the backdrop rendered `zoom` larger than the canvas, so the
    window is always at least canvas-sized and the photograph pasted over it
    never moves. See `BACKDROP_ZOOM` for why this exists.
    """
    start, end, zoom_in = _move_for(seed)
    plate = (max(canvas[0], round(canvas[0] * zoom)), max(canvas[1], round(canvas[1] * zoom)))
    return KenBurns(plate=plate, canvas=canvas, zoom=zoom, start=start, end=end, zoom_in=zoom_in)


def _move_for(seed: str) -> tuple[tuple[float, float], tuple[float, float], bool]:
    """Direction and zoom sense, from the photograph's own hash.

    So the same photograph moves the same way in every memory it appears in,
    on every machine - the engine's byte-for-byte promise reaching the video.
    """
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    start, end = _MOVES[digest[0] % len(_MOVES)]
    return start, end, bool(digest[1] & 1)


# --------------------------------------------------------------------------
# cards and captions


def _alpha(image: Image.Image, alpha: float) -> Image.Image:
    """An RGBA layer at `alpha`, scaling whatever alpha it already has."""
    out = image.copy()
    band = out.getchannel("A").point(lambda v: int(v * alpha))
    out.putalpha(band)
    return out


def _text_layer(canvas: tuple[int, int], draw_on: Callable[[ImageDraw.ImageDraw], None]):
    layer = Image.new("RGBA", canvas, (0, 0, 0, 0))
    draw_on(ImageDraw.Draw(layer))
    return layer


def title_layers(title: str, subtitle: str, canvas: tuple[int, int]) -> Image.Image:
    """The title card's TEXT only, on transparency, ready to be faded in."""
    card = static.title_card(title, subtitle, canvas)
    # `title_card` paints on black, and the text is the only thing that is not
    # black - so the difference IS the text, with its anti-aliasing intact.
    # Rebuilding the layout here would be a second copy of the wrapping rules.
    layer = card.convert("RGBA")
    grey = card.convert("L")
    layer.putalpha(grey)
    return layer


def caption_layer(text: str, canvas: tuple[int, int]) -> Image.Image | None:
    """One caption, on transparency. None for an empty caption."""
    if not text:
        return None
    plate = Image.new("RGB", canvas, static.BACKGROUND)
    static.caption_frame(plate, text)
    layer = plate.convert("RGBA")
    layer.putalpha(plate.convert("L"))
    return layer


def _arrive(
    base: Image.Image,
    layer: Image.Image | None,
    progress: float,
    *,
    fade: float,
    rise: float,
    duration: float,
) -> Image.Image:
    """Composite `layer` onto `base`, arriving over the first `fade` seconds.

    Rises by `rise` of the frame height while it fades up, and then does not
    move again. Once it has arrived the composite is returned unchanged, which
    is also what makes this cheap: the expensive path runs for about a dozen
    frames per shot.
    """
    if layer is None:
        return base
    elapsed = progress * duration
    if elapsed >= fade:
        out = base.convert("RGBA")
        out.alpha_composite(layer)
        return out.convert("RGB")
    share = max(0.0, elapsed / fade) if fade > 0 else 1.0
    # Ease-out: fast at first, settling. `1 - (1 - x)^2` and nothing cleverer;
    # an easing curve nobody can name is a curve nobody can tune.
    eased = 1.0 - (1.0 - share) ** 2
    offset = int(round(base.height * rise * (1.0 - eased)))
    moved = Image.new("RGBA", base.size, (0, 0, 0, 0))
    moved.paste(layer, (0, offset))
    out = base.convert("RGBA")
    out.alpha_composite(_alpha(moved, eased))
    return out.convert("RGB")


# --------------------------------------------------------------------------
# one shot, ready to be sampled at any time


@dataclass
class Plate:
    """One shot, decoded once and then cheap to sample.

    Holds ONE image: either the canvas-sized static frame, or the motion plate
    (the source scaled so the tightest crop window is exactly the canvas).
    24 plates at a 2560-wide canvas is about the same memory as the list of
    frames the hard-cut path already builds.
    """

    image: Image.Image
    canvas: tuple[int, int]
    caption: str = ""
    motion: KenBurns | None = None
    placement: str = ""
    is_title: bool = False
    duration: float = 1.0
    #: A photograph pasted at a FIXED position over a moving plate. This is
    #: the sub-canvas case: the backdrop drifts, the photograph does not.
    overlay: Image.Image | None = None
    overlay_at: tuple[int, int] = (0, 0)
    _caption_layer: Image.Image | None = field(default=None, repr=False)
    _title_layer: Image.Image | None = field(default=None, repr=False)

    def frame_at(self, progress: float) -> Image.Image:
        if self.is_title:
            base = Image.new("RGB", self.canvas, static.BACKGROUND)
            return _arrive(
                base,
                self._title_layer,
                progress,
                fade=TITLE_FADE,
                rise=TITLE_RISE,
                duration=self.duration,
            )
        if self.motion is None:
            base = self.image
        else:
            base = self.image.resize(
                self.canvas,
                # BILINEAR, not LANCZOS: this runs 25 times a second and the
                # crop is a small downscale of an already-reduced plate, where
                # the two are hard to tell apart. Measured 3.7 ms against
                # 76 ms a frame, which is the difference between a memory that
                # encodes in a minute and one that takes twenty.
                Image.Resampling.BILINEAR,
                box=self.motion.window_at(progress),
            )
            if self.overlay is not None:
                # Pasted AFTER the resize and at a fixed position, so the
                # photograph is pixel-for-pixel what the still render would
                # show while the blurred field behind it drifts.
                base.paste(self.overlay, self.overlay_at)
        return _arrive(
            base,
            self._caption_layer,
            progress,
            fade=CAPTION_FADE,
            rise=CAPTION_RISE,
            duration=self.duration,
        )


@dataclass
class MotionReport:
    """How many shots panned, how many drifted, how many held, and why."""

    moving: int = 0
    #: Sub-canvas shots whose BACKDROP drifts while the photograph holds.
    drifting: int = 0
    still: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def hold(self, reason: str) -> None:
        self.still += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    @property
    def total(self) -> int:
        return self.moving + self.drifting + self.still


HOLD_TOO_CROPPED = "filling the canvas would crop too much of them"


def build_plates(
    spec: MemorySpec,
    canvas: tuple[int, int],
    timeline: Timeline,
    *,
    resolve: Callable[[str], Photo | None],
    locate: Callable[[Photo], Path | None],
    with_captions: bool = True,
    with_title: bool = True,
    motion: bool = True,
) -> tuple[list[Plate], static.FrameReport, MotionReport]:
    """Decode a spec into plates, one per rendered shot.

    Shares `frames.fit_photo`'s placement rules rather than reimplementing
    them: a shot that holds still is byte-identical to the hard-cut render of
    the same shot, backdrop and all.
    """
    report = static.FrameReport()
    moves = MotionReport()
    plates: list[Plate] = []
    beats = list(timeline.beats)

    def hold_for(i: int) -> float:
        return beats[i].duration if i < len(beats) else 1.0

    if with_title:
        plates.append(
            Plate(
                image=Image.new("RGB", canvas, static.BACKGROUND),
                canvas=canvas,
                is_title=True,
                duration=hold_for(0),
                _title_layer=title_layers(spec.title, spec.subtitle, canvas),
            )
        )

    for shot in spec.shots:
        report.requested += 1
        photo = resolve(shot.file_hash)
        if photo is None:
            report.drop(static.DROP_NOT_IN_INDEX, shot.file_hash[:12])
            continue
        path = locate(photo)
        if path is None:
            report.drop(static.DROP_MISSING, photo.paths[0].name if photo.paths else "")
            continue
        try:
            plate = _plate_for(shot, path, canvas, motion=motion, moves=moves)
        except (OSError, ValueError, Image.DecompressionBombError):
            report.drop(static.DROP_UNDECODABLE, path.name)
            continue
        plate.duration = hold_for(len(plates))
        if with_captions:
            plate.caption = shot.caption
            plate._caption_layer = caption_layer(shot.caption, canvas)
        plates.append(plate)
        report.rendered += 1
        report.placement[plate.placement] = report.placement.get(plate.placement, 0) + 1
    return plates, report, moves


def _plate_for(shot, path: Path, canvas, *, motion: bool, moves: MotionReport) -> Plate:
    from rekindle.memory.composition import FIT_DOWNSCALE, FIT_PAD, plan_placement
    from rekindle.meta.exif import open_upright

    if not motion:
        frame, mode = static.fit_photo(path, canvas)
        return Plate(image=frame, canvas=canvas, placement=mode)

    # No `draft` hint here. `fit_photo` may ask the JPEG decoder for a smaller
    # image because it only ever needs the canvas; a moving shot needs the
    # surplus resolution that draft would throw away, which is the whole
    # reason it is allowed to move.
    rgb = open_upright(path).convert("RGB")
    plan = plan_motion(rgb.size, canvas, shot.file_hash)
    if plan is not None:
        moves.moving += 1
        return Plate(
            image=rgb.resize(plan.plate, Image.Resampling.LANCZOS),
            canvas=canvas,
            motion=plan,
            placement=FIT_DOWNSCALE,
        )

    target, mode = plan_placement(rgb.size, canvas)
    if mode == FIT_PAD:
        # The photograph is below the canvas, so it may not be zoomed. Its
        # backdrop drifts instead. See BACKDROP_ZOOM.
        drift = plan_backdrop(canvas, shot.file_hash)
        moves.drifting += 1
        return Plate(
            image=static.backdrop_at(rgb, drift.plate),
            canvas=canvas,
            motion=drift,
            overlay=rgb.resize(target, Image.Resampling.LANCZOS),
            overlay_at=((canvas[0] - target[0]) // 2, (canvas[1] - target[1]) // 2),
            placement=mode,
        )

    frame, mode = static.fit_photo(path, canvas)
    moves.hold(HOLD_TOO_CROPPED)
    return Plate(image=frame, canvas=canvas, placement=mode)


# --------------------------------------------------------------------------
# the film


def frames_for(plates: list[Plate], timeline: Timeline) -> Iterator[Image.Image]:
    """Every output frame of the memory, in order.

    A generator, not a list: a 46-second memory at 25 fps and 2560x1440 is
    1,150 frames of 11 MB, and materialising them is 12 GB. The encoder
    consumes them one at a time.

    The dissolve is a straight `Image.blend`, with the incoming shot painted
    OVER the outgoing one rather than both fading to black. A dip to black
    between every shot is a different and much worse effect.
    """
    beats = [b for b in timeline.beats if b.index < len(plates)]
    if not beats:
        return
    cache: dict[int, tuple[float, Image.Image]] = {}

    def render(beat: Beat, t: float) -> Image.Image:
        progress = beat.progress_at(t)
        hit = cache.get(beat.index)
        if hit is not None and hit[0] == progress:
            return hit[1]
        image = plates[beat.index].frame_at(progress)
        cache[beat.index] = (progress, image)
        return image

    for t in timeline.times():
        active = [b for b in beats if b.start <= t < b.end]
        if not active:
            continue
        frame = render(active[0], t)
        for beat in active[1:]:
            alpha = beat.alpha_at(t)
            if alpha <= 0.0:
                continue
            incoming = render(beat, t)
            frame = incoming if alpha >= 1.0 else Image.blend(frame, incoming, alpha)
        yield frame
        # A beat that is finished cannot be needed again, and each cached
        # frame is 11 MB at a 2560-wide canvas.
        for index in [i for i, (_, _) in cache.items() if not any(b.index == i for b in active)]:
            cache.pop(index, None)


def describe_motion(report: MotionReport) -> str:
    if not report.total:
        return ""
    parts = [f"{report.moving} of {report.total} shots pan"]
    if report.drifting:
        parts.append(
            f"{report.drifting} are below the canvas, so their backdrop drifts "
            "and the photo itself holds at native size"
        )
    for reason, count in sorted(report.reasons.items(), key=lambda kv: (-kv[1], kv[0])):
        parts.append(f"{count} hold still - {reason}")
    return "; ".join(parts)
