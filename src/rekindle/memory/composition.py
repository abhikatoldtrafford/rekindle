"""Composition guardrails: what makes a set of photos watchable together.

Selection (the recipes) answers "which photos belong together". This module
answers "which of those can actually be shown side by side" - and it is the
second half of the same promise, because a memory that mixes a portrait phone
photo with a landscape panorama and a 101x24 barcode looks broken however well
chosen its contents were.

Every rule here REJECTS, every rejection is counted with a reason, and the
count is surfaced. A guardrail that drops photos silently is the defect, not
the feature: a user who expected a photo and did not get it must be able to
find out why.

All thresholds were measured against the reference library (18,201 live
images) rather than guessed, and the measurements are recorded beside each
one. The bias throughout is conservative - this library runs back to 2000 and
older phone photos are genuinely soft and small, so a threshold tuned on 2025
photos would quietly delete the early years.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from rekindle.models import MediaType, Photo

# --------------------------------------------------------------------------
# thresholds, each with the measurement that chose it


# Square if the long edge is within 5% of the short one.
#
# Measured: 178 live images are EXACTLY 1:1, 210 are within 1.02, and only 21
# more fall in the 1.02-1.05 band. There is a natural cliff at 1.02 (deliberate
# square crops) and the band above it is nearly empty, so the exact tolerance
# barely matters; 1.05 is forgiving of a crop that is a pixel off without
# swallowing a real 4:3 (1.33).
SQUARE_RATIO = 1.05

# Minimum short edge, in native pixels.
#
# Measured percentiles of the short edge: p1=240, p2=480, p5=600, p10=1080,
# median=2976. 480 sits exactly on the knee and removes 303 images (1.66%),
# which is barcodes, thumbnails and tiny re-saves. Raising it to 640 would
# remove 986 (5.42%) and start eating genuine early-2010s phone photos, so the
# conservative end of the knee is the right place to stand.
MIN_SHORT_EDGE = 480

# Beyond this, letterboxing turns the photo into a sliver.
#
# Measured: 33 images (0.18%) exceed 2.5:1, and the extreme is a 8874x943 VR
# panorama at 9.41:1. Excluding them costs essentially nothing and prevents a
# panorama rendering as a 1280x136 band inside a black frame.
MAX_ASPECT = 2.5

# Quality gates, measured on 1,149 real photos stratified across every year.
#
# Mean luminance percentiles: p1=33, p2=44, median=113, p99=186. The gates at
# 20 and 235 therefore sit far outside anything the library actually contains
# (0.26% and 0.09%) and catch only true pocket shots and blown frames.
MIN_BRIGHTNESS = 20.0
MAX_BRIGHTNESS = 235.0

# Sharpness gate, RE-DERIVED from measurement when the measure changed, and
# then re-derived AGAIN when the whole library disagreed with the sample.
#
# `memory.fingerprint.sharpness` is now a reblur ratio in [0, 1], not a mean
# gradient running to about 25. The old 1.5 is meaningless on that scale, and
# scaling it by a guess would have been the worst of both: the distribution
# changed shape, not just units.
#
# Measured over ALL 18,363 fingerprinted photos: p1=0.214, p2=0.252, p5=0.315,
# median=0.543, p95=0.714. A 120-per-year sample put p1 at 0.236 and suggested
# 0.24; the full library says that would have rejected 1.50% and, worse, would
# have taken 2.25% of 2008-2013 against 0.83% of 2020-2026. The sample was
# uniform per year, which over-weights the sparse early years - so the number
# it produced was wrong in exactly the direction this gate must not be wrong.
#
# 0.12 was then chosen by LOOKING at what each candidate rejects. In the band
# 0.12-0.22, roughly a quarter of a hand-graded sample of 16 were photographs
# worth keeping - an 800x600 portrait, a 2012 face at 240x320 - because the
# measure is unreliable on the heavily compressed sub-megapixel files that
# make up 53% of 2011 and 7% of the library. Below 0.12, 15 of 16 were
# indefensible: a blown-out sun, out-of-focus blobs, flat sky, motion smears.
#
# Measured rejection at 0.12, whole library: 0.19% (34 photos) against the old
# gate's 0.99% (182). No year loses more than 0.70%. 0.15% of 2008-2013 and
# 0.16% of 2020-2026 - flat, and gentler than the old gate in every early year
# (2011: 0.44% against 3.83%; 2008: 0.00% against 2.82%).
#
# This gate is DELIBERATELY weaker than the one it replaces, and that is not a
# retreat from the goal. A gate that removes 1% of a library cannot be what
# keeps mild blur out of a 24-shot memory drawn from a pool of hundreds; the
# RANKING is, and sharpness is the ranking signal. That ranking used to score
# a sharp photo above a mildly blurred one 51.9% of the time and now does so
# 90.3% of the time. The gate's only job is to stop the indefensible from
# being ranked at all, and a false positive here deletes an irreplaceable
# photograph outright - so it belongs below the 1st percentile, not near it.
MIN_SHARPNESS = 0.12

# Common phone and desktop screen sizes, either orientation. Used ONLY in
# conjunction with a total absence of camera metadata - see `is_screenshot`.
_SCREEN_SIZES = frozenset(
    {
        (1220, 2712),
        (1192, 2649),
        (1080, 2340),
        (1080, 2400),
        (1080, 2280),
        (1080, 1920),
        (1440, 2560),
        (1440, 3120),
        (1125, 2436),
        (1170, 2532),
        (1179, 2556),
        (1284, 2778),
        (828, 1792),
        (750, 1334),
        (720, 1280),
        (640, 1136),
        (768, 1024),
        (1536, 2048),
        (1668, 2388),
        (1620, 2160),
        (1366, 768),
        (1280, 800),
        (1920, 1200),
        (2560, 1600),
        (3840, 2160),
    }
)

_SCREENSHOT_NAME = re.compile(r"^(screenshot|screen[ _-]shot)", re.IGNORECASE)

# Rejection reasons. Every one is reported.
DROP_VIDEO = "video"
DROP_NO_DIMENSIONS = "no_dimensions"
DROP_TOO_SMALL = "too_small"
DROP_EXTREME_ASPECT = "extreme_aspect"
DROP_SCREENSHOT = "screenshot"
DROP_TOO_DARK = "too_dark"
DROP_TOO_BRIGHT = "too_bright"
DROP_OUT_OF_FOCUS = "out_of_focus"
DROP_UNREADABLE = "unreadable"
DROP_MINORITY_ORIENTATION = "minority_orientation"


class Orientation(StrEnum):
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"
    SQUARE = "square"
    UNKNOWN = "unknown"


@dataclass
class CompositionReport:
    """What the guardrails removed, and why.

    Counted per reason and surfaced by the CLI. `examples` keeps one filename
    per reason so a user can see WHICH kind of photo a rule is catching -
    filenames only, never full paths, and only for photos the user already
    owns.
    """

    considered: int = 0
    kept: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    examples: dict[str, str] = field(default_factory=dict)
    orientation: Orientation = Orientation.UNKNOWN
    canvas: tuple[int, int] | None = None

    def drop(self, reason: str, photo: Photo | None = None) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1
        if photo is not None and reason not in self.examples and photo.paths:
            self.examples[reason] = photo.paths[0].name

    @property
    def total_dropped(self) -> int:
        return sum(self.dropped.values())

    @property
    def accounted(self) -> bool:
        return self.considered == self.kept + self.total_dropped

    def merge(self, other: CompositionReport) -> None:
        self.considered += other.considered
        self.kept += other.kept
        for reason, n in other.dropped.items():
            self.dropped[reason] = self.dropped.get(reason, 0) + n
        for reason, name in other.examples.items():
            self.examples.setdefault(reason, name)


def dimensions(photo: Photo) -> tuple[int, int] | None:
    """Post-rotation width and height, or None.

    The index stores these already transposed - see `meta.exif.read_exif` and
    the repair in `memory.fingerprint`. Nothing in this module may re-apply
    the orientation tag: it is applied exactly once, at measurement time.
    """
    width, height = photo.meta.width, photo.meta.height
    if not width or not height:
        return None
    return width, height


def classify(photo: Photo, *, square_ratio: float = SQUARE_RATIO) -> Orientation:
    size = dimensions(photo)
    if size is None:
        return Orientation.UNKNOWN
    width, height = size
    ratio = max(width, height) / min(width, height)
    if ratio <= square_ratio:
        return Orientation.SQUARE
    return Orientation.LANDSCAPE if width > height else Orientation.PORTRAIT


def aspect(photo: Photo) -> float | None:
    size = dimensions(photo)
    if size is None:
        return None
    return max(size) / min(size)


def is_screenshot(photo: Photo) -> bool:
    """Screenshot detection from metadata that actually exists.

    Requires BOTH a total absence of camera metadata AND dimensions matching a
    known screen size (or a filename following the Android/Windows screenshot
    convention, which is a specific format string rather than a guess).

    The conjunction is the point. 1,801 live images (9.9%) have no camera make
    or model, and most of them are perfectly ordinary photos whose EXIF was
    stripped by a re-save or a messaging app - dropping all of them was
    explicitly declined. Measured on the reference library, all 89 files whose
    name follows the screenshot convention also have no camera metadata, and
    their dimensions are 1220x2712, 1080x2340, 1192x2649, 1080x1920 and
    720x1280 - all real screen sizes.

    The two branches are NOT equally strong, and measurement showed it:

    * The filename convention is a positive assertion by the operating system
      ("Screenshot_20161008-222024.png"). 89 files match it and all 89 have no
      camera metadata. Trusted on its own.
    * A matching screen size is only circumstantial. It flagged 370 more
      files - mostly downloaded wallpapers at 1920x1200, which is the right
      answer - but 58 of them carry Google FACE TAGS, and a photo that Google
      found a person in is a photograph, not a screen capture. Those 58 are
      re-compressed photos (many from WhatsApp) that happen to land on a
      common screen size.

    So the size branch additionally requires that nobody was detected in the
    image. Face tags are positive evidence of a photograph and cost nothing to
    consult, since they are already in the index.

    WHAT THIS CANNOT DETECT: a PHOTOGRAPH of a document, a receipt or a
    whiteboard. Those carry ordinary camera metadata and ordinary dimensions
    and are indistinguishable from any other photo without a model, which v1
    deliberately does not have. They will appear in memories. Equally, an
    untagged re-compressed photo at exactly a screen size is still dropped -
    face tags cover only 55.9% of this library, so the exemption is not
    available for the other 44%.

    THE SIZE BRANCH DOES NOT APPLY TO A VIDEO FRAME, and this was found by
    measurement, not by reasoning. When `rekindle fingerprint` began extracting
    a still per video, **87 of the reference library's 242 extracted frames
    (36%) were immediately classified as screenshots.** None of them is one.
    The cause is that the commonest video resolutions and the commonest screen
    sizes are the same numbers - 70 frames at 1920x1080, 29 at 1080x1920, 8 at
    1280x720 - and a video carries no camera make or model at all (0 of 242),
    so the conjunction that protects photographs protects nothing here.

    The size branch's justification is specifically about DOWNLOADED IMAGES:
    "mostly downloaded wallpapers at 1920x1200, which is the right answer".
    Nobody downloads a wallpaper as a video. A file the user recorded is a
    recording, and its frame dimensions carry no evidence either way.

    The FILENAME branch still applies. `Screenshot_20161008-222024.mp4` is a
    screen RECORDING, which is exactly the thing this gate exists to keep out,
    and the operating system said so itself.
    """
    if photo.meta.camera_make or photo.meta.camera_model:
        return False
    if photo.paths and _SCREENSHOT_NAME.match(photo.paths[0].name):
        return True
    if any(photo.meta.people):
        return False
    if photo.media_type is MediaType.VIDEO:
        return False
    size = dimensions(photo)
    if size is None:
        return False
    width, height = size
    return (width, height) in _SCREEN_SIZES or (height, width) in _SCREEN_SIZES


def _reject(photo: Photo, *, min_short_edge: int, max_aspect: float) -> str | None:
    """Per-photo gates, in a fixed order. None means the photo is usable."""
    if photo.media_type is MediaType.VIDEO and photo.meta.phash is None:
        # A video with no extracted still frame. `rekindle fingerprint` caches
        # one per standalone video when ffmpeg is present (see
        # `memory.videoframe`), and once it has, the video carries a phash,
        # dimensions and a colour signature like any photograph and falls
        # through to the same gates below.
        #
        # THE GATE IS "HAS A STILL", NOT "FFMPEG IS INSTALLED". That
        # distinction is the whole of the design: selection reads the index
        # and a cached file, never a capability, so two machines sharing a
        # data directory build the same memory whether or not either of them
        # can decode video today.
        return DROP_VIDEO
    if photo.meta.phash_error:
        # Tried and could not be decoded. Counted, never silently dropped.
        #
        # No video exemption here any more, and its absence is deliberate:
        # every reason a video carries means "no still frame", and the branch
        # above has already returned DROP_VIDEO for exactly those rows. A
        # video that reaches this line has a phash, which means the fingerprint
        # pass succeeded, which means it cleared `phash_error`. The old
        # `!= "video"` test was reachable when "video" was the only reason;
        # with four of them it was a string comparison guarding a branch
        # nothing can enter, and a guard no test can fail is worse than none.
        return DROP_UNREADABLE

    size = dimensions(photo)
    if size is None:
        return DROP_NO_DIMENSIONS
    width, height = size
    if min(width, height) < min_short_edge:
        return DROP_TOO_SMALL
    if max(width, height) / min(width, height) > max_aspect:
        return DROP_EXTREME_ASPECT
    if is_screenshot(photo):
        return DROP_SCREENSHOT

    brightness = photo.meta.brightness
    if brightness is not None:
        if brightness < MIN_BRIGHTNESS:
            return DROP_TOO_DARK
        if brightness > MAX_BRIGHTNESS:
            return DROP_TOO_BRIGHT
    sharpness = photo.meta.sharpness
    if sharpness is not None and sharpness < MIN_SHARPNESS:
        return DROP_OUT_OF_FOCUS
    # A photo with no measured brightness or sharpness has simply never been
    # fingerprinted. It is KEPT: the quality gates are a filter on measured
    # evidence, not a requirement to have measured. `rekindle memory` warns
    # once when the index is unfingerprinted rather than silently emptying
    # every memory.
    return None


def cohesive_orientation(photos: list[Photo]) -> Orientation:
    """The orientation a set should be rendered in.

    Majority wins and the minority is dropped, rather than splitting the set
    into one memory per orientation. Two reasons: several recipes are not
    splittable (`then_and_now` is exactly two shots, and `person_years` walks
    one person through time - cutting it in half by orientation destroys the
    narrative), and splitting doubles the number of memories while halving
    each one, which works against the "return fewer photos than you think"
    rule the whole engine follows.

    Ties break towards LANDSCAPE - 73% of this library is landscape, and a
    montage is watched in a landscape frame. SQUARE is counted with neither:
    it letterboxes acceptably into either canvas, so it is a compatible
    minority rather than a competing majority.
    """
    counts = Counter(classify(p) for p in photos)
    portrait = counts[Orientation.PORTRAIT]
    landscape = counts[Orientation.LANDSCAPE]
    if portrait == 0 and landscape == 0:
        return Orientation.SQUARE if counts[Orientation.SQUARE] else Orientation.UNKNOWN
    return Orientation.PORTRAIT if portrait > landscape else Orientation.LANDSCAPE


def canvas_for(photos: list[Photo]) -> tuple[int, int] | None:
    """The render canvas: the MEDIAN width and the MEDIAN height of the set.

    Not the minimum. The minimum was the first rule here and it was
    catastrophic on a library spanning 2000-2026: one 640x480 photo from 2014
    pinned an entire "Paramita over the years" memory to 640x480, rendering
    two 7008x4672 photos at a 120th of their pixel count. Measured across 45
    memories under the old rule, **11 of them landed on 640x480** and one on
    1105x510 - each because of whichever single weakest photo happened to be
    selected.

    The median keeps the intent (a canvas the set actually supports, derived
    from the photos rather than a constant) without letting the weakest member
    dictate it. Width and height are still minimised INDEPENDENTLY of each
    other, so a set mixing 4:3 and 16:9 gets a canvas both can sit in.

    Photos above the canvas are downscaled. Photos BELOW it are not excluded
    and not stretched - see `plan_placement`.
    """
    sizes = [s for s in (dimensions(p) for p in photos) if s is not None]
    if not sizes:
        return None
    return (_median(sorted(s[0] for s in sizes)), _median(sorted(s[1] for s in sizes)))


def _median(values: list[int]) -> int:
    """Lower median. An even-length set takes the smaller of the two middles
    rather than averaging them, so the canvas is always a size at least half
    the set can meet or exceed, and is always an integer without rounding."""
    return values[(len(values) - 1) // 2]


# How far a photo may be upscaled to meet the canvas before it is padded
# instead. A 25% linear stretch is about 1.6x the pixel count and is
# imperceptible at viewing size; beyond that the softness starts to show, and
# "mush" is exactly what the never-upscale rule exists to prevent.
#
# The tolerance matters because without it a photo 3% below the canvas would
# be padded, which reads as an inconsistency rather than as a deliberate
# signal that the photo is older and smaller.
UPSCALE_TOLERANCE = 1.25

FIT_DOWNSCALE = "downscale"
FIT_UPSCALE = "upscale"
FIT_PAD = "pad"


def plan_placement(
    size: tuple[int, int], canvas: tuple[int, int], *, tolerance: float = UPSCALE_TOLERANCE
) -> tuple[tuple[int, int], str]:
    """How one photo meets the canvas: (rendered size, mode).

    Three cases, and **nothing is ever excluded for being small**:

    * larger than the canvas - downscale to fit, preserving aspect.
    * slightly smaller (within `tolerance`) - upscale to fit. Imperceptible,
      and it avoids padding a photo that is only marginally below.
    * far smaller - render at NATIVE size and pad the remainder. A 640x480
      photo from 2014 then reads as an old photo rather than a broken one,
      and the memory keeps its earliest years.

    Excluding the small ones was considered and rejected: on this library
    small means OLD, so it would quietly delete the early years of exactly the
    memories - "person over the years" - whose whole subject is the span. Two
    individually reasonable rules would have combined to defeat each other.
    """
    width, height = size
    # scale > 1 means the CANVAS is larger, i.e. the photo would have to be
    # blown up. scale < 1 means the photo is larger and must come down.
    scale = min(canvas[0] / width, canvas[1] / height)
    if scale <= 1.0:
        return (max(1, round(width * scale)), max(1, round(height * scale))), FIT_DOWNSCALE
    if scale <= tolerance:
        return (max(1, round(width * scale)), max(1, round(height * scale))), FIT_UPSCALE
    return (width, height), FIT_PAD


def compose(
    photos: list[Photo],
    *,
    min_short_edge: int = MIN_SHORT_EDGE,
    max_aspect: float = MAX_ASPECT,
    enforce_orientation: bool = True,
) -> tuple[list[Photo], CompositionReport]:
    """Apply every composition guardrail. Returns (usable, report).

    Order matters: per-photo gates run first, so that the orientation majority
    is computed over photos that are actually usable. Deciding orientation
    first would let a pile of rejected portrait thumbnails outvote the real
    landscape photos and empty the memory.
    """
    report = CompositionReport(considered=len(photos))

    usable: list[Photo] = []
    for photo in photos:
        reason = _reject(photo, min_short_edge=min_short_edge, max_aspect=max_aspect)
        if reason:
            report.drop(reason, photo)
        else:
            usable.append(photo)

    if enforce_orientation and usable:
        target = cohesive_orientation(usable)
        report.orientation = target
        cohesive = []
        for photo in usable:
            orientation = classify(photo)
            # SQUARE is compatible with either canvas, so it is never the
            # minority that gets dropped.
            if orientation in (target, Orientation.SQUARE):
                cohesive.append(photo)
            else:
                report.drop(DROP_MINORITY_ORIENTATION, photo)
        usable = cohesive
    elif usable:
        report.orientation = cohesive_orientation(usable)

    report.kept = len(usable)
    report.canvas = canvas_for(usable)
    return usable, report


def fit_within(size: tuple[int, int], canvas: tuple[int, int]) -> tuple[int, int]:
    """Scale `size` to fit inside `canvas`, preserving aspect, NEVER upscaling.

    A hard bound, used where a maximum matters and padding does not - the MP4
    and GIF canvas caps. Per-photo placement goes through `plan_placement`,
    which has the tolerance and the padding case.
    """
    width, height = size
    scale = min(canvas[0] / width, canvas[1] / height, 1.0)
    return (max(1, round(width * scale)), max(1, round(height * scale)))


def describe_drops(report: CompositionReport) -> list[str]:
    """Human-readable lines for the CLI, most-dropped first."""
    lines = []
    for reason, count in sorted(report.dropped.items(), key=lambda kv: (-kv[1], kv[0])):
        example = report.examples.get(reason)
        suffix = f" (e.g. {example})" if example else ""
        lines.append(f"{count} {reason.replace('_', ' ')}{suffix}")
    return lines


def name_of(photo: Photo) -> str:
    return photo.paths[0].name if photo.paths else photo.file_hash


__all__ = [
    "Orientation",
    "plan_placement",
    "UPSCALE_TOLERANCE",
    "FIT_DOWNSCALE",
    "FIT_UPSCALE",
    "FIT_PAD",
    "CompositionReport",
    "classify",
    "aspect",
    "is_screenshot",
    "cohesive_orientation",
    "canvas_for",
    "compose",
    "fit_within",
    "describe_drops",
    "dimensions",
    "name_of",
    "Path",
]
