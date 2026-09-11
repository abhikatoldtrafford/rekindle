"""Photos that are only correct if the EXIF orientation tag is APPLIED.

Every test that decodes pixels needs a way to ask *"is this the right way
up?"* - and a size check cannot answer it. A 400x300 file tagged 90 degrees
decodes to 300x400 whether the transpose ran forwards, backwards, or was
replaced by a bare `width, height = height, width`; and orientations 2, 3 and
4 (mirrored, 180 degrees, flipped) do not change the size at all, so half the
tag values are invisible to any assertion about dimensions.

That is exactly how a missing transpose survived: the index stored
post-rotation `width`/`height`, tests asserted those numbers, and two decode
sites still handed the encoder and the face detector a photo on its side.

These fixtures put the answer in the PIXELS. `write_oriented` writes the
bytes a camera would actually have written - the upright image transformed
*backwards* through the tag - and `quadrants` reads back which corner each
colour ended up in, so a test can assert the picture, not its shape.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

ORIENTATION_TAG = 0x0112
ALL_ORIENTATIONS = (1, 2, 3, 4, 5, 6, 7, 8)

# `ImageOps.exif_transpose` applies the FORWARD transform for each tag. A
# fixture has to go backwards: from the upright image a viewer must see, to
# the pixels on disk. Five of the eight are their own inverse; 6 and 8 are
# each other's, which is precisely the pair a careless fixture gets wrong.
_INVERSE: dict[int, Image.Transpose | None] = {
    1: None,
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_90,  # exif_transpose applies ROTATE_270
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_270,  # exif_transpose applies ROTATE_90
}

# Four saturated, mutually distant quadrant colours. All four differ so that
# every one of the eight orientations produces a DIFFERENT arrangement: a
# two-colour image cannot tell a mirror from a rotation, and a grey one
# cannot tell anything. Saturated so a JPEG round trip cannot move one colour
# into another's basin.
_COLOURS: dict[str, tuple[int, int, int]] = {
    "TL": (230, 20, 20),
    "TR": (20, 200, 20),
    "BL": (20, 20, 230),
    "BR": (230, 200, 20),
}
UPRIGHT = ("TL", "TR", "BL", "BR")


def upright_marker(size: tuple[int, int] = (400, 300)) -> Image.Image:
    """The image a viewer MUST see: four differently-coloured quadrants."""
    width, height = size
    im = Image.new("RGB", size)
    half_w, half_h = width // 2, height // 2
    im.paste(_COLOURS["TL"], (0, 0, half_w, half_h))
    im.paste(_COLOURS["TR"], (half_w, 0, width, half_h))
    im.paste(_COLOURS["BL"], (0, half_h, half_w, height))
    im.paste(_COLOURS["BR"], (half_w, half_h, width, height))
    return im


def write_oriented(path: Path, orientation: int, size: tuple[int, int] = (400, 300)) -> Image.Image:
    """Write the JPEG a camera holding `orientation` would have written.

    Returns the upright image it must decode back to, so a caller can compare
    against the real thing rather than a second hand-written expectation.
    """
    upright = upright_marker(size)
    inverse = _INVERSE[orientation]
    raw = upright if inverse is None else upright.transpose(inverse)
    exif = raw.getexif()
    exif[ORIENTATION_TAG] = orientation
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.save(path, "JPEG", quality=95, exif=exif)
    return upright


def write_untagged(path: Path, size: tuple[int, int] = (400, 300)) -> Image.Image:
    """The same marker with NO orientation tag at all - 1,852 live images."""
    upright = upright_marker(size)
    path.parent.mkdir(parents=True, exist_ok=True)
    upright.save(path, "JPEG", quality=95)
    return upright


def _nearest(pixel: tuple[int, int, int]) -> str:
    r, g, b = pixel[:3]
    return min(
        _COLOURS,
        key=lambda name: sum((c - p) ** 2 for c, p in zip(_COLOURS[name], (r, g, b), strict=True)),
    )


def quadrants(
    image: Image.Image, box: tuple[int, int, int, int] | None = None
) -> tuple[str, str, str, str]:
    """Which marker colour sits in each quadrant, as (TL, TR, BL, BR).

    Sampled at each quadrant's CENTRE - a quarter and three quarters of the
    way across - so JPEG ringing at a colour boundary, a resize, or a caption
    burnt into the bottom edge cannot decide the answer. `box` restricts the
    read to the photo's area inside a larger, letterboxed frame.
    """
    left, top, right, bottom = box or (0, 0, image.width, image.height)
    width, height = right - left, bottom - top
    rgb = image.convert("RGB")
    xs = (left + width // 4, left + 3 * width // 4)
    ys = (top + height // 4, top + 3 * height // 4)
    return (
        _nearest(rgb.getpixel((xs[0], ys[0]))),
        _nearest(rgb.getpixel((xs[1], ys[0]))),
        _nearest(rgb.getpixel((xs[0], ys[1]))),
        _nearest(rgb.getpixel((xs[1], ys[1]))),
    )
