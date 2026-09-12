"""Ken Burns, crossfades and cards, with no encoder anywhere in sight.

The assertion this file exists for is the first one: **over every frame of
every plan, no crop window is ever narrower or shorter than the canvas.** That
is the canvas rule - photographs above the canvas downscale, photographs below
it are never upscaled - expressed as an arithmetic invariant instead of a
promise, and it is swept over a grid of real photo shapes rather than checked
on one example.
"""

from __future__ import annotations

import itertools

import pytest
from PIL import Image

from rekindle.memory.render import motion, timeline
from rekindle.memory.spec import FactSheet, MemorySpec, Shot

# Real shapes from the reference library, plus the canvas shapes they land on.
SOURCES = [
    (4032, 3024),  # phone 4:3
    (4032, 2268),  # phone 16:9
    (7008, 4672),  # the biggest camera in the library
    (3984, 2988),  # the median
    (1600, 1200),  # a 2011 compact
    (640, 480),  # a 2008 phone
    (2048, 2048),  # square
    (3000, 1200),  # a wide crop
    # Found by sweeping for the tightest plate: at a 1280x960 canvas this
    # needs a 1046.4px-tall window from a 1046px plate, so the clamp in
    # `window_at` is what stops it panning off the bottom edge. Without a case
    # like this in the grid, deleting that clamp fails nothing.
    (1399, 1049),
]
CANVASES = [(2560, 1440), (2560, 1920), (1280, 960), (3984, 2988), (960, 1280)]


@pytest.mark.parametrize(("source", "canvas"), list(itertools.product(SOURCES, CANVASES)))
def test_a_crop_window_is_never_smaller_than_the_canvas(source, canvas):
    """THE invariant. A window smaller than the canvas is an upscale.

    Swept over every frame of the move, not just its ends: the interpolation
    is where an off-by-one would hide.
    """
    plan = motion.plan_motion(source, canvas, "seed")
    if plan is None:
        return
    for step in range(0, 101):
        left, top, right, bottom = plan.window_at(step / 100)
        assert right - left >= canvas[0], (step, source, canvas)
        assert bottom - top >= canvas[1], (step, source, canvas)


@pytest.mark.parametrize(("source", "canvas"), list(itertools.product(SOURCES, CANVASES)))
def test_a_crop_window_never_leaves_the_plate(source, canvas):
    """Off the edge is a black stripe down the side of someone's photograph."""
    plan = motion.plan_motion(source, canvas, "seed")
    if plan is None:
        return
    for step in range(0, 101):
        left, top, right, bottom = plan.window_at(step / 100)
        assert left >= 0 and top >= 0
        assert right <= plan.plate[0] and bottom <= plan.plate[1]


def test_a_photo_below_the_canvas_gets_no_motion_at_all():
    """The collision the whole design is about. A 640x480 photo on a
    2560x1440 canvas is drawn at native size on a backdrop, and a pan that
    zoomed into it would upscale it into exactly the mush the canvas rule
    exists to prevent."""
    assert motion.plan_motion((640, 480), (2560, 1440), "x") is None
    assert motion.plan_motion((1600, 1200), (2560, 1920), "x") is None


def test_a_photo_the_same_size_as_the_canvas_gets_no_motion():
    """There is no surplus to travel through."""
    assert motion.plan_motion((2560, 1440), (2560, 1440), "x") is None


def test_a_photo_that_would_be_cropped_too_hard_gets_no_motion():
    """A 4:3 photograph filling a 16:9 canvas loses a quarter of its height.

    Cropping a quarter off someone's photograph to make it move is a worse
    trade than letting it hold still.
    """
    assert motion.plan_motion((4032, 3024), (2560, 1440), "x") is None
    # The same photograph on a 4:3 canvas has almost nothing to lose.
    assert motion.plan_motion((4032, 3024), (2560, 1920), "x") is not None


def test_the_zoom_is_capped_however_much_surplus_there_is():
    """A 7008px photo on a 1280px canvas could zoom 5x. A 5x zoom is a zoom
    effect; Ken Burns is a drift you barely notice."""
    plan = motion.plan_motion((7008, 5256), (1280, 960), "x")
    assert plan is not None
    assert plan.zoom == pytest.approx(motion.MAX_ZOOM)


def test_the_move_is_the_same_for_the_same_photograph_every_time():
    """Derived from the file hash, so a photograph moves the same way in every
    memory it appears in and on every machine."""
    a = motion.plan_motion((4032, 3024), (2560, 1920), "abc123")
    b = motion.plan_motion((4032, 3024), (2560, 1920), "abc123")
    c = motion.plan_motion((4032, 3024), (2560, 1920), "def456")
    assert a == b
    assert (a.start, a.end, a.zoom_in) != (c.start, c.end, c.zoom_in)


def test_a_window_wider_than_its_plate_is_clamped_rather_than_panning_off():
    """`KenBurns` is public and constructible; the planners are not the only
    way one is made. A window wider than the plate reads as a black stripe
    down the side of the photograph."""
    plan = motion.KenBurns(
        plate=(100, 80),
        canvas=(120, 90),
        zoom=1.2,
        start=(0.9, 0.9),
        end=(0.1, 0.1),
        zoom_in=True,
    )
    for step in range(0, 101):
        left, top, right, bottom = plan.window_at(step / 100)
        assert left >= 0 and top >= 0
        assert right <= 100 and bottom <= 80


def test_the_plate_is_large_enough_for_the_whole_zoom():
    """The plate is the source scaled so the TIGHTEST window is the canvas.

    Scale it by the wide window instead and the plate is too small for the
    zoom, `window_at`'s clamp silently pins the window to the plate, and the
    shot pans without ever zooming - a slower, more expensive still.
    """
    for source in SOURCES:
        for canvas in CANVASES:
            plan = motion.plan_motion(source, canvas, "seed")
            if plan is None:
                continue
            assert plan.plate[0] >= canvas[0] * plan.zoom - 1
            assert plan.plate[1] >= canvas[1] * plan.zoom - 1


def test_the_zoom_actually_changes_the_window_size():
    """Not just its position. A pan with no zoom is half the effect."""
    for source in SOURCES:
        for canvas in CANVASES:
            plan = motion.plan_motion(source, canvas, "seed")
            if plan is None:
                continue
            first, last = plan.window_at(0.0), plan.window_at(1.0)
            assert (first[2] - first[0]) != (last[2] - last[0]), (source, canvas)


def test_the_window_actually_travels():
    """A plan whose start and end windows are identical is a still frame with
    the cost of a pan."""
    plan = motion.plan_motion((4032, 3024), (2560, 1920), "abc123")
    assert plan.window_at(0.0) != plan.window_at(1.0)


# --------------------------------------------------------------------------
# the backdrop drift


def test_the_backdrop_plate_is_larger_than_the_canvas():
    plan = motion.plan_backdrop((2560, 1440), "x")
    assert plan.plate[0] > 2560 and plan.plate[1] > 1440


def test_the_backdrop_window_is_never_smaller_than_the_canvas():
    plan = motion.plan_backdrop((2560, 1440), "x")
    for step in range(0, 101):
        left, top, right, bottom = plan.window_at(step / 100)
        assert right - left >= 2560
        assert bottom - top >= 1440
        assert right <= plan.plate[0] and bottom <= plan.plate[1]


def test_the_photograph_over_a_drifting_backdrop_does_not_move_one_pixel():
    """The whole point of the drift, and the thing that keeps the canvas rule.

    The backdrop moves; the photograph is pasted at a fixed position at
    exactly native size. If this ever changes, a sub-canvas photograph is
    being scaled, which is the failure the rule exists to prevent.
    """
    canvas = (400, 300)
    photo = Image.new("RGB", (160, 120), (200, 30, 30))
    plan = motion.plan_backdrop(canvas, "x")
    # A GRADIENT, not a flat colour: a flat backdrop looks identical however
    # far it drifts, and this test would pass with the drift deleted.
    plate = Image.linear_gradient("L").resize(plan.plate).convert("RGB")
    at = ((canvas[0] - 160) // 2, (canvas[1] - 120) // 2)
    plate_obj = motion.Plate(image=plate, canvas=canvas, motion=plan, overlay=photo, overlay_at=at)
    first = plate_obj.frame_at(0.0)
    last = plate_obj.frame_at(1.0)
    box = (at[0], at[1], at[0] + 160, at[1] + 120)
    assert first.crop(box).tobytes() == photo.tobytes()
    assert last.crop(box).tobytes() == photo.tobytes()
    # ...and the backdrop DID move, or this test proves nothing.
    assert first.tobytes() != last.tobytes()


# --------------------------------------------------------------------------
# cards that arrive


def test_the_title_fades_up_and_settles():
    canvas = (400, 300)
    plate = motion.Plate(
        image=Image.new("RGB", canvas),
        canvas=canvas,
        is_title=True,
        duration=3.5,
        _title_layer=motion.title_layers("A trip", "12 photos", canvas),
    )
    brightness = [_mean(plate.frame_at(p)) for p in (0.0, 0.05, 0.15, 0.5, 1.0)]
    assert brightness[0] < brightness[1] < brightness[2]
    # Arrived, and then still: nothing moves after the fade.
    assert plate.frame_at(0.5).tobytes() == plate.frame_at(1.0).tobytes()


def test_the_title_RISES_as_it_arrives():
    """Text that arrives rather than appears. Brightness alone does not
    distinguish a fade from a fade plus a rise, so this measures WHERE the
    text is: mid-fade it must sit lower than where it settles.
    """
    canvas = (400, 300)
    plate = motion.Plate(
        image=Image.new("RGB", canvas),
        canvas=canvas,
        is_title=True,
        duration=3.5,
        _title_layer=motion.title_layers("A trip", "12 photos", canvas),
    )
    early = _centroid_row(plate.frame_at(0.02))
    settled = _centroid_row(plate.frame_at(1.0))
    assert early > settled + 1.0, (early, settled)


def _centroid_row(image: Image.Image) -> float:
    """The brightness-weighted mean row. Where the text is."""
    grey = image.convert("L")
    pixels = list(grey.tobytes())
    width = grey.width
    total = weighted = 0.0
    for y in range(grey.height):
        row = sum(pixels[y * width : (y + 1) * width])
        total += row
        weighted += row * y
    return weighted / total if total else 0.0


def test_a_caption_arrives_and_then_holds():
    canvas = (400, 300)
    plate = motion.Plate(
        image=Image.new("RGB", canvas, (90, 90, 90)),
        canvas=canvas,
        caption="October 2019",
        duration=2.5,
        _caption_layer=motion.caption_layer("October 2019", canvas),
    )
    assert _mean(plate.frame_at(0.0)) < _mean(plate.frame_at(0.15))
    assert plate.frame_at(0.5).tobytes() == plate.frame_at(0.9).tobytes()


def test_no_caption_is_no_layer_and_no_cost():
    assert motion.caption_layer("", (400, 300)) is None


def _mean(image: Image.Image) -> float:
    from PIL import ImageStat

    return ImageStat.Stat(image.convert("L")).mean[0]


# --------------------------------------------------------------------------
# the film


def _plates(n: int, canvas=(80, 60)) -> list[motion.Plate]:
    return [
        motion.Plate(image=Image.new("RGB", canvas, (i * 40 % 256, 0, 0)), canvas=canvas)
        for i in range(n)
    ]


def test_the_frame_count_matches_the_timeline():
    plan = timeline.plan(3, seconds=1.0, title_seconds=1.0, crossfade=0.2, fps=10)
    frames = list(motion.frames_for(_plates(3), plan))
    assert len(frames) == plan.frames


def test_a_dissolve_produces_a_frame_that_is_neither_shot():
    """The one thing a crossfade has to do."""
    plan = timeline.plan(2, seconds=1.0, title_seconds=1.0, crossfade=0.4, fps=10)
    plates = _plates(2)
    frames = list(motion.frames_for(plates, plan))
    mid = plan.beats[1].start + plan.beats[1].fade / 2
    frame = frames[int(mid * plan.fps)]
    assert frame.getpixel((10, 10)) not in (
        plates[0].image.getpixel((10, 10)),
        plates[1].image.getpixel((10, 10)),
    )


def test_the_incoming_shot_is_painted_over_the_outgoing_one():
    """Not both fading to black. A dip to black between every shot is a
    different and much worse effect."""
    canvas = (40, 30)
    plates = [
        motion.Plate(image=Image.new("RGB", canvas, (255, 255, 255)), canvas=canvas),
        motion.Plate(image=Image.new("RGB", canvas, (255, 255, 255)), canvas=canvas),
    ]
    plan = timeline.plan(2, seconds=1.0, title_seconds=1.0, crossfade=0.4, fps=10)
    for frame in motion.frames_for(plates, plan):
        assert frame.getpixel((5, 5)) == (255, 255, 255), "white into white must stay white"


def test_more_plates_than_beats_is_not_an_index_error():
    """A shot can fail to decode after the timeline was planned."""
    plan = timeline.plan(2, fps=5)
    assert list(motion.frames_for(_plates(5), plan))


def test_fewer_plates_than_beats_drops_the_missing_beats():
    plan = timeline.plan(5, fps=5)
    frames = list(motion.frames_for(_plates(2), plan))
    assert frames
    assert len(frames) < plan.frames


def test_no_plates_yields_no_frames():
    assert list(motion.frames_for([], timeline.plan(3))) == []


# --------------------------------------------------------------------------
# decoding a spec


def _fake_photo():
    from types import SimpleNamespace

    return SimpleNamespace(paths=[])


def _spec(hashes) -> MemorySpec:
    return MemorySpec(
        recipe="scenery",
        key="sea",
        title="The sea",
        subtitle="",
        shots=tuple(Shot(h, "", "2019-01-01T00:00:00", True) for h in hashes),
        facts=FactSheet(title="The sea", recipe="scenery", photo_count=len(hashes)),
        public_safe=True,
    )


def test_a_missing_file_is_dropped_and_counted_not_rendered_black(tmp_path):
    """Half a rendered memory with an honest report beats a traceback, and
    beats a black slot nobody can explain."""
    spec = _spec(["a"])
    plates, report, moves = motion.build_plates(
        spec,
        (200, 150),
        timeline.plan(2),
        resolve=lambda h: _fake_photo(),
        locate=lambda p: None,
    )
    assert report.rendered == 0
    assert report.dropped == {"file_missing": 1}
    assert report.accounted
    # The title card is still there: a memory whose photos are all on an
    # unmounted drive should still say what it was.
    assert len(plates) == 1 and plates[0].is_title


def test_motion_off_gives_the_same_placement_as_the_hard_cut_renderer(tmp_path):
    """`--style cuts` must be exactly what it always was."""
    from rekindle.memory.render import frames as fr

    path = tmp_path / "p.jpg"
    Image.new("RGB", (800, 600), (30, 120, 200)).save(path)
    expected, mode = fr.fit_photo(path, (400, 300))
    plate = motion._plate_for(
        Shot("h", "", None, True), path, (400, 300), motion=False, moves=motion.MotionReport()
    )
    assert plate.placement == mode
    assert plate.image.tobytes() == expected.tobytes()
    assert plate.motion is None


def test_the_report_accounts_for_every_shot():
    report = motion.MotionReport()
    report.moving += 2
    report.drifting += 1
    report.hold(motion.HOLD_TOO_CROPPED)
    assert report.total == 4
    assert "2 of 4 shots pan" in motion.describe_motion(report)
    assert "backdrop drifts" in motion.describe_motion(report)


def test_describe_motion_says_nothing_about_nothing():
    assert motion.describe_motion(motion.MotionReport()) == ""
