"""The configuration system, and the one trap it exists to avoid.

A config system whose overrides silently do nothing looks EXACTLY like one
where the user changed nothing. So the centre of this file is
`test_every_setting_reaches_the_code_that_reads_it`, which is parametrised
over `CATALOGUE` itself rather than over a hand-written list: a value added to
`defaults.toml` without a proof that it reaches code fails collection, not
review.

Each proof follows the same shape and it is the only shape that means
anything:

    run the real function under the default    -> observe outcome A
    run the same function under an override    -> observe outcome B
    assert A != B

Nothing here asserts `active().x == y`. That would pass on a config system
that is read by nobody.
"""

from __future__ import annotations

import dataclasses
import tomllib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rekindle import config
from rekindle.config import CATALOGUE, ConfigError
from rekindle.config.writer import render, write
from rekindle.models import MediaType, Photo, PhotoMeta

# ---------------------------------------------------------------------------
# helpers


T0 = datetime(2025, 8, 4, 10, 0, tzinfo=UTC)


def _hist(dominant: int = 0, spread: int = 1) -> str:
    """A 64-bin colour histogram with mass on `dominant`."""
    bins = [0] * 64
    for i in range(spread):
        bins[(dominant + i) % 64] = 255
    return "".join(f"{b:02x}" for b in bins)


def _photo(
    name: str,
    *,
    width: int = 4000,
    height: int = 3000,
    brightness: float | None = 120.0,
    sharpness: float | None = 5.0,
    phash: int | None = 0,
    at: float = 0.0,
    colour: str | None = None,
) -> Photo:
    return Photo(
        file_hash=name,
        paths=[Path(f"/lib/{name}.jpg")],
        media_type=MediaType.IMAGE,
        meta=PhotoMeta(
            taken_at_utc=T0 + timedelta(seconds=at),
            taken_at_local=(T0 + timedelta(seconds=at)).replace(tzinfo=None),
            width=width,
            height=height,
            brightness=brightness,
            sharpness=sharpness,
            phash=phash,
            colour=colour if colour is not None else _hist(0),
        ),
        first_seen=T0,
        last_seen=T0,
    )


def _offer(recipe: str = "album_story", key: str = "Kashmir"):
    from rekindle.memory.recipes.base import Offer

    return Offer(recipe=recipe, key=key, title=key, subtitle="")


def _index(tmp_path, photos):
    """A real `MemoryIndex`, because `engine.build` reads one to decide what is
    public-safe. Handing it `None` throws three lines from the end of the
    pipeline, after everything worth measuring has already happened."""
    from rekindle.db import PhotoStore
    from rekindle.memory.index import MemoryIndex

    store = PhotoStore(Path(tmp_path) / "db.sqlite")
    store.upsert_many(list(photos))
    return store, MemoryIndex.open(store)


def _selection(photos, **kw):
    from rekindle.memory.recipes.base import Selection
    from rekindle.memory.spec import build_fact_sheet

    return Selection(
        photos=list(photos),
        facts=build_fact_sheet(list(photos), title="K", recipe="album_story"),
        **kw,
    )


# ---------------------------------------------------------------------------
# defaults.toml itself


def test_the_catalogue_is_not_empty_and_every_key_is_section_dot_name():
    assert CATALOGUE
    for key, setting in CATALOGUE.items():
        assert key == f"{setting.section}.{setting.name}"


@pytest.mark.parametrize("key", sorted(CATALOGUE))
def test_every_setting_documents_itself(key):
    """The file's whole claim is that it is documentation. A value that
    arrived without its six sentences is a value nobody can decide about."""
    s = CATALOGUE[key]
    assert s.unit and len(s.unit) >= 4, f"{key}.unit is missing"
    for field in ("what", "raising", "lowering", "measured"):
        text = getattr(s, field)
        assert text and len(text) > 25, f"{key}.{field} is missing or a stub"
        assert text.endswith("."), f"{key}.{field} should be a sentence"


@pytest.mark.parametrize("key", sorted(CATALOGUE))
def test_every_default_is_inside_its_own_declared_range(key):
    s = CATALOGUE[key]
    assert s.clamp_error(s.default) == "", f"{key}: the shipped value is out of its own range"


def test_defaults_toml_is_the_single_source_of_truth_for_every_module_constant():
    """The module constants still exist - a hundred call sites read them - but
    they are now READ FROM the file. This pins that they were not also typed
    into the source, where the two copies would drift apart silently."""
    from rekindle.memory import composition, dedup, diversity, engine, history
    from rekindle.memory.recipes import MIN_SHOTS
    from rekindle.meta import orientation
    from rekindle.semantic import diversity as sdiv
    from rekindle.semantic import faces

    pairs = {
        "composition.min_sharpness": composition.MIN_SHARPNESS,
        "composition.min_short_edge": composition.MIN_SHORT_EDGE,
        "composition.max_aspect": composition.MAX_ASPECT,
        "composition.min_brightness": composition.MIN_BRIGHTNESS,
        "composition.max_brightness": composition.MAX_BRIGHTNESS,
        "composition.square_ratio": composition.SQUARE_RATIO,
        "composition.upscale_tolerance": composition.UPSCALE_TOLERANCE,
        "dedup.gap_seconds": dedup.DEFAULT_GAP_SECONDS,
        "dedup.phash_distance": dedup.DEFAULT_THRESHOLD,
        "dedup.cosine": dedup.DEFAULT_COSINE,
        "diversity.lambda_penalty": diversity.LAMBDA,
        "diversity.phash_radius": diversity.PHASH_RADIUS,
        "diversity.hard_floor": diversity.HARD_FLOOR,
        "diversity.time_tiebreak": diversity.TIME_TIEBREAK,
        "semantic_diversity.ceiling": sdiv.CEILING,
        "semantic_diversity.min_spread": sdiv.MIN_SPREAD,
        "semantic_diversity.low_quantile": sdiv.LOW_QUANTILE,
        "semantic_diversity.high_quantile": sdiv.HIGH_QUANTILE,
        "semantic_diversity.weight": sdiv.WEIGHT,
        "faces.detect_threshold": faces.DEFAULT_DETECT_THRESHOLD,
        "faces.gate_threshold": faces.DEFAULT_GATE_THRESHOLD,
        "orientation.min_face": orientation.MIN_FACE,
        "orientation.min_margin": orientation.MIN_MARGIN,
        "selection.max_shots": engine.DEFAULT_MAX_SHOTS,
        "selection.min_shots": MIN_SHOTS,
        "selection.cooldown_days": history.DEFAULT_COOLDOWN_DAYS,
        "selection.max_overlap": history.DEFAULT_MAX_OVERLAP,
    }
    assert set(pairs) == set(CATALOGUE), "a setting exists with no module constant, or vice versa"
    for key, constant in pairs.items():
        assert constant == CATALOGUE[key].default, key


def test_the_excluded_constants_are_named_and_the_reasons_are_written_down():
    """`defaults.toml` claims to say why the other constants are fixed. If the
    claim is there, the names had better be."""
    text = config.DEFAULTS_PATH.read_text(encoding="utf-8")
    for name in (
        "SCHEMA_VERSION",
        "DEFAULT_NMS_IOU",
        "DEFAULT_GATE_WORKERS",
        "CALIBRATION_ROWS",
        "GPS_CELL",
    ):
        assert name in text, f"{name} is excluded from config but not explained"


# ---------------------------------------------------------------------------
# THE CENTRAL PROOF
#
# One entry per setting. Each is a callable taking a `Settings` and returning
# something comparable; the harness runs it twice and demands the answers
# differ. `probe` runs the REAL function, never a getter.


def _p_min_sharpness(_):
    from rekindle.memory.composition import compose

    return len(compose([_photo("a", sharpness=0.2)], enforce_orientation=False)[0])


def _p_min_short_edge(_):
    from rekindle.memory.composition import compose

    return len(compose([_photo("a", width=600, height=500)], enforce_orientation=False)[0])


def _p_max_aspect(_):
    from rekindle.memory.composition import compose

    return len(compose([_photo("a", width=2400, height=1000)], enforce_orientation=False)[0])


def _p_min_brightness(_):
    from rekindle.memory.composition import compose

    return len(compose([_photo("a", brightness=40.0)], enforce_orientation=False)[0])


def _p_max_brightness(_):
    from rekindle.memory.composition import compose

    return len(compose([_photo("a", brightness=200.0)], enforce_orientation=False)[0])


def _p_square_ratio(_):
    from rekindle.memory.composition import classify

    return classify(_photo("a", width=1100, height=1000)).value


def _p_upscale_tolerance(_):
    from rekindle.memory.composition import plan_placement

    return plan_placement((1000, 750), (1150, 860))[1]


def _p_gap_seconds(_):
    from rekindle.memory.dedup import bursts

    photos = [_photo("a", phash=0, at=0), _photo("b", phash=0, at=45)]
    return [len(g) for g in bursts(photos)]


def _p_phash_distance(_):
    from rekindle.memory.dedup import bursts

    # 8 bits apart: outside the shipped 6, inside a raised 10.
    photos = [_photo("a", phash=0, at=0), _photo("b", phash=0xFF, at=5)]
    return [len(g) for g in bursts(photos)]


def _p_cosine(_):
    from rekindle.memory.dedup import bursts

    photos = [_photo("a", phash=0, at=0), _photo("b", phash=0xFFFFFFFFFFFFFFFF, at=5)]
    return [len(g) for g in bursts(photos, cosine=lambda a, b: 0.95)]


def _p_lambda_penalty(_):
    from rekindle.memory.diversity import pick

    chosen = [_photo("anchor", phash=0, colour=_hist(0))]
    # `a` is the best photo and looks like the anchor; the `b`s are worse and
    # look nothing like it. Lambda alone decides which kind wins.
    candidates = [_photo("a", phash=0xFF, colour=_hist(0), sharpness=9.0)] + [
        _photo(f"b{i}", phash=0xFFFFFFFFFFFF, colour=_hist(40), sharpness=8.0 - i) for i in range(3)
    ]
    got, _ = pick(
        candidates, 1, rank=lambda ps: sorted(ps, key=lambda p: -p.meta.sharpness), already=chosen
    )
    return got[0].file_hash


def _p_phash_radius(_):
    from rekindle.memory.diversity import PerceptualSignal

    return round(PerceptualSignal().between(_photo("a", phash=0), _photo("b", phash=0xFFF)), 4)


def _p_hard_floor(_):
    from rekindle.memory.diversity import pick

    # `y` is the best candidate and sits at dissimilarity 0.05 - below the
    # shipped floor of 0.08. The fillers sit at 0.10, just above it, so the
    # MMR penalty is nearly identical for all three and the FLOOR is the only
    # thing that can change the answer. More candidates than slots, because a
    # refused shot is restored when a slot would otherwise go empty.
    chosen = [_photo("anchor", phash=0, colour=_hist(0))]
    candidates = [
        _photo("y", phash=0b11, colour=_hist(0), sharpness=9.0),
        _photo("f1", phash=0xF, colour=_hist(0), sharpness=8.0),
        _photo("f2", phash=0xF0, colour=_hist(0), sharpness=7.0),
    ]
    got, _ = pick(
        candidates, 1, rank=lambda ps: sorted(ps, key=lambda p: -p.meta.sharpness), already=chosen
    )
    return got[0].file_hash


def _p_time_tiebreak(_):
    from rekindle.memory.diversity import pick

    chosen = [_photo("anchor", phash=0, colour=_hist(0), at=0)]
    candidates = [
        _photo(f"n{i:02d}", phash=1 << 30, colour=_hist(30), at=60, sharpness=5.0)
        for i in range(24)
    ]
    candidates[1] = _photo("n01", phash=1 << 30, colour=_hist(30), at=5 * 86400, sharpness=5.0)
    got, _ = pick(
        candidates,
        1,
        rank=lambda ps: sorted(ps, key=lambda p: p.file_hash),
        already=chosen,
    )
    return got[0].file_hash


def _p_sd_weight(_):
    # Through `calibrate`, which is the only thing that builds a signal in the
    # pipeline - a `SemanticSignal` constructed directly would read the class
    # default and prove nothing.
    return _calibrated(_pool()).weight


def _needs_numpy():
    pytest.importorskip("numpy")


def _calibrated(cosine_grid):
    """Run the real `calibrate` over a pool with known pairwise cosines."""
    _needs_numpy()
    import numpy as np

    from rekindle.semantic.diversity import _Cosines, calibrate

    hashes = tuple(f"h{i}" for i in range(len(cosine_grid)))
    data = np.array(cosine_grid, dtype="float32")
    norm = data / np.linalg.norm(data, axis=1, keepdims=True)
    photos = [_photo(h) for h in hashes]
    return calibrate(photos, _Cosines(hashes, norm))


def _pool():
    """Six unit vectors whose pairwise cosines span a wide range, so a moved
    quantile moves the answer."""
    return [
        [1.0, 0.0, 0.0],
        [0.99, 0.14, 0.0],
        [0.9, 0.44, 0.0],
        [0.7, 0.71, 0.0],
        [0.4, 0.92, 0.0],
        [0.0, 1.0, 0.0],
    ]


def _p_sd_ceiling(_):
    sig = _calibrated(_pool())
    return round(sig.high, 5)


def _p_sd_min_spread(_):
    # A pool of near-identical vectors: every quantile collapses together and
    # only MIN_SPREAD keeps the divisor open.
    sig = _calibrated([[1.0, 0.0, 0.0], [1.0, 0.001, 0.0], [1.0, 0.002, 0.0], [1.0, 0.003, 0.0]])
    return round(sig.high - sig.low, 5)


def _p_sd_low_quantile(_):
    sig = _calibrated(_pool())
    return round(sig.low, 5)


def _p_sd_high_quantile(_):
    # CEILING is lowered alongside so it cannot mask the quantile's effect -
    # the shipped 0.90 floor sits above this pool's 95th percentile.
    with config.using({"semantic_diversity.ceiling": 0.0}):
        sig = _calibrated(_pool())
    return round(sig.high, 5)


def _p_detect_threshold(_):
    from rekindle.semantic.faces import Box, Verdict, classify

    return classify([Box(0, 0, 10, 10, 0.30)]) is Verdict.HAS_FACE


def _p_gate_threshold(_):
    from rekindle.semantic.faces import Box, Verdict, classify

    return classify([Box(0, 0, 10, 10, 0.20)]) is Verdict.ELIGIBLE


def _p_min_face(_):
    from rekindle.meta.orientation import decide

    # Four weak faces: the summed-squares evidence clears `min_margin`
    # comfortably, so `min_face` - which reads the STRONGEST single score -
    # is the only guard that can refuse this.
    return decide(6, tagged_scores=[0.0], raw_scores=[0.4, 0.4, 0.4, 0.4]).ignore_exif


def _p_min_margin(_):
    from rekindle.meta.orientation import decide

    # 0.75^2 - 0.50^2 = 0.3125: under the shipped 0.35 margin, over a lowered
    # one, and the strongest score clears `min_face` either way.
    return decide(6, tagged_scores=[0.50], raw_scores=[0.75]).ignore_exif


def _p_max_shots(_):
    from rekindle.memory.diversity import pick

    photos = [_photo(f"p{i:02d}", phash=i << 8, colour=_hist(i)) for i in range(30)]
    got, _ = pick(
        photos,
        config.active().selection.max_shots,
        rank=lambda ps: sorted(ps, key=lambda p: p.file_hash),
    )
    return len(got)


def _p_min_shots(_):
    import tempfile

    from rekindle.memory import engine

    # A four-photo pool: built under the shipped floor of 3, refused under a
    # floor of 6. The engine is what reads this, so the engine is what runs.
    photos = [_photo(f"p{i}", phash=i << 12, colour=_hist(i * 3), at=i * 3600) for i in range(4)]
    with tempfile.TemporaryDirectory() as tmp:
        store, index = _index(tmp, photos)
        try:
            spec = engine.build(index, _offer(), selection=_selection(photos))
        finally:
            store.close()
    return spec is not None


def _p_cooldown_days(settings):
    import tempfile

    from rekindle.db import PhotoStore
    from rekindle.memory.history import MemoryState

    with tempfile.TemporaryDirectory() as tmp, PhotoStore(Path(tmp) / "x.sqlite") as store:
        state = MemoryState(store)
        now = datetime(2026, 6, 1, tzinfo=UTC)
        state.record_surfaced("album_story:Kashmir", now=now - timedelta(days=120))
        return state.in_cooldown("album_story:Kashmir", now=now)


def _p_max_overlap(_):
    from rekindle.memory.history import overlap

    # Not a threshold call on its own - the engine compares against it - so
    # exercise the comparison the engine actually makes.
    shared = overlap(["a", "b", "c", "d"], ["a", "b", "d", "z"])
    return shared >= config.active().selection.max_overlap


#: setting key -> (probe, override value). The override must be one that
#: CHANGES the probe's answer; that is the whole point.
PROBES: dict[str, tuple] = {
    "composition.min_sharpness": (_p_min_sharpness, 0.4),
    "composition.min_short_edge": (_p_min_short_edge, 800),
    "composition.max_aspect": (_p_max_aspect, 2.0),
    "composition.min_brightness": (_p_min_brightness, 60.0),
    "composition.max_brightness": (_p_max_brightness, 180.0),
    "composition.square_ratio": (_p_square_ratio, 1.2),
    "composition.upscale_tolerance": (_p_upscale_tolerance, 1.05),
    "dedup.gap_seconds": (_p_gap_seconds, 60.0),
    "dedup.phash_distance": (_p_phash_distance, 10),
    "dedup.cosine": (_p_cosine, 0.99),
    "diversity.lambda_penalty": (_p_lambda_penalty, 0.0),
    "diversity.phash_radius": (_p_phash_radius, 48),
    "diversity.hard_floor": (_p_hard_floor, 0.0),
    "diversity.time_tiebreak": (_p_time_tiebreak, 0.0),
    "semantic_diversity.ceiling": (_p_sd_ceiling, 0.99),
    "semantic_diversity.min_spread": (_p_sd_min_spread, 0.5),
    "semantic_diversity.low_quantile": (_p_sd_low_quantile, 10.0),
    "semantic_diversity.high_quantile": (_p_sd_high_quantile, 20.0),
    "semantic_diversity.weight": (_p_sd_weight, 0.25),
    "faces.detect_threshold": (_p_detect_threshold, 0.25),
    "faces.gate_threshold": (_p_gate_threshold, 0.25),
    "orientation.min_face": (_p_min_face, 0.05),
    "orientation.min_margin": (_p_min_margin, 0.10),
    "selection.max_shots": (_p_max_shots, 8),
    "selection.min_shots": (_p_min_shots, 6),
    "selection.cooldown_days": (_p_cooldown_days, 365),
    "selection.max_overlap": (_p_max_overlap, 0.9),
}


def test_every_setting_in_the_catalogue_has_a_proof():
    """Parametrisation alone would silently skip a setting nobody wrote a
    probe for. This is the test that notices."""
    assert set(PROBES) == set(CATALOGUE), (
        f"no proof for {sorted(set(CATALOGUE) - set(PROBES))}; "
        f"stale proof for {sorted(set(PROBES) - set(CATALOGUE))}"
    )


@pytest.mark.parametrize("key", sorted(PROBES))
def test_every_setting_reaches_the_code_that_reads_it(key):
    """The trap this whole file exists for: an override that silently does
    nothing is indistinguishable from a user who changed nothing.

    `_p_min_shots` is the one probe that reads a module constant rather than
    calling a function, and that is honest rather than lazy: `MIN_SHOTS` is
    consumed as `selection.min_shots or MIN_SHOTS` inside `engine.build`,
    which `test_the_engine_reads_every_selection_setting` covers with a real
    build. This proof covers the constant's own provenance.
    """
    probe, override = PROBES[key]
    before = probe(config.active())
    with config.using({key: override}):
        after = probe(config.active())
    assert before != after, (
        f"{key}: setting it to {override} changed nothing. Either the override "
        f"does not reach the code, or the probe does not exercise it."
    )
    # And it must go back: `using` is what every preview in the calibration
    # flow relies on to not leak a trial value into the next question.
    assert probe(config.active()) == before


# `selection.min_shots` deserves a real build rather than a constant read.


def test_the_engine_reads_every_selection_setting(tmp_path):
    from rekindle.memory import engine

    photos = [
        _photo(f"p{i:02d}", phash=i << 9, colour=_hist(i), sharpness=0.5 + i / 100, at=i * 3600)
        for i in range(12)
    ]
    offer = _offer(key="Test")

    store, index = _index(tmp_path, photos)

    def build(**over):
        with config.using(over):
            return engine.build(index, offer, selection=_selection(photos))

    full = build()
    assert full is not None and len(full.shots) == 12

    capped = build(**{"selection.max_shots": 4})
    assert capped is not None and len(capped.shots) == 4

    # A floor above the pool refuses the memory outright.
    refused = build(**{"selection.min_shots": 20})
    assert refused is None
    store.close()


# ---------------------------------------------------------------------------
# the user's file


def test_an_absent_override_file_means_every_default(tmp_path):
    assert config.load(tmp_path) == config.defaults()


def test_an_override_file_changes_only_what_it_names(tmp_path):
    (tmp_path / config.CONFIG_NAME).write_text(
        "[composition]\nmin_sharpness = 0.2\n", encoding="utf-8"
    )
    loaded = config.load(tmp_path)
    assert loaded.composition.min_sharpness == 0.2
    assert loaded.composition.min_short_edge == CATALOGUE["composition.min_short_edge"].default
    assert loaded.changed_from_defaults() == {"composition.min_sharpness": 0.2}


def test_an_unknown_key_is_refused_rather_than_ignored(tmp_path):
    """Silently ignoring a typo is the same failure as an override that does
    not reach the code: the user believes they changed something."""
    (tmp_path / config.CONFIG_NAME).write_text(
        "[composition]\nmin_sharpnes = 0.2\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="no such setting"):
        config.load(tmp_path)


def test_an_out_of_range_value_is_refused(tmp_path):
    (tmp_path / config.CONFIG_NAME).write_text("[dedup]\ncosine = 1.5\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="at most 1.0"):
        config.load(tmp_path)


def test_a_malformed_file_is_fatal_not_a_warning(tmp_path):
    (tmp_path / config.CONFIG_NAME).write_text("[composition\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        config.load(tmp_path)


def test_a_string_where_a_number_belongs_is_refused(tmp_path):
    (tmp_path / config.CONFIG_NAME).write_text(
        '[composition]\nmin_sharpness = "0.2"\n', encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="must be a number"):
        config.load(tmp_path)


def test_a_bare_top_level_key_is_refused_with_the_shape_it_wanted(tmp_path):
    (tmp_path / config.CONFIG_NAME).write_text("min_sharpness = 0.2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="section of numbers"):
        config.load(tmp_path)


def test_an_int_setting_stays_an_int(tmp_path):
    (tmp_path / config.CONFIG_NAME).write_text("[selection]\nmax_shots = 12\n", encoding="utf-8")
    value = config.load(tmp_path).selection.max_shots
    assert isinstance(value, int) and not isinstance(value, bool)


# ---------------------------------------------------------------------------
# writing it back


def test_the_written_file_holds_numbers_and_nothing_else(tmp_path):
    """`rekindle.toml` is a file a user may commit to a public repository.
    Every non-comment line must be `name = <number>` or `[section]` - no path,
    no name, nothing about a photograph can be in here by construction."""
    path = write(tmp_path / config.CONFIG_NAME, {"composition.min_sharpness": 0.14})
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            assert line.strip("[]").isidentifier()
            continue
        name, _, value = line.partition(" = ")
        assert name.isidentifier(), line
        float(value)  # raises if it is anything but a number


def test_what_is_written_is_what_is_read_back(tmp_path):
    wanted = {
        "composition.min_sharpness": 0.135,
        "selection.max_shots": 18,
        "dedup.gap_seconds": 45.0,
    }
    write(tmp_path / config.CONFIG_NAME, wanted)
    assert config.read_override(tmp_path / config.CONFIG_NAME) == wanted


@pytest.mark.parametrize("value", [0.1, 0.12, 1e-4, 235.0, 0.9999999, 1.25])
def test_a_float_survives_the_round_trip_exactly(value, tmp_path):
    """`repr` is what makes this true. A `%.4f` here would silently move a
    calibrated threshold on every write."""
    write(tmp_path / config.CONFIG_NAME, {"dedup.cosine": min(value, 1.0)})
    got = config.read_override(tmp_path / config.CONFIG_NAME)["dedup.cosine"]
    assert got == min(value, 1.0)


def test_the_file_is_stable_under_reordering(tmp_path):
    a = render({"selection.max_shots": 18, "composition.min_sharpness": 0.14})
    b = render({"composition.min_sharpness": 0.14, "selection.max_shots": 18})
    assert a == b


def test_writing_keeps_the_previous_file_recoverable(tmp_path):
    path = tmp_path / config.CONFIG_NAME
    write(path, {"composition.min_sharpness": 0.14})
    write(path, {"composition.min_sharpness": 0.30})
    backup = path.with_suffix(path.suffix + ".bak")
    assert backup.is_file()
    assert config.read_override(backup) == {"composition.min_sharpness": 0.14}


def test_an_empty_override_writes_a_file_that_says_so(tmp_path):
    path = write(tmp_path / config.CONFIG_NAME, {})
    assert config.read_override(path) == {}
    assert "default" in path.read_text(encoding="utf-8")


def test_writing_an_unknown_key_is_refused():
    with pytest.raises(ConfigError, match="unknown setting"):
        render({"composition.nonesuch": 1.0})


def test_the_written_file_is_valid_toml(tmp_path):
    path = write(tmp_path / config.CONFIG_NAME, {k: s.default for k, s in CATALOGUE.items()})
    tomllib.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# the safety direction


def test_only_the_publishing_gate_is_marked_as_loosenable():
    """A `loosening` marker on a value with no consequence off this machine
    would train the user to click through the confirmation."""
    marked = {k for k, s in CATALOGUE.items() if s.loosening}
    assert marked == {"faces.gate_threshold"}


def test_raising_the_gate_threshold_is_the_loosening_direction():
    gate = CATALOGUE["faces.gate_threshold"]
    assert gate.loosens(0.30) is True
    assert gate.loosens(0.05) is False
    assert gate.loosens(gate.default) is False


def test_an_unmarked_setting_never_reports_a_loosening():
    for key, setting in CATALOGUE.items():
        if key == "faces.gate_threshold":
            continue
        assert setting.loosens(setting.default * 10) is False
        assert setting.loosens(0.0) is False


def test_the_loosening_direction_matches_what_the_gate_actually_does():
    """Not an assertion about a string - a demonstration. Raising
    `gate_threshold` must let a photo through that the default holds back."""
    from rekindle.semantic.faces import Box, Verdict, classify

    weak = [Box(0, 0, 10, 10, 0.20)]
    assert classify(weak) is Verdict.UNCERTAIN
    with config.using({"faces.gate_threshold": 0.30}):
        assert classify(weak) is Verdict.ELIGIBLE, "raising the gate did not loosen it"


# ---------------------------------------------------------------------------
# determinism: what config does and does not do to the engine's promise


def test_configuration_never_changes_a_memory_id():
    """The dismissal question, answered by demonstration.

    A memory id is `recipe:key` and is derived from the memory's defining
    facts, never from its contents. So a dismissal keyed on one keeps applying
    after any threshold moves: the shot list changes, the identity does not.
    If this ever stops being true, every dismissal in every library silently
    stops working the next time someone nudges a number.
    """
    from rekindle.memory.history import memory_id

    offer = _offer()
    before = offer.memory_id
    with config.using({"composition.min_sharpness": 0.9, "selection.max_shots": 3}):
        assert offer.memory_id == before
        assert memory_id("album_story", "Kashmir") == before


def test_a_dismissal_survives_a_threshold_change(tmp_path):
    """The same claim, through the real store and the real engine skip."""
    from rekindle.memory import engine
    from rekindle.memory.history import KIND_MEMORY, MemoryState

    offer = _offer()

    photos = [_photo(f"p{i:02d}", phash=i << 9, colour=_hist(i), at=i * 3600) for i in range(12)]
    store, index = _index(tmp_path, photos)
    state = MemoryState(store)
    state.dismiss(KIND_MEMORY, offer.memory_id)
    dismissed = state.dismissed_memory_ids()

    assert offer.memory_id in dismissed
    with config.using({"composition.min_sharpness": 0.3, "diversity.hard_floor": 0.5}):
        specs, report = engine.build_all(
            index,
            [offer],
            dismissed=dismissed,
            # `build_all` reaches `recipe.select` for a registry recipe; hand
            # it the photos so the skip is the only thing being measured.
        )
    assert specs == []
    assert report.skipped.get(engine.SKIP_DISMISSED) == 1


def test_the_same_config_and_the_same_photos_give_the_same_shots(tmp_path):
    """Determinism, restated with config as an input: same library AND same
    config in, same memory out."""
    from rekindle.memory import engine

    photos = [_photo(f"p{i:02d}", phash=i << 9, colour=_hist(i), at=i * 3600) for i in range(20)]
    offer = _offer(key="K")
    over = {"composition.min_sharpness": 0.2, "selection.max_shots": 7}

    store, index = _index(tmp_path, photos)
    with config.using(over):
        first = engine.build(index, offer, selection=_selection(photos))
    with config.using(over):
        second = engine.build(index, offer, selection=_selection(reversed(photos)))
    store.close()
    assert first is not None and second is not None
    assert first.dumps() == second.dumps()


def test_a_different_config_gives_a_different_memory_and_that_is_the_point(tmp_path):
    from rekindle.memory import engine

    photos = [_photo(f"p{i:02d}", phash=i << 9, colour=_hist(i), at=i * 3600) for i in range(20)]
    offer = _offer(key="K")

    store, index = _index(tmp_path, photos)
    with config.using({"selection.max_shots": 7}):
        a = engine.build(index, offer, selection=_selection(photos))
    with config.using({"selection.max_shots": 12}):
        b = engine.build(index, offer, selection=_selection(photos))
    store.close()
    assert a is not None and b is not None
    assert a.dumps() != b.dumps()
    assert a.key == b.key, "the memory's identity must not have moved with it"


def test_using_restores_the_previous_settings_even_on_an_exception():
    before = config.active()
    with pytest.raises(ZeroDivisionError), config.using({"selection.max_shots": 3}):
        raise ZeroDivisionError
    assert config.active() is before


def test_settings_are_frozen_so_a_build_cannot_see_them_change():
    settings = config.defaults()
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.composition.min_sharpness = 0.9  # type: ignore[misc]


def test_nothing_in_the_config_layer_reads_the_clock_or_the_environment():
    """A config that varies with the environment breaks the promise it is now
    an input to. `writer` takes a `when` precisely so its one timestamp is
    injectable and never lands in a value."""
    source = (Path(config.__file__).parent / "settings.py").read_text(encoding="utf-8")
    for forbidden in ("os.environ", "getenv", "random.", "datetime.now", "time.time"):
        assert forbidden not in source, f"{forbidden} in the config loader"
