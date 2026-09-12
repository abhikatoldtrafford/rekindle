"""Reading `defaults.toml`, and the user override that sits on top of it.

TWO FILES, TWO SHAPES, AND THAT IS DELIBERATE
---------------------------------------------
`defaults.toml` ships inside the package and is replaced on every upgrade. It
is a table per setting, carrying the number AND the six things a user needs to
decide whether to change it:

    [composition.min_sharpness]
    value = 0.12
    unit = "..."   what = "..."   raising = "..."
    lowering = "..."   measured = "..."   range = [0.0, 1.0]

The user's `rekindle.toml` is a flat number per key, and nothing else:

    [composition]
    min_sharpness = 0.14

The two shapes cannot be confused for each other, which is the point: an
upgrade replaces the documentation without touching the user's choices, and a
`git diff` of the user's file shows exactly what they changed and nothing
else. It also enforces the privacy rule mechanically - `rekindle.toml` is a
file a user may commit to a public repository, and a loader that accepts only
`section.key = number` cannot be talked into carrying a path or a name.

WHY VALUES ARE READ AT CALL TIME
--------------------------------
Every consumer reads `active()` inside the function that needs the number,
not at import. Binding at import would make `rekindle.toml` arrive too late
for any module already imported - the failure mode the brief calls out by
name, where "an override silently does nothing" is indistinguishable from
"the user changed nothing".

The module-level constants (`composition.MIN_SHARPNESS` and friends) still
exist and are still the shipped numbers, because a hundred call sites and
tests read them - but they are now READ FROM `defaults.toml` rather than
typed twice, so the file and the code cannot drift. `test_config.py` pins
that.

DETERMINISM
-----------
The engine's promise is same-library-in, same-memory-out, byte for byte.
Configuration is an input to that promise, not an exception to it: the
promise is now same-library AND same-config in, same memory out. Nothing here
reads the clock, the environment, or a random source, and `Settings` is
frozen, so a build cannot observe a value changing underneath it.

What configuration must NOT touch is memory IDENTITY. A memory id is
`recipe:key` - `album_story:Kashmir`, `year_in_review:2016` - derived from the
memory's defining facts and never from its contents (see `memory.history`).
So a dismissal keyed on a memory id keeps applying after any threshold moves:
the shot list changes, the identity does not. That is the whole answer to
"can a memory built under one config be rebuilt under another and still be
the same memory", and `test_config.py` pins it against a real threshold
change rather than by assertion.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

#: The user's override file, alongside `exclusions.toml` in the data dir.
CONFIG_NAME = "rekindle.toml"

DEFAULTS_PATH = Path(__file__).with_name("defaults.toml")

#: Every field `defaults.toml` must carry for a setting, in the order the
#: onboarding flow shows them.
REQUIRED_FIELDS = ("value", "unit", "what", "raising", "lowering", "measured")

#: A setting whose `loosening` says "raise" gets weaker as the number goes up.
LOOSEN_RAISE = "raise"
LOOSEN_LOWER = "lower"


class ConfigError(RuntimeError):
    """The config file exists but cannot be used.

    Fatal, for the same reason `policy.PolicyError` is: continuing with a
    half-read config would silently run on defaults the user believes they
    replaced, and a warning scrolls past.
    """


@dataclass(frozen=True)
class Setting:
    """One tunable, with everything needed to decide whether to move it."""

    key: str  # "composition.min_sharpness"
    section: str
    name: str
    default: float | int
    unit: str
    what: str
    raising: str
    lowering: str
    measured: str
    minimum: float | int | None = None
    maximum: float | int | None = None
    #: "raise", "lower", or "" - the direction in which this weakens a safety
    #: gate whose failure has a consequence off this machine.
    loosening: str = ""

    @property
    def is_int(self) -> bool:
        return isinstance(self.default, int) and not isinstance(self.default, bool)

    def loosens(self, new: float) -> bool:
        """Does moving to `new` weaken this gate?

        False for every setting that is not marked, and false for a move that
        does not change the value. `rekindle calibrate` refuses to make a move
        for which this is true without a separate, explicit confirmation.
        """
        if not self.loosening:
            return False
        if self.loosening == LOOSEN_RAISE:
            return new > self.default
        return new < self.default

    def clamp_error(self, value: float) -> str:
        """Empty if `value` is in range, else the sentence to show."""
        if self.minimum is not None and value < self.minimum:
            return f"{self.key} must be at least {self.minimum} (got {value})"
        if self.maximum is not None and value > self.maximum:
            return f"{self.key} must be at most {self.maximum} (got {value})"
        return ""


def _load_catalogue() -> dict[str, Setting]:
    raw = tomllib.loads(DEFAULTS_PATH.read_text(encoding="utf-8"))
    out: dict[str, Setting] = {}
    for section, entries in raw.items():
        for name, body in entries.items():
            key = f"{section}.{name}"
            missing = [f for f in REQUIRED_FIELDS if f not in body]
            if missing:
                raise ConfigError(f"{key} in defaults.toml is missing {', '.join(missing)}")
            span = body.get("range", [None, None])
            out[key] = Setting(
                key=key,
                section=section,
                name=name,
                default=body["value"],
                unit=body["unit"],
                what=body["what"],
                raising=body["raising"],
                lowering=body["lowering"],
                measured=body["measured"],
                minimum=span[0],
                maximum=span[1],
                loosening=body.get("loosening", ""),
            )
    return out


#: Every shipped tunable, keyed "section.name". Read once, at import: the file
#: is inside the package and cannot change while the process runs.
CATALOGUE: dict[str, Setting] = _load_catalogue()


# ---------------------------------------------------------------------------
# the typed view
#
# Attribute access rather than a dict lookup, so a typo is an AttributeError
# at the call site instead of a KeyError three frames down - or worse, a
# `.get(key, default)` that silently returns the shipped number forever.


@dataclass(frozen=True)
class Composition:
    min_sharpness: float
    min_short_edge: int
    max_aspect: float
    min_brightness: float
    max_brightness: float
    square_ratio: float
    upscale_tolerance: float


@dataclass(frozen=True)
class Dedup:
    gap_seconds: float
    phash_distance: int
    cosine: float


@dataclass(frozen=True)
class Diversity:
    lambda_penalty: float
    phash_radius: int
    hard_floor: float
    time_tiebreak: float


@dataclass(frozen=True)
class SemanticDiversity:
    ceiling: float
    min_spread: float
    low_quantile: float
    high_quantile: float
    weight: float


@dataclass(frozen=True)
class Faces:
    detect_threshold: float
    gate_threshold: float


@dataclass(frozen=True)
class Orientation:
    min_face: float
    min_margin: float


@dataclass(frozen=True)
class Selection:
    max_shots: int
    min_shots: int
    cooldown_days: int
    max_overlap: float


_SECTIONS: dict[str, type] = {
    "composition": Composition,
    "dedup": Dedup,
    "diversity": Diversity,
    "semantic_diversity": SemanticDiversity,
    "faces": Faces,
    "orientation": Orientation,
    "selection": Selection,
}


@dataclass(frozen=True)
class Settings:
    """The active numbers. Immutable, so a build cannot see them change."""

    composition: Composition
    dedup: Dedup
    diversity: Diversity
    semantic_diversity: SemanticDiversity
    faces: Faces
    orientation: Orientation
    selection: Selection

    def flat(self) -> dict[str, float | int]:
        """Every value as `{"section.name": number}`, in catalogue order."""
        out: dict[str, float | int] = {}
        for key, setting in CATALOGUE.items():
            out[key] = getattr(getattr(self, setting.section), setting.name)
        return out

    def get(self, key: str) -> float | int:
        setting = CATALOGUE[key]
        return getattr(getattr(self, setting.section), setting.name)

    def with_values(self, changes: dict[str, float | int]) -> Settings:
        """A copy with `changes` applied. Raises `ConfigError` on a bad key or
        an out-of-range value - the same validation the file loader runs, so
        a value cannot enter through the UI that the file would refuse."""
        return from_flat({**self.flat(), **_validated(changes)})

    def changed_from_defaults(self) -> dict[str, float | int]:
        """Only what differs from the shipped numbers. This is what gets
        written to `rekindle.toml`: an override file records choices, not a
        snapshot, so next year's better default still reaches this user."""
        return {k: v for k, v in self.flat().items() if v != CATALOGUE[k].default}


def _validated(changes: dict[str, float | int]) -> dict[str, float | int]:
    clean: dict[str, float | int] = {}
    for key, value in changes.items():
        setting = CATALOGUE.get(key)
        if setting is None:
            raise ConfigError(f"unknown setting {key!r}")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{key} must be a number, not {type(value).__name__}")
        problem = setting.clamp_error(value)
        if problem:
            raise ConfigError(problem)
        clean[key] = int(value) if setting.is_int else float(value)
    return clean


def from_flat(values: dict[str, float | int]) -> Settings:
    """Build `Settings` from a complete flat mapping."""
    built: dict[str, Any] = {}
    for section, cls in _SECTIONS.items():
        built[section] = cls(
            **{f.name: values[f"{section}.{f.name}"] for f in fields(cls)}  # type: ignore[arg-type]
        )
    return Settings(**built)


def defaults() -> Settings:
    """The shipped numbers, with no user file anywhere in sight.

    This is what CI runs on and what an install with no data directory runs
    on, so it must never touch the filesystem beyond the packaged file the
    catalogue already read.
    """
    return from_flat({k: s.default for k, s in CATALOGUE.items()})


# ---------------------------------------------------------------------------
# the user's file


def read_override(path: Path) -> dict[str, float | int]:
    """Parse `rekindle.toml` into `{"section.name": number}`.

    Absent is fine and means "everything default". Malformed is fatal.
    """
    if not path.is_file():
        return {}
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{path} could not be read: {exc}") from exc

    flat: dict[str, float | int] = {}
    for section, body in raw.items():
        if not isinstance(body, dict):
            raise ConfigError(
                f"{path}: `{section}` must be a section of numbers, e.g. "
                f"[composition]\\nmin_sharpness = 0.14"
            )
        for name, value in body.items():
            key = f"{section}.{name}"
            if key not in CATALOGUE:
                raise ConfigError(
                    f"{path}: no such setting `{key}`. "
                    f"Run `rekindle config list` to see every name."
                )
            flat[key] = value
    return _validated(flat)


def load(data_dir: Path | None = None) -> Settings:
    """Defaults, with the user's overrides on top. Never partially applied."""
    if data_dir is None:
        return defaults()
    return defaults().with_values(read_override(Path(data_dir) / CONFIG_NAME))


# ---------------------------------------------------------------------------
# the active settings
#
# One module-level frozen object, replaced wholesale. Reads need no lock
# because they are a single attribute load of an immutable value, which
# matters: the web server is threaded.

_active: Settings = defaults()


def active() -> Settings:
    """What the code should use right now."""
    return _active


def activate(settings: Settings) -> Settings:
    """Install `settings` as active and return what was there before."""
    global _active
    previous = _active
    _active = settings
    return previous


def activate_from(data_dir: Path | None) -> Settings:
    """Load `data_dir/rekindle.toml` and make it active. Returns the previous."""
    return activate(load(data_dir))


class using:
    """Context manager that activates `settings` for a block.

    Used by tests and by the calibration preview, which has to answer "what
    would this library look like at 0.14?" without writing anything.
    """

    def __init__(self, settings: Settings | dict[str, float | int]) -> None:
        self._wanted = (
            settings if isinstance(settings, Settings) else active().with_values(settings)
        )
        self._previous: Settings | None = None

    def __enter__(self) -> Settings:
        self._previous = activate(self._wanted)
        return self._wanted

    def __exit__(self, *exc: object) -> None:
        assert self._previous is not None
        activate(self._previous)
