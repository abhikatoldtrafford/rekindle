"""Configuration: read-only shipped defaults, plus the user's overrides.

    from rekindle.config import active
    ...
    if sharpness < active().composition.min_sharpness:

Read `settings.py` for why every consumer calls `active()` inside the function
rather than binding at import, and for what configuration does and does not do
to the engine's determinism promise.
"""

from rekindle.config.settings import (
    CATALOGUE,
    CONFIG_NAME,
    DEFAULTS_PATH,
    LOOSEN_LOWER,
    LOOSEN_RAISE,
    ConfigError,
    Setting,
    Settings,
    activate,
    activate_from,
    active,
    defaults,
    from_flat,
    load,
    read_override,
    using,
)

__all__ = [
    "CATALOGUE",
    "CONFIG_NAME",
    "DEFAULTS_PATH",
    "LOOSEN_LOWER",
    "LOOSEN_RAISE",
    "ConfigError",
    "Setting",
    "Settings",
    "activate",
    "activate_from",
    "active",
    "defaults",
    "from_flat",
    "load",
    "read_override",
    "using",
]
