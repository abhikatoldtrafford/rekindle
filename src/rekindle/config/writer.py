"""Writing `rekindle.toml`, by hand.

Python's standard library reads TOML (`tomllib`) and does not write it. The
binding constraint on this project is zero new dependencies - the default
install is four packages and the web UI added none - so `tomli-w` is not an
option and this is the alternative: about forty lines, for a file whose whole
grammar is `[section]` and `name = <number>`.

That is safe to hand-write precisely because the grammar is that small. Every
key is an identifier drawn from `CATALOGUE` - never from user input - and
every value is an `int` or a `float` that has already been range-checked. No
strings are ever written, so there is no escaping problem; no paths and no
names are ever written, so there is nothing here that could leak. If this file
ever needs to emit a string, that is the moment to reconsider, not the moment
to add a quoting function.

`repr()` is what turns a float into text. It round-trips exactly - the shortest
decimal that reads back as the same float - so a value written and read is the
value that was calibrated, and 0.12 stays "0.12" rather than becoming
"0.12000000000000001".
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from rekindle.config.settings import CATALOGUE, ConfigError

HEADER = """\
# rekindle.toml - your overrides, and nothing else.
#
# Only the values you changed are here. Everything absent comes from the
# defaults that ship with rekindle, so an upgrade improves what you did not
# choose without colliding with what you did. Delete a line to go back to the
# shipped value; delete the file to go back to all of them.
#
# `rekindle config explain <name>` prints what each one does, what raising and
# lowering it costs, and the measurement behind the default.
#
# This file contains numbers only. No paths, no names, nothing about your
# photographs - it is safe to commit.
"""


def render(values: dict[str, float | int], *, when: datetime | None = None) -> str:
    """The text of a `rekindle.toml` holding exactly `values`.

    Sections and keys come out in `defaults.toml` order, not in the caller's,
    so the same set of choices always produces the same file and a diff shows
    a changed number rather than a reshuffle.
    """
    for key in values:
        if key not in CATALOGUE:
            raise ConfigError(f"refusing to write unknown setting {key!r}")

    stamp = (when or datetime.now(UTC)).strftime("%Y-%m-%d")
    lines = [HEADER, f"# Last written {stamp}.", ""]

    if not values:
        lines.append("# Nothing overridden: every value is rekindle's default.")
        return "\n".join(lines) + "\n"

    section = ""
    for key, setting in CATALOGUE.items():
        if key not in values:
            continue
        if setting.section != section:
            section = setting.section
            lines.append(f"[{section}]")
        lines.append(f"{setting.name} = {_number(values[key], setting.is_int)}")
        lines.append("")

    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def _number(value: float | int, want_int: bool) -> str:
    if want_int:
        return str(int(value))
    # A float that is written as "1" reads back as an int, and an int where a
    # float belongs is a type error waiting three modules away. Force the
    # point on.
    text = repr(float(value))
    return text if ("." in text or "e" in text or "n" in text) else text + ".0"


def write(path: Path, values: dict[str, float | int], *, when: datetime | None = None) -> Path:
    """Write `rekindle.toml`, keeping the previous one recoverable.

    Calibration is never destructive: the file it replaces is moved to
    `rekindle.toml.bak` first, so a user who calibrated badly has the old
    numbers on disk and not merely in their memory of what they typed.
    """
    path = Path(path)
    text = render(values, when=when)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(text, encoding="utf-8")
    return path
