"""How to install an optional extra, phrased for how THIS copy was installed.

WHY THIS EXISTS
---------------
Nine places in this program tell the user that an optional feature needs an
extra, and every one of them said the same thing:

    Install it with:  uv sync --extra semantic

That is the right command in a git checkout and useless everywhere else. The
audience for `rekindle` is people with a Google Takeout export, not Python
developers; the whole point of shipping a wheel is that they never clone
anything. Telling somebody who ran `pipx install rekindle` to run `uv sync` is
telling them to install a build tool and clone a repository to get a feature
they already paid for.

So the command is derived from where this module is actually running from.
Three cases, in the order they are checked:

  * A SOURCE CHECKOUT - `pyproject.toml` two directories above the package,
    which is exactly the `src/rekindle` layout this project uses. Contributors
    get `uv sync`, which is what CONTRIBUTING.md tells them to run.
  * A PIPX VENV - `pipx_metadata.json` beside the interpreter. pipx owns the
    environment and `pip install` into it is not the supported gesture;
    `pipx install --force` with the extra is.
  * ANYTHING ELSE - a plain `pip install` into a venv or a user site.

Nothing here imports anything but the standard library, and it is called from
`doctor`, which must run on the smallest possible install.

WHAT IT DELIBERATELY DOES NOT DO: guess. If none of the three signals is
present it falls through to the pip form, which is correct for every remaining
case worth naming and is at worst a command that does nothing surprising.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: The extras this project publishes, so a caller cannot invent one and have
#: it silently printed at a user.
SEMANTIC = "semantic"
SEMANTIC_GPU = "semantic-gpu"
HEIC = "heic"
EXTRAS = frozenset({SEMANTIC, SEMANTIC_GPU, HEIC})

DISTRIBUTION = "rekindle"


def in_source_checkout() -> bool:
    """Is this package being imported from the repository it lives in?

    `src/rekindle/extras.py` -> parents[2] is the repository root. An
    installed wheel puts the package in `site-packages`, whose grandparent is
    a virtual environment and holds no `pyproject.toml`.
    """
    try:
        return (Path(__file__).resolve().parents[2] / "pyproject.toml").is_file()
    except (OSError, IndexError):  # pragma: no cover - exotic loaders
        return False


def in_pipx() -> bool:
    """Is this running inside a pipx-managed virtual environment?

    pipx writes `pipx_metadata.json` at the root of every environment it
    creates, and has done since 0.16. Its absence is not proof of anything, so
    this only ever selects a friendlier command, never a broken one.
    """
    try:
        return (Path(sys.prefix) / "pipx_metadata.json").is_file()
    except OSError:  # pragma: no cover - exotic prefixes
        return False


def install_command(*extras: str) -> str:
    """The exact command THIS user should run to add `extras`.

    Several extras are one command, not several: installing them one at a time
    with pipx would uninstall the previous one, which is the kind of advice
    that is worse than none.
    """
    unknown = [e for e in extras if e not in EXTRAS]
    if unknown:
        raise ValueError(f"unknown extra(s): {', '.join(unknown)}")
    if not extras:
        raise ValueError("name at least one extra")
    joined = ",".join(extras)
    if in_source_checkout():
        return "uv sync " + " ".join(f"--extra {e}" for e in extras)
    if in_pipx():
        return f"pipx install --force '{DISTRIBUTION}[{joined}]'"
    return f"pip install '{DISTRIBUTION}[{joined}]'"


def install_hint(*extras: str) -> str:
    """`install_command`, wrapped in the sentence the CLI prints."""
    return f"Install it with:  {install_command(*extras)}"


def markup_safe(text: str) -> str:
    """`text` with Rich's markup delimiter escaped.

    **`pip install 'rekindle[semantic]'` is Rich markup.** `[semantic]` reads
    as a style tag, and Rich silently deletes it - the first pipx install of
    this package printed `pipx install --force 'rekindle'`, a command that
    installs the package without the extra it was recommending. Silent, and
    the resulting command RUNS, which is the worst kind of wrong.

    Escaping lives here rather than at twenty `console.print` call sites, and
    `install_command` stays unescaped so that a caller who is not Rich - a
    test, a log line, a subprocess - gets a command it can actually run.
    """
    return text.replace("[", r"\[")


def install_command_markup(*extras: str) -> str:
    """`install_command`, safe to interpolate into a Rich markup string."""
    return markup_safe(install_command(*extras))
