"""The wheel, the sdist and the install hints.

WHY THESE ARE STATIC CHECKS. Building a wheel needs `hatchling`, which is a
BUILD dependency and is deliberately not in the venv - CI has no network and
must stay green, so a test that shells out to `uv build` would either fail or
teach everyone to ignore it. The build-based assertions are therefore skipped
unless a wheel already exists in `dist/`, and everything else is checked
against `pyproject.toml` and the source tree, which is where the mistakes
actually live.

The one that matters most is negative: **the sdist must not be able to package
personal data.** This repository sits next to a 61 GB photo export, and the
first sdist built from it reached into `.claude/worktrees/*/` because
`.gitignore` contains a `!.env.example` negation that un-ignores that name at
any depth. Nothing secret escaped. The mechanism would have packaged anything
else those directories held, into a public package with no undo.
"""

from __future__ import annotations

import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
PROJECT = PYPROJECT["project"]


# --------------------------------------------------------------------- sdist

#: Anything matching one of these, anywhere in a distribution, is personal data
#: or somebody else's workspace. Spelled out rather than inferred, so a new
#: hazard has to be added deliberately.
#: Substrings, matched against every path in a built distribution. Directory
#: names are anchored (`/data/`, not `data`) because `rekindle/semantic/` and
#: `tests/fixtures/` are legitimate SOURCE paths that happen to share a word
#: with a data directory - the first version of this list failed on its own
#: package.
FORBIDDEN = (
    ".claude",
    "/Takeout/",
    "/takeout/",
    "/data/",
    "/memories/",
    "/music/",
    "/frames/",
    ".sqlite",
    ".jpg",
    ".jpeg",
    ".heic",
    ".mp4",
    ".mp3",
)


#: Directory names that must never appear as a path SEGMENT of a whitelist
#: entry. Compared segment by segment, casefolded and with a leading dot
#: stripped, because `/.claude` and `/claude` are the same hazard and the
#: first version of this check only caught the second.
FORBIDDEN_SEGMENTS = frozenset(
    {"claude", "takeout", "data", "memories", "music", "frames", "photos", "venv", "dist"}
)


def test_the_sdist_is_a_whitelist():
    """Hatchling's default is "everything git does not ignore". That default
    plus one `!` negation in `.gitignore` is what packaged three agent
    worktrees; a whitelist cannot be widened by anything anyone drops in this
    directory later."""
    sdist = PYPROJECT["tool"]["hatch"]["build"]["targets"]["sdist"]
    assert "include" in sdist, "the sdist must name what it wants"
    assert "exclude" not in sdist, "an exclude list is the failure mode, not the fix"


@pytest.mark.parametrize(
    "entry", PYPROJECT["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
)
def test_every_whitelisted_path_exists(entry):
    """A typo in the whitelist silently ships less, and nothing would notice
    until somebody tried to install the sdist."""
    pattern = entry.lstrip("/")
    assert list(ROOT.glob(pattern)), f"{entry} matches nothing in the repository"


@pytest.mark.parametrize(
    "entry", PYPROJECT["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
)
def test_no_whitelisted_path_can_carry_personal_data(entry):
    segments = {s.casefold().lstrip(".") for s in entry.split("/") if s}
    clash = segments & FORBIDDEN_SEGMENTS
    assert not clash, f"{entry} would package {', '.join(sorted(clash))}"


# ------------------------------------------------------------------ metadata


def test_the_console_script_points_at_something_that_exists():
    """`rekindle = "rekindle.cli:app"` is a string until somebody runs it. A
    rename that misses this line produces a wheel that installs fine and dies
    on first use."""
    target = PROJECT["scripts"]["rekindle"]
    module, _, attribute = target.partition(":")
    import importlib

    import typer

    app = getattr(importlib.import_module(module), attribute, None)
    # `isinstance`, not `callable`. `rekindle.cli` has other module-level
    # callables, so "it resolves to something callable" passes for a target
    # that would give the user a traceback instead of a CLI.
    assert isinstance(app, typer.Typer), f"{target} is {type(app).__name__}, not a Typer app"
    assert app.registered_commands, "the app has no commands"


def test_the_typed_classifier_is_backed_by_a_py_typed_marker():
    """`Typing :: Typed` without `py.typed` is a claim no type checker can
    act on - the hints are in the source and invisible to a consumer."""
    if "Typing :: Typed" in PROJECT["classifiers"]:
        assert (ROOT / "src" / "rekindle" / "py.typed").is_file()


def test_the_python_classifiers_do_not_overclaim():
    """Only versions this project is actually tested on. `requires-python` may
    be wider - it is a floor, not a promise about every future release - but a
    classifier is read as "we ran it"."""
    claimed = {
        c.rsplit(" :: ", 1)[1]
        for c in PROJECT["classifiers"]
        if c.startswith("Programming Language :: Python :: ") and c[-1].isdigit()
    }
    assert claimed == {"3.12"}, claimed


def test_the_metadata_a_package_page_shows_is_actually_filled_in():
    for field in ("description", "readme", "license", "authors", "keywords", "classifiers"):
        assert PROJECT.get(field), f"{field} is empty"
    assert len(PROJECT["description"]) > 40, "a one-line summary is the only thing search shows"
    assert PROJECT["urls"]["Repository"].startswith("https://")


def test_the_readme_has_no_relative_links():
    """PyPI renders `README.md` on its own page, where `](docs/x.md)` is a 404.
    17 of them were, before this was checked."""
    import re

    text = (ROOT / PROJECT["readme"]).read_text(encoding="utf-8")
    bad = [m.group(1) for m in re.finditer(r"\]\((?!https?://|#|mailto:)([^)]+)\)", text)]
    # HTML `<img src=...>` too. The markdown-only version of this test passed
    # while four relative image sources sat in the gallery, because the gallery
    # is a table of `<img>` tags - a rule that only inspects one syntax is a
    # rule the other syntax walks past.
    bad += [
        m.group(1)
        for m in re.finditer(r"""<img[^>]+src=["'](?!https?://|data:)([^"']+)["']""", text)
    ]
    assert bad == [], f"relative links break on a package page: {bad}"


def test_no_new_runtime_dependency_crept_in():
    """ "The default `uv sync` is four packages" is a stated constraint of this
    project, and a dependency added for one feature is paid for by everybody."""
    names = {d.split(">")[0].split("=")[0].split(";")[0].strip() for d in PROJECT["dependencies"]}
    assert names == {"pillow", "defusedxml", "typer", "rich", "tzdata"}


def test_every_extra_this_project_documents_is_declared():
    from rekindle.extras import EXTRAS

    assert set(PROJECT["optional-dependencies"]) == EXTRAS


# --------------------------------------------------- the wheel, if one exists

WHEELS = sorted((ROOT / "dist").glob("*.whl")) if (ROOT / "dist").is_dir() else []
needs_wheel = pytest.mark.skipif(not WHEELS, reason="no built wheel in dist/ (run `uv build`)")


@needs_wheel
def test_the_wheel_carries_every_data_file_the_code_reads():
    """A wheel of pure `.py` files installs cleanly and then cannot find its
    own festival corpus, its caption vocabulary, its model checksums, its
    default thresholds or the web UI's HTML. Every one of those is loaded by
    path at runtime - and `config/defaults.toml` is read at IMPORT, so a wheel
    without it does not degrade, it fails to start."""
    names = set(zipfile.ZipFile(WHEELS[-1]).namelist())
    for wanted in (
        "rekindle/memory/corpus/festivals.toml",
        "rekindle/memory/corpus/caption_vocab.toml",
        "rekindle/memory/corpus/scenery.toml",
        "rekindle/memory/corpus/prompt_tags.json",
        "rekindle/semantic/model-locks.json",
        "rekindle/config/defaults.toml",
        "rekindle/web/assets/index.html",
        "rekindle/web/assets/app.js",
        "rekindle/web/assets/app.css",
        "rekindle/py.typed",
    ):
        assert wanted in names, f"{wanted} is missing from the wheel"


@needs_wheel
def test_the_wheel_ships_no_bytecode_and_nothing_personal():
    names = zipfile.ZipFile(WHEELS[-1]).namelist()
    assert not [n for n in names if "__pycache__" in n or n.endswith(".pyc")]
    for bad in FORBIDDEN:
        assert not [n for n in names if bad in n], f"{bad} is in the wheel"


@needs_wheel
def test_the_wheel_declares_the_console_script():
    text = (
        zipfile.ZipFile(WHEELS[-1])
        .read(f"rekindle-{PROJECT['version']}.dist-info/entry_points.txt")
        .decode()
    )
    assert "[console_scripts]" in text
    assert "rekindle = rekindle.cli:app" in text


# ------------------------------------------------------------- install hints


def test_a_source_checkout_is_told_to_use_uv():
    from rekindle import extras

    assert extras.in_source_checkout(), "the tests run from a checkout, so this must be true here"
    assert extras.install_command("semantic") == "uv sync --extra semantic"


def test_an_installed_copy_is_never_told_to_run_uv_sync(monkeypatch):
    """The audience is people with a Takeout export, not Python developers.
    Telling somebody who ran `pipx install rekindle` to run `uv sync` is
    telling them to clone a repository to get a feature they already have."""
    from rekindle import extras

    monkeypatch.setattr(extras, "in_source_checkout", lambda: False)
    monkeypatch.setattr(extras, "in_pipx", lambda: False)
    assert extras.install_command("semantic") == "pip install 'rekindle[semantic]'"

    monkeypatch.setattr(extras, "in_pipx", lambda: True)
    assert extras.install_command("semantic") == "pipx install --force 'rekindle[semantic]'"


def test_several_extras_are_ONE_command(monkeypatch):
    """Installing them one at a time under pipx uninstalls the previous one,
    which is advice worse than none."""
    from rekindle import extras

    monkeypatch.setattr(extras, "in_source_checkout", lambda: False)
    monkeypatch.setattr(extras, "in_pipx", lambda: True)
    got = extras.install_command("semantic", "heic")
    assert got == "pipx install --force 'rekindle[semantic,heic]'"
    assert got.count("pipx install") == 1


def test_an_extra_that_does_not_exist_is_refused():
    from rekindle import extras

    with pytest.raises(ValueError, match="unknown extra"):
        extras.install_command("gpu")
    with pytest.raises(ValueError, match="at least one"):
        extras.install_command()


def test_the_bracket_that_rich_would_eat_is_escaped(monkeypatch):
    """`pip install 'rekindle[semantic]'` is valid Rich markup. Rich deletes
    `[semantic]` and prints a command that installs the package WITHOUT the
    extra it is recommending - silently, and the result runs."""
    from rich.console import Console

    from rekindle import extras

    monkeypatch.setattr(extras, "in_source_checkout", lambda: False)
    monkeypatch.setattr(extras, "in_pipx", lambda: True)

    console = Console(width=200, force_terminal=False)
    with console.capture() as capture:
        console.print(f"[red]{extras.install_command('semantic')}[/red]")
    assert "semantic]" not in capture.get(), "Rich did not eat it; this test is now pointless"

    with console.capture() as capture:
        console.print(f"[red]{extras.install_command_markup('semantic')}[/red]")
    assert "rekindle[semantic]" in capture.get()


def test_the_command_a_non_rich_caller_gets_is_runnable(monkeypatch):
    """`install_command` stays unescaped, so a test, a log or a subprocess gets
    something that can actually be run."""
    from rekindle import extras

    monkeypatch.setattr(extras, "in_source_checkout", lambda: False)
    monkeypatch.setattr(extras, "in_pipx", lambda: False)
    assert "\\" not in extras.install_command("semantic")


def test_the_two_version_strings_agree():
    """`pyproject.toml` and `rekindle.__version__` are written separately, and
    `release.yml` only checks the tag against the BUILT wheel - so a stale
    `__version__` ships happily and `rekindle --version` then lies about which
    release is running. Caught while bumping 0.1.0 -> 0.1.1, where the second
    string was missed."""
    import rekindle

    assert rekindle.__version__ == PROJECT["version"]
