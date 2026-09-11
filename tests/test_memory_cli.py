"""The memory CLI.

The recurring guarantee: a command that needs an index and does not find one
exits 2 and leaves NO database file behind. `PhotoStore` creates its file on
open, so without an explicit check every one of these would silently report a
healthy library of zero photos.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from rekindle.cli import app
from rekindle.db import PhotoStore
from rekindle.memory.history import MemoryState
from rekindle.models import MediaType, Photo, PhotoMeta

runner = CliRunner()
T0 = datetime(2020, 5, 1, 12, 0, tzinfo=UTC)

MEMORY_COMMANDS = [
    ["memories"],
    ["memory", "--recipe", "album_story", "--key", "X"],
    ["fingerprint"],
    ["dismiss", "album_story", "X"],
    ["undismiss", "album_story", "X"],
    ["exclude", "--person", "X"],
    ["dismissals"],
]


def _jpeg(path: Path, size=(1600, 1200), colour=(120, 90, 60)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path, "JPEG")
    return path


def _photo(h, path, *, local, albums=(), people=(), size=(1600, 1200), sharp=8.0) -> Photo:
    return Photo(
        file_hash=h,
        paths=[path],
        media_type=MediaType.IMAGE,
        meta=PhotoMeta(
            taken_at_utc=local.replace(tzinfo=UTC),
            taken_at_local=local,
            people=list(people),
            width=size[0],
            height=size[1],
            sharpness=sharp,
            brightness=110.0,
            phash=abs(hash(h)) % (1 << 60),
        ),
        first_seen=T0,
        last_seen=T0,
        albums=list(albums),
    )


def _library(tmp_path, *, n=8, album="Kashmir", people=("Abhik Maiti",)) -> Path:
    """A small real library on disk plus a matching index."""
    data = tmp_path / "data"
    photos = []
    for i in range(n):
        path = _jpeg(tmp_path / "lib" / f"p{i}.jpg", colour=(40 + i * 20, 90, 60))
        photos.append(
            _photo(
                f"h{i}",
                path,
                local=datetime(2020, 5, 1, 9, 0) + timedelta(days=i),
                albums=[album],
                people=people,
            )
        )
    with PhotoStore(data / "rekindle.sqlite") as store:
        store.upsert_many(photos)
    return data


# --------------------------------------------------------------------------
# no index: exit 2, and no stray database


@pytest.mark.parametrize("argv", MEMORY_COMMANDS, ids=lambda a: a[0])
def test_a_missing_index_exits_2_and_creates_no_database(tmp_path, argv):
    data = tmp_path / "data"
    result = runner.invoke(app, [*argv, "--data-dir", str(data)])

    assert result.exit_code == 2, result.output
    assert "No index" in result.output
    assert not (data / "rekindle.sqlite").exists(), "a failed command left a stray database"


def test_watch_with_no_index_exits_2(tmp_path):
    (tmp_path / "lib").mkdir()
    result = runner.invoke(
        app, ["watch", str(tmp_path / "lib"), "--once", "--data-dir", str(tmp_path / "data")]
    )
    assert result.exit_code == 2
    assert not (tmp_path / "data" / "rekindle.sqlite").exists()


# --------------------------------------------------------------------------
# help: catches a typer signature error at import


@pytest.mark.parametrize(
    "name",
    [
        "fingerprint",
        "memories",
        "memory",
        "dismiss",
        "undismiss",
        "exclude",
        "dismissals",
        "watch",
    ],
)
def test_every_command_has_working_help(name):
    result = runner.invoke(app, [name, "--help"])
    assert result.exit_code == 0, result.output


# --------------------------------------------------------------------------
# memories


def test_memories_lists_offers(tmp_path):
    data = _library(tmp_path)
    result = runner.invoke(app, ["memories", "--data-dir", str(data)])
    assert result.exit_code == 0
    assert "Kashmir" in result.output


def test_memories_on_an_empty_library_exits_0(tmp_path):
    data = tmp_path / "data"
    with PhotoStore(data / "rekindle.sqlite") as store:
        store.count()
    result = runner.invoke(app, ["memories", "--data-dir", str(data)])
    assert result.exit_code == 0
    assert "No photos" in result.output


def test_memories_can_filter_by_recipe(tmp_path):
    data = _library(tmp_path)
    result = runner.invoke(app, ["memories", "--recipe", "year_in_review", "--data-dir", str(data)])
    assert result.exit_code == 0
    assert "album_story" not in result.output


# --------------------------------------------------------------------------
# memory


def test_memory_writes_a_spec_and_a_gif(tmp_path):
    data = _library(tmp_path)
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        # --no-mp4 so the test does not depend on ffmpeg being installed.
        [
            "memory",
            "--recipe",
            "album_story",
            "--key",
            "Kashmir",
            "--out",
            str(out),
            "--no-mp4",
            "--data-dir",
            str(data),
        ],
    )
    assert result.exit_code == 0, result.output
    folders = list(out.iterdir())
    assert len(folders) == 1
    assert (folders[0] / "memory.json").is_file()
    assert (folders[0] / "memory.gif").is_file()
    assert not (folders[0] / "memory.mp4").exists()


def test_the_cli_tells_the_music_layer_which_memory_it_is_rendering(tmp_path, monkeypatch):
    """`resolve_music` picks a track from the memory's stable id, and it falls
    back to the first track when it is not given one. So the whole per-memory
    feature can be correct in `music.py` and dead in the product, with every
    unit test green, if the CLI simply does not pass the id. That is the shape
    of defect this project keeps finding, so the wiring gets its own test.

    Renders through the MP4 branch with ffmpeg stubbed out, because that is
    the only branch that resolves music at all.
    """
    from rekindle.memory import cli as memory_cli
    from rekindle.memory.render.mp4 import Mp4Result

    seen = []

    def spy(explicit=None, folder=None, memory_id=None):
        seen.append(memory_id)
        return None

    monkeypatch.setattr(memory_cli, "resolve_music", spy)
    monkeypatch.setattr(
        memory_cli, "write_mp4", lambda *a, **k: Mp4Result(path=None, skipped="stubbed")
    )

    data = _library(tmp_path)
    result = runner.invoke(
        app,
        [
            "memory",
            "--recipe",
            "album_story",
            "--key",
            "Kashmir",
            "--out",
            str(tmp_path / "out"),
            "--data-dir",
            str(data),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen == ["album_story:Kashmir"], (
        "the CLI did not hand the music layer the memory's id, so every "
        "memory would get the same track"
    )


def test_naming_one_memory_overrides_its_cooldown(tmp_path):
    """The cooldown stops `--auto` repeating itself next week. It must not
    refuse a person who names one memory and asks for it - which is exactly
    what you type after changing the code that builds it, and what this
    milestone's re-render ran into."""
    data = _library(tmp_path)
    out = tmp_path / "out"
    argv = [
        "memory",
        "--recipe",
        "album_story",
        "--key",
        "Kashmir",
        "--out",
        str(out),
        "--no-mp4",
        "--data-dir",
        str(data),
    ]

    assert runner.invoke(app, argv).exit_code == 0
    again = runner.invoke(app, argv)
    assert again.exit_code == 0, again.output
    assert "cooldown" not in again.output
    assert (next(out.iterdir()) / "memory.json").is_file()


def test_an_unnamed_build_still_respects_the_cooldown(tmp_path):
    """The other half. Without it "override the cooldown" becomes "there is no
    cooldown", and `--auto` shows the same memory every day."""
    data = _library(tmp_path)
    out = tmp_path / "out"
    base = ["memory", "--out", str(out), "--no-mp4", "--data-dir", str(data)]

    assert runner.invoke(app, [*base, "--recipe", "album_story", "--key", "Kashmir"]).exit_code == 0
    again = runner.invoke(app, base)
    assert "cooldown" in again.output, again.output


def test_a_dismissed_memory_is_still_refused_when_named(tmp_path):
    """Dismissal is deliberate and permanent; `undismiss` is how it is taken
    back. Only the automatic cooldown is overridden."""
    data = _library(tmp_path)
    assert (
        runner.invoke(app, ["dismiss", "album_story", "Kashmir", "--data-dir", str(data)]).exit_code
        == 0
    )
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "memory",
            "--recipe",
            "album_story",
            "--key",
            "Kashmir",
            "--out",
            str(out),
            "--no-mp4",
            "--data-dir",
            str(data),
        ],
    )
    assert not out.exists() or list(out.iterdir()) == []
    assert "dismissed" in result.output.lower()


def test_a_merged_album_is_reported_not_silent(tmp_path):
    """A silent merge is the same defect as a silent drop: the user goes
    looking for `Christmas 2025`, finds `Christmas`, and has no way to know
    why. The line also names the override, because "you can change this" is
    useless without "here is what to type"."""
    data = tmp_path / "data"
    photos = [
        _photo(
            "c1",
            _jpeg(tmp_path / "lib" / "c1.jpg"),
            local=datetime(2015, 12, 25, 9, 0),
            albums=["Christmas 15"],
        ),
        _photo(
            "c2",
            _jpeg(tmp_path / "lib" / "c2.jpg"),
            local=datetime(2025, 12, 25, 9, 0),
            albums=["Christmas 2025"],
        ),
        _photo(
            "c3",
            _jpeg(tmp_path / "lib" / "c3.jpg"),
            local=datetime(2025, 12, 26, 9, 0),
            albums=["Christmas 2025"],
        ),
    ]
    with PhotoStore(data / "rekindle.sqlite") as store:
        store.upsert_many(photos)

    result = runner.invoke(app, ["memories", "--data-dir", str(data)])
    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    assert "Albums merged as 'Christmas'" in flat, result.output
    assert "album_aliases" in flat
    assert "exclusions.toml" in flat


def test_nothing_is_said_when_no_album_was_merged(tmp_path):
    data = _library(tmp_path, album="Kashmir")
    result = runner.invoke(app, ["memories", "--data-dir", str(data)])
    assert result.exit_code == 0
    assert "merged" not in result.output.lower()


def test_recipe_alone_builds_only_that_recipe(tmp_path):
    """`rekindle memory --recipe on_this_month` used to fall through to "build
    anything" and cheerfully render album stories - the command doing
    something other than what its own --help says, silently. Found by running
    it, not by reading it."""
    data = _library(tmp_path, n=8)
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "memory",
            "--recipe",
            "album_story",
            "--limit",
            "5",
            "--out",
            str(out),
            "--no-mp4",
            "--data-dir",
            str(data),
        ],
    )
    assert result.exit_code == 0, result.output
    built = [json.loads((d / "memory.json").read_text(encoding="utf-8")) for d in out.iterdir()]
    assert built, "nothing was built"
    assert {s["recipe"] for s in built} == {"album_story"}


def test_a_recipe_with_no_offers_exits_2_rather_than_building_something_else(tmp_path):
    data = _library(tmp_path, n=8)
    result = runner.invoke(
        app,
        [
            "memory",
            "--recipe",
            "place_cluster",
            "--out",
            str(tmp_path / "o"),
            "--no-mp4",
            "--data-dir",
            str(data),
        ],
    )
    assert result.exit_code == 2, result.output
    assert "No memory" in result.output
    assert not (tmp_path / "o").exists() or list((tmp_path / "o").iterdir()) == []


def test_key_without_recipe_is_refused_not_guessed(tmp_path):
    """ "10" is a month to `on_this_month` and half a date to `on_this_day`."""
    data = _library(tmp_path, n=8)
    result = runner.invoke(
        app,
        [
            "memory",
            "--key",
            "10",
            "--out",
            str(tmp_path / "o"),
            "--no-mp4",
            "--data-dir",
            str(data),
        ],
    )
    assert result.exit_code == 2, result.output
    assert "--key needs --recipe" in result.output


def test_memory_with_an_unknown_key_exits_2(tmp_path):
    data = _library(tmp_path)
    result = runner.invoke(
        app,
        [
            "memory",
            "--recipe",
            "album_story",
            "--key",
            "NoSuchAlbum",
            "--no-mp4",
            "--data-dir",
            str(data),
            "--out",
            str(tmp_path / "o"),
        ],
    )
    assert result.exit_code == 2
    assert "No memory" in result.output


def test_the_public_safe_marker_is_written_only_when_it_qualifies(tmp_path):
    """The marker exists so the user can choose what to publish with `ls`
    rather than by reading JSON."""
    data = _library(tmp_path, people=("Abhik Maiti",))
    (data / "exclusions.toml").write_text('public_safe_allow = ["Abhik Maiti"]\n', encoding="utf-8")
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "memory",
            "--recipe",
            "album_story",
            "--key",
            "Kashmir",
            "--public-safe",
            "--out",
            str(out),
            "--no-mp4",
            "--data-dir",
            str(data),
        ],
    )
    assert result.exit_code == 0, result.output
    folder = next(iter(out.iterdir()))
    assert (folder / "PUBLIC-SAFE").is_file()
    assert "face tags" in (folder / "PUBLIC-SAFE").read_text(encoding="utf-8")


def test_no_marker_when_a_shot_is_not_public_safe(tmp_path):
    data = _library(tmp_path, people=("Abhik Maiti", "Someone Else"))
    (data / "exclusions.toml").write_text('public_safe_allow = ["Abhik Maiti"]\n', encoding="utf-8")
    out = tmp_path / "out"
    runner.invoke(
        app,
        [
            "memory",
            "--recipe",
            "album_story",
            "--key",
            "Kashmir",
            "--out",
            str(out),
            "--no-mp4",
            "--data-dir",
            str(data),
        ],
    )
    folders = list(out.iterdir())
    if folders:
        assert not (folders[0] / "PUBLIC-SAFE").exists()


def test_public_safe_mode_can_empty_the_library_cleanly(tmp_path):
    """43.7% of a real library is untagged and none of it is public-safe, so
    an empty result is a normal outcome - exit 0 with an explanation, not a
    crash."""
    data = _library(tmp_path, people=())
    result = runner.invoke(
        app,
        [
            "memory",
            "--public-safe",
            "--no-mp4",
            "--data-dir",
            str(data),
            "--out",
            str(tmp_path / "o"),
        ],
    )
    assert result.exit_code == 0
    assert "No photos available" in result.output


def test_a_malformed_exclusions_file_exits_2(tmp_path):
    """Continuing with an unparsed exclusion list would surface exactly the
    photos the user asked never to see."""
    data = _library(tmp_path)
    (data / "exclusions.toml").write_text("people = [unclosed\n", encoding="utf-8")
    result = runner.invoke(app, ["memories", "--data-dir", str(data)])
    assert result.exit_code == 2


def test_an_unfingerprinted_index_warns_rather_than_producing_nothing(tmp_path):
    data = tmp_path / "data"
    photos = []
    for i in range(6):
        path = _jpeg(tmp_path / "lib" / f"p{i}.jpg")
        photo = _photo(
            f"h{i}", path, local=datetime(2020, 5, 1, 9, 0) + timedelta(days=i), albums=["A"]
        )
        photo.meta.phash = None
        photo.meta.sharpness = None
        photo.meta.brightness = None
        photos.append(photo)
    with PhotoStore(data / "rekindle.sqlite") as store:
        store.upsert_many(photos)

    result = runner.invoke(
        app,
        [
            "memory",
            "--recipe",
            "album_story",
            "--key",
            "A",
            "--no-mp4",
            "--out",
            str(tmp_path / "o"),
            "--data-dir",
            str(data),
        ],
    )
    assert result.exit_code == 0, result.output
    # Match the WARNING TEXT, not the bare word "fingerprint".
    #
    # pytest names its temp directory after the test, so `tmp_path` here
    # contains "...test_an_unfingerprinted_index_warns..." - and that path is
    # echoed in the output. An assertion on "fingerprint" alone therefore
    # passed with the warning deleted: the test could not fail. Caught by
    # mutating the call away and watching it stay green.
    assert "no perceptual fingerprint" in result.output
    assert "rekindle fingerprint" in result.output
    assert (tmp_path / "o").is_dir()


# --------------------------------------------------------------------------
# dismissal


def test_dismiss_then_the_memory_is_not_offered(tmp_path):
    data = _library(tmp_path)
    assert (
        runner.invoke(app, ["dismiss", "album_story", "Kashmir", "--data-dir", str(data)]).exit_code
        == 0
    )

    result = runner.invoke(
        app,
        [
            "memory",
            "--recipe",
            "album_story",
            "--key",
            "Kashmir",
            "--no-mp4",
            "--out",
            str(tmp_path / "o"),
            "--data-dir",
            str(data),
        ],
    )
    assert result.exit_code == 0
    assert "Nothing to build" in result.output
    assert not (tmp_path / "o").exists() or list((tmp_path / "o").iterdir()) == []


def test_dismiss_survives_the_library_growing(tmp_path):
    """The identity test, through the CLI. A memory id derived from the photo
    set would let one new photo resurrect a dismissed memory."""
    data = _library(tmp_path)
    runner.invoke(app, ["dismiss", "album_story", "Kashmir", "--data-dir", str(data)])

    with PhotoStore(data / "rekindle.sqlite") as store:
        extra = [
            _photo(
                f"new{i}",
                _jpeg(tmp_path / "lib" / f"new{i}.jpg"),
                local=datetime(2026, 1, 1, 9, 0) + timedelta(days=i),
                albums=["Kashmir"],
            )
            for i in range(20)
        ]
        store.upsert_many(extra)

    result = runner.invoke(
        app,
        [
            "memory",
            "--recipe",
            "album_story",
            "--key",
            "Kashmir",
            "--no-mp4",
            "--out",
            str(tmp_path / "o"),
            "--data-dir",
            str(data),
        ],
    )
    assert "Nothing to build" in result.output


def test_undismiss_restores_it(tmp_path):
    data = _library(tmp_path)
    runner.invoke(app, ["dismiss", "album_story", "Kashmir", "--data-dir", str(data)])
    result = runner.invoke(app, ["undismiss", "album_story", "Kashmir", "--data-dir", str(data)])
    assert result.exit_code == 0
    assert "Restored" in result.output

    with PhotoStore(data / "rekindle.sqlite") as store:
        assert MemoryState(store).dismissed_memory_ids() == frozenset()


def test_undismissing_something_that_was_not_dismissed_is_not_an_error(tmp_path):
    data = _library(tmp_path)
    result = runner.invoke(app, ["undismiss", "album_story", "Nope", "--data-dir", str(data)])
    assert result.exit_code == 0


def test_dismissals_lists_what_has_been_dismissed(tmp_path):
    data = _library(tmp_path)
    runner.invoke(app, ["dismiss", "album_story", "Kashmir", "--data-dir", str(data)])
    result = runner.invoke(app, ["dismissals", "--data-dir", str(data)])
    assert "album_story:Kashmir" in result.output


def test_dismissals_says_so_when_there_are_none(tmp_path):
    data = _library(tmp_path)
    result = runner.invoke(app, ["dismissals", "--data-dir", str(data)])
    assert result.exit_code == 0
    assert "Nothing has been dismissed" in result.output


# --------------------------------------------------------------------------
# exclude


def test_exclude_person_removes_their_photos_everywhere(tmp_path):
    data = _library(tmp_path, people=("Abhik Maiti",))
    assert (
        runner.invoke(
            app, ["exclude", "--person", "Abhik Maiti", "--data-dir", str(data)]
        ).exit_code
        == 0
    )

    result = runner.invoke(app, ["memories", "--data-dir", str(data)])
    assert "No photos available" in result.output


def test_exclude_a_date_range(tmp_path):
    data = _library(tmp_path)
    result = runner.invoke(
        app, ["exclude", "--from", "2020-01-01", "--to", "2020-12-31", "--data-dir", str(data)]
    )
    assert result.exit_code == 0
    assert "No photos available" in runner.invoke(app, ["memories", "--data-dir", str(data)]).output


def test_exclude_rejects_a_malformed_date(tmp_path):
    data = _library(tmp_path)
    result = runner.invoke(
        app, ["exclude", "--from", "yesterday", "--to", "today", "--data-dir", str(data)]
    )
    assert result.exit_code == 2


def test_exclude_with_no_arguments_exits_2(tmp_path):
    data = _library(tmp_path)
    result = runner.invoke(app, ["exclude", "--data-dir", str(data)])
    assert result.exit_code == 2


def test_an_excluded_album_is_honoured(tmp_path):
    data = _library(tmp_path, album="Hospital")
    runner.invoke(app, ["exclude", "--album", "hospital", "--data-dir", str(data)])
    result = runner.invoke(app, ["memories", "--data-dir", str(data)])
    assert "No photos available" in result.output


# --------------------------------------------------------------------------
# fingerprint


def test_fingerprint_runs_and_is_idempotent(tmp_path):
    data = _library(tmp_path)
    with PhotoStore(data / "rekindle.sqlite") as store:
        for photo in list(store.iter_photos()):
            photo.meta.phash = None
            photo.meta.phash_error = None
            store.update_photo(photo)

    first = runner.invoke(app, ["fingerprint", "--data-dir", str(data)])
    assert first.exit_code == 0
    assert "Fingerprinted" in first.output

    second = runner.invoke(app, ["fingerprint", "--data-dir", str(data)])
    assert second.exit_code == 0
    assert "Nothing to do" in second.output


# --------------------------------------------------------------------------
# watch


def test_watch_once_prints_and_renders_nothing(tmp_path):
    data = _library(tmp_path)
    out = tmp_path / "memories"
    out.mkdir()
    result = runner.invoke(app, ["watch", str(tmp_path / "lib"), "--once", "--data-dir", str(data)])
    assert result.exit_code == 0, result.output
    assert "Watching" in result.output
    assert list(out.iterdir()) == [], "the watcher must never render anything"


# --------------------------------------------------------------------------
# the optional GPT caption layer, from the CLI


def test_captions_defaults_to_deterministic_and_makes_no_network_call(tmp_path, monkeypatch):
    """The default configuration makes no network calls at all. Asserted by
    forbidding sockets for the whole invocation."""
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("the default configuration attempted a network call")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)

    data = _library(tmp_path)
    result = runner.invoke(
        app,
        [
            "memory",
            "--recipe",
            "album_story",
            "--key",
            "Kashmir",
            "--no-mp4",
            "--out",
            str(tmp_path / "o"),
            "--data-dir",
            str(data),
        ],
    )
    assert result.exit_code == 0, result.output


def test_an_unknown_captions_mode_exits_2(tmp_path):
    data = _library(tmp_path)
    result = runner.invoke(
        app,
        [
            "memory",
            "--captions",
            "magic",
            "--recipe",
            "album_story",
            "--key",
            "Kashmir",
            "--no-mp4",
            "--out",
            str(tmp_path / "o"),
            "--data-dir",
            str(data),
        ],
    )
    assert result.exit_code == 2
    assert "deterministic" in result.output


def test_captions_gpt_without_a_key_falls_back_cleanly(tmp_path, monkeypatch):
    """A missing key is a supported configuration, not an error: deterministic
    captions, one warning, exit 0."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    data = _library(tmp_path)
    out = tmp_path / "o"
    result = runner.invoke(
        app,
        [
            "memory",
            "--captions",
            "gpt",
            "--recipe",
            "album_story",
            "--key",
            "Kashmir",
            "--no-mp4",
            "--out",
            str(out),
            "--data-dir",
            str(data),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "OPENAI_API_KEY is not set" in result.output
    assert (next(iter(out.iterdir())) / "memory.gif").is_file()


def test_deleting_the_llm_module_leaves_a_working_product(tmp_path, monkeypatch):
    """The test of whether the layer is really ADDITIVE.

    Simulates `rm src/rekindle/memory/llm.py` by making its import fail, then
    builds and renders a memory. If anything outside the --captions gpt path
    reached into it, this breaks.
    """
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "rekindle.memory.llm" or name.endswith(".llm"):
            raise ModuleNotFoundError("No module named 'rekindle.memory.llm'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    monkeypatch.delitem(__import__("sys").modules, "rekindle.memory.llm", raising=False)

    data = _library(tmp_path)
    out = tmp_path / "o"
    result = runner.invoke(
        app,
        [
            "memory",
            "--recipe",
            "album_story",
            "--key",
            "Kashmir",
            "--no-mp4",
            "--out",
            str(out),
            "--data-dir",
            str(data),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (next(iter(out.iterdir())) / "memory.gif").is_file()
