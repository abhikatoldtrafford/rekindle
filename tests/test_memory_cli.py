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
import typer
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


def test_force_rebuilds_an_unnamed_build_that_the_cooldown_would_refuse(tmp_path):
    """The gap between the two tests above. Naming ONE memory overrode the
    cooldown; a bulk rebuild had no override at all, and refused 29 of 34
    album stories on a deliberate re-render after the dedup code changed.
    `--force` is that override. Without the flag this exact argv reports
    "cooldown" and writes nothing, which is what the sibling test pins."""
    data = _library(tmp_path)
    out = tmp_path / "out"
    base = ["memory", "--out", str(out), "--no-mp4", "--data-dir", str(data)]

    assert runner.invoke(app, [*base, "--recipe", "album_story", "--key", "Kashmir"]).exit_code == 0
    forced = runner.invoke(app, [*base, "--force"])
    assert forced.exit_code == 0, forced.output
    assert "cooldown" not in forced.output, forced.output
    assert (out / next(iter(p.name for p in out.iterdir())) / "memory.json").is_file()


def test_force_does_not_override_a_dismissal(tmp_path):
    """`--force` bypasses the resurfacing clock and NOTHING else. If it also
    resurrected dismissals, "never show me this again" would be undone by a
    flag whose help text promises only to ignore a cooldown."""
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
            "--force",
            "--out",
            str(out),
            "--no-mp4",
            "--data-dir",
            str(data),
        ],
    )
    assert not out.exists() or not any(out.iterdir()), result.output


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


# --------------------------------------------------------------------------
# Prompt memories
#
# `rekindle memory "some words"`. The retriever is injected in every test
# below, so none of these needs torch, an embedding store or a network - and
# the two that DO exercise the real semantic path assert the message, not the
# model.


def _festival_library(tmp_path) -> Path:
    """Four Octobers of a made-up festival, plus an unrelated May."""
    data = tmp_path / "data"
    photos = []
    for year in (2019, 2020, 2021, 2022):
        for i in range(6):
            path = _jpeg(tmp_path / "lib" / f"f{year}{i}.jpg", colour=(40 + i * 20, 90, 60))
            photos.append(
                _photo(
                    f"f{year}{i}",
                    path,
                    local=datetime(year, 10, 4, 9, i * 5),
                    people=["Abhik Maiti"],
                )
            )
    for i in range(6):
        path = _jpeg(tmp_path / "lib" / f"m{i}.jpg", colour=(200, 30 + i * 10, 30))
        photos.append(_photo(f"m{i}", path, local=datetime(2019, 5, 1, 9, i * 5)))
    with PhotoStore(data / "rekindle.sqlite") as store:
        store.upsert_many(photos)
    return data


def _fake_retriever(hashes):
    def retrieve(text, k):
        return [(h, 1.0 - i * 0.01) for i, h in enumerate(hashes[:k])]

    return retrieve


def _run_prompt(data, text, tmp_path, hashes, **kw):
    from rekindle.memory.cli import prompt_cmd

    return prompt_cmd(
        data,
        text,
        tmp_path / "out",
        no_mp4=True,
        retrieve=_fake_retriever(hashes),
        **kw,
    )


def test_a_prompt_memory_writes_a_spec_keyed_on_the_normalised_prompt(tmp_path, capsys):
    data = _festival_library(tmp_path)
    seeds = [f"f{y}{i}" for y in (2019, 2020, 2021, 2022) for i in range(2)]
    _run_prompt(data, "  A Made Up Festival  ", tmp_path, seeds)

    folders = list((tmp_path / "out").glob("*prompt*"))
    assert len(folders) == 1
    spec = json.loads((folders[0] / "memory.json").read_text(encoding="utf-8"))
    assert spec["recipe"] == "prompt"
    assert spec["key"] == "a made up festival"
    assert spec["title"] == "a made up festival"
    assert spec["facts"]["title_substantiated"] is False
    assert (folders[0] / "memory.gif").is_file()


def test_the_same_prompt_twice_gives_a_byte_identical_spec(tmp_path):
    data = _festival_library(tmp_path)
    seeds = [f"f{y}{i}" for y in (2019, 2020, 2021, 2022) for i in range(2)]
    _run_prompt(data, "a made up festival", tmp_path, seeds)
    first = next((tmp_path / "out").glob("*prompt*")) / "memory.json"
    text = first.read_text(encoding="utf-8")
    _run_prompt(data, "A MADE UP FESTIVAL", tmp_path, seeds)
    assert first.read_text(encoding="utf-8") == text


def test_the_report_names_the_tags_the_seed_days_and_the_caveat(tmp_path, capsys):
    from rekindle.memory.cli import PROMPT_CAVEAT

    data = _festival_library(tmp_path)
    seeds = [f"f{y}{i}" for y in (2019, 2020, 2021, 2022) for i in range(2)]
    _run_prompt(data, "a beach", tmp_path, seeds)
    # Rich wraps to the terminal width, so the assertions are made against the
    # text with its line breaks collapsed rather than against the layout.
    out = " ".join(capsys.readouterr().out.split())

    # The tags ARE the query; a user who cannot see them cannot tell a bad
    # memory from a bad description of what they asked for.
    assert "waves breaking on wet sand" in out
    assert "2019-10-04" in out and "2022-10-04" in out
    # The month histogram, so the user can see a festival split themselves.
    assert "months: October 24" in out
    assert "shots have face tags" in out
    for line in PROMPT_CAVEAT.splitlines():
        assert line in out


def test_a_word_that_narrowed_nothing_is_named(tmp_path, capsys):
    data = _festival_library(tmp_path)
    seeds = [f"f{y}{i}" for y in (2019, 2020, 2021, 2022) for i in range(2)]
    _run_prompt(data, "christmas in midnapur", tmp_path, seeds)
    out = " ".join(capsys.readouterr().out.split())
    assert "midnapur" in out
    assert "narrowed nothing" in out
    assert "gazetteer" in out


def test_the_weak_path_is_named_when_no_source_describes_the_prompt(tmp_path, capsys):
    from rekindle.memory.cli import WEAK_PATH_WARNING

    data = _festival_library(tmp_path)
    seeds = [f"f{y}{i}" for y in (2019, 2020, 2021, 2022) for i in range(2)]
    _run_prompt(data, "a thing nobody has ever described", tmp_path, seeds)
    out = " ".join(capsys.readouterr().out.split())
    assert "from your own words" in out
    assert WEAK_PATH_WARNING.split(".")[0] in out


def test_a_dismissed_prompt_memory_is_not_rendered(tmp_path, capsys):
    data = _festival_library(tmp_path)
    seeds = [f"f{y}{i}" for y in (2019, 2020, 2021, 2022) for i in range(2)]
    with PhotoStore(data / "rekindle.sqlite") as store:
        MemoryState(store).dismiss("memory", "prompt:a made up festival")
    _run_prompt(data, "a made up festival", tmp_path, seeds)
    assert "Dismissed" in capsys.readouterr().out
    assert not list((tmp_path / "out").glob("*prompt*"))


def test_a_prompt_that_finds_no_agreeing_day_builds_nothing(tmp_path, capsys):
    data = _festival_library(tmp_path)
    # One hit per day: never reaches MIN_SEEDS.
    _run_prompt(data, "a made up festival", tmp_path, ["f20190", "f20200", "f20210"])
    out = " ".join(capsys.readouterr().out.split())
    assert "No capture day had enough agreement" in out
    assert not list((tmp_path / "out").glob("*prompt*"))


def test_a_prompt_memory_never_becomes_an_offer(tmp_path):
    """The hard rule. `rekindle memories` and `--auto` both walk the registry,
    so a prompt memory reaching either of them would mean it had been
    registered - which `tests/test_engine.py` also forbids from the other
    side."""
    data = _festival_library(tmp_path)
    seeds = [f"f{y}{i}" for y in (2019, 2020, 2021, 2022) for i in range(2)]
    _run_prompt(data, "a made up festival", tmp_path, seeds)

    listing = runner.invoke(app, ["memories", "--data-dir", str(data)])
    assert listing.exit_code == 0
    assert "made up festival" not in listing.output

    auto = runner.invoke(
        app,
        ["memory", "--auto", "--no-mp4", "--out", str(tmp_path / "auto"), "--data-dir", str(data)],
    )
    assert "made up festival" not in auto.output


def test_a_prompt_cannot_be_combined_with_recipe_key_or_auto(tmp_path):
    data = _festival_library(tmp_path)
    for extra in (["--recipe", "album_story"], ["--key", "X"], ["--auto"]):
        result = runner.invoke(app, ["memory", "words", *extra, "--data-dir", str(data)])
        assert result.exit_code == 2, result.output
        assert "cannot be combined" in result.output


def test_an_empty_prompt_is_refused(tmp_path):
    data = _festival_library(tmp_path)
    result = runner.invoke(app, ["memory", "   ", "--data-dir", str(data)])
    assert result.exit_code == 2
    assert "empty prompt" in result.output


def test_a_library_that_cannot_be_searched_says_which_problem_it_is(tmp_path):
    """Three different failures, three different fixes, never folded together.

    Without the extra: install it. With the extra but no embeddings: run
    `rekindle semantic embed`. Neither may look like SKIP_EMPTY, which means
    "your prompt found nothing" and would send the user off rewording a prompt
    that was never searched.
    """
    data = _festival_library(tmp_path)
    result = runner.invoke(
        app,
        ["memory", "a beach", "--no-mp4", "--out", str(tmp_path / "o"), "--data-dir", str(data)],
    )
    flat = " ".join(result.output.split())
    assert result.exit_code in (2, 3), result.output
    if result.exit_code == 3:
        # The extra is missing. The message names the command that installs it.
        assert "uv sync --extra semantic" in flat
    else:
        # The extra is here; the library has simply never been embedded.
        assert "rekindle semantic embed" in flat
    assert "no_candidates" not in flat
    assert not list((tmp_path / "o").glob("*"))


def test_the_judge_can_refuse_before_anything_is_built(tmp_path, monkeypatch, capsys):
    """Gate one. The transport is injected; the assertion is on the BEHAVIOUR -
    exit code 4 and nothing rendered - not on a mock having been called."""
    from rekindle.memory import tags as tags_mod
    from rekindle.memory.cli import EXIT_REFUSED, prompt_cmd

    data = _festival_library(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        tags_mod,
        "generator_from_env",
        lambda transport=None: tags_mod.TagGenerator(
            "test-key", transport=lambda payload, key: {"output_text": "REFUSE: that is gibberish"}
        ),
    )
    with pytest.raises(typer.Exit) as exit_info:
        prompt_cmd(
            data,
            "qwertyuiop asdfgh",
            tmp_path / "out",
            no_mp4=True,
            retrieve=_fake_retriever(["f20190"]),
        )
    assert exit_info.value.exit_code == EXIT_REFUSED
    assert "that is gibberish" in capsys.readouterr().out
    assert not list((tmp_path / "out").glob("*"))


def test_no_judge_builds_the_memory_the_judge_refused(tmp_path, monkeypatch):
    """Guards the test above: the refusal is the judge's doing, not the
    library's."""
    from rekindle.memory import tags as tags_mod

    data = _festival_library(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        tags_mod,
        "generator_from_env",
        lambda transport=None: tags_mod.TagGenerator(
            "test-key", transport=lambda payload, key: {"output_text": "REFUSE: that is gibberish"}
        ),
    )
    seeds = [f"f{y}{i}" for y in (2019, 2020, 2021, 2022) for i in range(2)]
    _run_prompt(data, "a beach", tmp_path, seeds, judge=False)
    assert list((tmp_path / "out").glob("*prompt*"))


# --------------------------------------------------------------------------
# what the build says about the optional embedding store
#
# Absence is a supported configuration, not a failure - it is what CI runs and
# what every install without the extra runs - so the message has to name the
# command that would change it rather than say nothing at all.


def _flat(output: str) -> str:
    """Console output with its wrapping removed.

    `rich` hard-wraps to the terminal width, so "uv sync --extra semantic" can
    arrive with a newline inside it. Asserting on the raw string makes a test
    that passes or fails on the width of the machine running it.
    """
    return " ".join(output.split())


def _fake_store(monkeypatch, support):
    """Make the REAL `_semantic_support` find `support` (or nothing).

    Patching `_semantic_support` itself would skip the messages that are the
    whole point of these tests.
    """
    monkeypatch.setattr("rekindle.semantic.diversity.open_support", lambda *a, **k: support)


def _build(tmp_path, data, *extra):
    return runner.invoke(
        app,
        ["memory", "--data-dir", str(data), "--out", str(tmp_path / "out"), "--no-mp4", *extra],
    )


class _FakeSupport:
    """Stands in for a real store so the reporting can be tested with no
    models, no GPU and no network - which is what CI has."""

    def __init__(self, cosine_value=0.0, signal=None):
        self.cosine_value = cosine_value
        self.signal = signal

    def __len__(self):
        return 1234

    def cosine(self, a, b):
        return self.cosine_value

    def signal_for(self, photos):
        return self.signal


def test_a_library_with_no_embeddings_says_so_and_still_builds(tmp_path, monkeypatch):
    # The extra IS installed in this environment, so the "never embedded"
    # branch has to be reached deliberately rather than by luck.
    _fake_store(monkeypatch, None)
    result = _build(tmp_path, _library(tmp_path))
    assert result.exit_code == 0, result.output
    assert "no embeddings yet" in _flat(result.output)
    assert "rekindle semantic embed" in _flat(result.output)


def test_a_machine_without_the_extra_is_told_which_command_installs_it(tmp_path, monkeypatch):
    from rekindle.semantic.availability import Availability

    monkeypatch.setattr(
        "rekindle.semantic.availability.probe",
        lambda: Availability(cpu=False, gpu=False, missing_cpu=("numpy",), missing_gpu=("torch",)),
    )
    result = _build(tmp_path, _library(tmp_path))
    assert result.exit_code == 0, result.output
    assert "uv sync --extra semantic" in _flat(result.output)


def test_a_corrupt_store_is_reported_and_the_build_carries_on(tmp_path, monkeypatch):
    """A broken optional feature must not take down a command that works
    perfectly well without it."""

    def explode(*a, **k):
        raise RuntimeError("vectors.f32 is truncated")

    monkeypatch.setattr("rekindle.semantic.diversity.open_support", explode)
    result = _build(tmp_path, _library(tmp_path))
    assert result.exit_code == 0, result.output
    assert "truncated" in _flat(result.output)
    assert "continuing without it" in _flat(result.output)


def test_a_working_store_is_announced_with_how_many_vectors(tmp_path, monkeypatch):
    _fake_store(monkeypatch, _FakeSupport())
    result = _build(tmp_path, _library(tmp_path))
    assert result.exit_code == 0, result.output
    assert "1234 vectors" in _flat(result.output)


def test_the_semantic_collapses_are_reported_apart_from_the_others(tmp_path, monkeypatch):
    """ "N near-duplicate frames collapsed" does not tell a user which pile to
    look in. These are the frames the perceptual hash called UNRELATED, and a
    user checking the guardrail by eye needs to know which they were."""
    _fake_store(monkeypatch, _FakeSupport(cosine_value=0.99))
    # Every photo on the same second, so the 30-second gate is wide open and
    # the cosine is the only thing deciding.
    data = tmp_path / "data"
    photos = []
    for i in range(8):
        path = _jpeg(tmp_path / "lib" / f"p{i}.jpg", colour=(40 + i * 20, 90, 60))
        photos.append(_photo(f"h{i}", path, local=datetime(2020, 5, 1, 9, 0), albums=["Kashmir"]))
    with PhotoStore(data / "rekindle.sqlite") as store:
        store.upsert_many(photos)
    result = _build(tmp_path, data)
    assert "near-duplicate frames collapsed" in _flat(result.output)
    assert "perceptual hash missed" in _flat(result.output)


def test_nothing_is_claimed_about_the_embedding_when_it_collapsed_nothing(tmp_path, monkeypatch):
    _fake_store(monkeypatch, _FakeSupport(cosine_value=0.0))
    result = _build(tmp_path, _library(tmp_path))
    assert result.exit_code == 0, result.output
    assert "perceptual hash missed" not in _flat(result.output)
    assert "chosen differently" not in _flat(result.output)


def test_shots_the_embedding_reordered_are_reported(tmp_path, monkeypatch):
    class _Shuns:
        """h1 looks like a duplicate of whatever is already chosen; nothing
        else resembles anything. So h1 loses its slot to a worse photo."""

        name = "shuns"
        weight = 1.0

        def between(self, a, b):
            return 0.0 if "h1" in {a.file_hash, b.file_hash} else 1.0

    _fake_store(monkeypatch, _FakeSupport(signal=_Shuns()))
    result = _build(tmp_path, _library(tmp_path), "--max-shots", "3")
    assert result.exit_code == 0, result.output
    assert "chosen differently" in _flat(result.output)


# --------------------------------------------------------------- --all-recipes


def _spread_library(tmp_path) -> Path:
    """Five distinct albums, plus album-LESS photos of a person nobody else
    appears with.

    Distinct albums so five album stories can coexist without the overlap
    guard eating four of them, and the album-less block so at least one other
    recipe has photographs no album story can claim. Both shapes are taken
    from the reference library, which has 34 real albums and 3,930 photos in
    no album but a Takeout year bucket.
    """
    data = tmp_path / "data"
    photos = []
    n = 0
    for year, album in ((2016, "Kashmir"), (2017, "Ladakh"), (2018, "Goa"), (2019, "Sikkim")):
        for i in range(10):
            path = _jpeg(
                tmp_path / "lib" / f"p{n}.jpg", colour=((n * 37) % 240, (n * 11) % 200, 60)
            )
            photos.append(
                _photo(
                    f"h{n:03d}",
                    path,
                    local=datetime(year, 6, 11, 9, 0) + timedelta(hours=i),
                    albums=[album],
                    people=("Abhik Maiti",),
                )
            )
            n += 1
    for year in (2016, 2018, 2020):
        for i in range(6):
            path = _jpeg(tmp_path / "lib" / f"p{n}.jpg", colour=((n * 53) % 250, 40, (n * 7) % 200))
            photos.append(
                _photo(
                    f"h{n:03d}",
                    path,
                    local=datetime(year, 11, 2, 9, 0) + timedelta(hours=i),
                    albums=[],
                    people=("Riya",),
                )
            )
            n += 1
    with PhotoStore(data / "rekindle.sqlite") as store:
        store.upsert_many(photos)
    return data


def _built(out: Path) -> dict[str, int]:
    if not out.exists():
        return {}
    counts: dict[str, int] = {}
    for d in out.iterdir():
        spec = json.loads((d / "memory.json").read_text(encoding="utf-8"))
        counts[spec["recipe"]] = counts.get(spec["recipe"], 0) + 1
    return counts


def _memory(data, out, *argv):
    return runner.invoke(
        app, ["memory", *argv, "--out", str(out), "--no-mp4", "--data-dir", str(data)]
    )


def test_all_recipes_reaches_recipes_a_bigger_limit_never_does(tmp_path):
    """The whole point, measured against the thing it replaces.

    Raising `--limit` does not help: offers come in registry order, so a
    bigger cap just builds more of the FIRST recipe. `--all-recipes` spends
    the cap per recipe instead.
    """
    data = _spread_library(tmp_path)
    plain = _memory(data, tmp_path / "o1", "--limit", "2")
    assert plain.exit_code == 0, plain.output
    assert _built(tmp_path / "o1") == {"album_story": 2}

    bigger = _memory(data, tmp_path / "o2", "--limit", "4", "--force")
    assert bigger.exit_code == 0, bigger.output
    assert set(_built(tmp_path / "o2")) == {"album_story"}, "a bigger cap must not help"

    every = _memory(data, tmp_path / "o3", "--all-recipes", "--limit", "2", "--force")
    assert every.exit_code == 0, every.output
    spread = _built(tmp_path / "o3")
    assert spread.get("album_story") == 2
    assert len(spread) > 1, f"--all-recipes built only {spread}"


def test_all_recipes_with_limit_zero_builds_every_offer_it_can(tmp_path):
    data = _spread_library(tmp_path)
    capped = _memory(data, tmp_path / "o1", "--all-recipes", "--limit", "1")
    assert capped.exit_code == 0, capped.output
    uncapped = _memory(data, tmp_path / "o2", "--all-recipes", "--limit", "0", "--force")
    assert uncapped.exit_code == 0, uncapped.output
    assert sum(_built(tmp_path / "o2").values()) > sum(_built(tmp_path / "o1").values())
    assert _built(tmp_path / "o2")["album_story"] == 4


def test_all_recipes_reports_every_recipe_including_the_silent_ones(tmp_path):
    """A run that builds nothing from six of nine recipes has to say so.

    The reported failure exited 0 with an empty directory and no explanation.
    Silence is the bug; the table is the fix.
    """
    data = _spread_library(tmp_path)
    result = _memory(data, tmp_path / "o", "--all-recipes", "--limit", "1")
    assert result.exit_code == 0, result.output
    assert "9 recipes registered" in result.output
    for name in ("album_story", "year_in_review", "place_cluster", "recurring_event"):
        assert name in result.output


def test_all_recipes_never_reports_zero_recipes(tmp_path):
    """The registry cannot be empty, so the announcement cannot say it is."""
    data = _spread_library(tmp_path)
    result = _memory(data, tmp_path / "o", "--all-recipes", "--limit", "1")
    assert "0 recipes registered" not in result.output


@pytest.mark.parametrize(
    "conflict",
    [
        ["--recipe", "album_story"],
        ["--recipe", "album_story", "--key", "Kashmir"],
        ["--auto"],
    ],
    ids=["recipe", "recipe+key", "auto"],
)
def test_all_recipes_refuses_a_contradictory_argument(tmp_path, conflict):
    """ "Every recipe" and "this one memory" are contradictory instructions.
    Resolving them by precedence is how this project's signature defect - a
    command silently doing something other than what its arguments say - gets
    in."""
    data = _spread_library(tmp_path)
    out = tmp_path / "o"
    result = _memory(data, out, "--all-recipes", *conflict)
    assert result.exit_code == 2, result.output
    assert "--all-recipes cannot be combined" in result.output
    assert not out.exists() or list(out.iterdir()) == []


def test_all_recipes_refuses_a_prompt(tmp_path):
    data = _spread_library(tmp_path)
    out = tmp_path / "o"
    result = runner.invoke(
        app,
        [
            "memory",
            "durga puja",
            "--all-recipes",
            "--out",
            str(out),
            "--no-mp4",
            "--data-dir",
            str(data),
        ],
    )
    assert result.exit_code == 2, result.output
    assert "--all-recipes cannot be combined" in result.output


def test_without_all_recipes_nothing_changes(tmp_path):
    """The default path must be byte-identical to what it was."""
    data = _spread_library(tmp_path)
    first = _memory(data, tmp_path / "o1", "--limit", "3")
    assert first.exit_code == 0, first.output
    assert "--all-recipes" not in first.output
    assert "every recipe, and what it built" not in first.output
