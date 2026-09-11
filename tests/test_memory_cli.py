"""The memory CLI.

The recurring guarantee: a command that needs an index and does not find one
exits 2 and leaves NO database file behind. `PhotoStore` creates its file on
open, so without an explicit check every one of these would silently report a
healthy library of zero photos.
"""

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
