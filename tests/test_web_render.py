"""Rendering, and the promise that the printed command reproduces the edit.

The test that matters is `test_the_reproduce_command_rebuilds_the_same_files`:
it edits a memory the way a person would - drop two shots, reverse the rest -
renders it through the API, then runs the argv the UI hands back through the
real CLI and compares the bytes. If the reproduce command ever stops
reproducing, that fails.

No browser, no network, no ffmpeg: every assertion is about the WebP and GIF,
which Pillow writes locally, and `--no-mp4` keeps ffmpeg out of it.
"""

from __future__ import annotations

import json
import shutil

import pytest
from PIL import Image
from typer.testing import CliRunner

from rekindle.cli import app as cli
from rekindle.memory.spec import MemorySpec
from rekindle.web import api
from rekindle.web.renderer import RenderOptions, render_spec
from tests.fixtures.web import ALBUM, exclude_person, make_library, make_workshop, open_library

runner = CliRunner()


def built(tmp_path, data_dir):
    workshop = make_workshop(tmp_path, data_dir)
    events = list(api.build_events(workshop, recipe="album_story", key=ALBUM))
    assert events[-1]["event"] == "ready", events[-1]
    return workshop, events[-1]["data"]["session_id"]


def test_the_reproduce_command_rebuilds_the_same_files(tmp_path):
    data_dir, _ = make_library(tmp_path)
    workshop, session_id = built(tmp_path, data_dir)

    # Edit it the way a person would.
    state = api.session_state(workshop, workshop.get(session_id))
    api.edit(workshop, session_id, {"op": "remove", "file_hash": state["order"][0]})
    state = api.session_state(workshop, workshop.get(session_id))
    api.edit(workshop, session_id, {"op": "remove", "file_hash": state["order"][2]})
    state = api.session_state(workshop, workshop.get(session_id))
    api.edit(workshop, session_id, {"op": "reorder", "order": list(reversed(state["order"]))})
    api.edit(workshop, session_id, {"op": "pace", "frame_ms": 900, "title_ms": 1800})

    state = api.render(workshop, session_id, {"no_mp4": True})
    folder = workshop.get(session_id).last_render.folder
    argv = state["reproduce"]
    assert argv[:3] == ["rekindle", "render", str(folder / "memory.json")]
    assert "--frame-ms" in argv and "900" in argv

    # Keep what the browser produced, then run the command it printed.
    kept = tmp_path / "kept"
    shutil.copytree(folder, kept)

    result = runner.invoke(
        cli, [*argv[1:], "--no-mp4", "--data-dir", str(data_dir)], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output

    for name in ("memory.json", "memory.webp", "memory.gif"):
        assert (folder / name).read_bytes() == (kept / name).read_bytes(), name


def test_the_reproduce_command_carries_the_users_order(tmp_path):
    """Not just the same bytes - the RIGHT bytes: the user's own sequence."""
    data_dir, _ = make_library(tmp_path)
    workshop, session_id = built(tmp_path, data_dir)
    state = api.session_state(workshop, workshop.get(session_id))
    wanted = list(reversed(state["order"]))
    api.edit(workshop, session_id, {"op": "reorder", "order": wanted})
    api.render(workshop, session_id, {"no_mp4": True})

    folder = workshop.get(session_id).last_render.folder
    spec = MemorySpec.loads((folder / "memory.json").read_text(encoding="utf-8"))
    assert [s.file_hash for s in spec.shots] == wanted

    # And rendering that spec from the shell keeps the order.
    result = runner.invoke(
        cli,
        ["render", str(folder / "memory.json"), "--no-mp4", "--data-dir", str(data_dir)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    again = MemorySpec.loads((folder / "memory.json").read_text(encoding="utf-8"))
    assert [s.file_hash for s in again.shots] == wanted


def test_render_leaves_the_spec_it_was_given_untouched(tmp_path):
    data_dir, _ = make_library(tmp_path)
    workshop, session_id = built(tmp_path, data_dir)
    api.render(workshop, session_id, {"no_mp4": True})
    spec_path = workshop.get(session_id).last_render.folder / "memory.json"

    # A hand-edit a user might make: a caption they preferred, a note of their
    # own, and whatever indentation their editor uses. None of it round-trips
    # through `MemorySpec.dumps`, which is the point - if `render` rewrote the
    # file, all three would be silently normalised away.
    raw = json.loads(spec_path.read_text(encoding="utf-8"))
    raw["shots"][0]["caption"] = "the one I like"
    raw["_note"] = "hand-edited, do not touch"
    spec_path.write_text(json.dumps(raw, indent=4) + "\n", encoding="utf-8")
    before = spec_path.read_bytes()
    assert b"_note" in before and b"\n    " in before

    result = runner.invoke(
        cli,
        ["render", str(spec_path), "--no-mp4", "--data-dir", str(data_dir)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert spec_path.read_bytes() == before, "an input a command overwrites is not an input"


def test_a_spec_naming_an_excluded_photo_renders_without_it(tmp_path):
    """A memory.json is safe to keep: excluding someone tomorrow removes them
    from every spec already on disk, with no file edited."""
    data_dir, _ = make_library(tmp_path)
    workshop, session_id = built(tmp_path, data_dir)
    api.render(workshop, session_id, {"no_mp4": True})
    spec_path = workshop.get(session_id).last_render.folder / "memory.json"
    spec = MemorySpec.loads(spec_path.read_text(encoding="utf-8"))
    doomed = [s for s in spec.shots if s.file_hash.startswith("excluded")]
    assert doomed, "the spec must name the photos the exclusion will withhold"
    frames_before = _frame_count(spec_path.parent / "memory.webp")

    exclude_person(data_dir)
    result = runner.invoke(
        cli,
        ["render", str(spec_path), "--no-mp4", "--data-dir", str(data_dir)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "no longer admitted by the guardrails" in " ".join(result.output.split())
    assert _frame_count(spec_path.parent / "memory.webp") == frames_before - len(doomed)
    # The spec still names them - the file was not rewritten - and the render
    # still refuses them. That is the guardrail, not a stale file.
    again = MemorySpec.loads(spec_path.read_text(encoding="utf-8"))
    assert [s.file_hash for s in again.shots] == [s.file_hash for s in spec.shots]


def test_render_refuses_a_spec_that_is_not_one(tmp_path):
    data_dir, _ = make_library(tmp_path)
    bad = tmp_path / "memory.json"
    bad.write_text('{"spec_version": 99}', encoding="utf-8")
    result = runner.invoke(cli, ["render", str(bad), "--data-dir", str(data_dir)])
    assert result.exit_code == 2
    # rich hard-wraps the console, so compare on collapsed whitespace.
    assert "not a readable MemorySpec" in " ".join(result.output.split())


def test_render_accepts_the_folder_as_well_as_the_file(tmp_path):
    data_dir, _ = make_library(tmp_path)
    workshop, session_id = built(tmp_path, data_dir)
    api.render(workshop, session_id, {"no_mp4": True})
    folder = workshop.get(session_id).last_render.folder
    result = runner.invoke(
        cli,
        ["render", str(folder), "--no-mp4", "--data-dir", str(data_dir)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output


def test_render_without_an_index_exits_two_and_leaves_no_database(tmp_path):
    spec = tmp_path / "memory.json"
    spec.write_text(
        json.dumps(
            {
                "spec_version": 1,
                "recipe": "album_story",
                "key": "X",
                "title": "X",
                "subtitle": "",
                "public_safe": False,
                "shots": [],
                "facts": {
                    "title": "X",
                    "recipe": "album_story",
                    "photo_count": 0,
                    "years": [],
                    "per_year": {},
                    "people": {},
                    "albums": [],
                    "video_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    empty = tmp_path / "nowhere"
    result = runner.invoke(cli, ["render", str(spec), "--data-dir", str(empty)])
    assert result.exit_code == 2
    assert not (empty / "rekindle.sqlite").exists()


def test_the_pace_reaches_the_file(tmp_path):
    """`--frame-ms` is not decoration: it changes the frame durations."""
    data_dir, _ = make_library(tmp_path)
    library = open_library(data_dir)
    workshop, session_id = built(tmp_path, data_dir)
    api.render(workshop, session_id, {"no_mp4": True})
    spec_path = workshop.get(session_id).last_render.folder / "memory.json"
    spec = MemorySpec.loads(spec_path.read_text(encoding="utf-8"))

    slow = tmp_path / "slow"
    fast = tmp_path / "fast"
    render_spec(spec, library.require_index(), slow, RenderOptions(frame_ms=2000, no_mp4=True))
    render_spec(spec, library.require_index(), fast, RenderOptions(frame_ms=400, no_mp4=True))
    assert _durations(slow / "memory.gif")[1] == 2000
    assert _durations(fast / "memory.gif")[1] == 400


def test_a_photo_added_by_hand_reaches_the_rendered_frames(tmp_path):
    data_dir, _ = make_library(tmp_path)
    workshop, session_id = built(tmp_path, data_dir)
    state = api.session_state(workshop, workshop.get(session_id))
    spare = next(c["file_hash"] for c in state["candidates"] if c["state"] == "rejected")

    api.render(workshop, session_id, {"no_mp4": True})
    before = _frame_count(workshop.get(session_id).last_render.folder / "memory.webp")

    api.edit(workshop, session_id, {"op": "add", "file_hash": spare})
    api.render(workshop, session_id, {"no_mp4": True})
    folder = workshop.get(session_id).last_render.folder
    assert _frame_count(folder / "memory.webp") == before + 1
    spec = MemorySpec.loads((folder / "memory.json").read_text(encoding="utf-8"))
    assert spare in {s.file_hash for s in spec.shots}


def test_rendering_an_empty_memory_is_refused(tmp_path):
    data_dir, _ = make_library(tmp_path)
    workshop, session_id = built(tmp_path, data_dir)
    state = api.session_state(workshop, workshop.get(session_id))
    for file_hash in list(state["order"]):
        api.edit(workshop, session_id, {"op": "remove", "file_hash": file_hash})
    with pytest.raises(api.ApiError, match="empty memory"):
        api.render(workshop, session_id, {"no_mp4": True})


def _frame_count(path) -> int:
    with Image.open(path) as image:
        return getattr(image, "n_frames", 1)


def _durations(path) -> list[int]:
    out = []
    with Image.open(path) as image:
        for index in range(getattr(image, "n_frames", 1)):
            image.seek(index)
            out.append(image.info.get("duration", 0))
    return out
