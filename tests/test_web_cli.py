"""`rekindle ui` and `rekindle render` as commands.

The wiring is a real place for defects: the root `cli.py` already binds the
name `render` to `doctor.render`, so registering the memory renderer as a bare
`def render` would shadow it and break `rekindle doctor` and `rekindle index`
while every test of those two kept passing. `test_the_doctor_renderer_is_not_
shadowed` is that guard.
"""

from __future__ import annotations

import http.client
import json
import threading

import pytest
import typer
from typer.testing import CliRunner

from rekindle.cli import app as cli
from tests.fixtures.web import make_library
from tests.helpers import unwrapped

runner = CliRunner()


def stop(server) -> None:
    """Shut a test server down without ever hanging.

    `BaseServer.shutdown()` waits for `serve_forever` to acknowledge, and on a
    server whose loop was never started it waits FOREVER. That is only
    reachable from a broken build - but a broken build that hangs the suite is
    strictly worse than one that fails it, so the wait is bounded here.
    """
    stopper = threading.Thread(target=server.shutdown, daemon=True)
    stopper.start()
    stopper.join(timeout=10)
    server.server_close()
    assert not stopper.is_alive(), "the server was never being served"


def test_both_verbs_are_registered_and_documented():
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    collapsed = " ".join(result.output.split())
    assert "ui " in collapsed and "render " in collapsed
    assert "127.0.0.1" in collapsed


def test_the_doctor_renderer_is_not_shadowed():
    """`rekindle render` is registered under an explicit command name because
    `render` is already bound in that module. If it were not, this import
    would give the typer command instead of the doctor's renderer and
    `rekindle doctor` would fail at runtime, not at import."""
    from rekindle import cli as module
    from rekindle.doctor import render as doctor_render

    assert module.render is doctor_render


def test_ui_without_an_index_exits_two_and_leaves_no_database(tmp_path, capsys):
    """`PhotoStore` creates its file on open, so without an explicit check the
    UI would open on a healthy library of zero photos and leave a stray
    database behind - the same trap every other memory command guards.

    Driven through `ui_cmd` rather than through `CliRunner`, deliberately: the
    real command blocks in `serve_forever`, so a build that lost this guard
    would HANG here instead of failing, and a hanging test looks exactly like
    a slow one.
    """
    from rekindle.web.cli import ui_cmd

    empty = tmp_path / "nowhere"
    with pytest.raises(typer.Exit) as caught:
        ui_cmd(
            empty,
            out_dir=tmp_path / "memories",
            music_dir=tmp_path / "music",
            port=0,
            open_browser=False,
            serve_forever=False,
        )
    assert caught.value.exit_code == 2
    assert "rekindle index" in " ".join(capsys.readouterr().out.split())
    assert not (empty / "rekindle.sqlite").exists()


def test_ui_serves_the_page_on_loopback_with_a_token(tmp_path, capsys):
    """The whole command, wired as the CLI wires it."""
    from rekindle.web.cli import ui_cmd

    data_dir, _ = make_library(tmp_path)
    server = ui_cmd(
        data_dir,
        out_dir=tmp_path / "memories",
        music_dir=tmp_path / "music",
        port=0,
        open_browser=False,
        serve_forever=False,
    )
    try:
        app = server.app
        assert server.server_address[0] == "127.0.0.1"
        printed = capsys.readouterr().out
        assert app.config.token in "".join(printed.split("\n"))

        connection = http.client.HTTPConnection("127.0.0.1", app.config.port, timeout=30)
        connection.request("GET", f"/?t={app.config.token}")
        response = connection.getresponse()
        body = response.read()
        connection.close()
        assert response.status == 200
        assert b"<title>rekindle</title>" in body

        # And the same page without the token is refused, so the printed URL
        # is not decoration.
        connection = http.client.HTTPConnection("127.0.0.1", app.config.port, timeout=30)
        connection.request("GET", "/")
        response = connection.getresponse()
        response.read()
        connection.close()
        assert response.status == 403
    finally:
        stop(server)


def test_two_uis_cannot_share_a_port(tmp_path):
    """A second instance must fail loudly rather than split traffic with the
    first. On Windows SO_REUSEADDR would otherwise allow exactly that."""
    import sys

    from rekindle.web.cli import ui_cmd

    if sys.platform != "win32":
        pytest.skip("SO_REUSEADDR only permits the hijack on Windows")
    data_dir, _ = make_library(tmp_path)
    first = ui_cmd(
        data_dir,
        out_dir=tmp_path / "memories",
        music_dir=tmp_path / "music",
        port=0,
        open_browser=False,
        serve_forever=False,
    )
    try:
        with pytest.raises(typer.Exit) as caught:
            ui_cmd(
                data_dir,
                out_dir=tmp_path / "memories",
                music_dir=tmp_path / "music",
                port=first.app.config.port,
                open_browser=False,
                serve_forever=False,
            )
        assert caught.value.exit_code == 2
    finally:
        stop(first)


def test_render_help_names_the_spec(tmp_path):
    result = runner.invoke(cli, ["render", "--help"])
    assert result.exit_code == 0
    collapsed = unwrapped(result.output)
    assert unwrapped("memory.json") in collapsed
    assert unwrapped("--frame-ms") in collapsed


def test_the_ui_reports_a_missing_semantic_extra_without_failing(tmp_path, capsys, monkeypatch):
    from rekindle.semantic import availability
    from rekindle.web.cli import ui_cmd

    monkeypatch.setattr(
        availability,
        "probe",
        lambda: availability.Availability(
            cpu=False, gpu=False, missing_cpu=("numpy",), missing_gpu=("torch",)
        ),
    )
    data_dir, _ = make_library(tmp_path)
    server = ui_cmd(
        data_dir,
        out_dir=tmp_path / "memories",
        music_dir=tmp_path / "music",
        port=0,
        open_browser=False,
        serve_forever=False,
    )
    try:
        printed = " ".join(capsys.readouterr().out.split())
        assert unwrapped("uv sync --extra semantic") in unwrapped(printed)
        assert "Everything else works without it" in printed
        # And the server is up regardless: the extra gates one panel, not the
        # command.
        connection = http.client.HTTPConnection("127.0.0.1", server.app.config.port, timeout=30)
        connection.request(
            "GET", "/api/status", headers={"X-Rekindle-Token": server.app.config.token}
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        connection.close()
        assert response.status == 200
        assert payload["semantic_installed"] is False
        assert payload["state"] in ("ready", "loading")
    finally:
        stop(server)
