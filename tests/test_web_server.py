"""The HTTP surface, over a real socket.

`http.client` rather than a helper library, because the security tests need to
send headers a well-behaved client would not - a forged `Host`, a foreign
`Origin`, no token at all - and a convenience wrapper that fixes those up
would turn every one of them into a test that cannot fail.

Nothing here needs a browser, a network, a GPU, a model or a photo library
beyond the four JPEGs the fixture writes.
"""

from __future__ import annotations

import http.client
import json

import pytest

from rekindle.web.server import build_app, free_port, serve_in_thread
from tests.fixtures.web import ALBUM, exclude_person, make_library


@pytest.fixture
def running(tmp_path):
    data_dir, _ = make_library(tmp_path)
    app = build_app(
        data_dir,
        out_dir=tmp_path / "memories",
        music_dir=tmp_path / "music",
        port=free_port(),
        token="test-token",
    )
    app.library.load()
    server, _thread = serve_in_thread(app)
    try:
        yield app, server
    finally:
        server.shutdown()
        server.server_close()


class Client:
    """A deliberately dumb HTTP client: it sends exactly what it is told."""

    def __init__(self, port: int, token: str = "test-token") -> None:
        self.port = port
        self.token = token

    def request(self, method, path, *, headers=None, body=None, token=True):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        sent = dict(headers or {})
        if token and "X-Rekindle-Token" not in sent:
            sent["X-Rekindle-Token"] = self.token
        payload = None
        if body is not None:
            payload = json.dumps(body).encode()
            sent["Content-Type"] = "application/json"
        try:
            connection.request(method, path, body=payload, headers=sent)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def json(self, method, path, **kwargs):
        status, headers, body = self.request(method, path, **kwargs)
        return status, json.loads(body.decode()) if body else {}

    def stream(self, path):
        """Read a text/event-stream to its end and parse the frames."""
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        connection.request("GET", path, headers={"X-Rekindle-Token": self.token})
        response = connection.getresponse()
        assert response.status == 200, response.read()
        assert response.getheader("Content-Type").startswith("text/event-stream")
        raw = response.read().decode("utf-8")
        connection.close()
        events = []
        for block in raw.split("\n\n"):
            if not block.strip():
                continue
            name = data = None
            for line in block.splitlines():
                if line.startswith("event: "):
                    name = line[len("event: ") :]
                elif line.startswith("data: "):
                    data = json.loads(line[len("data: ") :])
            events.append({"event": name, "data": data})
        return events


def client(app) -> Client:
    return Client(app.config.port, app.config.token)


# ------------------------------------------------------------------ binding


def test_the_socket_binds_loopback_only(running):
    _app, server = running
    assert server.server_address[0] == "127.0.0.1"


def test_there_is_no_option_to_bind_anywhere_else():
    """A flag exposing a photo library on a LAN would be a foot-gun.

    Asserted on the module, not on a running server: the point is that no
    caller CAN ask for another interface.
    """
    import inspect

    from rekindle.web import server as module

    assert module.HOST == "127.0.0.1"
    assert "host" not in inspect.signature(module.build_app).parameters
    assert "host" not in inspect.signature(module.Server.__init__).parameters


# ------------------------------------------------------------------ the token


def test_no_token_is_refused(running):
    app, _ = running
    status, payload = client(app).json("GET", "/api/status", token=False)
    assert status == 403
    assert "token" in payload["error"].lower()


def test_a_wrong_token_is_refused(running):
    app, _ = running
    status, _ = client(app).json(
        "GET", "/api/status", headers={"X-Rekindle-Token": "not-it"}, token=False
    )
    assert status == 403


def test_the_right_token_is_accepted(running):
    app, _ = running
    status, payload = client(app).json("GET", "/api/status")
    assert status == 200
    assert payload["state"] == "ready"


def test_a_token_in_the_query_string_works_for_img_tags(running):
    """`<img src=...>` cannot set a header, so thumbnails carry `?t=`."""
    app, _ = running
    status, _ = client(app).json("GET", f"/api/status?t={app.config.token}", token=False)
    assert status == 200


# ------------------------------------------------- rebinding and cross-origin


def test_a_forged_host_header_is_refused(running):
    app, _ = running
    status, payload = client(app).json("GET", "/api/status", headers={"Host": "photos.example.com"})
    assert status == 403
    assert "127.0.0.1" in payload["error"]


def test_a_foreign_origin_is_refused(running):
    app, _ = running
    status, _ = client(app).json(
        "POST",
        "/api/edit",
        headers={"Origin": "https://evil.example"},
        body={"session_id": "x", "op": "remove", "file_hash": "y"},
    )
    assert status == 403


def test_the_pages_own_origin_is_accepted(running):
    app, _ = running
    origin = f"http://127.0.0.1:{app.config.port}"
    status, payload = client(app).json(
        "POST",
        "/api/edit",
        headers={"Origin": origin},
        body={"session_id": "nope", "op": "remove", "file_hash": "y"},
    )
    # Past the origin gate, refused for the real reason instead.
    assert status == 404
    assert "session" in payload["error"].lower()


def test_a_cross_site_post_without_an_origin_is_refused(running):
    app, _ = running
    status, _ = client(app).json(
        "POST",
        "/api/edit",
        headers={"Sec-Fetch-Site": "cross-site"},
        body={"session_id": "x", "op": "remove", "file_hash": "y"},
    )
    assert status == 403


# ------------------------------------------------------------------- assets


def test_the_page_and_its_assets_are_served_locally(running):
    app, _ = running
    status, _headers, body = client(app).request("GET", "/")
    assert status == 200
    text = body.decode()
    assert "<title>rekindle</title>" in text
    # The whole promise of this page: nothing is fetched from anywhere else.
    for marker in ("http://", "https://"):
        assert marker not in text.replace("http://www.w3.org/2000/svg", "")
    status, _headers, script = client(app).request("GET", "/assets/app.js")
    assert status == 200 and b"rekindle" in script


def test_assets_are_an_allow_list_not_a_path_join(running):
    app, _ = running
    for path in (
        "/assets/../__init__.py",
        "/assets/..%2f__init__.py",
        "/assets/server.py",
        "/assets/",
    ):
        status, _ = client(app).json("GET", path)
        assert status == 404, path


def test_responses_carry_a_content_security_policy(running):
    app, _ = running
    _status, headers, _body = client(app).request("GET", "/")
    policy = headers["Content-Security-Policy"]
    assert "default-src 'self'" in policy
    assert "connect-src 'self'" in policy
    assert headers["Referrer-Policy"] == "no-referrer"


# --------------------------------------------------------------- thumbnails


def test_a_library_photo_has_a_thumbnail(running):
    app, _ = running
    status, headers, body = client(app).request("GET", "/api/thumb/ok00?w=320")
    assert status == 200
    assert headers["Content-Type"] == "image/jpeg"
    assert body[:2] == b"\xff\xd8", "a JPEG"


def test_an_archived_photo_has_no_thumbnail(running):
    """The critical rule. A 404 here and a 200 is a defect, not a nicety."""
    app, _ = running
    status, _ = client(app).json("GET", "/api/thumb/archived?w=320")
    assert status == 404


def test_an_excluded_photo_loses_its_thumbnail(tmp_path):
    data_dir, _ = make_library(tmp_path)
    app = build_app(
        data_dir,
        out_dir=tmp_path / "memories",
        music_dir=tmp_path / "music",
        port=free_port(),
        token="t",
    )
    app.library.load()
    server, _thread = serve_in_thread(app)
    try:
        api = Client(app.config.port, "t")
        assert api.request("GET", "/api/thumb/excluded?w=320")[0] == 200
    finally:
        server.shutdown()
        server.server_close()

    exclude_person(data_dir)
    app2 = build_app(
        data_dir,
        out_dir=tmp_path / "memories",
        music_dir=tmp_path / "music",
        port=free_port(),
        token="t",
    )
    app2.library.load()
    server2, _thread2 = serve_in_thread(app2)
    try:
        api2 = Client(app2.config.port, "t")
        assert api2.request("GET", "/api/thumb/excluded?w=320")[0] == 404
    finally:
        server2.shutdown()
        server2.server_close()


def test_thumbnails_are_cached_and_revalidate(running):
    app, _ = running
    status, headers, first = client(app).request("GET", "/api/thumb/ok01?w=320")
    assert status == 200
    etag = headers["ETag"]
    cached = app.thumbs.path_for("ok01", 320)
    assert cached.is_file(), "the second page load must not decode the original again"
    status, _headers, body = client(app).request(
        "GET", "/api/thumb/ok01?w=320", headers={"If-None-Match": etag}
    )
    assert status == 304 and body == b""
    assert cached.read_bytes() == first


def test_only_the_two_real_widths_ever_reach_the_cache(running):
    """A query string cannot fill the cache with a thousand sizes of a photo."""
    from rekindle.web.thumbs import SIZES

    app, _ = running
    legitimate = {}
    for width in SIZES:
        status, _headers, body = client(app).request("GET", f"/api/thumb/ok02?w={width}")
        assert status == 200
        legitimate[width] = body

    for junk in ("4096", "0", "-5", "321", "abc", ""):
        status, _headers, body = client(app).request("GET", f"/api/thumb/ok02?w={junk}")
        assert status == 200, junk
        assert body in legitimate.values(), junk

    on_disk = {d.name for d in app.thumbs.root.iterdir() if d.is_dir()}
    assert on_disk == {str(w) for w in SIZES}


def test_the_cache_itself_refuses_a_width_it_does_not_know(tmp_path):
    """Defence in depth: the route clamps, and the cache refuses anyway."""
    from rekindle.web.thumbs import ThumbnailCache, ThumbnailError
    from tests.fixtures.web import open_library

    data_dir, _ = make_library(tmp_path)
    index = open_library(data_dir).require_index()
    photo = index.get("ok00")
    cache = ThumbnailCache(tmp_path / "thumbs")
    with pytest.raises(ThumbnailError, match="unsupported"):
        cache.get(photo, index.resolve_path(photo), 4096)
    assert not (tmp_path / "thumbs" / "4096").exists()


# -------------------------------------------------------------- the build SSE


def test_building_a_recipe_memory_streams_to_ready(running):
    app, _ = running
    events = client(app).stream(f"/api/build?recipe=album_story&key={ALBUM}&t=test-token")
    names = [e["event"] for e in events]
    assert names[-1] == "ready"
    assert "stage" in names
    session = events[-1]["data"]
    assert session["order"], "the memory must contain shots"
    assert "archived" not in {c["file_hash"] for c in session["candidates"]}


def test_an_unknown_recipe_ends_the_stream_with_an_error(running):
    app, _ = running
    events = client(app).stream("/api/build?recipe=nope&key=x&t=test-token")
    assert events[-1]["event"] == "error"
    assert "nope" in events[-1]["data"]["message"]


def test_a_stream_that_blows_up_still_ends_with_a_terminal_event(running, monkeypatch):
    """A stream that just stops is a stream the browser RECONNECTS to, which
    silently re-runs the whole build. Every failure has to arrive as an
    event."""
    app, _ = running
    from rekindle.web import api as api_module

    def explode(*_args, **_kwargs):
        raise RuntimeError("the disk caught fire")

    monkeypatch.setattr(api_module, "session_state", explode)
    events = client(app).stream(f"/api/build?recipe=album_story&key={ALBUM}&t=test-token")
    assert events[-1]["event"] == "error"
    assert "disk caught fire" in events[-1]["data"]["message"]


# ------------------------------------------------------------------- editing


def build_session(app) -> dict:
    events = client(app).stream(f"/api/build?recipe=album_story&key={ALBUM}&t=test-token")
    assert events[-1]["event"] == "ready", events[-1]
    return events[-1]["data"]


def test_an_edit_survives_a_reload_of_the_session(running):
    app, _ = running
    session = build_session(app)
    victim = session["order"][0]

    status, after = client(app).json(
        "POST",
        "/api/edit",
        body={"session_id": session["session_id"], "op": "remove", "file_hash": victim},
    )
    assert status == 200
    assert victim not in after["order"]

    status, reloaded = client(app).json("GET", f"/api/session/{session['session_id']}")
    assert status == 200
    assert victim not in reloaded["order"], "a dropped photo stays dropped"


def test_adding_an_archived_photo_over_http_is_refused(running):
    app, _ = running
    session = build_session(app)
    status, payload = client(app).json(
        "POST",
        "/api/edit",
        body={"session_id": session["session_id"], "op": "add", "file_hash": "archived"},
    )
    assert status == 400
    assert "guardrails" in payload["error"]


def test_search_never_returns_a_withheld_photo(running):
    app, _ = running
    status, payload = client(app).json("GET", f"/api/search?q={ALBUM}&mode=metadata")
    assert status == 200
    found = {hit["file_hash"] for hit in payload["hits"]}
    assert found, "the album must match something"
    assert "archived" not in found


def test_semantic_search_without_the_extra_says_what_to_install(running, monkeypatch):
    app, _ = running
    from rekindle.semantic import availability

    monkeypatch.setattr(
        availability,
        "probe",
        lambda: availability.Availability(
            cpu=False, gpu=False, missing_cpu=("numpy",), missing_gpu=("torch",)
        ),
    )
    status, payload = client(app).json("GET", "/api/search?q=mountains&mode=semantic")
    assert status == 501
    assert "uv sync --extra semantic" in payload["error"]


def test_the_same_day_control_needs_no_model(running):
    app, _ = running
    status, payload = client(app).json("GET", "/api/day?hash=burst_keep")
    assert status == 200
    assert payload["day"] == "2020-05-20"
    assert {h["file_hash"] for h in payload["hits"]} == {"burst_keep", "burst_drop"}


# ------------------------------------------------------------------- outputs


def test_output_is_refused_before_anything_is_rendered(running):
    app, _ = running
    session = build_session(app)
    status, _ = client(app).json("GET", f"/api/output/{session['session_id']}/memory.webp")
    assert status == 404


def test_output_names_are_an_allow_list(running, tmp_path):
    """A traversal that WOULD reach a real file, and a real file with the
    wrong suffix. Both must 404 - a target that happens not to exist proves
    nothing about the check."""
    app, _ = running
    session = build_session(app)
    client(app).json(
        "POST", "/api/render", body={"session_id": session["session_id"], "no_mp4": True}
    )
    folder = app.workshop.get(session["session_id"]).last_render.folder

    escape = "../../data/rekindle.sqlite"
    assert (folder / escape).is_file(), "the traversal target must genuinely exist"
    (folder / "notes.txt").write_text("private", encoding="utf-8")

    for name in (escape, "..%2F..%2Fdata%2Frekindle.sqlite", "notes.txt", "memory.exe"):
        status, _headers, body = client(app).request(
            "GET", f"/api/output/{session['session_id']}/{name}"
        )
        assert status == 404, name
        assert b"private" not in body and b"SQLite" not in body, name

    status, _headers, body = client(app).request(
        "GET", f"/api/output/{session['session_id']}/memory.webp"
    )
    assert status == 200 and body[:4] == b"RIFF"


def test_a_range_request_returns_a_partial_body(running):
    app, _ = running
    session = build_session(app)
    client(app).json(
        "POST", "/api/render", body={"session_id": session["session_id"], "no_mp4": True}
    )
    status, headers, body = client(app).request(
        "GET",
        f"/api/output/{session['session_id']}/memory.webp",
        headers={"Range": "bytes=0-99"},
    )
    assert status == 206
    assert len(body) == 100
    assert headers["Content-Range"].startswith("bytes 0-99/")
