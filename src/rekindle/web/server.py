"""The HTTP layer: stdlib only, loopback only, and deliberately small.

WHY THERE IS NO WEB FRAMEWORK HERE
----------------------------------
Measured against this project's own resolver on 2026-09-11:

    flask               7 packages
    fastapi + uvicorn  12 packages
    starlette           4 packages (+3 more for an ASGI server to run it)
    bottle              1 package
    http.server         0 packages

`rekindle` ships four runtime dependencies. This is a single-user page served
on 127.0.0.1 that needs routing, JSON, static files, byte ranges and
server-sent events - 188 lines of the plumbing below, counted - and none of
the things a framework is actually for: no concurrency model to choose, no
WSGI/ASGI deployment, no request validation of untrusted input from the
internet, no templating (the page is one static HTML file). Buying twelve
packages for 188 lines would be the expensive choice, and it would put a
web framework in the dependency graph of `rekindle index`.

It also buys something concrete: **CI runs these tests.** A framework behind an
optional extra means the server tests skip on every machine that does not
install it, and this project's own `pyproject.toml` already states the position
- "a skipped test is not a passing test". `tests/test_web_server.py` runs on a
plain `uv sync`.

SECURITY, BECAUSE THESE ARE SOMEONE'S FAMILY PHOTOGRAPHS
--------------------------------------------------------
Four rules, each enforced here and pinned by a test:

1. **The socket binds to 127.0.0.1 and there is no option to change it.** Not
   a default - there is no `--host`. A flag to expose a photo library on a
   LAN is a foot-gun with no good use.
2. **Every route needs a per-run secret.** Loopback is not a permission
   boundary: any process and any other user on this machine can connect. The
   token is printed in the URL, held by the page, and sent on every request.
3. **The `Host` header must name the loopback.** A page on the internet can
   resolve its own hostname to 127.0.0.1 and make the browser send requests
   here (DNS rebinding); the token already stops that, and this stops it
   twice.
4. **`Origin` must be absent or loopback on anything that changes state.** A
   same-origin `fetch` sends no `Origin` on GET and the page's own origin on
   POST, so this costs the real page nothing and refuses a cross-site one.

Nothing is fetched from a CDN, no font is loaded from Google, no analytics
exist. Everything the page needs is in `assets/` and is served from there.
"""

from __future__ import annotations

import json
import secrets
import socket
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

from rekindle.web import api
from rekindle.web import calibrate_api as calibrate
from rekindle.web.library import Library, LibraryError
from rekindle.web.thumbs import GRID, SIZES, ThumbnailCache, ThumbnailError

HOST = "127.0.0.1"
ASSETS = Path(__file__).parent / "assets"

#: What `index.html` writes where the run's token goes. The file on disk
#: holds the placeholder; `_shell` substitutes the real token when it serves
#: the page, so no token is ever written to the source tree.
TOKEN_PLACEHOLDER = "__REKINDLE_TOKEN__"

#: Files the page may fetch. An allow-list rather than a path join, so no
#: amount of `..` or URL-encoding reaches a file that is not part of the UI.
STATIC = {
    "app.css": "text/css; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
}

#: What the rendered outputs may be served as. Same reasoning: the browser
#: asks for a name, and a name not on this list is not a file.
OUTPUT_TYPES = {
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".mp4": "video/mp4",
    ".json": "application/json; charset=utf-8",
}

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}


@dataclass
class Config:
    data_dir: Path
    out_dir: Path
    music_dir: Path
    port: int
    token: str
    public_safe: bool = False
    max_shots: int = 24


class App:
    """Everything a handler needs, hung off the server object."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.library = Library(config.data_dir, public_safe=config.public_safe)
        self.workshop = api.Workshop(
            self.library,
            out_dir=config.out_dir,
            music_dir=config.music_dir,
            max_shots=config.max_shots,
        )
        self.thumbs = ThumbnailCache(config.data_dir / "cache" / "thumbs")
        # Built lazily inside itself: the library loads on a background thread
        # and a calibration session needs its photographs.
        self.calibration = calibrate.CalibrationRoom(self.library, memories_root=config.out_dir)

    @property
    def origins(self) -> set[str]:
        port = self.config.port
        return {
            f"http://127.0.0.1:{port}",
            f"http://localhost:{port}",
            f"http://[::1]:{port}",
        }

    def url(self) -> str:
        return f"http://{HOST}:{self.config.port}/?t={self.config.token}"


class Handler(BaseHTTPRequestHandler):
    # HTTP/1.1 so the browser reuses connections for a grid of 200 thumbnails
    # instead of opening 200 sockets. Every response below therefore sends an
    # accurate Content-Length, or closes the connection (the SSE case).
    protocol_version = "HTTP/1.1"
    server_version = "rekindle"
    sys_version = ""

    @property
    def app(self) -> App:
        return self.server.app  # type: ignore[attr-defined]

    # ---- logging: silent by default. A request log of a photo library is a
    # record of which of someone's photographs they looked at, written to a
    # terminal they may be sharing. The CLI turns it on with --verbose.
    def log_message(self, fmt: str, *args) -> None:  # noqa: A002
        if getattr(self.server, "verbose", False):  # pragma: no cover
            super().log_message(fmt, *args)

    # ---- entry points

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        parts = urlsplit(self.path)
        path = unquote(parts.path)
        query = parse_qs(parts.query)
        try:
            self._guard(method, query)
            self._route(method, path, query)
        except api.ApiError as exc:
            self._json({"error": str(exc), "hint": exc.hint}, status=exc.status)
        except LibraryError as exc:
            self._json({"error": str(exc), "hint": ""}, status=503)
        except ConnectionError:  # pragma: no cover - the tab was closed
            # BrokenPipeError on Unix, ConnectionAbortedError on Windows, and
            # ConnectionResetError when a browser cancels an image it has
            # scrolled past - which a thumbnail grid does constantly. All three
            # mean "nobody is listening", not "something went wrong".
            self.close_connection = True

    # ---- the four security rules

    def _guard(self, method: str, query: dict[str, list[str]]) -> None:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        if host not in LOOPBACK_HOSTS:
            raise api.ApiError(
                "This server answers only to 127.0.0.1.", status=HTTPStatus.FORBIDDEN
            )
        origin = self.headers.get("Origin")
        if origin and origin not in self.app.origins:
            raise api.ApiError("Cross-origin requests are refused.", status=HTTPStatus.FORBIDDEN)
        site = self.headers.get("Sec-Fetch-Site")
        if method != "GET" and origin is None and site not in (None, "same-origin", "none"):
            raise api.ApiError("Cross-site requests are refused.", status=HTTPStatus.FORBIDDEN)

        supplied = self.headers.get("X-Rekindle-Token") or (query.get("t") or [""])[0]
        # compare_digest, not ==: a plain comparison leaks the length of the
        # matching prefix through timing, and a local attacker is exactly the
        # attacker who can measure that.
        if not secrets.compare_digest(supplied, self.app.config.token):
            raise api.ApiError(
                "Wrong or missing token. Open the URL that `rekindle ui` printed.",
                status=HTTPStatus.FORBIDDEN,
            )

    # ---- routing

    def _route(self, method: str, path: str, query: dict[str, list[str]]) -> None:
        workshop = self.app.workshop
        one = _first(query)

        if method == "GET":
            if path == "/":
                return self._shell()
            if path.startswith("/assets/"):
                name = path[len("/assets/") :]
                if name not in STATIC:
                    return self._json({"error": "no such asset"}, status=404)
                return self._file(ASSETS / name, STATIC[name])
            if path == "/api/status":
                return self._json(api.status(workshop))
            if path == "/api/offers":
                return self._json(api.offers(workshop, recipe=one("recipe") or None))
            if path == "/api/suggestions":
                return self._json(api.suggestions(workshop))
            if path == "/api/build":
                return self._stream(
                    api.build_events(
                        workshop,
                        recipe=one("recipe") or None,
                        key=one("key") if "key" in query else None,
                        prompt=one("prompt") or None,
                    )
                )
            if path.startswith("/api/thumb/"):
                return self._thumb(path[len("/api/thumb/") :], one("w"))
            if path == "/api/search":
                return self._json(
                    api.search(
                        workshop,
                        query=one("q"),
                        mode=one("mode") or "metadata",
                        limit=_int(one("limit"), 60, 1, 300),
                    )
                )
            if path == "/api/similar":
                return self._json(
                    api.neighbours(
                        workshop, file_hash=one("hash"), limit=_int(one("limit"), 40, 1, 200)
                    )
                )
            if path == "/api/day":
                return self._json(api.capture_day(workshop, file_hash=one("hash")))
            if path == "/api/calibrate":
                return self._json(calibrate.status(self.app.calibration))
            if path == "/api/calibrate/example":
                return self._json(calibrate.example(self.app.calibration, setting=one("setting")))
            if path == "/api/calibrate/consequence":
                return self._json(
                    calibrate.consequence(
                        self.app.calibration,
                        setting=one("setting"),
                        value=_float(one("value")),
                    )
                )
            if path == "/api/calibrate/affected":
                return self._json(calibrate.affected(self.app.calibration))
            if path == "/api/render/progress":
                return self._json(api.render_progress(workshop, one("session_id")))
            if path.startswith("/api/session/"):
                return self._json(
                    api.session_state(workshop, workshop.get(path[len("/api/session/") :]))
                )
            if path.startswith("/api/output/"):
                return self._output(path[len("/api/output/") :])

        if method == "POST":
            body = self._body()
            if path == "/api/edit":
                return self._json(api.edit(workshop, _need(body, "session_id"), body))
            if path == "/api/render":
                return self._json(api.render(workshop, _need(body, "session_id"), body))
            if path == "/api/dismiss":
                return self._json(api.dismiss(workshop, _need(body, "session_id")))
            if path == "/api/calibrate/begin":
                return self._json(calibrate.begin(self.app.calibration))
            if path == "/api/calibrate/answer":
                # The VERDICT only. The browser never sends back the measured
                # value: a hand-crafted POST could then place a judgement
                # anywhere on the scale, and the server already knows which
                # example it handed out.
                return self._json(
                    calibrate.answer(
                        self.app.calibration,
                        setting=_need(body, "setting"),
                        said_yes=bool(body.get("said_yes")),
                    )
                )
            if path == "/api/calibrate/choose":
                return self._json(
                    calibrate.choose(
                        self.app.calibration,
                        setting=_need(body, "setting"),
                        value=(None if body.get("value") is None else float(body["value"])),
                        accept=bool(body.get("accept")),
                    )
                )
            if path == "/api/calibrate/confirm":
                return self._json(
                    calibrate.confirm(
                        self.app.calibration,
                        setting=_need(body, "setting"),
                        phrase=str(body.get("phrase", "")),
                    )
                )
            if path == "/api/calibrate/skip":
                return self._json(
                    calibrate.skip(self.app.calibration, setting=_need(body, "setting"))
                )
            if path == "/api/calibrate/reset":
                return self._json(
                    calibrate.reset(self.app.calibration, setting=_need(body, "setting"))
                )
            if path == "/api/calibrate/finish":
                return self._json(calibrate.finish(self.app.calibration))
            if path == "/api/calibrate/not-now":
                return self._json(calibrate.dismiss_offer(self.app.calibration))

        self._json({"error": f"no route for {method} {path}"}, status=404)

    # ---- responses

    def _body(self) -> dict:
        length = _int(self.headers.get("Content-Length"), 0, 0, 1 << 20)
        if not length:
            return {}
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise api.ApiError("The request body was not JSON.") from exc
        if not isinstance(payload, dict):
            raise api.ApiError("The request body must be a JSON object.")
        return payload

    def _json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self._privacy_headers()
        self.end_headers()
        self.wfile.write(data)

    def _shell(self) -> None:
        """`index.html`, with the run's token stamped into its asset URLs.

        Security rule 2 is "every route needs the token", and `<link>` and
        `<script>` are in exactly the position `<img>` is: an element the
        browser fetches for you, which cannot be given a header. Thumbnails
        already solved this with `?t=`; the stylesheet and the script had not,
        so both were served a 403 and the page rendered as unstyled markup
        with no behaviour at all. They carry the token the same way now.

        Substituted here rather than written into the file, because the token
        changes every run and the file on disk must never hold one.
        """
        try:
            html = (ASSETS / "index.html").read_text(encoding="utf-8")
        except OSError:
            return self._json({"error": "missing asset"}, status=404)
        html = html.replace(TOKEN_PLACEHOLDER, quote(self.app.config.token, safe=""))
        return self._bytes(html.encode("utf-8"), "text/html; charset=utf-8")

    def _file(self, path: Path, content_type: str) -> None:
        try:
            data = path.read_bytes()
        except OSError:
            return self._json({"error": "missing asset"}, status=404)
        return self._bytes(data, content_type)

    def _bytes(self, data: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        # No caching for the shell: a redeployed asset must not be shadowed by
        # a stale copy when the user upgrades rekindle under the same port.
        self.send_header("Cache-Control", "no-store")
        self._privacy_headers()
        self.end_headers()
        self.wfile.write(data)

    def _privacy_headers(self) -> None:
        """Headers that stop the page leaking what is being viewed.

        `Referrer-Policy: no-referrer` so that if a link ever escapes this
        page, the target learns nothing. A CSP with `default-src 'self'` so a
        bug in `app.js` still cannot fetch a font, a script or a tracking
        pixel from anywhere else - which is the machine-checked version of the
        promise that nothing about someone's photographs leaves the machine.
        """
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; media-src 'self'; "
            "style-src 'self'; script-src 'self'; connect-src 'self'; "
            "font-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
        )

    def _stream(self, events: Iterator[dict]) -> None:
        """Server-sent events, one JSON object per frame.

        No Content-Length, so the connection delimits the body and must close
        at the end - `Connection: close` sets `close_connection` on this
        handler. `X-Accel-Buffering` is meaningless without a proxy and
        harmless with one; it is here because a user who puts this behind one
        would otherwise see the whole stream arrive at once.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self._privacy_headers()
        self.end_headers()

        def frame(name: str, data: dict) -> bytes:
            return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()

        try:
            for event in events:
                self.wfile.write(frame(event["event"], event["data"]))
                self.wfile.flush()
        except ConnectionError:  # pragma: no cover - the tab was closed
            raise
        except Exception as exc:  # noqa: BLE001
            # A stream that ends without a terminal event is a stream the
            # browser RECONNECTS to, which re-runs the whole build - so an
            # unexpected failure has to arrive as an event rather than as a
            # dropped connection. `app.js` closes the EventSource on `error`.
            self.wfile.write(frame("error", {"message": f"The build failed: {exc}", "hint": ""}))
            self.wfile.flush()

    def _thumb(self, file_hash: str, width_raw: str) -> None:
        """A thumbnail, or 404. THE guardrail, in four lines.

        `MemoryIndex.get` is the only lookup. An archived, trashed or excluded
        photo is not on the index, so it has no thumbnail and no error that
        distinguishes it from a hash that never existed - which is the correct
        answer to "is this photo in the library" for a photo the user asked
        never to see.
        """
        index = self.app.library.require_index()
        photo = index.get(file_hash)
        if photo is None:
            return self._json({"error": "no such photo"}, status=404)
        source = index.resolve_path(photo)
        if source is None:
            return self._json({"error": "the file is not on this machine"}, status=404)
        width = _int(width_raw, GRID, min(SIZES), max(SIZES))
        if width not in SIZES:
            width = GRID
        try:
            thumb = self.app.thumbs.get(photo, source, width)
        except ThumbnailError as exc:
            return self._json({"error": str(exc)}, status=415)
        if self.headers.get("If-None-Match") == thumb.etag:
            self.send_response(304)
            self.send_header("ETag", thumb.etag)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(thumb.data)))
        self.send_header("ETag", thumb.etag)
        self.send_header("Cache-Control", "private, max-age=3600")
        self._privacy_headers()
        self.end_headers()
        self.wfile.write(thumb.data)

    def _output(self, rest: str) -> None:
        """Serve a rendered file, and only from a session's own folder.

        The path is never joined from user input: the session id names a
        folder this process itself created, and the file name has to match one
        of four known outputs. `<video>` needs byte ranges, so a single-range
        request is honoured.
        """
        session_id, _, name = rest.partition("/")
        session = self.app.workshop.get(session_id)
        result = session.last_render
        if result is None:
            return self._json({"error": "nothing has been rendered yet"}, status=404)
        suffix = Path(name).suffix.lower()
        if suffix not in OUTPUT_TYPES or "/" in name or "\\" in name or ".." in name:
            return self._json({"error": "no such output"}, status=404)
        target = result.folder / name
        try:
            data = target.read_bytes()
        except OSError:
            return self._json({"error": "no such output"}, status=404)

        start, end = _range(self.headers.get("Range"), len(data))
        chunk = data[start : end + 1]
        self.send_response(206 if start or end != len(data) - 1 else 200)
        self.send_header("Content-Type", OUTPUT_TYPES[suffix])
        self.send_header("Content-Length", str(len(chunk)))
        self.send_header("Accept-Ranges", "bytes")
        if start or end != len(data) - 1:
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        self.send_header("Cache-Control", "no-store")
        self._privacy_headers()
        self.end_headers()
        self.wfile.write(chunk)


def _range(header: str | None, size: int) -> tuple[int, int]:
    """One byte range, clamped. Anything unparseable means the whole file."""
    if not header or not header.startswith("bytes=") or "," in header:
        return (0, size - 1)
    spec = header[len("bytes=") :]
    first, _, last = spec.partition("-")
    try:
        if not first:
            length = int(last)
            return (max(0, size - length), size - 1)
        start = int(first)
        end = int(last) if last else size - 1
    except ValueError:
        return (0, size - 1)
    start = max(0, min(start, size - 1))
    end = max(start, min(end, size - 1))
    return (start, end)


def _first(query: dict[str, list[str]]):
    def get(name: str) -> str:
        values = query.get(name) or [""]
        return values[0]

    return get


def _int(raw: str | None, default: int, low: int, high: int) -> int:
    try:
        value = int(raw or "")
    except (TypeError, ValueError):
        return default
    return max(low, min(value, high))


def _float(raw: str | None) -> float:
    """A query-string number, or a refusal naming the parameter.

    Not clamped here: `config.Setting.clamp_error` owns the range, and a
    second range check in the HTTP layer is a second thing to keep in step
    with `defaults.toml`.
    """
    try:
        return float(raw or "")
    except (TypeError, ValueError):
        raise api.ApiError("`value` must be a number.") from None


def _need(body: dict, field: str) -> str:
    value = body.get(field)
    if not isinstance(value, str) or not value:
        raise api.ApiError(f"{field} is required.")
    return value


class Server(ThreadingHTTPServer):
    daemon_threads = True
    # `HTTPServer` sets SO_REUSEADDR, and its meaning differs by platform.
    #
    # On Unix it only allows binding over sockets in TIME_WAIT, which is what
    # makes stop-and-restart work; two live listeners still collide. Keep it.
    #
    # On WINDOWS the same option lets a DIFFERENT PROCESS bind a port this one
    # is already listening on, and connections are then split between them.
    # For a server holding someone's photo library that is a hijack, not a
    # convenience, so it is off there and a second `rekindle ui` fails to bind
    # with a message naming `--port`.
    allow_reuse_address = sys.platform != "win32"

    def __init__(self, app: App) -> None:
        self.app = app
        self.verbose = False
        super().__init__((HOST, app.config.port), Handler)


def build_app(
    data_dir: Path,
    *,
    out_dir: Path,
    music_dir: Path,
    port: int,
    public_safe: bool = False,
    max_shots: int = 24,
    token: str | None = None,
) -> App:
    return App(
        Config(
            data_dir=data_dir,
            out_dir=out_dir,
            music_dir=music_dir,
            port=port,
            token=token or secrets.token_urlsafe(24),
            public_safe=public_safe,
            max_shots=max_shots,
        )
    )


def serve(app: App) -> Server:
    """Bind and return the server, without serving. The caller runs the loop."""
    try:
        return Server(app)
    except OSError as exc:
        raise LibraryError(
            f"Could not bind {HOST}:{app.config.port} ({exc}). "
            "Another rekindle ui may already be running; pass --port."
        ) from exc


def serve_in_thread(app: App) -> tuple[Server, threading.Thread]:
    """For the tests: a real socket, a real thread, torn down by the caller."""
    server = serve(app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def free_port() -> int:
    """An OS-assigned free port, for tests and for `--port 0`."""
    with socket.socket() as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])
