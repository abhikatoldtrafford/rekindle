"""`rekindle ui` and `rekindle render`, the two verbs this package adds.

Both keep working headless. `render` never imports `server.py`, never opens a
socket and never needs a browser - it is the command the UI hands back, and a
reproduce command that only works while the thing it reproduces is running
would not be one.
"""

from __future__ import annotations

import contextlib
import threading
import webbrowser
from pathlib import Path

import typer
from rich.console import Console

from rekindle.memory.render.gif import DEFAULT_FRAME_MS, DEFAULT_MAX_FRAMES, TITLE_MS
from rekindle.memory.render.mp4 import FFMPEG_MISSING
from rekindle.memory.spec import MemorySpec
from rekindle.web.renderer import SPEC_NAME, RenderOptions, render_spec

console = Console()


def ui_cmd(
    data_dir: Path,
    *,
    out_dir: Path,
    music_dir: Path,
    port: int,
    public_safe: bool = False,
    max_shots: int = 24,
    open_browser: bool = True,
    verbose: bool = False,
    serve_forever: bool = True,
) -> object:
    """Start the local editor. Binds 127.0.0.1 and nothing else."""
    from rekindle.web.library import LibraryError
    from rekindle.web.server import build_app, free_port, serve

    db_path = data_dir / "rekindle.sqlite"
    if not db_path.is_file():
        console.print(f"[red]No index at[/red] {db_path}. Run `rekindle index <folder>` first.")
        raise typer.Exit(code=2)

    app = build_app(
        data_dir,
        out_dir=out_dir,
        music_dir=music_dir,
        port=port or free_port(),
        public_safe=public_safe,
        max_shots=max_shots,
    )
    try:
        server = serve(app)
    except LibraryError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    server.verbose = verbose

    # Reading 19,480 rows and applying the policy takes a noticeable moment.
    # It starts now, in the background, so the page is already there when the
    # browser opens and the first request is the only one that ever waits.
    app.library.start()

    console.print(f"[green]rekindle[/green] is editing memories at [bold]{app.url()}[/bold]")
    console.print(
        "[dim]Bound to 127.0.0.1 only. The token in that URL is required on every "
        "request and changes every run. Nothing is fetched from the network.[/dim]"
    )
    if not app.library.semantic_available():
        console.print(
            "[dim]Prompt search is unavailable: install the semantic extra with "
            "`uv sync --extra semantic` (or --extra semantic-gpu) and run "
            "`rekindle semantic embed`. Everything else works without it.[/dim]"
        )
    console.print("[dim]Ctrl-C to stop.[/dim]")

    if open_browser:
        # `webbrowser` shells out to the platform's default handler. It never
        # touches the network itself, and a failure to find a browser must not
        # take the server down with it - on a headless Linux box there may be
        # no handler at all.
        with contextlib.suppress(Exception):  # pragma: no cover - platform dependent
            webbrowser.open(app.url())

    if not serve_forever:
        # For a caller that wants to drive this server rather than block on it.
        # The loop still RUNS: returning a bound-but-unserved socket would hand
        # back something that looks alive - the OS accepts connections into the
        # backlog - and answers nothing. The caller owns `shutdown()`.
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/dim]")
    finally:
        server.server_close()
    return server


def render_cmd(
    spec_path: Path,
    *,
    data_dir: Path,
    out_dir: Path | None = None,
    frame_ms: int = DEFAULT_FRAME_MS,
    title_ms: int = TITLE_MS,
    preview_frames: int = DEFAULT_MAX_FRAMES,
    preview_width: int = 0,
    mp4_width: int = 0,
    music: Path | None = None,
    no_mp4: bool = False,
) -> None:
    """Re-render a memory.json. The command the UI hands back.

    Deliberately NOT `rekindle memory --recipe ... --key ...`: that re-runs
    selection, and selection is exactly what a hand-edited memory has
    overruled. The spec is the edit, so rendering the spec is the reproduction
    - and because the shots are resolved through `MemoryIndex`, a photo
    excluded since the spec was written is dropped and counted rather than
    quietly reappearing.
    """
    from rekindle.memory.cli import open_index

    if not spec_path.is_file():
        console.print(f"[red]No spec at[/red] {spec_path}.")
        raise typer.Exit(code=2)
    try:
        spec = MemorySpec.loads(spec_path.read_text(encoding="utf-8"))
    except (ValueError, KeyError, OSError) as exc:
        console.print(f"[red]{spec_path} is not a readable MemorySpec:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    store, index = open_index(data_dir)
    try:
        # Default to the folder the spec already lives in, so re-rendering
        # rewrites the memory rather than minting a new dated folder beside it
        # every time - which would make the reproduce command reproduce
        # somewhere else on each run.
        folder = out_dir if out_dir is not None else spec_path.parent
        options = RenderOptions(
            frame_ms=frame_ms,
            title_ms=title_ms,
            preview_frames=preview_frames,
            preview_width=preview_width or RenderOptions.preview_width,
            mp4_width=mp4_width or RenderOptions.mp4_width,
            music=music,
            no_mp4=no_mp4,
            # The spec on disk IS the input. Rewriting it would silently
            # normalise a file the user may have hand-edited, and an input a
            # command overwrites is not an input.
            write_spec=folder.resolve() != spec_path.parent.resolve(),
        )
        result = render_spec(spec, index, folder, options)
    finally:
        store.close()

    if not result.ok:
        console.print(
            f"[red]No frames[/red] for {spec.title} - every photo was unreadable or withheld."
        )
        _report_drops(result)
        raise typer.Exit(code=1)

    # `shots` is what the MEMORY holds; the preview holds the first
    # `--preview-frames` of them. Reporting the preview's count as the
    # memory's printed a 24-shot memory as "16 of 16 shots".
    preview = ""
    if result.preview_rendered < result.shots:
        preview = f", first {result.preview_rendered} in the preview"
    line = (
        f"[green]{spec.title}[/green] - {result.shots} shots{preview}, "
        f"{result.preview_size[0]}x{result.preview_size[1]} preview "
        f"(WebP {result.webp_bytes // 1024} KB, GIF {result.gif_bytes // 1024} KB)"
    )
    if result.mp4 is not None:
        line += f", MP4 {result.mp4_bytes // 1024} KB"
    elif result.mp4_skipped == FFMPEG_MISSING:
        line += " (GIF only)"
    console.print(line)
    console.print(f"  [dim]{result.folder}[/dim]")
    if result.music_used is not None:
        console.print(f"  [dim]music: {result.music_used.name}[/dim]")
    if result.mp4_error:
        console.print(f"  [yellow]![/yellow] ffmpeg failed: {result.mp4_error.splitlines()[-1:]}")
    elif result.mp4_skipped:
        console.print(f"  [yellow]![/yellow] {result.mp4_skipped}")
    if result.padded:
        console.print(
            f"  [dim]{result.padded} of {result.preview_rendered} preview frames were below the "
            f"{result.canvas[0]}x{result.canvas[1]} canvas and are shown at native size.[/dim]"
        )
    _report_drops(result)


def _report_drops(result) -> None:
    if result.withheld:
        console.print(
            f"  [yellow]![/yellow] {result.withheld} shots in this spec are no longer "
            "admitted by the guardrails and were left out. That is the exclusion "
            "working, not a fault."
        )
    for reason, count in sorted(result.dropped.items()):
        example = result.examples.get(reason, "")
        suffix = f" (e.g. {example})" if example else ""
        console.print(f"  [yellow]![/yellow] {count} shots dropped: {reason}{suffix}")


def default_spec(folder: Path) -> Path:
    """`memories/<something>/` -> that folder's memory.json."""
    return folder / SPEC_NAME if folder.is_dir() else folder
