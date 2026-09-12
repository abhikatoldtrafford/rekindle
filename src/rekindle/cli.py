"""Command line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from rekindle import __version__
from rekindle.db import PhotoStore
from rekindle.doctor import diagnose, diagnose_index, render, render_enrich, render_index
from rekindle.enrich.takeout import EmptyIndexError, TakeoutEnricher
from rekindle.semantic.cli import register as register_semantic
from rekindle.sources.folder import FolderSource

# The default port only. `rekindle.web` imports nothing heavier than the
# standard library at module level, and `--help` has to print the number.
#
# NOTE: `render` is already bound in this module, to `doctor.render`. The
# memory renderer is therefore registered under an explicit command name and
# its function is called something else - a bare `def render` here would
# shadow the import and break `rekindle doctor` and `rekindle index`.
from rekindle.web import DEFAULT_PORT as WEB_PORT

app = typer.Typer(help="Turn your photo library into memories.", no_args_is_help=True)
console = Console()

# The semantic verbs live in rekindle.semantic.cli and attach themselves, so
# this file gains two lines rather than six command bodies. They import no
# heavy dependency at module level, so `rekindle --help` still works - and is
# still fast - on an install that has none of the optional extras.
register_semantic(app)

DataDir = Annotated[Path, typer.Option("--data-dir", help="Where rekindle stores its index.")]


def _check_root(root: Path) -> None:
    if not root.is_dir():
        console.print(f"[red]Not a directory:[/red] {root}")
        raise typer.Exit(code=2)


def _resolved(root: Path) -> Path:
    """Absolute and normalised, at the CLI boundary.

    `FolderSource` stores whatever Path it is handed, so `rekindle index
    Takeout` stores `Takeout/Photos from 2019/A.jpg` and `is_absolute()`
    is False. That breaks two things at once: the "Photo paths are absolute"
    claim the foreign-root warnings rest on, and the warnings themselves,
    which are raw string compares - `rekindle index Takeout` followed by
    `rekindle enrich Takeout` from a DIFFERENT directory compares two equal
    strings, fires no warning, and then matches nothing at all, which is the
    worst of both. Resolving here, before anything is stored and before
    anything is compared, is what makes the claim true. Every command routes
    its root through this, not just `enrich`.

    `Path.resolve()` is non-strict, so it is safe on the `doctor --from-index`
    path, where the root need not exist.
    """
    return root.resolve()


def _version_callback(value: bool) -> None:
    if value:
        console.print(__version__)
        raise typer.Exit()


@app.callback()
def main(
    _version: Annotated[
        bool, typer.Option("--version", callback=_version_callback, is_eager=True)
    ] = False,
) -> None:
    """Turn your photo library into memories."""


@app.command()
def doctor(
    root: Annotated[Path, typer.Argument(help="Folder of photos to inspect.")],
    from_index: Annotated[
        bool,
        typer.Option("--from-index", help="Report on the stored index instead of rescanning."),
    ] = False,
    data_dir: DataDir = Path("./data"),
) -> None:
    """Report what metadata a library has. Writes nothing."""
    root = _resolved(root)
    if from_index:
        # PhotoStore CREATES its database on open, so without this check
        # `doctor --from-index` on a machine that has never indexed would
        # silently report a healthy library of zero photos - and leave a
        # stray file behind, violating "doctor writes nothing".
        db_path = data_dir / "rekindle.sqlite"
        if not db_path.is_file():
            console.print(f"[red]No index at[/red] {db_path}. Run `rekindle index {root}` first.")
            raise typer.Exit(code=2)
        with PhotoStore(db_path) as store:
            diagnosis = diagnose_index(store)
        render_index(diagnosis, console)
        # `root` is otherwise unused on this path; warn rather than ignore it.
        if diagnosis.index_root and diagnosis.index_root != str(root):
            console.print(
                f"\n[yellow]![/yellow] This index was built from {diagnosis.index_root}, "
                f"not {root}. The report above describes the former."
            )
        return
    _check_root(root)
    _photos, report = FolderSource().scan(root)
    render(diagnose(report), console)


@app.command()
def index(
    root: Annotated[Path, typer.Argument(help="Folder of photos to index.")],
    data_dir: DataDir = Path("./data"),
) -> None:
    """Scan a folder and store its photos in the local index."""
    root = _resolved(root)
    _check_root(root)
    photos, report = FolderSource().scan(root)
    with PhotoStore(data_dir / "rekindle.sqlite") as store:
        inserted, updated = store.upsert_many(photos)
        store.set_meta("index_root", str(root))
        total = store.count()
    render(diagnose(report), console)
    console.print(
        f"\n[green]Indexed[/green] {inserted} new, {updated} updated. {total} photos in the index."
    )


@app.command()
def enrich(
    root: Annotated[Path, typer.Argument(help="Takeout folder whose sidecars to read.")],
    data_dir: DataDir = Path("./data"),
) -> None:
    """Read Google Takeout JSON sidecars into an index that already exists."""
    root = _resolved(root)
    _check_root(root)
    # Same reasoning as `doctor --from-index`: PhotoStore creates its file on
    # open, so without this check up front, running `enrich` before `index`
    # would correctly still exit 2 (`TakeoutEnricher.enrich` raises on an
    # empty store) but leave a stray, empty database file behind.
    db_path = data_dir / "rekindle.sqlite"
    if not db_path.is_file():
        console.print(f"[red]No index at[/red] {db_path}. Run `rekindle index {root}` first.")
        raise typer.Exit(code=2)
    with PhotoStore(db_path) as store:
        try:
            report = TakeoutEnricher().enrich(root, store)
        except EmptyIndexError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=2) from exc
        indexed_root = store.get_meta("index_root")
    render_enrich(report, console)
    if indexed_root and indexed_root != str(root):
        console.print(
            f"\n[yellow]![/yellow] The index was built from {indexed_root}, not {root}. "
            "Photo paths are absolute, so matches will be few."
        )


# --------------------------------------------------------------------------
# M2: memories
#
# The implementations live in rekindle.memory.cli so this module stays a thin
# argument-parsing layer. Imported lazily inside each command: typer builds
# every signature at import time, and pulling the whole memory engine (and
# Pillow's image plugins) in for `rekindle --version` is a measurable startup
# cost on a cold Windows filesystem.


# Repeated here rather than imported: importing `rekindle.fetch` at module
# scope would pull urllib into `rekindle --version`, and it would make the
# "nothing else imports the fetcher" guard depend on where a call happens to
# sit. `test_fetch.py` pins the two together.
_MUSIC_ITEM = "musopen-chopin"

music_app = typer.Typer(
    help="Music beds. `fetch` is the ONE command in rekindle that uses the network.",
    no_args_is_help=True,
)
app.add_typer(music_app, name="music")


@music_app.command("fetch")
def music_fetch(
    dest: Annotated[Path, typer.Option("--dest", help="Where to write the tracks.")] = Path(
        "music"
    ),
    count: Annotated[
        int, typer.Option("--count", help="How many tracks to fetch. 0 for all of them.")
    ] = 12,
    item: Annotated[str, typer.Option("--item", help="archive.org item identifier.")] = _MUSIC_ITEM,
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation prompt.")] = False,
) -> None:
    """Download CC0 music beds from archive.org. Never runs on its own.

    Nothing else in rekindle touches the network - not index, not enrich, not
    fingerprint, not memory, not watch. This exists so that wanting a
    soundtrack does not mean wiring a download into first run, which is how a
    third party's uptime becomes your program's correctness. freepd.com, the
    most obvious source for exactly this, shut down permanently in 2025.

    Filenames and checksums are resolved through the metadata API at fetch
    time; nothing is pinned in the source and no name is constructed by
    pattern.
    """
    from rekindle.fetch import FetchError, fetch_tracks, list_tracks

    try:
        source, tracks = list_tracks(item)
    except FetchError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    if not source.cc0:
        # Checked before a byte is downloaded, and refused rather than
        # warned about. An item can be relicensed after this code is
        # written, which is the whole reason the licence is read at fetch
        # time instead of asserted in a comment.
        console.print(
            f"[red]{item} does not declare CC0[/red] "
            f"(licence: {source.licence_url or 'none stated'}). Nothing downloaded."
        )
        raise typer.Exit(code=2)
    if not tracks:
        console.print(f"[red]{item} has no checksummed MP3s.[/red] Nothing downloaded.")
        raise typer.Exit(code=2)

    chosen = tracks if count <= 0 else tracks[:count]
    total_mb = sum(t.size for t in chosen) / 1_000_000
    console.print(f"[bold]{source.title}[/bold]")
    console.print(f"  source   {source.page}")
    console.print("  licence  CC0 1.0 - https://creativecommons.org/publicdomain/zero/1.0/")
    console.print(f"  fetching {len(chosen)} of {len(tracks)} tracks, about {total_mb:.0f} MB")
    console.print(f"  into     {dest}")

    if not yes and not typer.confirm("Download now?"):
        console.print("Nothing downloaded.")
        raise typer.Exit(code=0)

    def progress(index: int, total: int, track) -> None:
        console.print(f"  [dim]{index}/{total}[/dim] {track.filename}")

    report = fetch_tracks(chosen, dest, item=item, on_progress=progress)
    console.print(
        f"[green]{report.downloaded} downloaded[/green], {report.skipped} already present, "
        f"{report.failed} failed ({report.bytes_written / 1_000_000:.0f} MB written)"
    )
    for line in report.errors:
        console.print(f"  [yellow]![/yellow] {line}")
    if report.failed:
        raise typer.Exit(code=1)


@app.command()
def fingerprint(data_dir: DataDir = Path("./data")) -> None:
    """Compute perceptual fingerprints. One-time pass; needed for dedup."""
    from rekindle.memory.cli import fingerprint_cmd

    fingerprint_cmd(_resolved(data_dir))


@app.command()
def memories(
    recipe: Annotated[
        str | None, typer.Option("--recipe", help="Only show this recipe's memories.")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Rows per recipe.")] = 10,
    data_dir: DataDir = Path("./data"),
) -> None:
    """List every memory this library could produce. Builds nothing."""
    from rekindle.memory.cli import memories_cmd

    memories_cmd(_resolved(data_dir), recipe, limit)


@app.command()
def memory(
    text: Annotated[
        str | None,
        typer.Argument(
            help=(
                "Build a memory from your own words, e.g. "
                '`rekindle memory "durga puja over the years"`. Needs the '
                "semantic extra and an embedded library."
            ),
        ),
    ] = None,
    recipe: Annotated[str | None, typer.Option("--recipe", help="Recipe name.")] = None,
    key: Annotated[str | None, typer.Option("--key", help="Which memory, from `memories`.")] = None,
    auto: Annotated[
        bool, typer.Option("--auto", help="Today's anniversary, if there is one.")
    ] = False,
    out: Annotated[Path, typer.Option("--out", help="Where to write memories.")] = Path("memories"),
    public_safe: Annotated[
        bool,
        typer.Option(
            "--public-safe",
            help="Only use photos whose face tags are a subset of the allow-list.",
        ),
    ] = False,
    max_shots: Annotated[int, typer.Option("--max-shots")] = 24,
    gif_frames: Annotated[
        int, typer.Option("--preview-frames", help="Frames in the GIF/WebP preview.")
    ] = 16,
    preview_width: Annotated[
        int,
        typer.Option(
            "--preview-width",
            help="Width of the GIF/WebP preview. 0 uses the default (1280).",
        ),
    ] = 0,
    mp4_width: Annotated[
        int,
        typer.Option(
            "--mp4-width",
            help=(
                "Maximum MP4 width. 0 uses the default (2560). Raise it to go "
                "up to the native resolution the photos support."
            ),
        ),
    ] = 0,
    music: Annotated[Path | None, typer.Option("--music", help="Audio bed for the MP4.")] = None,
    no_mp4: Annotated[
        bool, typer.Option("--no-mp4", help="Skip the MP4 even if ffmpeg is here.")
    ] = False,
    limit: Annotated[int, typer.Option("--limit", help="Maximum memories to build.")] = 1,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help=(
                "Rebuild even memories inside the resurfacing cooldown. "
                "Dismissals are still honoured."
            ),
        ),
    ] = False,
    captions: Annotated[
        str,
        typer.Option(
            "--captions",
            help=(
                "'deterministic' (default), 'clip' or 'gpt'. 'clip' grounds "
                "each caption in what an image model recognised, from a closed "
                "vocabulary. 'gpt' has a language model phrase those terms and "
                "the fact sheet - it never sees your photos - and needs "
                "OPENAI_API_KEY."
            ),
        ),
    ] = "deterministic",
    seed_k: Annotated[
        int,
        typer.Option(
            "--seed-k",
            help="Prompt only: how many consensus-ranked photos seed the days. 0 uses 100.",
        ),
    ] = 0,
    min_seeds: Annotated[
        int,
        typer.Option("--min-seeds", help="Prompt only: seed photos a day needs. 0 uses 2."),
    ] = 0,
    tag_k: Annotated[
        int,
        typer.Option(
            "--tag-k", help="Prompt only: how deep each visual tag is searched. 0 uses 100."
        ),
    ] = 0,
    judge: Annotated[
        bool,
        typer.Option(
            "--judge/--no-judge",
            help=(
                "Prompt only: ask the language model whether the query is "
                "coherent before building. Needs OPENAI_API_KEY; without one "
                "it never runs."
            ),
        ),
    ] = True,
    data_dir: DataDir = Path("./data"),
) -> None:
    """Build memories into a folder. Renders a GIF, and an MP4 if ffmpeg is on PATH.

    With a TEXT argument this builds ONE memory from your own words. That kind
    of memory is a PREVIEW, not an offer: it never appears in
    `rekindle memories` and `--auto` will never show you one, because nothing
    can check that the photos match the words. Looking at the result is a
    required step rather than one a threshold pretends to replace.
    """
    from rekindle.memory.captioning import MODES
    from rekindle.memory.cli import memory_cmd, prompt_cmd

    if captions not in MODES:
        console.print(f"[red]--captions must be one of {', '.join(MODES)}, not {captions!r}[/red]")
        raise typer.Exit(code=2)

    if text is not None:
        if recipe or key or auto:
            console.print(
                "[red]A prompt cannot be combined with --recipe, --key or --auto.[/red] "
                "A prompt memory is built only when you ask for it by name."
            )
            raise typer.Exit(code=2)
        prompt_cmd(
            _resolved(data_dir),
            text,
            out,
            public_safe=public_safe,
            max_shots=max_shots,
            gif_frames=gif_frames,
            music=music,
            no_mp4=no_mp4,
            seed_k=seed_k,
            min_seeds=min_seeds,
            tag_k=tag_k,
            captions=captions,
            judge=judge,
            preview_width=preview_width,
            mp4_width=mp4_width,
        )
        return

    memory_cmd(
        _resolved(data_dir),
        recipe,
        key,
        auto,
        out,
        public_safe,
        max_shots,
        gif_frames,
        music,
        no_mp4,
        limit,
        force,
        captions,
        preview_width,
        mp4_width,
    )


@app.command()
def scenery(
    concept: Annotated[
        list[str] | None,
        typer.Argument(
            help=(
                "Which scenery concepts to build, e.g. `rekindle scenery sea "
                "mountains`. Run with --list to see them all."
            ),
        ),
    ] = None,
    list_: Annotated[
        bool,
        typer.Option("--list", help="Show the corpus and exit. Needs no index."),
    ] = False,
    all_: Annotated[bool, typer.Option("--all", help="Build every concept in the corpus.")] = False,
    out: Annotated[Path, typer.Option("--out", help="Where to write memories.")] = Path("memories"),
    public_safe: Annotated[
        bool,
        typer.Option(
            "--public-safe",
            help="Only use photos whose face tags are a subset of the allow-list.",
        ),
    ] = False,
    max_shots: Annotated[int, typer.Option("--max-shots")] = 24,
    gif_frames: Annotated[
        int, typer.Option("--preview-frames", help="Frames in the GIF/WebP preview.")
    ] = 16,
    preview_width: Annotated[
        int, typer.Option("--preview-width", help="0 uses the default (1280).")
    ] = 0,
    mp4_width: Annotated[int, typer.Option("--mp4-width", help="0 uses the default (2560).")] = 0,
    music: Annotated[Path | None, typer.Option("--music", help="Audio bed for the MP4.")] = None,
    no_mp4: Annotated[bool, typer.Option("--no-mp4", help="Skip the MP4.")] = False,
    min_votes: Annotated[
        int,
        typer.Option(
            "--min-votes",
            help="How many descriptions must agree on a photo. 0 uses 2.",
        ),
    ] = 0,
    tag_k: Annotated[
        int, typer.Option("--tag-k", help="How deep each description is searched. 0 uses 100.")
    ] = 0,
    captions: Annotated[
        str,
        typer.Option(
            "--captions",
            help=("'deterministic' (default), 'clip' or 'gpt'. See `rekindle memory --help`."),
        ),
    ] = "deterministic",
    style: Annotated[
        str,
        typer.Option("--style", help="Motion style: 'film' (default) or 'cuts'."),
    ] = "film",
    data_dir: DataDir = Path("./data"),
) -> None:
    """Build a memory whose subject is a SCENE - the sea, the hills, the food.

    A third of this library has no face tag, no album anyone named and no
    GPS, so no other recipe can be ABOUT it. These memories can. They are
    retrieved by what a photograph looks like, from the checked-in concept
    list in `corpus/scenery.toml`, and like a prompt memory they are built
    only when you ask for one by name.
    """
    from rekindle.memory.captioning import MODES
    from rekindle.memory.cli import scenery_cmd, scenery_list_cmd
    from rekindle.memory.render.timeline import STYLES

    if captions not in MODES:
        console.print(f"[red]--captions must be one of {', '.join(MODES)}, not {captions!r}[/red]")
        raise typer.Exit(code=2)
    if style not in STYLES:
        console.print(f"[red]--style must be one of {', '.join(STYLES)}, not {style!r}[/red]")
        raise typer.Exit(code=2)
    if list_:
        scenery_list_cmd()
        return
    names = list(concept or [])
    if not names and not all_:
        console.print(
            "[red]Name a concept, or pass --all.[/red] "
            "`rekindle scenery --list` shows what is available."
        )
        raise typer.Exit(code=2)
    scenery_cmd(
        _resolved(data_dir),
        names,
        out,
        all_concepts=all_,
        public_safe=public_safe,
        max_shots=max_shots,
        gif_frames=gif_frames,
        music=music,
        no_mp4=no_mp4,
        min_votes=min_votes,
        tag_k=tag_k,
        captions=captions,
        preview_width=preview_width,
        mp4_width=mp4_width,
        style=style,
    )


@app.command()
def dismiss(
    recipe: Annotated[str, typer.Argument(help="Recipe name.")],
    key: Annotated[str, typer.Argument(help="Memory key.")],
    data_dir: DataDir = Path("./data"),
) -> None:
    """Never show this memory again. Permanent, and undoable with `undismiss`."""
    from rekindle.memory.cli import dismiss_cmd

    dismiss_cmd(_resolved(data_dir), recipe, key)


@app.command()
def undismiss(
    recipe: Annotated[str, typer.Argument(help="Recipe name.")],
    key: Annotated[str, typer.Argument(help="Memory key.")],
    data_dir: DataDir = Path("./data"),
) -> None:
    """Undo a dismissal."""
    from rekindle.memory.cli import undismiss_cmd

    undismiss_cmd(_resolved(data_dir), recipe, key)


@app.command()
def exclude(
    person: Annotated[str | None, typer.Option("--person")] = None,
    album: Annotated[str | None, typer.Option("--album")] = None,
    from_: Annotated[str | None, typer.Option("--from", help="YYYY-MM-DD")] = None,
    to: Annotated[str | None, typer.Option("--to", help="YYYY-MM-DD")] = None,
    data_dir: DataDir = Path("./data"),
) -> None:
    """Exclude a person, an album or a date range from every memory."""
    from rekindle.memory.cli import exclude_cmd

    exclude_cmd(_resolved(data_dir), person, album, from_, to)


@app.command()
def dismissals(data_dir: DataDir = Path("./data")) -> None:
    """Show everything that has been dismissed or excluded."""
    from rekindle.memory.cli import dismissals_cmd

    dismissals_cmd(_resolved(data_dir))


@app.command()
def watch(
    root: Annotated[Path, typer.Argument(help="Folder to watch.")],
    interval: Annotated[float, typer.Option("--interval", help="Seconds between polls.")] = 300.0,
    once: Annotated[bool, typer.Option("--once", help="Run a single cycle and exit.")] = False,
    data_dir: DataDir = Path("./data"),
) -> None:
    """Watch a library in the FOREGROUND. Re-indexes on change and prints
    today's anniversary. Never renders anything on its own."""
    from rekindle.memory.cli import watch_cmd

    root = _resolved(root)
    _check_root(root)
    watch_cmd(root, _resolved(data_dir), interval, once)


# --------------------------------------------------------------------------
# M4: the interactive memory builder
#
# Two verbs, both additive. `ui` is a lens onto everything above; `render`
# turns a hand-edited MemorySpec back into files, which is what makes an
# afternoon in the browser reproducible from a shell.
#
# Imported lazily for the same reason every other command here is: `rekindle
# --version` must not pay for Pillow, the memory engine or an HTTP server.


@app.command()
def ui(
    port: Annotated[
        int,
        typer.Option("--port", help="Port on 127.0.0.1. 0 asks the OS for a free one."),
    ] = WEB_PORT,
    out: Annotated[Path, typer.Option("--out", help="Where to write memories.")] = Path("memories"),
    music_dir: Annotated[
        Path, typer.Option("--music-dir", help="Folder of audio beds to choose from.")
    ] = Path("music"),
    public_safe: Annotated[
        bool,
        typer.Option(
            "--public-safe",
            help="Only offer photos whose face tags are a subset of the allow-list.",
        ),
    ] = False,
    max_shots: Annotated[int, typer.Option("--max-shots")] = 24,
    open_browser: Annotated[
        bool, typer.Option("--open/--no-open", help="Open the page in your browser.")
    ] = True,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Log every request. Off by default: a request log is a record of "
            "which of your photographs you looked at.",
        ),
    ] = False,
    data_dir: DataDir = Path("./data"),
) -> None:
    """Edit memories in a local browser page. Binds 127.0.0.1 and nothing else.

    The page is a LENS onto the same engine these other commands use: it
    selects nothing of its own, it cannot show a photo the guardrails refuse,
    and every edit ends as a `rekindle render` command you can re-run.
    """
    from rekindle.web.cli import ui_cmd

    ui_cmd(
        _resolved(data_dir),
        out_dir=out,
        music_dir=music_dir,
        port=port,
        public_safe=public_safe,
        max_shots=max_shots,
        open_browser=open_browser,
        verbose=verbose,
    )


@app.command("render")
def render_memory(
    spec: Annotated[Path, typer.Argument(help="A memory.json, or the folder holding one.")],
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Where to write. Defaults to the spec's own folder."),
    ] = None,
    frame_ms: Annotated[
        int, typer.Option("--frame-ms", help="How long each photo holds, in milliseconds.")
    ] = 1400,
    title_ms: Annotated[
        int, typer.Option("--title-ms", help="How long the title card holds.")
    ] = 2200,
    preview_frames: Annotated[
        int, typer.Option("--preview-frames", help="Frames in the GIF/WebP preview.")
    ] = 16,
    preview_width: Annotated[
        int, typer.Option("--preview-width", help="0 uses the default (1280).")
    ] = 0,
    mp4_width: Annotated[int, typer.Option("--mp4-width", help="0 uses the default (2560).")] = 0,
    music: Annotated[Path | None, typer.Option("--music", help="Audio bed for the MP4.")] = None,
    no_mp4: Annotated[bool, typer.Option("--no-mp4", help="Skip the MP4.")] = False,
    data_dir: DataDir = Path("./data"),
) -> None:
    """Render a MemorySpec that already exists. No selection is re-run.

    This is how an edited memory becomes repeatable: `rekindle ui` writes the
    spec, prints this command, and running it rebuilds the same files. Shots
    are resolved through the same guardrailed index as everything else, so a
    photo excluded since the spec was written is left out and counted.
    """
    from rekindle.web.cli import default_spec, render_cmd

    render_cmd(
        default_spec(spec),
        data_dir=_resolved(data_dir),
        out_dir=out,
        frame_ms=frame_ms,
        title_ms=title_ms,
        preview_frames=preview_frames,
        preview_width=preview_width,
        mp4_width=mp4_width,
        music=music,
        no_mp4=no_mp4,
    )
