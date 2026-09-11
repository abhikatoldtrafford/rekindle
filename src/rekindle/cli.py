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
from rekindle.sources.folder import FolderSource

app = typer.Typer(help="Turn your photo library into memories.", no_args_is_help=True)
console = Console()

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
