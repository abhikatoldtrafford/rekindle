"""Command line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from rekindle import __version__
from rekindle.db import PhotoStore
from rekindle.doctor import diagnose, render
from rekindle.sources.folder import FolderSource

app = typer.Typer(help="Turn your photo library into memories.", no_args_is_help=True)
console = Console()

DataDir = Annotated[Path, typer.Option("--data-dir", help="Where rekindle stores its index.")]


def _check_root(root: Path) -> None:
    if not root.is_dir():
        console.print(f"[red]Not a directory:[/red] {root}")
        raise typer.Exit(code=2)


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
) -> None:
    """Report what metadata a library has. Writes nothing."""
    _check_root(root)
    _photos, report = FolderSource().scan(root)
    render(diagnose(report), console)


@app.command()
def index(
    root: Annotated[Path, typer.Argument(help="Folder of photos to index.")],
    data_dir: DataDir = Path("./data"),
) -> None:
    """Scan a folder and store its photos in the local index."""
    _check_root(root)
    photos, report = FolderSource().scan(root)
    with PhotoStore(data_dir / "rekindle.sqlite") as store:
        inserted, updated = store.upsert_many(photos)
        total = store.count()
    render(diagnose(report), console)
    console.print(
        f"\n[green]Indexed[/green] {inserted} new, {updated} updated. {total} photos in the index."
    )
