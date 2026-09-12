"""`rekindle config` - see the numbers, see why, see what you changed.

Three verbs and no fourth. Editing happens in `rekindle calibrate`, where a
number is changed by looking at photographs, or in `rekindle.toml`, where a
number is changed by typing one. A `config set` would be a third way to do it
with none of the safeguards of either.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from rekindle import config

console = Console()


def register(app: typer.Typer) -> None:
    """Attach the `config` sub-app. Imports nothing heavy at module level."""
    app.add_typer(config_app, name="config")


config_app = typer.Typer(
    help="Show rekindle's thresholds, and which of them you have changed.",
    no_args_is_help=True,
)

DataDir = Annotated[Path, typer.Option("--data-dir", help="Where rekindle stores its index.")]


@config_app.command("list")
def list_cmd(data_dir: DataDir = Path("./data")) -> None:
    """Every tunable, its value, and whether you changed it."""
    try:
        settings = config.load(data_dir)
    except config.ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    changed = settings.changed_from_defaults()
    table = Table(show_header=True, header_style="bold")
    table.add_column("setting")
    table.add_column("value", justify="right")
    table.add_column("default", justify="right")
    table.add_column("unit")
    for key, setting in config.CATALOGUE.items():
        value = settings.get(key)
        table.add_row(
            f"[yellow]{key}[/yellow]" if key in changed else key,
            f"[yellow]{value}[/yellow]" if key in changed else str(value),
            str(setting.default),
            setting.unit,
        )
    console.print(table)
    path = Path(data_dir) / config.CONFIG_NAME
    if changed:
        console.print(f"\n{len(changed)} of {len(config.CATALOGUE)} changed, from {path}.")
    else:
        console.print(
            "\nNothing overridden - every value is rekindle's default. "
            "Run `rekindle calibrate` to derive them from your own photographs."
        )


@config_app.command("explain")
def explain_cmd(
    name: Annotated[str, typer.Argument(help="A setting name, e.g. composition.min_sharpness.")],
    data_dir: DataDir = Path("./data"),
) -> None:
    """What one threshold does, what moving it costs, and where it came from."""
    setting = config.CATALOGUE.get(name)
    if setting is None:
        matches = [k for k in config.CATALOGUE if name in k]
        console.print(f"[red]No setting called[/red] {name}.")
        if matches:
            console.print("Did you mean: " + ", ".join(matches))
        raise typer.Exit(code=2)

    try:
        current = config.load(data_dir).get(name)
    except config.ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    console.print(f"[bold]{setting.key}[/bold] = {current}  [dim]({setting.unit})[/dim]")
    if current != setting.default:
        console.print(f"[dim]rekindle's default is {setting.default}.[/dim]")
    console.print()
    console.print(f"[bold]What it does[/bold]\n  {setting.what}\n")
    console.print(f"[bold]Raising it[/bold]\n  {setting.raising}\n")
    console.print(f"[bold]Lowering it[/bold]\n  {setting.lowering}\n")
    console.print(f"[bold]How the default was measured[/bold]\n  {setting.measured}")
    if setting.loosening:
        direction = "Raising" if setting.loosening == config.LOOSEN_RAISE else "Lowering"
        console.print(
            f"\n[red bold]{direction} this weakens a safety gate.[/red bold] "
            f"`rekindle calibrate` will not do it without a separate confirmation."
        )


@config_app.command("diff")
def diff_cmd(data_dir: DataDir = Path("./data")) -> None:
    """Only what you changed, and by how much."""
    try:
        settings = config.load(data_dir)
    except config.ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    changed = settings.changed_from_defaults()
    if not changed:
        console.print("No overrides. Every value is rekindle's default.")
        return
    for key, value in changed.items():
        setting = config.CATALOGUE[key]
        arrow = "higher" if value > setting.default else "lower"
        console.print(f"{key}: {setting.default} -> [yellow]{value}[/yellow]  ({arrow})")
