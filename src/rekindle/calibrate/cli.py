"""`rekindle calibrate` - the guided sequence, in a terminal.

The terminal cannot show a photograph, so it prints the path and waits. That
sounds like a compromise and mostly is, but it has one real advantage the
browser does not: it works over ssh, on a machine with no display, and inside
a test that types its answers. The web UI in `rekindle ui` is the pleasant
one; this is the one that can be automated, and it is the one that was used to
calibrate rekindle's own reference library.

Every question is asked through `calibrate.session`, so the two front ends
cannot disagree about what a threshold means, about how many judgements it
rests on, or about whether the publishing gate may be widened.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from rekindle import config
from rekindle.calibrate import impact, labels, plan, preview, session, state
from rekindle.calibrate.plan import MODE_BLIND, MODE_DEFAULT, MODE_SLIDER, Step
from rekindle.config.writer import write as write_toml

console = Console()

DataDir = Annotated[Path, typer.Option("--data-dir", help="Where rekindle stores its index.")]

#: How many blind judgements one step asks for before offering to stop. Ten
#: bisections narrow a band by a factor of a thousand, which is far more
#: resolution than any of these thresholds has; the number is a comfort limit,
#: not an information one, and the user can always answer fewer.
DEFAULT_ROUNDS = 10


def register(app: typer.Typer) -> None:
    app.command("calibrate")(calibrate_cmd)


def calibrate_cmd(
    data_dir: DataDir = Path("./data"),
    memories: Annotated[
        Path, typer.Option("--memories", help="Where built memories live, for the exit question.")
    ] = Path("memories"),
    only: Annotated[
        str | None,
        typer.Option(
            "--only", help="Jump straight to one threshold, e.g. composition.min_sharpness."
        ),
    ] = None,
    redo: Annotated[
        bool, typer.Option("--redo", help="Forget previous answers and start again.")
    ] = False,
    rounds: Annotated[
        int, typer.Option("--rounds", help="Blind judgements per threshold.")
    ] = DEFAULT_ROUNDS,
    status: Annotated[
        bool, typer.Option("--status", help="Say what has been calibrated, and stop.")
    ] = False,
) -> None:
    """Derive rekindle's thresholds from your own photographs. Once."""
    from rekindle.memory.cli import open_index

    store, index = open_index(data_dir)
    try:
        photos = index.all()
        sess = session.Session(
            data_dir,
            photos,
            availability=session.detect(
                photos,
                semantic=_semantic(data_dir),
                faces=_faces(data_dir),
            ),
        )

        if status:
            _print_status(sess)
            return

        if redo:
            for step in sess.steps:
                sess.reset(step)
            sess.record.finished_at = None
            sess.save()

        _print_intro(sess)
        sess.start()

        steps = sess.steps
        if only:
            step = plan.BY_SETTING.get(only)
            if step is None:
                console.print(f"[red]No threshold called[/red] {only}.")
                raise typer.Exit(code=2)
            if step not in steps:
                console.print(
                    f"[yellow]{only} cannot be calibrated on this library.[/yellow] "
                    f"{sess.availability.reasons.get(step.needs, '')}"
                )
                raise typer.Exit(code=3)
            steps = [step]
            sess.reset(step)

        for step in steps:
            if step.setting in sess.done and not only:
                continue
            if not _run_step(sess, step, rounds=rounds):
                console.print(
                    "\n[dim]Stopped. Your answers are saved; run "
                    "`rekindle calibrate` again to pick up here.[/dim]"
                )
                return

        _finish(sess, index, memories)
    finally:
        store.close()


def _semantic(data_dir: Path):
    """The embedding store, or None. Never an error.

    A missing semantic extra is a supported configuration, not a degraded one:
    the sequence simply has one fewer step and says why. Importing it lazily
    is what keeps `rekindle calibrate --help` from pulling in onnxruntime.
    """
    try:
        from rekindle.semantic.diversity import open_support

        return open_support(data_dir)
    except Exception:  # noqa: BLE001 - any failure here means "not available"
        return None


def _faces(data_dir: Path):
    """The face detector, or None. Same reasoning as `_semantic`."""
    try:
        from rekindle.semantic.faces import load_detector
        from rekindle.semantic.setup import cache_dir_for

        return load_detector(cache_dir_for(data_dir))
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# the sequence


def _print_intro(sess: session.Session) -> None:
    offer = sess.offer()
    console.print()
    if offer.reason == state.OFFER_RESUME:
        console.print(f"[bold]{offer.headline}[/bold]")
    elif sess.record.finished:
        console.print(
            f"[bold]You calibrated on {sess.record.finished_at[:10]}, against "
            f"{sess.record.library_size:,} photographs; you now have {len(sess.photos):,}.[/bold]"
        )
    else:
        console.print(
            f"[bold]{len(sess.photos):,} photographs.[/bold] rekindle's thresholds were "
            f"derived from somebody else's library. Let's derive them from yours."
        )
    for missing in sorted(sess.availability.reasons.values()):
        console.print(f"[dim]{missing}[/dim]")
    console.print()


def _print_status(sess: session.Session) -> None:
    record = sess.record
    if record.finished:
        console.print(
            f"Calibrated {record.finished_at[:10]}, against {record.library_size:,} photographs. "
            f"You now have {len(sess.photos):,}."
        )
    elif record.in_progress:
        console.print(
            f"In progress since {record.started_at[:10]}: "
            f"{len(record.completed)} of {len(sess.steps)} done."
        )
    else:
        console.print("Never calibrated. Run `rekindle calibrate`.")

    table = Table(show_header=True, header_style="bold")
    table.add_column("threshold")
    table.add_column("state")
    table.add_column("value", justify="right")
    table.add_column("rests on")
    for step in sess.steps:
        setting = config.CATALOGUE[step.setting]
        if step.setting in record.completed:
            mark = "[green]done[/green]"
        elif step.setting in record.skipped:
            mark = "[yellow]skipped[/yellow]"
        else:
            mark = "[dim]not reached[/dim]"
        value = sess.chosen.get(step.setting, setting.default)
        found = sess.derived(step)
        table.add_row(
            step.setting,
            mark,
            f"{value:g}" + ("" if value == setting.default else " *"),
            found.confidence() if found.judgements else "-",
        )
    console.print(table)

    # Said out loud, because it is a file the user did not ask for and it
    # holds judgements about their own photographs. Counts only - naming a
    # photograph here would defeat the point of keeping the log private.
    kept = labels.summary(sess.data_dir)
    if kept:
        console.print(
            f"\n[dim]{sum(kept.values())} judgement"
            f"{'s' if sum(kept.values()) != 1 else ''} on record across "
            f"{len(kept)} threshold{'s' if len(kept) != 1 else ''} in "
            f"{labels.log_path(sess.data_dir)}.\n"
            f"That file is append-only, so a re-calibration adds to it rather than "
            f"replacing it. It is personal data: it stays on this machine and is "
            f"never committed. Nothing reads it - see "
            f"docs/decision-log-calibration-labels.md.[/dim]"
        )


def _run_step(sess: session.Session, step: Step, *, rounds: int) -> bool:
    """One step. Returns False if the user asked to stop."""
    at, total = sess.position(step)
    setting = config.CATALOGUE[step.setting]
    console.rule(f"[bold]{at} of {total} - {step.title}[/bold]")
    console.print(f"[dim]{step.area}. {setting.what}[/dim]")
    if step.note:
        console.print(f"[dim]{step.note}[/dim]")
    console.print()

    mode = _choose_mode(step)
    if mode is None:
        return False
    if mode == "skip":
        sess.skip(step)
        return True

    if mode == MODE_BLIND:
        return _blind(sess, step, rounds=rounds)
    if mode == MODE_SLIDER:
        return _slider(sess, step)
    return _show_default(sess, step)


#: One letter per mode, and the SAME letter on every step.
#:
#: Numbered options were tried first and were a mistake: not every step offers
#: every mode, so "2" meant "show me the default" on one screen and "let me
#: pick a number" on the next. Someone working through eleven steps learns the
#: position, not the meaning, and then answers the wrong question. A letter
#: tied to the mode cannot drift.
MODE_KEYS = {
    MODE_BLIND: ("p", "show me photographs and ask me about each one"),
    MODE_SLIDER: ("n", "let me pick a number and see what it does to my library"),
    MODE_DEFAULT: ("d", "show me what rekindle's default does here, and I will accept or nudge it"),
}


def _choose_mode(step: Step) -> str | None:
    if len(step.modes) == 1:
        return step.modes[0]
    for mode in step.modes:
        key, label = MODE_KEYS[mode]
        console.print(f"  [bold]{key}[/bold]. {label}")
    console.print("  [bold]s[/bold]. skip this one    [bold]q[/bold]. stop for now")
    first = MODE_KEYS[step.modes[0]][0]
    answer = typer.prompt("How would you like to do this", default=first).strip().lower()
    if answer == "q":
        return None
    if answer == "s":
        return "skip"
    for mode in step.modes:
        if answer == MODE_KEYS[mode][0]:
            return mode
    return step.modes[0]


def _blind(sess: session.Session, step: Step, *, rounds: int) -> bool:
    pool = sess.pool(step)
    if not pool:
        console.print("[yellow]Nothing in this library to ask about. Keeping the default.[/yellow]")
        sess.skip(step)
        return True

    console.print(
        f"[dim]{len(pool):,} candidates. Open each file and answer. "
        f"You will never be shown a number - that is the point.[/dim]\n"
    )
    asked = len(sess.judgements(step))
    while asked < rounds:
        example = sess.next_example(step)
        if example is None:
            break
        for path in example.paths:
            console.print(f"  [cyan]{path}[/cyan]")
        if example.caption:
            console.print(f"  [dim]{example.caption}[/dim]")
        reply = (
            typer.prompt(
                f"  {step.question} [{step.yes[0]}]{step.yes[1:]} / "
                f"[{step.no[0]}]{step.no[1:]} / (q)uit",
                default=step.no[0],
            )
            .strip()
            .lower()
        )
        if reply.startswith("q"):
            return False
        said_yes = reply.startswith(step.yes[0].lower())
        found = sess.answer(step, example, said_yes)
        asked += 1
        console.print(f"  [dim]{asked}/{rounds}[/dim]\n")
        # Clean separation and enough evidence. Offering to stop early is not
        # a shortcut: more questions inside a resolved interval add nothing,
        # and asking them anyway is how sixty judgements became the number
        # people dread.
        settled = found.usable and asked >= 6 and found.disagreements == 0
        if settled and not typer.confirm(
            "  Your answers separate cleanly. Keep going?", default=False
        ):
            break

    found = sess.derived(step)
    if not found.usable:
        console.print(f"[yellow]{found.refusal}[/yellow]")
        sess.accept_default(step)
        return True

    console.print(f"\n[bold]That gives {found.value:g}[/bold] - {found.confidence()}")
    return _commit(sess, step, found.value)


def _slider(sess: session.Session, step: Step) -> bool:
    setting = config.CATALOGUE[step.setting]
    current = sess.base.get(step.setting)
    console.print(f"[dim]Raising it: {setting.raising}[/dim]")
    console.print(f"[dim]Lowering it: {setting.lowering}[/dim]\n")

    if preview.countable(step.setting):
        _print_ladder(sess, step, current)

    while True:
        raw = typer.prompt(
            f"  A number in {setting.unit} (currently {current:g}), or (k)eep, or (q)uit",
            default="k",
        ).strip()
        if raw.lower().startswith("q"):
            return False
        if raw.lower().startswith("k"):
            sess.accept_default(step)
            return True
        try:
            value = float(raw)
        except ValueError:
            console.print("  [red]That is not a number.[/red]")
            continue
        problem = setting.clamp_error(value)
        if problem:
            console.print(f"  [red]{problem}[/red]")
            continue
        return _commit(sess, step, value)


def _print_ladder(sess: session.Session, step: Step, current: float) -> None:
    """What a handful of candidate values do to this library.

    This is the "live preview" the slider mode is for, in the form a terminal
    can show: the same numbers a dragged slider would recompute, all at once.
    """
    setting = config.CATALOGUE[step.setting]
    lo = setting.minimum if setting.minimum is not None else current / 4
    hi = setting.maximum if setting.maximum is not None else current * 4
    candidates = sorted(
        {_clamp(current * f, lo, hi) for f in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)}
    )
    deletes = step.setting in preview.DEDUP
    table = Table(show_header=True, header_style="bold")
    table.add_column("value", justify="right")
    table.add_column(
        "photographs collapsed away" if deletes else "photographs usable", justify="right"
    )
    table.add_column("against now", justify="right")
    for value in candidates:
        c = preview.consequence(sess.photos, step.setting, value, settings=sess.base)
        # For a dedup threshold the column that matters is the DELETION count,
        # not the survivor count. Both are the same arithmetic; only one of
        # them makes a user hesitate before turning 6 into 21.
        shown = c.collapsed_then if deletes else c.kept_then
        against = c.collapsed_now if deletes else c.kept_now
        delta = shown - against
        mark = "" if delta == 0 else (f"+{delta:,}" if delta > 0 else f"{delta:,}")
        label = f"{value:g}" + ("  (now)" if value == current else "")
        table.add_row(label, f"{shown:,}", mark)
    console.print(table)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _show_default(sess: session.Session, step: Step) -> bool:
    setting = config.CATALOGUE[step.setting]
    current = sess.base.get(step.setting)
    console.print(f"[bold]rekindle's value: {current:g}[/bold] ({setting.unit})")
    console.print(f"[dim]{setting.measured}[/dim]\n")
    if preview.countable(step.setting):
        c = preview.consequence(sess.photos, step.setting, current, settings=sess.base)
        if step.setting in preview.DEDUP:
            # For a dedup threshold the interesting number is what it DELETES,
            # not what survives. "Keeps 16,124" and "collapses 3,194 away" are
            # the same fact and only one of them is worth reading.
            console.print(
                f"On your library that collapses [bold]{c.collapsed_now:,}[/bold] of "
                f"{c.considered:,} photographs away as duplicates."
            )
        else:
            console.print(
                f"On your library that keeps [bold]{c.kept_now:,}[/bold] of "
                f"{c.considered:,} photographs."
            )
    reply = typer.prompt("  (a)ccept, (n)udge to a number, (q)uit", default="a").strip().lower()
    if reply.startswith("q"):
        return False
    if reply.startswith("n"):
        return _slider(sess, step)
    sess.accept_default(step)
    return True


def _commit(sess: session.Session, step: Step, value: float) -> bool:
    """Show the consequence, then stage it - and never the other way round."""
    if preview.countable(step.setting):
        c = preview.consequence(sess.photos, step.setting, value, settings=sess.base)
        console.print(f"  {c.sentence()}")
        for photo in c.newly_dropped[:3]:
            console.print(f"    [red]would now be dropped:[/red] {photo.paths[0]}")
        for photo in c.newly_kept[:3]:
            console.print(f"    [green]would now be kept:[/green] {photo.paths[0]}")

    refusal = sess.propose(step, value)
    if refusal is not None:
        console.print()
        console.print("[red bold]This widens the publishing gate.[/red bold]")
        console.print(refusal.meaning)
        console.print()
        console.print("[dim]Type this exactly to continue, or anything else to cancel:[/dim]")
        console.print(f"[bold]{refusal.phrase}[/bold]")
        typed = typer.prompt("  ", default="").strip()
        if not sess.confirm_loosening(step, typed):
            console.print("[green]Left alone. The gate keeps its current value.[/green]")
            sess.accept_default(step)
            return True
        refusal2 = sess.propose(step, value)
        if refusal2 is not None:  # pragma: no cover - confirm_loosening just cleared it
            console.print("[red]Still refused.[/red]")
            return True
        console.print("[yellow]Widened, on your explicit confirmation.[/yellow]")

    console.print(f"  [green]Set to {sess.chosen.get(step.setting, value):g}.[/green]\n")
    return True


# ---------------------------------------------------------------------------
# the way out


def _finish(sess: session.Session, index, memories_root: Path) -> None:
    overrides = sess.overrides()
    console.rule("[bold]Done[/bold]")

    if not overrides:
        console.print(
            "You kept every one of rekindle's defaults. That is a real result: "
            "they were derived from a different library and they fit yours."
        )
        sess.finish()
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("threshold")
    table.add_column("was", justify="right")
    table.add_column("now", justify="right")
    table.add_column("rests on")
    for key, value in overrides.items():
        found = sess.derived(plan.BY_SETTING[key]) if key in plan.BY_SETTING else None
        table.add_row(
            key,
            f"{config.CATALOGUE[key].default:g}",
            f"{value:g}",
            (found.confidence() if found and found.judgements else "you chose it directly"),
        )
    console.print(table)

    path = sess.data_dir / config.CONFIG_NAME
    if not typer.confirm(f"\nWrite these to {path}?", default=True):
        console.print("[dim]Nothing written. Your answers are saved.[/dim]")
        return
    write_toml(path, overrides)
    console.print(f"[green]Written.[/green] The previous file, if any, is at {path}.bak")
    config.activate_from(sess.data_dir)
    sess.finish()

    _exit_question(sess, index, memories_root)


def _exit_question(sess: session.Session, index, memories_root: Path) -> None:
    """Which memories would actually change - and the offer to rebuild those."""
    paths = impact.find_specs(memories_root)
    if not paths:
        console.print("\nNo memories built yet. Run `rekindle memory --auto` when you want some.")
        return

    console.print(
        f"\n[dim]Re-running selection for {len(paths)} memories - no frames, no encoding...[/dim]"
    )
    report = impact.analyse(index, memories_root, settings=sess.settings())
    console.print(f"\n[bold]{report.sentence()}[/bold]")

    for change in impact.rebuildable(report):
        bits = []
        if change.gained:
            bits.append(f"+{change.gained}")
        if change.lost:
            bits.append(f"-{change.lost}")
        console.print(
            f"  {change.memory_id}  {change.before} -> {change.after}  [dim]{' '.join(bits)}[/dim]"
        )
    for change in report.uncheckable:
        console.print(f"  [yellow]{change.memory_id}[/yellow]  [dim]{change.note}[/dim]")

    if not report.affected:
        return
    console.print(
        f"\n[dim]Rebuilding is the expensive half. "
        f"`rekindle render --recipe <r> --key <k>` rebuilds one; "
        f"the {len(report.affected)} above are the only ones worth redoing.[/dim]"
    )
