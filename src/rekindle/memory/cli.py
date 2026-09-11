"""The memory commands, kept out of cli.py so that module stays readable.

Every command that needs an index checks the database file exists FIRST.
`PhotoStore` creates its file on open, so without that check `rekindle memory`
on a machine that has never indexed would report a healthy library of zero
photos and leave a stray file behind - the same trap `doctor --from-index`
and `enrich` already guard against.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, date, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from rekindle.db import PhotoStore
from rekindle.memory import engine
from rekindle.memory.composition import FIT_PAD, describe_drops
from rekindle.memory.history import (
    KIND_ALBUM,
    KIND_DATES,
    KIND_MEMORY,
    KIND_PERSON,
    MemoryState,
    memory_id,
)
from rekindle.memory.index import MemoryIndex
from rekindle.memory.policy import CONFIG_NAME, PolicyError, load_policy
from rekindle.memory.recipes import registered
from rekindle.memory.render.frames import build_frames
from rekindle.memory.render.gif import DEFAULT_WIDTH as PREVIEW_WIDTH
from rekindle.memory.render.gif import preview_canvas, write_gif, write_webp
from rekindle.memory.render.mp4 import DEFAULT_WIDTH as MP4_WIDTH
from rekindle.memory.render.mp4 import mp4_canvas, write_mp4
from rekindle.memory.render.music import NO_MUSIC_HINT, resolve_music
from rekindle.memory.spec import MemorySpec, safe_slug

console = Console()

DB_NAME = "rekindle.sqlite"
PUBLIC_SAFE_MARKER = "PUBLIC-SAFE"
DEFAULT_OUT = Path("memories")


def open_index(data_dir: Path, *, public_safe: bool = False) -> tuple[PhotoStore, MemoryIndex]:
    """Open the store and build a guardrailed index, or exit 2.

    Dismissals are merged into the file policy here, at the single place every
    command goes through, so a dismissed person is enforced identically
    whether it came from `rekindle dismiss` or from exclusions.toml.
    """
    db_path = data_dir / DB_NAME
    if not db_path.is_file():
        console.print(f"[red]No index at[/red] {db_path}. Run `rekindle index <folder>` first.")
        raise typer.Exit(code=2)
    try:
        policy = load_policy(data_dir / CONFIG_NAME)
    except PolicyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    store = PhotoStore(db_path)
    try:
        policy = MemoryState(store).apply_to(policy).with_public_safe(public_safe)
        index = MemoryIndex.open(store, policy)
    except (ValueError, PolicyError) as exc:
        store.close()
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    return store, index


def _warn_unfingerprinted(index: MemoryIndex) -> None:
    """One clear warning, not a silent no-op.

    Without fingerprints, dedup cannot run and the quality gates have nothing
    to read. The memories are still built - a warning is the right response,
    not an empty result.
    """
    missing = sum(1 for p in index.all() if p.meta.phash is None and not p.meta.phash_error)
    if missing:
        console.print(
            f"[yellow]![/yellow] {missing} photos have no perceptual fingerprint, so "
            "burst dedup and the quality filters are inactive for them. "
            "Run `rekindle fingerprint` once to enable both."
        )


def fingerprint_cmd(data_dir: Path) -> None:
    from rekindle.memory.fingerprint import run_fingerprints

    db_path = data_dir / DB_NAME
    if not db_path.is_file():
        console.print(f"[red]No index at[/red] {db_path}. Run `rekindle index <folder>` first.")
        raise typer.Exit(code=2)

    with PhotoStore(db_path) as store:
        total = sum(1 for _ in store.iter_unfingerprinted())
        if not total:
            console.print("[green]Nothing to do[/green] - every photo already has a fingerprint.")
            return
        console.print(f"Fingerprinting {total} photos. This is a one-time pass.")
        with console.status("decoding...") as status:

            def progress(done: int, of: int) -> None:
                status.update(f"decoding... {done}/{of}")

            report = run_fingerprints(store, on_progress=progress)

    console.print(
        f"[green]Fingerprinted[/green] {report.hashed} photos. "
        f"{report.skipped_video} videos skipped (never deduped by design), "
        f"{report.failed} could not be decoded."
    )
    for reason, count in sorted(report.errors.items()):
        console.print(f"  [yellow]![/yellow] {count} {reason}")


def memories_cmd(data_dir: Path, recipe_filter: str | None, limit: int) -> None:
    store, index = open_index(data_dir)
    try:
        if index.count() == 0:
            console.print("[yellow]No photos available.[/yellow]")
            _render_exclusions(index)
            return
        state = MemoryState(store)
        dismissed = state.dismissed_memory_ids()
        cooling = state.cooling()

        table = Table(title="Available memories", show_header=True)
        table.add_column("Recipe")
        table.add_column("Key")
        table.add_column("Title")
        table.add_column("Detail")
        table.add_column("", justify="right")

        shown = 0
        for recipe in registered():
            if recipe_filter and recipe.name != recipe_filter:
                continue
            offers = recipe.offers(index)
            for offer in offers[:limit]:
                flag = ""
                if offer.memory_id in dismissed:
                    flag = "[dim]dismissed[/dim]"
                elif offer.memory_id in cooling:
                    flag = "[dim]recent[/dim]"
                table.add_row(recipe.name, offer.key, offer.title, offer.subtitle, flag)
                shown += 1
            if len(offers) > limit:
                table.add_row(recipe.name, f"[dim]... {len(offers) - limit} more[/dim]", "", "", "")
        console.print(table)
        console.print(f"\n{shown} shown. Render one with: rekindle memory --recipe R --key K")
        _render_exclusions(index)
    finally:
        store.close()


def _render_exclusions(index: MemoryIndex) -> None:
    report = index.report
    if not report.by_reason:
        return
    parts = ", ".join(f"{n} {reason}" for reason, n in sorted(report.by_reason.items()))
    console.print(f"[dim]{report.excluded} photos withheld by the guardrails: {parts}[/dim]")


def memory_cmd(
    data_dir: Path,
    recipe: str | None,
    key: str | None,
    auto: bool,
    out_dir: Path,
    public_safe: bool,
    max_shots: int,
    gif_frames: int,
    music: Path | None,
    no_mp4: bool,
    limit: int,
    captions: str = "deterministic",
    preview_width: int = 0,
    mp4_width: int = 0,
    today: datetime | None = None,
) -> None:
    store, index = open_index(data_dir, public_safe=public_safe)
    try:
        if index.count() == 0:
            console.print("[yellow]No photos available after the guardrails.[/yellow]")
            _render_exclusions(index)
            return
        _warn_unfingerprinted(index)
        state = MemoryState(store)

        if auto:
            moment = today or datetime.now(UTC)
            offers = engine.offers_for_today(index, moment)
            if not offers:
                console.print("[yellow]No anniversary for today.[/yellow] Try `rekindle memories`.")
                return
        elif recipe and key:
            found = [o for o in engine.all_offers(index) if o.recipe == recipe and o.key == key]
            if not found:
                console.print(
                    f"[red]No memory[/red] for recipe={recipe!r} key={key!r}. "
                    "Run `rekindle memories` to see what is available."
                )
                raise typer.Exit(code=2)
            offers = found
        else:
            offers = engine.all_offers(index)

        specs, report = engine.build_all(
            index,
            offers,
            max_shots=max_shots,
            dismissed=state.dismissed_memory_ids(),
            cooling=state.cooling(),
            limit=limit,
        )
        if not specs:
            console.print("[yellow]Nothing to build.[/yellow]")
            _render_build_report(report)
            return

        specs = _maybe_caption(specs, captions)
        for spec in specs:
            _render_one(spec, index, out_dir, gif_frames, music, no_mp4, preview_width, mp4_width)
            state.record_surfaced(memory_id(spec.recipe, spec.key), title=spec.title)
        _render_build_report(report)
    finally:
        store.close()


def _maybe_caption(specs: list[MemorySpec], mode: str) -> list[MemorySpec]:
    """The ONE place the optional LLM layer is reached from.

    Off unless `--captions gpt`. A missing key, an unreachable service or a
    rejected caption all fall back to the deterministic captions with a
    warning and exit 0 - `rekindle` must keep working for everyone who never
    sets an API key, which is the default configuration.
    """
    if mode != "gpt":
        return specs

    from rekindle.memory.llm import LLMUnavailable, apply_captions, captioner_from_env

    try:
        captioner = captioner_from_env()
    except LLMUnavailable as exc:
        console.print(f"[yellow]![/yellow] {exc}")
        return specs

    out: list[MemorySpec] = []
    accepted = requested = 0
    rejected: dict[str, int] = {}
    for spec in specs:
        rewritten, report = apply_captions(spec, captioner)
        out.append(rewritten)
        accepted += report.accepted
        requested += report.requested
        for reason, count in report.rejected.items():
            rejected[reason] = rejected.get(reason, 0) + count
        if report.error:
            console.print(f"[yellow]![/yellow] {report.error}")
            # One failure means the service is unreachable; stop asking.
            out.extend(specs[len(out) :])
            break
    detail = ", ".join(f"{n} {r}" for r, n in sorted(rejected.items()))
    console.print(
        f"[dim]GPT captions: {accepted}/{requested} accepted"
        + (f"; rejected {detail}" if detail else "")
        + "[/dim]"
    )
    return out


def _render_build_report(report: engine.BuildReport) -> None:
    if report.skipped:
        parts = ", ".join(f"{n} {r.replace('_', ' ')}" for r, n in sorted(report.skipped.items()))
        console.print(f"[dim]{report.total_skipped} candidate memories skipped: {parts}[/dim]")
    if report.deduped:
        console.print(f"[dim]{report.deduped} near-duplicate frames collapsed.[/dim]")
    drops = describe_drops(report.composition)
    if drops:
        console.print(f"[dim]Composition guardrails dropped: {'; '.join(drops)}[/dim]")
    for stratum in report.strata:
        # A period that could not be represented is a CORRECT outcome when the
        # quality gates emptied it - but a silent one looks exactly like the
        # clustering bug that stratification fixed, so it is always named.
        if stratum.keys_lost_to_gates:
            shown = ", ".join(stratum.keys_lost_to_gates[:8])
            more = (
                f" (+{len(stratum.keys_lost_to_gates) - 8} more)"
                if len(stratum.keys_lost_to_gates) > 8
                else ""
            )
            console.print(
                f"  [yellow]![/yellow] {stratum.memory_id}: no usable photos from "
                f"{shown}{more} - every candidate there failed a guardrail."
            )
        if stratum.unslotted:
            console.print(
                f"  [dim]{stratum.memory_id}: {stratum.unslotted} further "
                f"{stratum.dimension}s had photos but no room in "
                f"{stratum.used} slots.[/dim]"
            )


def _render_one(
    spec: MemorySpec,
    index: MemoryIndex,
    out_dir: Path,
    gif_frames: int,
    music: Path | None,
    no_mp4: bool,
    preview_width: int = 0,
    mp4_width: int = 0,
) -> None:
    from rekindle.memory.composition import canvas_for

    photos = [index.get(s.file_hash) for s in spec.shots]
    # The MEDIAN size of the chosen photos, not the minimum - see
    # composition.canvas_for for why the minimum was catastrophic.
    canvas = canvas_for([p for p in photos if p is not None]) or (1280, 960)

    folder = out_dir / f"{date.today().isoformat()}-{spec.recipe}-{safe_slug(spec.key)}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "memory.json").write_text(spec.dumps(), encoding="utf-8")

    preview_size = preview_canvas(canvas, preview_width or PREVIEW_WIDTH)
    frames, frame_report = build_frames(
        spec,
        preview_size,
        resolve=index.get,
        locate=index.resolve_path,
        limit=gif_frames,
    )
    if frame_report.rendered == 0:
        console.print(f"[red]No frames[/red] for {spec.title} - every photo file was unreadable.")
        _render_frame_drops(frame_report)
        return

    # WebP first: it is the preview worth looking at. GIF is written too
    # because it embeds everywhere, but it quantises to 256 colours and bands
    # photographs regardless of resolution.
    webp_bytes = write_webp(frames, folder / "memory.webp")
    gif_bytes = write_gif(frames, folder / "memory.gif")
    line = (
        f"[green]{spec.title}[/green] - {len(spec.shots)} shots, "
        f"{preview_size[0]}x{preview_size[1]} preview "
        f"(WebP {webp_bytes // 1024} KB, GIF {gif_bytes // 1024} KB)"
    )

    if not no_mp4:
        video_size = mp4_canvas(canvas, mp4_width or MP4_WIDTH)
        mp4_frames, mp4_report = build_frames(
            spec, video_size, resolve=index.get, locate=index.resolve_path
        )
        bed = resolve_music(music, memory_id=memory_id(spec.recipe, spec.key))
        result = write_mp4(mp4_frames, folder / "memory.mp4", music=bed)
        if result.ok:
            line += f", MP4 {video_size[0]}x{video_size[1]} {result.size // 1024} KB"
        elif result.skipped:
            line += " (GIF only)"
        else:
            line += " (MP4 failed)"
        frame_report = mp4_report if mp4_report.total_dropped else frame_report
        if result.error:
            console.print(f"[yellow]![/yellow] ffmpeg failed: {result.error.splitlines()[-1:]}")
        elif result.skipped:
            console.print(f"[yellow]![/yellow] {result.skipped}")
        if result.ok and bed is None:
            console.print(f"  [dim]{NO_MUSIC_HINT}[/dim]")

    if spec.public_safe:
        (folder / PUBLIC_SAFE_MARKER).write_text(
            "Every photo in this memory has face tags that are a subset of the "
            "configured public-safe allow-list.\n"
            "Untagged photos are NEVER treated as public-safe: face tags cover "
            "only part of a Google Takeout library, so an untagged photo may "
            "still contain other people.\n",
            encoding="utf-8",
        )
        line += " [cyan]PUBLIC-SAFE[/cyan]"

    console.print(line)
    console.print(f"  [dim]{folder}[/dim]")
    padded = frame_report.placement.get(FIT_PAD, 0)
    if padded:
        # Not a fault: it says the photos of that period are genuinely smaller
        # than the rest of the memory. Worth surfacing because a memory that
        # is mostly padded is telling the user something real.
        console.print(
            f"  [dim]{padded} of {frame_report.rendered} shots were below the "
            f"{canvas[0]}x{canvas[1]} canvas and are shown at native size.[/dim]"
        )
    _render_frame_drops(frame_report)


def _render_frame_drops(report) -> None:
    for reason, count in sorted(report.dropped.items()):
        example = report.names.get(reason, "")
        suffix = f" (e.g. {example})" if example else ""
        console.print(f"  [yellow]![/yellow] {count} shots dropped: {reason}{suffix}")


def dismiss_cmd(data_dir: Path, recipe: str, key: str) -> None:
    store, _ = open_index(data_dir)
    try:
        MemoryState(store).dismiss(KIND_MEMORY, memory_id(recipe, key))
        console.print(
            f"[green]Dismissed[/green] {recipe}:{key}. It will never be surfaced again.\n"
            "[dim]Undo with `rekindle undismiss`.[/dim]"
        )
    finally:
        store.close()


def undismiss_cmd(data_dir: Path, recipe: str, key: str) -> None:
    store, _ = open_index(data_dir)
    try:
        if MemoryState(store).undismiss(KIND_MEMORY, memory_id(recipe, key)):
            console.print(f"[green]Restored[/green] {recipe}:{key}.")
        else:
            console.print(f"[yellow]{recipe}:{key} was not dismissed.[/yellow]")
    finally:
        store.close()


def exclude_cmd(
    data_dir: Path, person: str | None, album: str | None, from_: str | None, to: str | None
) -> None:
    """Add a standing exclusion. Dismissal is the everyday path; this is for a
    person or a period the user wants gone before anything surfaces it."""
    store, _ = open_index(data_dir)
    try:
        state = MemoryState(store)
        if person:
            state.dismiss(KIND_PERSON, person, reason="excluded")
            console.print(f"[green]Excluded[/green] every photo containing {person}.")
        if album:
            state.dismiss(KIND_ALBUM, album, reason="excluded")
            console.print(f"[green]Excluded[/green] album {album}.")
        if from_ and to:
            try:
                date.fromisoformat(from_), date.fromisoformat(to)
            except ValueError as exc:
                console.print(f"[red]Dates must be YYYY-MM-DD: {exc}[/red]")
                raise typer.Exit(code=2) from exc
            state.dismiss(KIND_DATES, f"{from_}/{to}", reason="excluded")
            console.print(f"[green]Excluded[/green] {from_} to {to}.")
        if not (person or album or (from_ and to)):
            console.print("[red]Nothing to exclude.[/red] Pass --person, --album, or --from/--to.")
            raise typer.Exit(code=2)
    finally:
        store.close()


def dismissals_cmd(data_dir: Path) -> None:
    store, _ = open_index(data_dir)
    try:
        rows = MemoryState(store).dismissals()
        if not rows:
            console.print("Nothing has been dismissed or excluded.")
            return
        table = Table(title="Dismissals and exclusions")
        table.add_column("Kind")
        table.add_column("Value")
        table.add_column("Why")
        table.add_column("When")
        for row in rows:
            table.add_row(row.kind, row.value, row.reason, row.created_at[:10])
        console.print(table)
    finally:
        store.close()


def watch_cmd(
    root: Path, data_dir: Path, interval: float, once: bool, now: datetime | None = None
) -> None:
    from rekindle.doctor import diagnose, render
    from rekindle.enrich.takeout import EmptyIndexError, TakeoutEnricher
    from rekindle.memory.watch import watch
    from rekindle.sources.folder import FolderSource

    db_path = data_dir / DB_NAME
    if not db_path.is_file():
        console.print(f"[red]No index at[/red] {db_path}. Run `rekindle index {root}` first.")
        raise typer.Exit(code=2)

    def reindex(_state) -> None:
        console.print("[cyan]Library changed[/cyan] - re-indexing...")
        photos, report = FolderSource().scan(root)
        with PhotoStore(db_path) as store:
            store.upsert_many(photos)
            # An empty index here means `index` has not run yet for this
            # root; the upsert above has just fixed that, but on the FIRST
            # cycle of a fresh watch the enricher can still see nothing.
            with contextlib.suppress(EmptyIndexError):
                TakeoutEnricher().enrich(root, store)
        render(diagnose(report), console)

    def announce(today: date) -> list[str]:
        store, index = open_index(data_dir)
        try:
            offers = engine.offers_for_today(index, today)
            if not offers:
                return [f"[dim]{today.isoformat()}: no anniversary today.[/dim]"]
            lines = []
            for offer in offers[:3]:
                lines.append(f"[green]{offer.title}[/green] - {offer.subtitle}")
                lines.append(
                    f'  [dim]rekindle memory --recipe {offer.recipe} --key "{offer.key}"[/dim]'
                )
            return lines
        finally:
            store.close()

    console.print(
        f"Watching {root} every {interval:.0f}s. "
        "[dim]Nothing is rendered automatically - a command is printed.[/dim]"
    )
    watch(
        root,
        on_change=reindex,
        announce=announce,
        emit=console.print,
        interval=interval,
        cycles=1 if once else None,
        now=(lambda: now) if now else (lambda: datetime.now(UTC)),
        sleep=None,
    )
