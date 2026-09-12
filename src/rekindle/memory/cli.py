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
from rekindle.memory import captioning, engine
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
from rekindle.memory.recipes import Offer, registered
from rekindle.memory.render.frames import build_frames
from rekindle.memory.render.gif import DEFAULT_WIDTH as PREVIEW_WIDTH
from rekindle.memory.render.gif import preview_canvas, write_gif, write_webp
from rekindle.memory.render.mp4 import DEFAULT_WIDTH as MP4_WIDTH
from rekindle.memory.render.mp4 import mp4_canvas, write_mp4
from rekindle.memory.render.music import NO_MUSIC_HINT, resolve_music
from rekindle.memory.render.timeline import STYLE_FILM
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
        _render_album_merges(index)
        _render_exclusions(index)
    finally:
        store.close()


def _render_exclusions(index: MemoryIndex) -> None:
    report = index.report
    if not report.by_reason:
        return
    parts = ", ".join(f"{n} {reason}" for reason, n in sorted(report.by_reason.items()))
    console.print(f"[dim]{report.excluded} photos withheld by the guardrails: {parts}[/dim]")


def _render_album_merges(index: MemoryIndex) -> None:
    """Say which albums were treated as one, because it changes memory ids.

    A silent merge is the same defect as a silent drop: the user goes looking
    for `Christmas 2025`, finds `Christmas`, and has no way to know why. The
    override is named here so that whoever reads the line knows what to type.
    """
    merges = getattr(index, "album_merges", {})
    if not merges:
        return
    targets: dict[str, list[str]] = {}
    for name, target in sorted(merges.items()):
        targets.setdefault(target, []).append(name)
    for target, names in sorted(targets.items()):
        joined = ", ".join(repr(n) for n in names)
        console.print(
            f"[dim]Albums merged as {target!r}: {joined} - override with "
            f"album_aliases in exclusions.toml[/dim]"
        )


def _semantic_support(data_dir: Path, *, announce: bool = True):
    """The embedding store as a diversity signal, or None with a sentence.

    The ONE other place `rekindle.semantic` is reached from this module, and
    like `_retriever_for` the import is inside the body: `rekindle --help`
    must never load torch, and a machine without the extra must get a message
    rather than an ImportError.

    Absence is a supported configuration, not a failure, so this never raises
    and never exits - it says what is missing, names the command that would
    fix it, and lets the build run on the two pixel signals exactly as it did
    before embeddings existed.
    """
    from rekindle.semantic.availability import probe

    found = probe()
    if not found.any:
        if announce:
            console.print(
                "[dim]No embeddings available (the 'semantic' extra is not "
                "installed), so near-duplicates are judged on the perceptual "
                "hash and colour alone. `uv sync --extra semantic` to add "
                "them.[/dim]"
            )
        return None
    try:
        from rekindle.semantic.diversity import open_support

        support = open_support(data_dir)
    except Exception as exc:  # pragma: no cover - a corrupt store, reported
        console.print(f"[yellow]Embedding store unusable ({exc}); continuing without it.[/yellow]")
        return None
    if support is None:
        if announce:
            console.print(
                "[dim]This library has no embeddings yet, so near-duplicates "
                "are judged on the perceptual hash and colour alone. Run "
                "`rekindle semantic embed` to add them.[/dim]"
            )
        return None
    if announce:
        console.print(f"[dim]Semantic near-duplicate detection on: {len(support)} vectors.[/dim]")
    return support


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
    force: bool = False,
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
        elif recipe or key:
            # `--recipe` ALONE used to fall through to "build anything", so
            # `rekindle memory --recipe on_this_month` cheerfully rendered
            # album stories - the command doing something other than what its
            # own `--help` says, silently, which is this project's signature
            # defect. `--key` alone is refused rather than guessed at: two
            # recipes can share a key shape ("10" is a month and, for
            # `on_this_day`, half a date).
            if key and not recipe:
                console.print(
                    "[red]--key needs --recipe[/red]: the same key can belong to "
                    "more than one recipe. Run `rekindle memories` to see both."
                )
                raise typer.Exit(code=2)
            found = [
                o
                for o in engine.all_offers(index)
                if o.recipe == recipe and (key is None or o.key == key)
            ]
            if not found:
                target = f"recipe={recipe!r}" + (f" key={key!r}" if key else "")
                console.print(
                    f"[red]No memory[/red] for {target}. "
                    "Run `rekindle memories` to see what is available."
                )
                raise typer.Exit(code=2)
            offers = found
        else:
            offers = engine.all_offers(index)

        # Naming ONE memory overrides the cooldown. The cooldown exists to
        # stop `--auto` and an unfiltered build from showing the same memory
        # again next week; `rekindle memory --recipe X --key Y` is a person
        # pointing at one memory and saying build this, and answering
        # "1 in cooldown" is the tool second-guessing an explicit instruction
        # with no way to override it. Found while trying to re-render a memory
        # after changing the code that builds it, which is the commonest
        # reason to type that command at all.
        #
        # DISMISSAL still applies. "Never show me this again" is deliberate
        # and permanent, and `rekindle undismiss` is how it is taken back.
        # `--force` extends that same reasoning to a filtered or unfiltered
        # build. Re-rendering after changing the code, the guardrails or the
        # exclusion list is the commonest reason to rebuild in bulk, and the
        # cooldown had no override at all - it refused 29 of 34 album stories
        # on a deliberate rebuild. DISMISSAL is still honoured; only the
        # resurfacing clock is bypassed.
        named = bool(recipe and key) or force
        specs, report = engine.build_all(
            index,
            offers,
            max_shots=max_shots,
            dismissed=state.dismissed_memory_ids(),
            cooling=frozenset() if named else state.cooling(),
            limit=limit,
            semantic=_semantic_support(data_dir),
        )
        if not specs:
            console.print("[yellow]Nothing to build.[/yellow]")
            _render_build_report(report)
            return

        specs = _maybe_caption(specs, captions, index=index, store=store, data_dir=data_dir)
        for spec in specs:
            _render_one(spec, index, out_dir, gif_frames, music, no_mp4, preview_width, mp4_width)
            state.record_surfaced(memory_id(spec.recipe, spec.key), title=spec.title)
        _render_build_report(report)
    finally:
        store.close()


def _vision_support(data_dir: Path):
    """The caption vocabulary scored against this library, or None + a sentence.

    Like `_semantic_support`, the import is inside the body and absence is a
    supported configuration rather than a failure: without it captions are the
    deterministic ones, which is what ships and what CI runs.
    """
    from rekindle.semantic.availability import probe

    if not probe().any:
        console.print(
            "[dim]Grounded captions need the 'semantic' extra "
            "(`uv sync --extra semantic`); the deterministic captions were "
            "used.[/dim]"
        )
        return None
    try:
        from rekindle.semantic import describe
        from rekindle.semantic.encoder import load_encoder
        from rekindle.semantic.registry import embed_model
        from rekindle.semantic.setup import cache_dir_for
        from rekindle.semantic.store import EmbeddingStore, store_root

        spec = embed_model(None)
        root = store_root(data_dir, spec.key)
        if not (root / "manifest.sqlite").is_file():
            console.print(
                "[dim]This library has no embeddings, so nothing could ground a "
                "caption. Run `rekindle semantic embed`.[/dim]"
            )
            return None
        store = EmbeddingStore(
            root,
            dim=spec.dim,
            model_key=spec.key,
            model_revision=spec.pin("torch").revision if spec.torch else "",
        )
        encoder = load_encoder(spec.key, device="auto", cache_dir=cache_dir_for(data_dir)).encoder
        return describe.build(store, encoder)
    except Exception as exc:  # pragma: no cover - reported, never fatal
        console.print(f"[yellow]Caption grounding unavailable ({exc}); continuing.[/yellow]")
        return None


def _maybe_caption(
    specs: list[MemorySpec],
    mode: str,
    *,
    index: MemoryIndex | None = None,
    store: PhotoStore | None = None,
    data_dir: Path | None = None,
    vision=None,
) -> list[MemorySpec]:
    """The ONE place the optional caption layers are reached from.

    `deterministic` returns the specs untouched. `clip` adds what an image
    model recognised, from a closed vocabulary. `gpt` adds a language model
    phrasing those terms - never the pixels - and verifies every sentence
    against the index before it renders.

    Every failure degrades one level and exits 0. A missing extra, an
    un-embedded library, a missing key, an unreachable service and a rejected
    caption all end with the deterministic caption in place, because
    `rekindle` must keep working for everyone who has none of these - which is
    the default configuration and is what CI runs.
    """
    if mode == captioning.MODE_DETERMINISTIC:
        return specs

    if vision is None and data_dir is not None:
        vision = _vision_support(data_dir)

    groundings: dict[str, dict] = {}
    if vision is not None:
        out: list[MemorySpec] = []
        totals = captioning.GroundingReport()
        for spec in specs:
            found, report = captioning.ground(spec, vision)
            groundings[spec.key] = found
            totals.requested += report.requested
            totals.grounded += report.grounded
            totals.silent += report.silent
            totals.unembedded += report.unembedded
            for facet, n in report.facets.items():
                totals.facets[facet] = totals.facets.get(facet, 0) + n
            out.append(
                captioning.apply_clip(spec, found, store=store)
                if mode == captioning.MODE_CLIP
                else spec
            )
        console.print(f"[dim]{captioning.describe_grounding(totals)}[/dim]")
        if mode == captioning.MODE_CLIP:
            return out

    if mode != captioning.MODE_GPT:
        return specs

    from rekindle.memory.llm import MODEL, LLMUnavailable, apply_captions, captioner_from_env

    try:
        captioner = captioner_from_env()
    except LLMUnavailable as exc:
        console.print(f"[yellow]![/yellow] {exc}")
        # Grounded captions are strictly better than the deterministic ones
        # and cost nothing more, so a missing key falls back to CLIP rather
        # than all the way to the year.
        if vision is not None:
            return [captioning.apply_clip(s, groundings.get(s.key, {}), store=store) for s in specs]
        return specs

    cache = captioning.gpt_cache(store, MODEL) if store is not None else None
    out = []
    accepted = requested = cached = 0
    rejected: dict[str, int] = {}
    for spec in specs:
        found = groundings.get(spec.key, {})

        def context(shot, _found=found):
            grounding = _found.get(shot.file_hash)
            terms = grounding.says if grounding else ()
            return captioning.shot_facts(index, shot, terms) if index is not None else None

        rewritten, report = apply_captions(
            spec, captioner, context if index is not None else None, cache=cache
        )
        out.append(rewritten)
        accepted += report.accepted
        cached += report.cached
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
        f"[dim]GPT captions: {accepted}/{requested} accepted, {cached} from cache"
        + (f"; rejected {detail}" if detail else "")
        + "[/dim]"
    )
    return out


def _render_build_report(report: engine.BuildReport) -> None:
    if report.skipped:
        parts = ", ".join(f"{n} {r.replace('_', ' ')}" for r, n in sorted(report.skipped.items()))
        console.print(f"[dim]{report.total_skipped} candidate memories skipped: {parts}[/dim]")
    if report.deduped:
        # The semantic count is broken out rather than folded in: these are
        # the frames the perceptual hash called UNRELATED, so a user who wants
        # to check the guardrail needs to know which pile to look in.
        extra = (
            f" {report.deduped_semantically} of them on visual similarity the "
            f"perceptual hash missed."
            if report.deduped_semantically
            else ""
        )
        console.print(f"[dim]{report.deduped} near-duplicate frames collapsed.{extra}[/dim]")
    if report.diversity.displaced:
        console.print(
            f"[dim]{report.diversity.displaced} shots chosen differently because a "
            f"semantically near-identical shot was already in the memory.[/dim]"
        )
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
    style: str = STYLE_FILM,
    onsets: bool = True,
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
        bed = resolve_music(music, memory_id=memory_id(spec.recipe, spec.key))
        if style == STYLE_FILM:
            result, mp4_report = _render_film(spec, index, folder, video_size, bed, onsets)
        else:
            mp4_frames, mp4_report = build_frames(
                spec, video_size, resolve=index.get, locate=index.resolve_path
            )
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


def _render_film(spec, index, folder: Path, video_size, bed: Path | None, onsets: bool):
    """The `film` style: dissolves, Ken Burns, and cards that arrive.

    **The previews are not touched.** GIF and WebP stay hard cuts and static
    frames, because they exist to be small enough to embed and motion at 1280
    wide costs several megabytes for something nobody asked a preview to do.
    Motion belongs in the MP4.
    """
    from rekindle.memory.render import motion as motion_mod
    from rekindle.memory.render import onsets as onsets_mod
    from rekindle.memory.render import timeline as tl
    from rekindle.memory.render.mp4 import write_film

    plan = tl.plan(len(spec.shots) + 1, has_title=True)
    if onsets and bed is not None:
        found, note = onsets_mod.detect(bed)
        if found:
            snapped = tl.snap_to(plan, found)
            moved = sum(
                1
                for a, b in zip(plan.beats, snapped.beats, strict=True)
                if abs(a.start - b.start) > 0.01
            )
            plan = snapped
            console.print(
                f"  [dim]Beat sync: {len(found)} onsets in {bed.name}; "
                f"{moved} of {len(plan.beats) - 1} cuts moved onto one.[/dim]"
            )
        elif note:
            console.print(f"  [dim]Beat sync off: {note}[/dim]")

    plates, report, moves = motion_mod.build_plates(
        spec, video_size, plan, resolve=index.get, locate=index.resolve_path
    )
    if report.rendered == 0:
        from rekindle.memory.render.mp4 import Mp4Result

        return Mp4Result(path=None, error="no frames to encode"), report
    # The timeline was planned for every shot; some were dropped, so it is
    # re-planned for what actually decoded. Rendering the original would leave
    # the last shots on screen for no time at all - or, with beat snapping,
    # leave a hole.
    if len(plates) != len(plan.beats):
        plan = tl.plan(len(plates), has_title=True)
    console.print(f"  [dim]{tl.describe(plan)}. {motion_mod.describe_motion(moves)}.[/dim]")
    result = write_film(
        motion_mod.frames_for(plates, plan),
        folder / "memory.mp4",
        video_size,
        fps=plan.fps,
        music=bed,
    )
    return result, report


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


# --------------------------------------------------------------------------
# Prompt memories
#
# This is the ONLY place `rekindle.semantic` is reached from the memory CLI,
# and the import is inside the function body. `rekindle --help` must never
# load torch, and a machine without the extra must get a sentence naming the
# command to run rather than an ImportError traceback.

EXIT_UNAVAILABLE = 3
EXIT_REFUSED = 4

PROMPT_CAVEAT = (
    "No automated check can tell whether these photos match your words.\n"
    "Look at the memory before you keep it."
)

WEAK_PATH_WARNING = (
    "Nothing describes this prompt visually, so the words themselves were "
    "searched. That is the measured-bad path: it put 9 of 24 shots of a Kali "
    "Puja memory on a Durga Puja day. Add an entry to the tag cache, name a "
    "festival the corpus knows, or set OPENAI_API_KEY."
)

NO_VIDEO_NOTE = (
    "No video can appear in a prompt memory: videos are not embedded, so the "
    "search cannot see them."
)

_MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


class PromptUnavailable(RuntimeError):
    """The extra is installed but this library has no embeddings yet."""


def _retriever_for(data_dir: Path, model_key: str | None):
    """A `prompt.Retriever` backed by the real embedding store.

    Raises `SemanticUnavailable` when the extra is not installed and
    `PromptUnavailable` when it is installed but the library has never been
    embedded. Two different problems with two different fixes, so they are
    never folded into one message - an un-embedded library must not look like
    a missing dependency, or like a query that found nothing.
    """
    from rekindle.semantic.availability import require
    from rekindle.semantic.encoder import load_encoder
    from rekindle.semantic.registry import embed_model
    from rekindle.semantic.search import SemanticSearch, embed_query
    from rekindle.semantic.setup import cache_dir_for
    from rekindle.semantic.store import EmbeddingStore, store_root

    require(feature="Prompt memories")
    spec = embed_model(model_key)
    root = store_root(data_dir, spec.key)
    if not (root / "manifest.sqlite").is_file():
        raise PromptUnavailable(
            f"No embeddings for '{spec.key}' at {root}. Run `rekindle semantic embed` first."
        )
    store = EmbeddingStore(
        root,
        dim=spec.dim,
        model_key=spec.key,
        model_revision=spec.pin("torch").revision if spec.torch else "",
    )
    search = SemanticSearch(store, None)
    if not len(search.matrix):
        raise PromptUnavailable(
            f"The embedding store at {root} holds no vectors. Run `rekindle semantic embed` first."
        )
    encoder = load_encoder(spec.key, device="auto", cache_dir=cache_dir_for(data_dir)).encoder

    def retrieve(text: str, k: int) -> list[tuple[str, float]]:
        hits = search.search_vector(embed_query(encoder, text), k=k)
        return [(h.file_hash, h.score) for h in hits]

    return retrieve, f"{len(search.matrix)} vectors, model {spec.key}"


def prompt_cmd(
    data_dir: Path,
    text: str,
    out_dir: Path,
    *,
    public_safe: bool = False,
    max_shots: int = engine.DEFAULT_MAX_SHOTS,
    gif_frames: int = 16,
    music: Path | None = None,
    no_mp4: bool = False,
    seed_k: int = 0,
    min_seeds: int = 0,
    tag_k: int = 0,
    captions: str = "deterministic",
    judge: bool = True,
    preview_width: int = 0,
    mp4_width: int = 0,
    retrieve=None,
) -> None:
    """Build ONE memory from the user's own words.

    A prompt memory is a PREVIEW, never an offer. It is built only when asked
    for by name; it never enters `all_offers`, never appears in
    `rekindle memories`, and is never eligible for `--auto`. That is a hard
    rule rather than a default, and it is the honest response to a limit that
    cannot be engineered away: there is no statistic that says whether these
    photos match these words, so the only working verifier is the person
    looking at the result.

    The cooldown is bypassed for the same reason `--recipe --key` bypasses it:
    this is a person pointing at one memory and asking for it. Dismissal still
    applies, and `record_surfaced` still runs.
    """
    from rekindle.memory import prompt as prompt_mod
    from rekindle.memory import tags as tags_mod
    from rekindle.memory.llm import LLMUnavailable

    if not prompt_mod.normalise(text):
        console.print("[red]An empty prompt cannot build anything.[/red]")
        raise typer.Exit(code=2)

    store, index = open_index(data_dir, public_safe=public_safe)
    try:
        if index.count() == 0:
            console.print("[yellow]No photos available after the guardrails.[/yellow]")
            _render_exclusions(index)
            return
        _warn_unfingerprinted(index)

        query = prompt_mod.parse(text, index)
        try:
            generator = tags_mod.generator_from_env()
        except LLMUnavailable:
            generator = None

        if _refused_by_judge(generator, query, index, judge):
            raise typer.Exit(code=EXIT_REFUSED)

        resolution = tags_mod.resolve(
            query,
            data_dir=data_dir,
            generator=generator,
            people=list(index.people_counts()),
        )
        if resolution.source == tags_mod.SOURCE_MODEL:
            written = tags_mod.write_cache_entry(data_dir, query.text, resolution.tags)
            console.print(
                f"[dim]Tags cached in {written}; the same prompt will give the "
                "same memory from now on.[/dim]"
            )

        if retrieve is None:
            retrieve = _open_retriever(data_dir)

        build = prompt_mod.build_selection(
            index,
            query,
            resolution.tags,
            retrieve,
            seed_k=seed_k or prompt_mod.SEED_K,
            min_seeds=min_seeds or prompt_mod.MIN_SEEDS,
            tag_k=tag_k or prompt_mod.TAG_K,
            months=resolution.months,
            festival=resolution.festival,
            known_words=_festival_names(resolution, query),
        )
        _render_prompt_tags(resolution, build)

        report = engine.BuildReport(offered=1)
        spec = None
        if build.selection is None:
            report.skip(engine.SKIP_EMPTY)
        else:
            spec = engine.build(
                index,
                Offer(recipe=prompt_mod.RECIPE, key=query.text, title=query.text),
                selection=build.selection,
                max_shots=max_shots,
                report=report,
                # A prompt memory is the homogeneous case by construction -
                # every shot is the thing that was asked for - so it is where
                # the relative calibration matters most and an absolute
                # threshold would have done the most damage. See
                # `rekindle.semantic.diversity`.
                semantic=_semantic_support(data_dir, announce=False),
            )
        _render_prompt_report(build, spec, index)
        if spec is None:
            console.print("[yellow]Nothing to build from that prompt.[/yellow]")
            _render_build_report(report)
            return

        state = MemoryState(store)
        if prompt_mod.memory_key(query) in state.dismissed_memory_ids():
            console.print(
                f"[yellow]Dismissed[/yellow]: {prompt_mod.memory_key(query)}. "
                "Undo it with `rekindle undismiss`."
            )
            return

        for built in _maybe_caption([spec], captions, index=index, store=store, data_dir=data_dir):
            _render_one(built, index, out_dir, gif_frames, music, no_mp4, preview_width, mp4_width)
            state.record_surfaced(memory_id(built.recipe, built.key), title=built.title)
        _render_build_report(report)
        console.print(f"\n[yellow]{PROMPT_CAVEAT}[/yellow]")
    finally:
        store.close()


def _open_retriever(data_dir: Path):
    from rekindle.semantic.availability import SemanticUnavailable

    try:
        retrieve, note = _retriever_for(data_dir, None)
    except PromptUnavailable as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    except SemanticUnavailable as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=EXIT_UNAVAILABLE) from exc
    console.print(f"[dim]{note}[/dim]")
    return retrieve


def _refused_by_judge(generator, query, index, judge: bool) -> bool:
    """Gate one: is the QUERY coherent and consistent with this library?

    Available only with an API key, and deliberately biased towards letting
    things through - a wrong refusal leaves the user with nothing to look at,
    which is worse than a wrong acceptance they can see and discard. It is
    UNMEASURED on the reference library, because measuring it needs a key.
    """
    from rekindle.memory import tags as tags_mod
    from rekindle.memory.llm import LLMUnavailable

    if generator is None or not judge:
        return False
    try:
        reason = generator.plausible(query.text, tags_mod.vocabulary_of(index))
    except LLMUnavailable as exc:
        console.print(f"[yellow]![/yellow] {exc}")
        return False
    if reason is None:
        return False
    console.print(f"[red]Refused[/red]: {reason}")
    console.print("[dim]Pass --no-judge to build it anyway.[/dim]")
    return True


def _festival_names(resolution, query) -> tuple[str, ...]:
    if not resolution.festival:
        return ()
    from rekindle.memory import festivals as festivals_mod

    matched = festivals_mod.match(query.subject)
    return matched.names if matched else ()


def _render_prompt_tags(resolution, build) -> None:
    """Say what was actually searched for.

    The tags ARE the query. A user who cannot see them cannot tell a bad
    memory from a bad description of what they wanted.
    """
    from rekindle.memory import tags as tags_mod

    origin = {
        tags_mod.SOURCE_CORPUS: f"the festival corpus ({resolution.festival})",
        tags_mod.SOURCE_CACHE: "the tag cache",
        tags_mod.SOURCE_MODEL: "the language model",
        tags_mod.SOURCE_PROMPT: "your own words",
    }[resolution.source]
    console.print(f"Searched for {len(build.tags)} visual descriptions, from {origin}:")
    for tag in build.tags:
        console.print(f"  [dim]-[/dim] {tag}")
    if resolution.source == tags_mod.SOURCE_PROMPT:
        console.print(f"[yellow]![/yellow] {WEAK_PATH_WARNING}")
    if resolution.months:
        months = ", ".join(_MONTH_NAMES[m] for m in resolution.months)
        console.print(f"[dim]Narrowed to {months}, from the corpus window.[/dim]")
    if resolution.rejected:
        detail = ", ".join(sorted(set(resolution.rejected)))
        console.print(f"[dim]Some generated tags were rejected: {detail}.[/dim]")


def _render_prompt_report(build, spec, index) -> None:
    """The honesty surface. Everything a person needs to judge the result for
    themselves, because nothing else can judge it for them."""
    if build.seed_days:
        days = ", ".join(f"{s.iso} ({s.hits} photos, {s.tags} tags)" for s in build.seed_days)
        console.print(f"\n{len(build.seed_days)} days looked like this: {days}")
        console.print(f"[dim]Expanded to {build.pool} candidate photos.[/dim]")
    else:
        console.print("\n[yellow]No capture day had enough agreement between the tags.[/yellow]")
    if build.albums:
        named = ", ".join(repr(a) for a in build.albums)
        console.print(f"[dim]Your own albums {named} were added whole.[/dim]")
    if build.unmatched:
        words = ", ".join(repr(w) for w in build.unmatched)
        console.print(
            f"[yellow]![/yellow] {words} narrowed nothing: no album, person or "
            "month in your library matches. It was searched for as a PICTURE and "
            "nothing else - so if one of those words is a place, note that "
            "rekindle has no gazetteer and never filtered by it."
        )
    console.print(
        f"[dim]Tag agreement {build.agreement:.0%}. This number does NOT say "
        "whether the concept is in your library: measured over 16 concepts this "
        "library has and 16 it does not, it fails to separate them, so nothing "
        "is refused on it.[/dim]"
    )
    if spec is None:
        return

    months: dict[int, int] = {}
    years: dict[int, int] = {}
    faces = gps = 0
    for shot in spec.shots:
        photo = index.get(shot.file_hash)
        local = photo.meta.taken_at_local
        months[local.month] = months.get(local.month, 0) + 1
        years[local.year] = years.get(local.year, 0) + 1
        faces += 1 if any(photo.meta.people) else 0
        gps += 1 if photo.meta.gps is not None else 0
    console.print(
        f"Built {len(spec.shots)} shots across {len(years)} years: "
        + ", ".join(f"{y} ({n})" for y, n in sorted(years.items()))
    )
    ordered = sorted(months.items(), key=lambda kv: (-kv[1], kv[0]))
    console.print("  months: " + ", ".join(f"{_MONTH_NAMES[m]} {n}" for m, n in ordered))
    console.print(f"  {faces} of {len(spec.shots)} shots have face tags, {gps} have GPS.")
    console.print(f"[dim]{NO_VIDEO_NOTE}[/dim]")


# --------------------------------------------------------------------------
# scenery


SCENERY_CAVEAT = (
    "A scenery memory is retrieved by what a photograph LOOKS like, and no "
    "automated check can confirm it got that right.\n"
    "Look at the memory before you keep it - and if a concept keeps getting "
    "it wrong, the descriptions are in `corpus/scenery.toml` and are yours "
    "to edit."
)


def scenery_list_cmd() -> None:
    """Print the corpus. Builds nothing, needs no index and no embeddings."""
    from rekindle.memory import scenery as scenery_mod

    table = Table(title="The scenery corpus", show_lines=True)
    table.add_column("key")
    table.add_column("title")
    table.add_column("looks like")
    table.add_column("months")
    for concept in scenery_mod.all_concepts():
        months = ", ".join(_MONTH_NAMES[m] for m in concept.months) if concept.months else "any"
        table.add_row(concept.key, concept.title, "\n".join(concept.tags), months)
    console.print(table)
    console.print(
        "[dim]This file is data. Edit "
        f"{scenery_mod.CORPUS_PATH} to change what a concept looks for, or to "
        "add one of your own.[/dim]"
    )


def scenery_cmd(
    data_dir: Path,
    concepts: list[str],
    out_dir: Path,
    *,
    all_concepts: bool = False,
    public_safe: bool = False,
    max_shots: int = engine.DEFAULT_MAX_SHOTS,
    gif_frames: int = 16,
    music: Path | None = None,
    no_mp4: bool = False,
    min_votes: int = 0,
    tag_k: int = 0,
    captions: str = "deterministic",
    preview_width: int = 0,
    mp4_width: int = 0,
    style: str = "film",
    retrieve=None,
) -> None:
    """Build memories whose subject is a SCENE.

    Like a prompt memory this is built only when asked for: it does not enter
    `all_offers`, does not appear in `rekindle memories` and is never eligible
    for `--auto`. The reason is narrower than the prompt one - the concepts
    here are curated and checkable, where a prompt can say anything - and it
    is structural: selection needs a retriever, and the `Recipe` protocol has
    no way to receive one. CI has no embeddings at all.

    Dismissal applies. The cooldown does not, for the same reason it does not
    for `--recipe --key`: this is a person naming one memory and asking for it.
    """
    from rekindle.memory import scenery as scenery_mod

    try:
        wanted = scenery_mod.concept_keys(None if all_concepts else concepts)
    except KeyError as exc:
        console.print(f"[red]{exc.args[0]}[/red]")
        raise typer.Exit(code=2) from exc

    store, index = open_index(data_dir, public_safe=public_safe)
    try:
        if index.count() == 0:
            console.print("[yellow]No photos available after the guardrails.[/yellow]")
            _render_exclusions(index)
            return
        _warn_unfingerprinted(index)

        if retrieve is None:
            retrieve = _open_retriever(data_dir)

        state = MemoryState(store)
        dismissed = state.dismissed_memory_ids()
        support = _semantic_support(data_dir, announce=False)
        # Built once for the whole run, not once per concept: the expensive
        # part is scoring every vocabulary probe against all 18,201 vectors,
        # and it depends on the library rather than on the memory.
        vision = _vision_support(data_dir) if captions != captioning.MODE_DETERMINISTIC else None
        built = 0
        for key in wanted:
            concept = scenery_mod.get(key)
            assert concept is not None  # concept_keys only returns known keys
            memory = memory_id(scenery_mod.RECIPE, concept.key)
            if memory in dismissed:
                console.print(
                    f"[yellow]Dismissed[/yellow]: {memory}. Undo it with `rekindle undismiss`."
                )
                continue
            build = scenery_mod.build_selection(
                index,
                concept,
                retrieve,
                tag_k=tag_k or scenery_mod.TAG_K,
                min_votes=min_votes or scenery_mod.MIN_VOTES,
            )
            spec = _build_scenery(index, build, max_shots, support)
            if spec is None:
                continue
            for finished in _maybe_caption(
                [spec], captions, index=index, store=store, vision=vision
            ):
                _render_one(
                    finished,
                    index,
                    out_dir,
                    gif_frames,
                    music,
                    no_mp4,
                    preview_width,
                    mp4_width,
                    style=style,
                )
                state.record_surfaced(
                    memory_id(finished.recipe, finished.key), title=finished.title
                )
                _render_scenery_report(build, finished, index)
                built += 1
        if built:
            console.print(f"\n[yellow]{SCENERY_CAVEAT}[/yellow]")
    finally:
        store.close()


def _build_scenery(index, build, max_shots: int, support):
    """One SceneryBuild through the engine, or None with the reason printed."""
    from rekindle.memory import scenery as scenery_mod

    concept = build.concept
    console.print(
        f"\n[bold]{concept.title}[/bold] [dim](scenery:{concept.key})[/dim] - "
        f"{len(concept.tags)} visual descriptions:"
    )
    for tag in concept.tags:
        console.print(f"  [dim]-[/dim] {tag}")
    if concept.months:
        months = ", ".join(_MONTH_NAMES[m] for m in concept.months)
        console.print(f"  [dim]Narrowed to {months}, from the corpus window.[/dim]")
    console.print(
        f"  [dim]{build.pool} photos matched at least one description; "
        f"{build.reached} were reached by {scenery_mod.MIN_VOTES} or more "
        f"({scenery_mod.describe_votes(build.votes)}), across "
        f"{len(build.years)} years.[/dim]"
    )
    if build.selection is None:
        console.print("  [yellow]Nothing agreed on enough to build a memory.[/yellow]")
        return None

    report = engine.BuildReport(offered=1)
    spec = engine.build(
        index,
        Offer(recipe=scenery_mod.RECIPE, key=concept.key, title=concept.title),
        selection=build.selection,
        max_shots=max_shots,
        report=report,
        semantic=support,
    )
    if spec is None:
        reasons = ", ".join(r.replace("_", " ") for r in sorted(report.skipped))
        console.print(f"  [yellow]Not built: {reasons}.[/yellow]")
        _render_build_report(report)
        return None
    _render_build_report(report)
    return spec


def _render_scenery_report(build, spec, index) -> None:
    """The honesty surface: how much of this memory no other recipe could reach."""
    from rekindle.memory import scenery as scenery_mod

    years: dict[int, int] = {}
    orphans = 0
    for shot in spec.shots:
        photo = index.get(shot.file_hash)
        if photo is None:  # pragma: no cover - the spec was built from the index
            continue
        years[photo.meta.taken_at_local.year] = years.get(photo.meta.taken_at_local.year, 0) + 1
        orphans += 1 if scenery_mod.orphan(photo) else 0
    console.print(
        f"  {len(spec.shots)} shots across {len(years)} years: "
        + ", ".join(f"{y} ({n})" for y, n in sorted(years.items()))
    )
    console.print(
        f"  [dim]{orphans} of {len(spec.shots)} have no face tag, no named album "
        "and no GPS - no other recipe could have made them the subject of "
        "anything.[/dim]"
    )
