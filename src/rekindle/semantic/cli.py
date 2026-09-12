"""Semantic CLI verbs, packaged so `rekindle.cli` needs only two added lines.

M2 is editing `src/rekindle/cli.py` in another worktree at the same time as
this milestone. Every command below therefore lives HERE and is attached by
`register(app)`, so the shared file gains one import and one call and nothing
else - a merge conflict you can resolve by reading two lines, rather than one
spread over six command bodies.

NOTHING HEAVY IS IMPORTED AT MODULE LEVEL. `rekindle --help` on a default
install must not import numpy, torch or onnxruntime, and
`tests/test_semantic_imports.py` fails if it starts to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

# `meta.orientation` imports nothing heavier than json and pathlib - PIL is
# deferred inside its functions - so naming its default here costs `rekindle
# --help` nothing and stops the CLI and the module disagreeing about it.
from rekindle.extras import markup_safe
from rekindle.meta.orientation import MIN_MARGIN
from rekindle.semantic.availability import SemanticUnavailable
from rekindle.semantic.embed import PREFETCH_BATCHES
from rekindle.semantic.faces import DEFAULT_GATE_WORKERS

#: Exit code for "the optional dependency or the model is not here". Distinct
#: from 2 ("you pointed me at the wrong thing"), because a wrapper script
#: should be able to tell "install the extra" from "bad arguments".
EXIT_UNAVAILABLE = 3

console = Console()

semantic_app = typer.Typer(
    help="Embeddings, semantic search, scene clusters, aesthetics and the face gate.",
    no_args_is_help=True,
)

DataDir = Annotated[Path, typer.Option("--data-dir", help="Where rekindle stores its index.")]
ModelOpt = Annotated[str | None, typer.Option("--model", help="Embedding model key.")]
DeviceOpt = Annotated[
    str, typer.Option("--device", help="auto | cuda | cpu. 'cuda' fails rather than falling back.")
]


def _db_path(data_dir: Path) -> Path:
    return data_dir / "rekindle.sqlite"


def _fail(message: str, code: int = EXIT_UNAVAILABLE) -> typer.Exit:
    # ESCAPED. Every message that reaches here may name an install command,
    # and `pip install 'rekindle[semantic]'` is valid Rich markup that Rich
    # deletes - see `extras.markup_safe`.
    console.print(f"[red]{markup_safe(message)}[/red]")
    return typer.Exit(code=code)


def _reader(data_dir: Path):
    from rekindle.semantic.photos import IndexUnavailable, PhotoIndexReader

    try:
        return PhotoIndexReader(_db_path(data_dir))
    except IndexUnavailable as exc:
        raise _fail(str(exc), code=2) from exc


def _open_store(data_dir: Path, model_key: str | None, *, create: bool):
    from rekindle.semantic.registry import embed_model
    from rekindle.semantic.store import EmbeddingStore, StoreError, store_root

    spec = embed_model(model_key)
    root = store_root(data_dir, spec.key)
    if not create and not (root / "manifest.sqlite").is_file():
        raise _fail(
            f"No embeddings for '{spec.key}' at {root}. Run `rekindle semantic embed` first."
        )
    try:
        store = EmbeddingStore(
            root,
            dim=spec.dim,
            model_key=spec.key,
            model_revision=spec.pin("torch").revision if spec.torch else "",
        )
    except StoreError as exc:
        raise _fail(str(exc), code=2) from exc
    return spec, store


# --------------------------------------------------------------------- doctor


@semantic_app.command("doctor")
def semantic_doctor(
    device: DeviceOpt = "auto",
    data_dir: DataDir = Path("./data"),
) -> None:
    """Report the semantic runtime: extras, device, models cached. Writes nothing."""
    from rekindle.semantic.availability import probe
    from rekindle.semantic.registry import EMBED_MODELS, embed_model
    from rekindle.semantic.runtime import DeviceUnavailable, resolve_device
    from rekindle.semantic.setup import cache_dir_for
    from rekindle.semantic.store import store_root

    found = probe()

    def _state(ok: bool, missing: tuple[str, ...]) -> str:
        if ok:
            return "[green]yes[/green]"
        return f"[yellow]no[/yellow] -> missing {', '.join(missing)}"

    console.print("[bold]extras[/bold]")
    console.print(f"  semantic      (numpy + onnxruntime):  {_state(found.cpu, found.missing_cpu)}")
    console.print(f"  semantic-gpu  (torch + transformers): {_state(found.gpu, found.missing_gpu)}")

    console.print("\n[bold]device[/bold]")
    try:
        report = resolve_device(device)  # type: ignore[arg-type]
    except DeviceUnavailable as exc:
        console.print(f"  [red]{markup_safe(str(exc))}[/red]")
    else:
        for line in report.lines():
            style = "yellow" if line.startswith("!") else ""
            console.print(f"  [{style}]{line}[/{style}]" if style else f"  {line}")
        if report.fell_back_to_cpu:
            console.print(
                "  [yellow]This is a defect on this machine, not a graceful degradation.[/yellow]"
            )

    cache = cache_dir_for(data_dir)
    console.print(f"\n[bold]model cache[/bold]  {cache}")
    console.print(f"  present: {'yes' if cache.is_dir() else 'no'}")

    console.print("\n[bold]embedding stores[/bold]")
    any_store = False
    for key in EMBED_MODELS:
        root = store_root(data_dir, key)
        manifest = root / "manifest.sqlite"
        if not manifest.is_file():
            continue
        any_store = True
        from rekindle.semantic.store import EmbeddingStore

        spec = embed_model(key)
        with EmbeddingStore(root, dim=spec.dim, model_key=key) as store:
            console.print(f"  {key}: {store.count()} vectors, dim {store.dim}, at {root}")
    if not any_store:
        console.print("  none yet - run `rekindle semantic embed <folder>`")


# ---------------------------------------------------------------------- setup


@semantic_app.command("setup")
def semantic_setup(
    model: ModelOpt = None,
    runtime: Annotated[str, typer.Option("--runtime", help="torch | onnx | both.")] = "torch",
    face: Annotated[
        str | None, typer.Option("--face", help="Also fetch this face detector.")
    ] = None,
    aesthetic: Annotated[
        bool, typer.Option("--aesthetic/--no-aesthetic", help="Also fetch the aesthetic head.")
    ] = True,
    allow_unlocked: Annotated[
        bool,
        typer.Option(
            "--allow-unlocked",
            help="Accept files that have no checksum in model-locks.json.",
        ),
    ] = False,
    write_lock: Annotated[
        bool, typer.Option("--write-lock", help="Record the checksums that were fetched.")
    ] = False,
    data_dir: DataDir = Path("./data"),
) -> None:
    """Download and verify model weights. The only command that uses the network."""
    from rekindle.semantic.registry import (
        DEFAULT_AESTHETIC_MODEL,
        DEFAULT_FACE_MODEL,
        embed_model,
    )
    from rekindle.semantic.setup import SetupError, run_setup
    from rekindle.semantic.setup import write_lock as save_lock

    try:
        spec = embed_model(model)
    except KeyError as exc:
        raise _fail(str(exc), code=2) from exc
    runtimes = ("torch", "onnx") if runtime == "both" else (runtime,)
    if runtime not in {"torch", "onnx", "both"}:
        raise _fail(f"--runtime must be torch, onnx or both (got {runtime!r})", code=2)

    face_keys = (face,) if face else (DEFAULT_FACE_MODEL,)
    aesthetic_keys = (DEFAULT_AESTHETIC_MODEL,) if aesthetic else ()

    console.print(f"Fetching into [bold]{data_dir / 'models'}[/bold] ...")
    try:
        report = run_setup(
            data_dir,
            embed_keys=(spec.key,),
            aesthetic_keys=aesthetic_keys,
            face_keys=face_keys,
            runtimes=runtimes,
            offline=False,
            allow_unlocked=allow_unlocked or write_lock,
        )
    except SemanticUnavailable as exc:
        raise _fail(str(exc)) from exc
    except SetupError as exc:
        raise _fail(str(exc), code=4) from exc

    table = Table(title="fetched", show_lines=False)
    table.add_column("repo")
    table.add_column("file")
    table.add_column("MB", justify="right")
    table.add_column("sha256")
    table.add_column("verified")
    for f in report.files:
        table.add_row(
            f.repo_id,
            f.filename,
            f"{f.size / 1e6:.1f}",
            f.sha256[:16] + "...",
            "[green]yes[/green]" if f.verified else "[yellow]unlocked[/yellow]",
        )
    console.print(table)
    console.print(
        f"{len(report.files)} files, {report.total_bytes / 1e6:.0f} MB, "
        f"cached in {report.cache_dir}"
    )
    console.print("\n[bold]licences[/bold]")
    for key, lic in report.licences.items():
        console.print(f"  {key}: [bold]{lic.spdx}[/bold]  {lic.url}")
        if lic.note:
            console.print(f"      {lic.note}")
    if write_lock:
        save_lock(report)
        console.print("\n[green]Wrote[/green] model-locks.json")
    console.print(
        "\nEverything else runs offline from this cache. Verify with: "
        "HF_HUB_OFFLINE=1 rekindle semantic doctor"
    )


@semantic_app.command("licences")
def semantic_licences() -> None:
    """Print the licence of every model rekindle can fetch. Fetches nothing."""
    from rekindle.semantic.registry import all_specs

    table = Table(title="model licences")
    table.add_column("key")
    table.add_column("kind")
    table.add_column("licence")
    table.add_column("source")
    for spec in all_specs():
        table.add_row(spec.key, spec.kind, spec.licence.spdx, spec.licence.url)
    console.print(table)
    for spec in all_specs():
        if spec.licence.note:
            console.print(f"[yellow]{spec.key}[/yellow]: {spec.licence.note}")


# ----------------------------------------------------------------------- embed


@semantic_app.command("embed")
def semantic_embed(
    model: ModelOpt = None,
    device: DeviceOpt = "auto",
    batch: Annotated[int, typer.Option("--batch", help="Images per forward pass.")] = 32,
    workers: Annotated[int, typer.Option("--workers", help="JPEG decode threads.")] = 6,
    prefetch: Annotated[
        int,
        typer.Option(
            "--prefetch",
            help="Batches decoded+preprocessed ahead of the device. "
            "This is the throughput knob; --batch is not.",
        ),
    ] = PREFETCH_BATCHES,
    limit: Annotated[
        int | None, typer.Option("--limit", help="Do at most N photos; report the rest.")
    ] = None,
    redo: Annotated[
        list[str] | None,
        typer.Option(
            "--redo",
            help="Recompute these file hashes whatever the plan says. Repeatable.",
        ),
    ] = None,
    redo_unverified: Annotated[
        bool,
        typer.Option(
            "--redo-unverified",
            help="Also recompute vectors written before the store recorded "
            "what they were decoded from. On a pre-existing store that is all "
            "of them.",
        ),
    ] = False,
    include_archived: Annotated[
        bool, typer.Option("--include-archived", help="Embed archived photos too.")
    ] = False,
    data_dir: DataDir = Path("./data"),
) -> None:
    """Embed every indexed photo with no vector, and every photo whose vector is STALE.

    A vector is stale when the pixels it was computed from are not the pixels
    this build now decodes - most often because `rekindle semantic orient`
    has since proved that file's EXIF orientation tag a lie. The file's bytes
    are unchanged, so its hash is unchanged; only the decode moved. Resumable.
    """
    from rekindle.semantic.embed import embed_photos
    from rekindle.semantic.encoder import load_encoder
    from rekindle.semantic.photos import ReadFilter
    from rekindle.semantic.runtime import DeviceUnavailable
    from rekindle.semantic.setup import cache_dir_for

    reader = _reader(data_dir)
    try:
        spec, store = _open_store(data_dir, model, create=True)
        try:
            loaded = load_encoder(spec.key, device=device, cache_dir=cache_dir_for(data_dir))  # type: ignore[arg-type]
        except (SemanticUnavailable, DeviceUnavailable) as exc:
            store.close()
            raise _fail(str(exc)) from exc

        for line in loaded.device.lines():
            console.print(f"  {line}")
        for warning in loaded.device.warnings:
            console.print(f"[yellow]![/yellow] {warning}")
        console.print(f"  runtime: {loaded.runtime}")

        where = ReadFilter(images_only=True, include_archived=include_archived)
        total = reader.count(where)
        console.print(f"\n{total} indexed images to consider, {store.count()} already embedded.")

        with typer.progressbar(length=max(1, total), label="embedding") as bar:
            state = {"last": 0}

            def tick(done: int, _total: int) -> None:
                bar.update(done - state["last"])
                state["last"] = done

            report = embed_photos(
                reader.iter_photos(where),
                loaded.encoder,
                store,
                batch_size=batch,
                workers=workers,
                prefetch=prefetch,
                target_px=spec.image_size,
                limit=limit,
                progress=tick,
                runtime=loaded.runtime,
                device=loaded.device.device,
                redo=redo or (),
                redo_unverified=redo_unverified,
            )
        store.close()
    finally:
        reader.close()

    console.print(
        f"\n[green]Embedded[/green] {report.embedded} "
        f"({report.newly_embedded} new, {report.recomputed} recomputed, "
        f"{report.images_per_s:.1f} img/s on {report.device}). "
        f"{report.already_embedded} already had vectors."
    )
    # The plan, always - including when it is all zeroes. A staleness detector
    # that never fires looks exactly like one with nothing to do, and the only
    # way to tell them apart is for the tool to say what it examined.
    console.print(
        f"  provenance: {report.fresh} fresh, {report.stale} stale, "
        f"{report.unverified} unverified"
        + (f", {report.duplicates} duplicate hash(es)" if report.duplicates else "")
    )
    if report.unverified and not redo_unverified:
        console.print(
            f"  [yellow]![/yellow] {report.unverified} vector(s) predate provenance "
            "recording, so staleness cannot be judged for them. "
            "`--redo-unverified` recomputes them."
        )
    if report.deferred:
        console.print(
            f"  [yellow]![/yellow] --limit deferred {report.deferred} photo(s) that need work."
        )
    if report.redo_unknown:
        console.print(
            f"  [yellow]![/yellow] --redo named {len(report.redo_unknown)} hash(es) the index "
            f"did not offer: {', '.join(report.redo_unknown[:5])}"
        )
    if report.missing_file or report.unreadable:
        console.print(
            f"[yellow]![/yellow] {report.missing_file} file(s) missing, "
            f"{report.unreadable} unreadable."
        )
        for path, why in report.failures[:10]:
            console.print(f"    {path}: {why}")
    if not report.accounted:
        console.print("[red]![/red] Accounting does not balance - this is a bug.")


# ---------------------------------------------------------------------- search


@semantic_app.command("find")
def semantic_find(
    query: Annotated[str, typer.Argument(help='What to look for, e.g. "sunsets".')],
    k: Annotated[int, typer.Option("-k", "--limit", help="How many results.")] = 20,
    model: ModelOpt = None,
    device: DeviceOpt = "auto",
    min_score: Annotated[
        float | None, typer.Option("--min-score", help="Drop hits below this cosine.")
    ] = None,
    data_dir: DataDir = Path("./data"),
) -> None:
    """Find photos matching a description."""
    from rekindle.semantic.encoder import load_encoder
    from rekindle.semantic.runtime import DeviceUnavailable
    from rekindle.semantic.search import SemanticSearch
    from rekindle.semantic.setup import cache_dir_for

    reader = _reader(data_dir)
    try:
        spec, store = _open_store(data_dir, model, create=False)
        try:
            loaded = load_encoder(spec.key, device=device, cache_dir=cache_dir_for(data_dir))  # type: ignore[arg-type]
            hits = SemanticSearch(store, reader).search_text(
                loaded.encoder, query, k=k, min_score=min_score
            )
        except (SemanticUnavailable, DeviceUnavailable) as exc:
            raise _fail(str(exc)) from exc
        finally:
            store.close()
    finally:
        reader.close()

    if not hits:
        console.print(f"No photos matched [bold]{query}[/bold].")
        return
    table = Table(title=f'"{query}"')
    table.add_column("score", justify="right")
    table.add_column("date")
    table.add_column("albums")
    table.add_column("path")
    for hit in hits:
        photo = hit.photo
        table.add_row(
            f"{hit.score:.4f}",
            (photo.taken_at_local or photo.taken_at_utc or "")[:10] if photo else "",
            ", ".join(photo.real_albums) if photo else "",
            hit.path_str,
        )
    console.print(table)


# --------------------------------------------------------------------- cluster


@semantic_app.command("cluster")
def semantic_cluster(
    k: Annotated[
        int | None, typer.Option("-k", help="Number of clusters. Default sqrt(n/2).")
    ] = None,
    model: ModelOpt = None,
    device: DeviceOpt = "auto",
    seed: Annotated[
        int, typer.Option("--seed", help="Clustering seed. Same seed, same answer.")
    ] = 0,
    untagged_only: Annotated[
        bool,
        typer.Option(
            "--untagged-only/--all",
            help="Only photos with no face tag and no real album (the population this exists for).",
        ),
    ] = False,
    show: Annotated[int, typer.Option("--show", help="How many clusters to print.")] = 25,
    label: Annotated[
        bool, typer.Option("--label/--no-label", help="Name clusters from a scene vocabulary.")
    ] = True,
    data_dir: DataDir = Path("./data"),
) -> None:
    """Group photos into scenes."""
    from rekindle.semantic.cluster import (
        SCENE_VOCABULARY,
        label_clusters,
        spherical_kmeans,
        suggest_k,
    )
    from rekindle.semantic.encoder import load_encoder
    from rekindle.semantic.photos import ReadFilter
    from rekindle.semantic.runtime import DeviceUnavailable
    from rekindle.semantic.setup import cache_dir_for
    from rekindle.semantic.vectors import Matrix, load_matrix

    reader = _reader(data_dir)
    try:
        spec, store = _open_store(data_dir, model, create=False)
        try:
            matrix = load_matrix(store)
            if untagged_only:
                wanted = {
                    p.file_hash
                    for p in reader.iter_photos(ReadFilter(without_people=True))
                    if not p.real_albums
                }
                keep = [i for i, h in enumerate(matrix.hashes) if h in wanted]
                if not keep:
                    raise _fail("No embedded photo is both untagged and un-albumed.", code=2)
                matrix = Matrix(tuple(matrix.hashes[i] for i in keep), matrix.data[keep])
            chosen_k = k or suggest_k(len(matrix))
            console.print(f"clustering {len(matrix)} photos into k={chosen_k} (seed {seed})")
            clustering = spherical_kmeans(matrix, chosen_k, seed=seed)
            if label:
                loaded = load_encoder(spec.key, device=device, cache_dir=cache_dir_for(data_dir))  # type: ignore[arg-type]
                vectors = loaded.encoder.encode_texts([f"a photo of {v}" for v in SCENE_VOCABULARY])
                clustering = label_clusters(clustering, SCENE_VOCABULARY, vectors)
        except (SemanticUnavailable, DeviceUnavailable) as exc:
            raise _fail(str(exc)) from exc
        finally:
            store.close()
    finally:
        reader.close()

    console.print(
        f"{len(clustering.clusters)} non-empty clusters in {clustering.iterations} "
        f"iterations ({'converged' if clustering.converged else 'hit the iteration cap'})"
    )
    table = Table()
    table.add_column("id", justify="right")
    table.add_column("n", justify="right")
    table.add_column("cohesion", justify="right")
    table.add_column("nearest", justify="right")
    table.add_column("looks like")
    for cluster in clustering.clusters[:show]:
        table.add_row(
            str(cluster.cluster_id),
            str(len(cluster)),
            f"{cluster.cohesion:.3f}",
            f"{cluster.separation:.3f}",
            f"{cluster.label} ({cluster.label_score:.3f})" if cluster.label else "",
        )
    console.print(table)


# ------------------------------------------------------------------- aesthetic


@semantic_app.command("rank")
def semantic_rank(
    album: Annotated[
        str | None, typer.Option("--album", help="Rank within this album only.")
    ] = None,
    k: Annotated[int, typer.Option("-k", "--limit", help="How many to show.")] = 20,
    model: ModelOpt = None,
    spread: Annotated[
        bool,
        typer.Option("--spread/--no-spread", help="Suppress near-duplicates in the picks."),
    ] = True,
    data_dir: DataDir = Path("./data"),
) -> None:
    """Rank photos by predicted aesthetic quality."""
    from rekindle.semantic.aesthetic import (
        AestheticError,
        load_head,
        pick_best,
        score_store,
    )
    from rekindle.semantic.photos import ReadFilter
    from rekindle.semantic.setup import cache_dir_for
    from rekindle.semantic.vectors import load_matrix

    reader = _reader(data_dir)
    try:
        _spec, store = _open_store(data_dir, model, create=False)
        try:
            head = load_head(cache_dir_for(data_dir))
            matrix = load_matrix(store)
            scored = score_store(store, head, matrix=matrix)
            if album:
                wanted = {p.file_hash for p in reader.iter_photos(ReadFilter(albums=(album,)))}
                if not wanted:
                    raise _fail(f"No photos in album {album!r}.", code=2)
                scored = [s for s in scored if s.file_hash in wanted]
            vectors = None
            if spread:
                index = {h: i for i, h in enumerate(matrix.hashes)}
                vectors = {
                    s.file_hash: matrix.data[index[s.file_hash]]
                    for s in scored
                    if s.file_hash in index
                }
            best = pick_best(scored, k, vectors=vectors)
            photos = reader.get_many([s.file_hash for s in best])
        except (SemanticUnavailable, AestheticError) as exc:
            raise _fail(str(exc)) from exc
        finally:
            store.close()
    finally:
        reader.close()

    table = Table(title=f"best of {album}" if album else "best overall")
    table.add_column("score", justify="right")
    table.add_column("date")
    table.add_column("path")
    for s in best:
        photo = photos.get(s.file_hash)
        table.add_row(
            f"{s.score:.3f}",
            (photo.taken_at_local or "")[:10] if photo else "",
            str(photo.path) if photo else s.file_hash,
        )
    console.print(table)


# ------------------------------------------------------------------- face gate


@semantic_app.command("facegate")
def semantic_facegate(
    allow: Annotated[
        list[str] | None,
        typer.Option("--allow", help="A person whose presence does not block. Repeatable."),
    ] = None,
    limit: Annotated[int | None, typer.Option("--limit", help="Check at most N photos.")] = None,
    detect_threshold: Annotated[float, typer.Option("--detect-threshold")] = 0.45,
    gate_threshold: Annotated[float, typer.Option("--gate-threshold")] = 0.15,
    face: Annotated[str | None, typer.Option("--face-model")] = None,
    workers: Annotated[
        int,
        typer.Option("--workers", help="Decode+detect threads. 4 measured best on 6 cores."),
    ] = DEFAULT_GATE_WORKERS,
    untagged_only: Annotated[
        bool,
        typer.Option("--untagged-only/--all", help="Only photos with no face tag at all."),
    ] = True,
    data_dir: DataDir = Path("./data"),
) -> None:
    """Propose photos that contain no face. NEVER publishes anything."""
    from rekindle.semantic.faces import gate_photos, load_detector
    from rekindle.semantic.photos import ReadFilter
    from rekindle.semantic.setup import cache_dir_for

    reader = _reader(data_dir)
    try:
        try:
            detector = load_detector(cache_dir_for(data_dir), face)
        except SemanticUnavailable as exc:
            raise _fail(str(exc)) from exc
        where = ReadFilter(images_only=True, without_people=untagged_only, limit=limit)
        photos = list(reader.iter_photos(where))
        console.print(
            f"{len(photos)} photos through [bold]{detector.spec.key}[/bold] "
            f"({', '.join(detector.providers)})"
        )
        with typer.progressbar(length=max(1, len(photos)), label="detecting") as bar:
            state = {"last": 0}

            def tick(done: int, _total: int) -> None:
                bar.update(done - state["last"])
                state["last"] = done

            report = gate_photos(
                photos,
                detector,
                detect_threshold=detect_threshold,
                gate_threshold=gate_threshold,
                allow_people=tuple(allow or ()),
                workers=workers,
                progress=tick,
            )
    finally:
        reader.close()

    console.print(
        f"\n[bold]{report.eligible}[/bold] proposed as face-free, "
        f"{report.has_face} contain a face, {report.uncertain} uncertain, "
        f"{report.errors} unreadable "
        f"({report.images_per_s:.1f} img/s)"
    )
    console.print(
        "[yellow]This is a PROPOSAL, not a decision.[/yellow] Nothing is published. "
        "Review the queue below before publishing anything."
    )
    table = Table(title="review queue (most in need of a human first)")
    table.add_column("verdict")
    table.add_column("faces", justify="right")
    table.add_column("top", justify="right")
    table.add_column("boxes")
    table.add_column("path")
    for det in report.review_queue()[:40]:
        table.add_row(
            det.verdict.value,
            str(det.face_count),
            f"{det.top_score:.3f}",
            "; ".join(str(b.as_tuple()) for b in det.boxes[:3]),
            str(det.path),
        )
    console.print(table)
    if not report.accounted:
        console.print("[red]![/red] Accounting does not balance - this is a bug.")


@semantic_app.command("orient")
def semantic_orient(
    limit: Annotated[int | None, typer.Option("--limit", help="Examine at most N photos.")] = None,
    margin: Annotated[
        float,
        typer.Option("--margin", help="Evidence a tag-free decode must win by."),
    ] = MIN_MARGIN,
    face: Annotated[str | None, typer.Option("--face-model")] = None,
    workers: Annotated[int, typer.Option("--workers")] = DEFAULT_GATE_WORKERS,
    dry_run: Annotated[
        bool, typer.Option("--dry-run/--write", help="Report without recording anything.")
    ] = False,
    review: Annotated[
        bool,
        typer.Option("--review", help="List the near misses a human should settle. Runs nothing."),
    ] = False,
    data_dir: DataDir = Path("./data"),
) -> None:
    """Find files whose EXIF orientation tag is STALE, and stop applying it.

    A one-time, resumable pass. Only files carrying a 90/270-degree tag are
    decoded at all, and only the hypothesis "this tag is stale" is ever
    tested - it can never invent a rotation for a file whose tag says upright.
    """
    from rekindle.db import PhotoStore
    from rekindle.meta import orientation
    from rekindle.semantic.faces import load_detector
    from rekindle.semantic.setup import cache_dir_for

    db_path = _db_path(data_dir)
    if not db_path.is_file():
        raise _fail(f"No index at {db_path}. Run `rekindle index <folder>` first.", code=2)
    if review:
        # Reads the evidence the pass already stored. No detector is loaded,
        # so this works on a machine with no model and no `semantic` extra -
        # which is the whole point of a queue a HUMAN works through.
        _print_review_queue(db_path, limit)
        return
    try:
        detector = load_detector(cache_dir_for(data_dir), face)
    except SemanticUnavailable as exc:
        # The whole feature is optional. Without a model the library keeps
        # decoding by EXIF alone, which is what it has always done.
        raise _fail(f"{exc}\nWithout it, rekindle decodes by the EXIF tag alone.") from exc

    with PhotoStore(db_path) as store:
        todo = list(store.iter_unoriented())
        if limit is not None:
            todo = todo[:limit]
        if not todo:
            counts = store.orientation_counts()
            console.print(
                "[green]Nothing to do[/green] - every photo has been examined. "
                f"{counts['examined']} examined, "
                f"{counts['ignoring_exif']} decoding with the tag IGNORED."
            )
            return
        console.print(
            f"Examining {len(todo)} photos with [bold]{detector.spec.key}[/bold] "
            f"({', '.join(detector.providers)}). Only 90/270-tagged files are decoded."
        )
        target = _PassTarget(store, todo, write=not dry_run)
        with typer.progressbar(length=max(1, len(todo)), label="examining") as bar:
            state = {"last": 0}

            def tick(done: int, _total: int) -> None:
                bar.update(done - state["last"])
                state["last"] = done

            report = orientation.run_orientation(
                target, detector, workers=workers, min_margin=margin, progress=tick
            )

    console.print(
        f"\n[bold]{report.tag_ignored}[/bold] files have a STALE orientation tag and will now "
        f"decode without it.\n"
        f"  {report.no_tag} had no axis-swapping tag (never examined further), "
        f"{report.tag_trusted} had one the picture agrees with,\n"
        f"  {report.no_face} had one but NO FACE to argue from - unreachable by this method - "
        f"and {report.errors} could not be read.\n"
        f"  {report.examined} examined in {report.elapsed_s:.0f}s."
    )
    if not report.accounted:
        console.print("[red]![/red] Accounting does not balance - this is a bug.")
    if dry_run:
        console.print("[yellow]--dry-run: nothing was recorded.[/yellow]")
    if report.corrected:
        table = Table(title="every rotation applied, and the evidence for it")
        table.add_column("tag")
        table.add_column("evidence with tag", justify="right")
        table.add_column("without", justify="right")
        table.add_column("margin", justify="right")
        table.add_column("best face", justify="right")
        table.add_column("file")
        for path, v in sorted(report.corrected, key=lambda c: -c[1].margin):
            table.add_row(
                str(v.tag),
                f"{v.e_tag:.3f}",
                f"{v.e_raw:.3f}",
                f"{v.margin:+.3f}",
                f"{v.m_raw:.3f}",
                Path(path).name,
            )
        console.print(table)
        console.print(
            f"[yellow]{len(report.invalidated)} photos changed their decoded PIXELS.[/yellow]\n"
            "  Their fingerprints have been cleared - run `rekindle fingerprint` to recompute\n"
            "  them (this also repairs the stored width/height, which were swapped).\n"
            "  Their EMBEDDINGS are in a separate store with no invalidation path: re-run\n"
            "  `rekindle semantic embed` after deleting that store, or those vectors stay\n"
            "  sideways. See docs/known-limitations.md."
        )


def _print_review_queue(db_path: Path, limit: int | None) -> None:
    """The near misses: the tag is contradicted, but not decisively."""
    from rekindle.db import PhotoStore
    from rekindle.meta import orientation

    with PhotoStore(db_path) as store:
        counts = store.orientation_counts()
        rows = store.orientation_review_queue(limit or 40)
    if not counts["examined"]:
        console.print(
            "[yellow]Nothing examined yet.[/yellow] Run `rekindle semantic orient` first."
        )
        return
    console.print(
        f"{counts['ignoring_exif']} files were corrected outright. Below are "
        f"{len(rows)} that were NOT, though the picture leans against the tag. "
        "They failed one of two\nguards, and the `why` column says which: the "
        f"margin did not reach {MIN_MARGIN}, or nothing in the tag-free decode "
        "was confident\nenough to count as a face. Neither is evidence the tag is "
        "RIGHT - only that the arithmetic would not commit."
    )
    table = Table(title="orientation review queue (closest call first)")
    table.add_column("tag")
    table.add_column("with tag", justify="right")
    table.add_column("without", justify="right")
    table.add_column("margin", justify="right")
    table.add_column("best face", justify="right")
    table.add_column("why")
    table.add_column("file")
    for path, ev in rows:
        table.add_row(
            str(ev.get("tag")),
            f"{ev.get('e_tag', 0):.3f}",
            f"{ev.get('e_raw', 0):.3f}",
            f"{ev.get('margin', 0):+.3f}",
            f"{ev.get('m_raw', 0):.3f}",
            "margin" if ev.get("reason") == orientation.MARGIN_TOO_SMALL else "no face",
            path.name,
        )
    console.print(table)
    console.print(
        "Nothing here has been changed. To act on one, re-run with a lower "
        "[bold]--margin[/bold] after deciding the band is worth it - "
        "see docs/known-limitations.md for what that costs."
    )


class _PassTarget:
    """What the pass writes through: applies `--limit`, honours `--dry-run`.

    Both are properties of THIS INVOCATION, not of the index, so they live
    here and `PhotoStore` keeps its one honest answer to "what still needs
    examining?". `--dry-run` swallowing the write at this seam is also the
    only version of it that cannot half-apply: the pass itself has no branch
    in it, so there is no path on which some verdicts are written and others
    are not.
    """

    def __init__(self, store, todo, *, write: bool) -> None:
        self._store = store
        self._todo = todo
        self._write = write

    def iter_unoriented(self):
        return iter(self._todo)

    def set_orientations(self, rows):
        if not self._write:
            return sum(1 for _ in rows)
        return self._store.set_orientations(rows)


def register(app: typer.Typer) -> None:
    """Attach the semantic verbs to the main CLI."""
    app.add_typer(semantic_app, name="semantic")
