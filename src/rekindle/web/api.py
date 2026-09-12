"""What each route does, with no knowledge that HTTP exists.

Every function here takes a `Workshop` and plain arguments and returns plain
data or raises `ApiError`. `server.py` does nothing but parse a request into
those arguments and serialise what comes back, which is what lets the whole
API be tested by calling functions - no socket, no browser, no event loop.

The build is a GENERATOR of events rather than a return value, because that is
the shape the streaming requirement actually has: a prompt search over 18,201
vectors runs one CLIP query per visual tag, and a user watching a blank page
for all of them has no idea whether it is working. The generator yields a
stage as each one completes; `server.py` turns those into `text/event-stream`
frames and the same generator is consumed synchronously by the tests.
"""

from __future__ import annotations

import secrets
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from rekindle.memory import engine
from rekindle.memory.history import KIND_MEMORY, memory_id
from rekindle.memory.recipes import Offer, registered
from rekindle.memory.render.music import available_tracks
from rekindle.memory.spec import safe_slug
from rekindle.web import renderer
from rekindle.web.draft import (
    ORIGIN_PROMPT,
    ORIGIN_RECIPE,
    Draft,
    DraftError,
    Origin,
    draft_from_selection,
    skip_explanation,
)
from rekindle.web.library import Library, LibraryError
from rekindle.web.renderer import RenderOptions, RenderResult, render_spec


class ApiError(RuntimeError):
    """A refusal with an HTTP status and a sentence for the user."""

    def __init__(self, message: str, status: int = 400, hint: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.hint = hint


@dataclass
class Session:
    session_id: str
    draft: Draft
    note: str = ""
    last_render: RenderResult | None = None
    spec_path: Path | None = None


class Workshop:
    """Sessions, and the one place a draft is created or found.

    Sessions live in memory and die with the process. That is deliberate: the
    durable artefact is `memory.json`, which the render step writes and which
    `rekindle render` rebuilds from - so the thing worth keeping is on disk in
    a format the CLI already understands, and there is no second, private
    database of half-finished edits to migrate or leak.
    """

    def __init__(
        self,
        library: Library,
        *,
        out_dir: Path,
        music_dir: Path,
        max_shots: int = engine.DEFAULT_MAX_SHOTS,
    ) -> None:
        self.library = library
        self.out_dir = out_dir
        self.music_dir = music_dir
        self.max_shots = max_shots
        self._sessions: dict[str, Session] = {}
        self._progress: dict[str, dict] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise ApiError("That editing session is gone. Build the memory again.", status=404)
        return session

    def put(self, draft: Draft, note: str = "") -> Session:
        session = Session(session_id=secrets.token_urlsafe(9), draft=draft, note=note)
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    # ---- the live state of a render that is happening right now
    #
    # A render is a blocking POST that takes over a minute, so the page cannot
    # learn anything from the response it is waiting on. It polls this instead,
    # on a second connection, which `ThreadingHTTPServer` serves concurrently.
    # Hence the lock: the writer is the rendering thread and the readers are
    # the polls.

    def set_progress(self, session_id: str, payload: dict) -> None:
        with self._lock:
            self._progress[session_id] = payload

    def progress_of(self, session_id: str) -> dict | None:
        with self._lock:
            return self._progress.get(session_id)


# --------------------------------------------------------------------------
# status and offers


def status(workshop: Workshop) -> dict:
    library = workshop.library
    payload: dict = {
        "state": library.state,
        "error": library.error,
        "data_dir": str(library.data_dir),
        "out_dir": str(workshop.out_dir),
        "semantic_installed": library.semantic_available(),
        "semantic_note": library.semantic_note,
        "semantic_error": library.semantic_error,
        "max_shots": workshop.max_shots,
        "music": [t.name for t in available_tracks(workshop.music_dir)],
    }
    if library.index is not None:
        report = library.index.report
        payload["photos"] = report.allowed
        payload["withheld"] = report.excluded
        payload["withheld_by_reason"] = dict(sorted(report.by_reason.items()))
        payload["unfingerprinted"] = sum(
            1 for p in library.index.all() if p.meta.phash is None and not p.meta.phash_error
        )
        # The oldest and newest year with a photograph in it. The page prints
        # it under the wordmark, and "19,318 photographs, 2000-2026" is a
        # shorter and truer description of a library than any label rekindle
        # could invent for it.
        years = library.index.years()
        payload["span"] = [years[0], years[-1]] if years else []
    return payload


def offers(workshop: Workshop, *, recipe: str | None = None, limit: int = 200) -> dict:
    """Every memory this library could produce - `rekindle memories`, as data.

    Built from `recipe.offers(index)` through the registry, so a recipe added
    tomorrow appears here with no change to this file, and a recipe that never
    registers (the prompt path) correctly does not.
    """
    index = _index(workshop)
    dismissed = workshop.library.with_state(lambda state: state.dismissed_memory_ids())
    cooling = workshop.library.with_state(lambda state: state.cooling())
    rows = []
    for definition in registered():
        if recipe and definition.name != recipe:
            continue
        for offer in definition.offers(index)[:limit]:
            rows.append(
                {
                    "recipe": offer.recipe,
                    "key": offer.key,
                    "title": offer.title,
                    "subtitle": offer.subtitle,
                    "size": offer.size,
                    "dismissed": offer.memory_id in dismissed,
                    "cooling": offer.memory_id in cooling,
                }
            )
    return {
        "offers": rows,
        "recipes": [r.name for r in registered()],
        "album_merges": dict(getattr(index, "album_merges", {})),
    }


# --------------------------------------------------------------------------
# what to type into an empty box


#: Enough evidence to put a suggestion in front of someone. A name that
#: appears in four photographs is a face tag somebody made once, not a person
#: this library has a memory of, and offering it teaches the wrong thing.
MIN_SUGGESTION_PHOTOS = 60
MIN_SUGGESTION_YEARS = 3

#: How many of each kind to keep before interleaving. Small on purpose: the
#: empty state is a lesson in what a prompt looks like, and eight examples
#: teach it better than forty.
PER_KIND = 2

MONTH_NAMES = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)


def suggestions(workshop: Workshop, *, limit: int = 8) -> dict:
    """Prompts worth typing, derived from THIS library.

    Nobody knows what to type into an empty box, and a canned list teaches
    nothing about the library in front of them. Every row here is counted off
    `MemoryIndex`, so a suggestion can only name people, months, albums and
    date windows the guardrails already admit - the same chokepoint the
    thumbnails and the search go through. A name on the exclusion list cannot
    be suggested, because `index.people_counts()` has never heard of it.

    Each row carries the count it was chosen on. That is not decoration: it is
    the difference between "rekindle thinks you like mountains" and "there are
    4,196 photographs of Paramita here, over eighteen years".
    """
    index = _index(workshop)
    rows: list[list[dict]] = [
        _suggest_festivals(index),
        _suggest_people(index),
        _suggest_months(index),
        _suggest_albums(index),
        _suggest_scenes(workshop),
    ]
    # `weight` is how each list was ranked and is nobody else's business;
    # a number in the payload is a number the page will eventually render.
    return {
        "suggestions": [
            {k: v for k, v in row.items() if k != "weight"} for row in _interleave(rows, limit)
        ]
    }


def _interleave(groups: list[list[dict]], limit: int) -> list[dict]:
    """One from each kind, then a second from each, until full.

    Round-robin rather than concatenation, because the point of the empty
    state is to show that a prompt can be a festival OR a person OR a month -
    and eight person rows in a row say the opposite.
    """
    out: list[dict] = []
    for rank in range(PER_KIND):
        for group in groups:
            if rank < len(group) and len(out) < limit:
                out.append(group[rank])
    return out


def _years_and_count(index, photos: list) -> tuple[int, int]:
    """How many distinct years, and how many photographs.

    `index.years_present` rather than a comprehension here, so a suggestion
    counts years the same way the recipes that build the memory do - local
    capture time, not UTC, and skipping the undated.
    """
    return len(index.years_present(photos)), len(photos)


def _suggest_festivals(index) -> list[dict]:
    """Festivals from the corpus whose MONTH WINDOW this library has years of.

    A photograph in October is not a photograph of Durga Puja and this never
    says it is - the note names the window that would be searched, which is
    exactly what the prompt path does with the festival's visual tags. So the
    suggestion is honest about being a place to look rather than a finding.
    """
    from rekindle.memory import festivals as festivals_mod

    rows = []
    seen_windows: set[tuple[int, ...]] = set()
    for festival in festivals_mod.all_festivals():
        # ONE festival per window. Kali Puja and Jagaddhatri Puja share
        # October-November, so counting photographs ranks them identically and
        # the empty state offered both - two rows carrying one fact. Corpus
        # order breaks the tie, which is the only ranking that exists: the
        # month counts genuinely cannot tell these two apart.
        if festival.window in seen_windows:
            continue
        photos = [p for month in festival.window for p in index.by_month(month)]
        years, count = _years_and_count(index, photos)
        if years < MIN_SUGGESTION_YEARS or count < MIN_SUGGESTION_PHOTOS:
            continue
        seen_windows.add(festival.window)
        window = "–".join(MONTH_NAMES[m - 1][:3] for m in festival.window)
        rows.append(
            {
                "text": f"{festival.names[0]} over the years",
                "note": f"{years} years · {window}",
                "kind": "festival",
                "weight": count,
            }
        )
    rows.sort(key=lambda r: -r["weight"])
    return rows


def _suggest_people(index) -> list[dict]:
    rows = []
    for name, count in index.people_counts().most_common(12):
        years, _ = _years_and_count(index, index.by_person(name))
        if years < MIN_SUGGESTION_YEARS or count < MIN_SUGGESTION_PHOTOS:
            continue
        rows.append(
            {
                "text": f"{name.lower()} over the years",
                "note": f"{years} years · {count:,} photographs",
                "kind": "person",
                "weight": count,
            }
        )
    return rows[:6]


def _suggest_months(index) -> list[dict]:
    """The months this library actually returns to, most-recurring first.

    Sorted on YEARS before photographs. A single 900-photo wedding in April
    makes April the biggest month and the dullest memory; twelve separate
    Augusts is the thing worth showing.
    """
    rows = []
    for month in index.months():
        photos = index.by_month(month)
        years, count = _years_and_count(index, photos)
        if years < MIN_SUGGESTION_YEARS or count < MIN_SUGGESTION_PHOTOS:
            continue
        rows.append(
            {
                "text": f"every {MONTH_NAMES[month - 1]}",
                "note": f"{years} years · {count:,} photographs",
                "kind": "month",
                "weight": (years, count),
            }
        )
    rows.sort(key=lambda r: r["weight"], reverse=True)
    return rows[:6]


def _suggest_albums(index) -> list[dict]:
    """Albums, filtered by the SAME rule `album_story` offers them under.

    `albums.presentable` is what keeps "Photos from 2019" out of the recipe
    list, and raw `album_counts()` does not apply it - suggesting a Takeout
    auto-album teaches someone that a prompt is a filing-system name. Sharing
    the predicate means an album this page suggests is always an album the
    engine would build a memory from.
    """
    from rekindle.memory import albums as albums_mod

    rows = []
    for name, count in index.album_counts().most_common():
        if count < MIN_SUGGESTION_PHOTOS or not albums_mod.presentable(name):
            continue
        rows.append(
            {
                "text": name.lower(),
                "note": f"your album · {count:,} photographs",
                "kind": "album",
                "weight": count,
            }
        )
        if len(rows) >= 6:
            break
    return rows


def _suggest_scenes(workshop: Workshop) -> list[dict]:
    """Scenery concepts - and NO count, because there is honestly not one.

    Whether this library has mountains in it is not a question the index can
    answer; it takes a CLIP search per concept over every embedding, which is
    seconds of work to draw an empty state. So these say what they are - a
    description to search for - and appear only when the search that would
    run them is installed. Suggesting "mountains" to someone who would get
    "install the semantic extra" is a worse empty state than nine rows.
    """
    if not workshop.library.semantic_available():
        return []
    from rekindle.memory import scenery as scenery_mod

    return [
        {
            "text": _scene_phrase(concept),
            "note": "a description to search for",
            "kind": "scene",
            "weight": 0,
        }
        for concept in scenery_mod.all_concepts()[:6]
    ]


def _scene_phrase(concept) -> str:
    """The name a person would type, not the corpus head-word.

    `names[0]` is the dictionary entry - "mountain", "flower", "sea" - and
    nobody types those. The title is the phrase ("The sea", "After dark"), so
    prefer it when it is itself a name the matcher accepts, then the plural,
    then give up and use the head-word. Every branch returns something in
    `names`, so a suggestion can never be a prompt `scenery.match` refuses.
    """
    names = [n.lower() for n in concept.names]
    for candidate in (concept.title.lower(), names[0] + "s"):
        if candidate in names:
            return candidate
    return names[0]


# --------------------------------------------------------------------------
# building, as a stream of events


def build_events(
    workshop: Workshop,
    *,
    recipe: str | None = None,
    key: str | None = None,
    prompt: str | None = None,
) -> Iterator[dict]:
    """Yield `{"event": ..., "data": ...}` until `ready` or `error`.

    The last event is always `ready` or `error`; the browser closes the
    EventSource on either, which is what stops it reconnecting in a loop when
    the server closes a finished stream.
    """
    try:
        if prompt:
            yield from _build_prompt(workshop, prompt)
        elif recipe and key is not None:
            yield from _build_recipe(workshop, recipe, key)
        else:
            raise ApiError("Name a recipe and a key, or give a prompt.")
    except ApiError as exc:
        yield {"event": "error", "data": {"message": str(exc), "hint": exc.hint}}
    except LibraryError as exc:
        yield {"event": "error", "data": {"message": str(exc), "hint": ""}}


def _build_recipe(workshop: Workshop, recipe: str, key: str) -> Iterator[dict]:
    index = _index(workshop)
    yield _stage("offers", f"looking up {recipe}:{key}")
    definition = next((r for r in registered() if r.name == recipe), None)
    if definition is None:
        raise ApiError(f"No recipe called {recipe!r}.", status=404)
    offer = next((o for o in definition.offers(index) if o.key == key), None)
    if offer is None:
        raise ApiError(f"No memory for recipe={recipe!r} key={key!r} in this library.", status=404)

    yield _stage("select", f"{definition.name} is choosing candidates")
    selection = definition.select(index, offer)
    if selection is None or not selection.photos:
        raise ApiError(f"{offer.title} has no candidates at all - nothing to edit.", status=404)

    yield _stage("guardrails", f"{len(selection.photos)} candidates through the guardrails")
    origin = Origin(
        kind=ORIGIN_RECIPE,
        recipe=offer.recipe,
        key=offer.key,
        title=offer.title,
        albums=selection.facts.albums,
        title_substantiated=selection.facts.title_substantiated,
    )
    draft = draft_from_selection(index, origin, offer, selection, max_shots=workshop.max_shots)
    yield from _finish(workshop, draft, note=skip_explanation(draft.report))


def _build_prompt(workshop: Workshop, text: str) -> Iterator[dict]:
    """The prompt path, streamed one visual tag at a time.

    Nothing about selection is re-decided here. `prompt.parse`,
    `tags.resolve` and `prompt.build_selection` are called exactly as
    `memory/cli.py:prompt_cmd` calls them; the only addition is that the
    injected retriever is wrapped so each tag's search emits a progress event.
    That wrapping is safe precisely because retrieval is injected - the
    consensus arithmetic never sees the wrapper.

    The LLM plausibility judge is NOT run. It exists to avoid spending a
    render on an incoherent query, and in this window the user sees the
    photographs before anything is rendered - which is the working verifier
    the judge was standing in for. `rekindle memory` keeps it.
    """
    from rekindle.memory import prompt as prompt_mod
    from rekindle.memory import tags as tags_mod
    from rekindle.memory.llm import LLMUnavailable

    index = _index(workshop)
    if not prompt_mod.normalise(text):
        raise ApiError("An empty prompt cannot build anything.")

    yield _stage("parse", "reading your words")
    query = prompt_mod.parse(text, index)

    try:
        generator = tags_mod.generator_from_env()
    except LLMUnavailable:
        generator = None

    yield _stage("tags", "turning the prompt into visual descriptions")
    resolution = tags_mod.resolve(
        query,
        data_dir=workshop.library.data_dir,
        generator=generator,
        people=list(index.people_counts()),
    )
    if resolution.source == tags_mod.SOURCE_MODEL:
        tags_mod.write_cache_entry(workshop.library.data_dir, query.text, resolution.tags)
    yield {
        "event": "tags",
        "data": {
            "tags": list(resolution.tags),
            "source": resolution.source,
            "festival": resolution.festival,
            "months": list(resolution.months),
            "rejected": list(resolution.rejected),
            "weak": resolution.source == tags_mod.SOURCE_PROMPT,
        },
    }

    retrieve = workshop.library.retriever()
    progress: list[dict] = []
    total = max(1, len(resolution.tags))
    seen = {"n": 0}

    def watched(tag_text: str, k: int) -> list[tuple[str, float]]:
        hits = retrieve(tag_text, k)
        seen["n"] += 1
        progress.append({"tag": tag_text, "hits": len(hits), "done": seen["n"], "of": total})
        return hits

    yield _stage("search", f"searching {total} descriptions over the embedded library")
    build = prompt_mod.build_selection(
        index,
        query,
        resolution.tags,
        watched,
        months=resolution.months,
        festival=resolution.festival,
        known_words=_festival_names(resolution, query),
    )
    for step in progress:
        yield {"event": "searched", "data": step}

    yield {
        "event": "found",
        "data": {
            "seed_days": [{"day": s.iso, "hits": s.hits, "tags": s.tags} for s in build.seed_days],
            "pool": build.pool,
            "albums": list(build.albums),
            "unmatched": list(build.unmatched),
            "agreement": build.agreement,
        },
    }
    if build.selection is None:
        raise ApiError(
            "No capture day had enough agreement between those descriptions, "
            "so there is nothing to edit.",
            status=404,
            hint=(
                "Tag agreement does not say whether the concept is in your library - "
                "it is measured not to. Try different words."
            ),
        )

    origin = Origin(
        kind=ORIGIN_PROMPT,
        recipe=prompt_mod.RECIPE,
        key=query.text,
        title=query.text,
        prompt=text,
        albums=build.albums,
        title_substantiated=False,
    )
    offer = Offer(recipe=prompt_mod.RECIPE, key=query.text, title=query.text)
    draft = draft_from_selection(
        index, origin, offer, build.selection, max_shots=workshop.max_shots
    )
    note = skip_explanation(draft.report) or (
        "No automated check can tell whether these photos match your words. "
        "Look at them before you keep this."
    )
    yield from _finish(workshop, draft, note=note)


def _festival_names(resolution, query) -> tuple[str, ...]:
    if not resolution.festival:
        return ()
    from rekindle.memory import festivals as festivals_mod

    matched = festivals_mod.match(query.subject)
    return matched.names if matched else ()


def _finish(workshop: Workshop, draft: Draft, *, note: str) -> Iterator[dict]:
    session = workshop.put(draft, note=note)
    for file_hash in draft.order:
        yield {"event": "shot", "data": {"file_hash": file_hash}}
    yield {"event": "ready", "data": session_state(workshop, session)}


def _stage(name: str, message: str) -> dict:
    return {"event": "stage", "data": {"stage": name, "message": message}}


# --------------------------------------------------------------------------
# session state and edits


def session_state(workshop: Workshop, session: Session) -> dict:
    draft = session.draft
    spec = draft.to_spec()
    stratum = draft.stratum
    return {
        "session_id": session.session_id,
        "note": session.note,
        "origin": {
            "kind": draft.origin.kind,
            "recipe": draft.origin.recipe,
            "key": draft.origin.key,
            "title": draft.origin.title,
            "prompt": draft.origin.prompt,
            "command": draft.origin.command(),
        },
        "title": spec.title,
        "subtitle": spec.subtitle,
        "public_safe": spec.public_safe,
        "order": list(draft.order),
        "engine_order": list(draft.engine_order),
        "removed": list(draft.removed),
        "added": list(draft.added),
        "max_shots": draft.max_shots,
        "pace": {
            "frame_ms": draft.pace.frame_ms,
            "title_ms": draft.pace.title_ms,
            "music": draft.pace.music.name if draft.pace.music else None,
        },
        "candidates": [c.to_json() for c in draft.catalogue()],
        "guardrails": {
            "considered": draft.composition.considered,
            "kept": draft.composition.kept,
            "dropped": dict(sorted(draft.composition.dropped.items())),
            "examples": dict(draft.composition.examples),
            "orientation": str(draft.composition.orientation),
            "collapsed": draft.report.deduped,
            "skipped": dict(draft.report.skipped),
        },
        "strata": None
        if stratum is None
        else {
            "dimension": stratum.dimension,
            "offered": stratum.offered,
            "surviving": stratum.surviving,
            "used": stratum.used,
            "lost_to_gates": list(stratum.keys_lost_to_gates),
            "without_slots": list(stratum.keys_without_slots),
        },
        "facts": spec.facts.to_json(),
        "last_render": session.last_render.to_json() if session.last_render else None,
        "reproduce": _reproduce(session),
    }


def _reproduce(session: Session) -> list[str] | None:
    if session.spec_path is None:
        return None
    return session.draft.reproduce_command(session.spec_path)


def edit(workshop: Workshop, session_id: str, payload: dict) -> dict:
    """Apply one edit. Every op routes into `Draft`, which is the guardrail."""
    session = workshop.get(session_id)
    draft = session.draft
    op = payload.get("op")
    try:
        if op == "remove":
            draft.remove(_need(payload, "file_hash"))
        elif op == "add":
            at = payload.get("at")
            draft.add(_need(payload, "file_hash"), at=int(at) if at is not None else None)
        elif op == "swap":
            draft.swap(_need(payload, "out"), _need(payload, "in"))
        elif op == "reorder":
            order = payload.get("order")
            if not isinstance(order, list) or not all(isinstance(h, str) for h in order):
                raise ApiError("A reorder needs a list of file hashes.")
            draft.reorder(order)
        elif op == "pace":
            music = payload.get("music")
            track = None
            clear = music is not None and music == ""
            if music:
                track = _resolve_track(workshop, music)
            draft.set_pace(
                frame_ms=_maybe_int(payload, "frame_ms"),
                title_ms=_maybe_int(payload, "title_ms"),
                music=track,
                clear_music=clear,
            )
        else:
            raise ApiError(f"Unknown edit {op!r}.")
    except DraftError as exc:
        raise ApiError(str(exc)) from exc
    return session_state(workshop, session)


def _resolve_track(workshop: Workshop, name: str) -> Path:
    """A music track BY NAME, from the music folder and nowhere else.

    The browser sends a file name, never a path. Resolving it against the
    directory listing rather than joining it means `../../etc/passwd` is not a
    track, and neither is any absolute path a page could be tricked into
    sending.
    """
    for track in available_tracks(workshop.music_dir):
        if track.name == name:
            return track
    raise ApiError(f"No track called {name!r} in {workshop.music_dir}.", status=404)


def _maybe_int(payload: dict, field: str) -> int | None:
    value = payload.get(field)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ApiError(f"{field} must be a whole number.") from exc


def _need(payload: dict, field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ApiError(f"{field} is required.")
    return value


# --------------------------------------------------------------------------
# finding more photos


def search(workshop: Workshop, *, query: str, mode: str = "metadata", limit: int = 60) -> dict:
    """Find photos to add. Both modes end at `MemoryIndex.resolve_many`.

    That is the whole guardrail story for this route: a semantic hit on an
    archived photo is a hash the index does not hold, so it is dropped on
    resolution with no filtering code here to forget.
    """
    index = _index(workshop)
    if mode == "semantic":
        try:
            retrieve = workshop.library.retriever()
        except LibraryError as exc:
            raise ApiError(str(exc), status=501) from exc
        raw = retrieve(query, limit * 3)
        photos = index.resolve_many([h for h, _ in raw])[:limit]
        scores = dict(raw)
        return {
            "mode": "semantic",
            "hits": [
                {"file_hash": p.file_hash, "score": scores.get(p.file_hash, 0.0), "why": "search"}
                for p in photos
            ],
        }
    hits = workshop.library.search_metadata(query, limit=limit)
    return {
        "mode": "metadata",
        "hits": [{"file_hash": h.file_hash, "score": h.score, "why": h.why} for h in hits],
    }


def neighbours(workshop: Workshop, *, file_hash: str, limit: int = 40) -> dict:
    """The photos nearest this one in the embedding space - "more like this".

    This is the scene-cluster control, asked one photo at a time. `rekindle
    semantic cluster` partitions the whole library with k-means but does not
    STORE the partition anywhere, so there is no cluster id to look up; a
    nearest-neighbour query over the same vectors answers the same question
    for the photo in front of the user and costs one cosine pass instead of
    twenty k-means iterations over 18,201 vectors.
    """
    index = _index(workshop)
    if index.get(file_hash) is None:
        raise ApiError("That photo is not in this library.", status=404)
    try:
        similar = workshop.library.similar()
    except LibraryError as exc:
        raise ApiError(str(exc), status=501) from exc
    raw = similar(file_hash, limit * 3)
    scores = dict(raw)
    photos = index.resolve_many([h for h, _ in raw])[:limit]
    return {
        "hits": [
            {"file_hash": p.file_hash, "score": scores.get(p.file_hash, 0.0), "why": "similar"}
            for p in photos
        ]
    }


def capture_day(workshop: Workshop, *, file_hash: str) -> dict:
    """Everything taken on the same local day as this photo.

    No model, no extra, and it is the control that matters most for a burst:
    "give me the rest of that afternoon". `MemoryIndex.by_date` is the same
    call the prompt path uses to expand a seed day, so the UI and the engine
    mean the same thing by "the same day".
    """
    index = _index(workshop)
    photo = index.get(file_hash)
    if photo is None or photo.meta.taken_at_local is None:
        raise ApiError("That photo is not in this library.", status=404)
    local = photo.meta.taken_at_local
    same = index.by_date(local.year, local.month, local.day)
    return {
        "day": local.date().isoformat(),
        "hits": [
            {"file_hash": p.file_hash, "score": 0.0, "why": "same day"}
            for p in sorted(same, key=lambda p: (p.meta.taken_at_utc, p.file_hash))
        ],
    }


# --------------------------------------------------------------------------
# rendering and dismissing


#: What share of a render each phase takes. MEASURED, not apportioned: a
#: 24-shot memory from this library, rendered twice on the reference machine
#: (Windows, ffmpeg on PATH), took 82-88s and split
#:
#:     preview 1.9%  webp 10.5%  gif 9.7%  video 8.8%  stills 22.1%  mp4 47.1%
#:
#: The first guess at these numbers had the MP4 at 15%. It is nearly half the
#: render, which is the difference between a bar that crawls and a bar that
#: jumps to 90% and stops - so they are measured and the measurement is
#: recorded here to be re-run when the renderer changes.
#:
#: `stills` and the two frame passes count their own items, so progress within
#: them is real. `mp4` is one ffmpeg subprocess and is NOT instrumented: the
#: bar holds at the start of that phase and the page says so rather than
#: inventing movement. Weights are renormalised over whichever phases will
#: actually run, so `no_mp4` still ends at 1.0.
PHASE_WEIGHTS = {
    renderer.PHASE_SPEC: 0.001,
    renderer.PHASE_PREVIEW: 0.019,
    renderer.PHASE_WEBP: 0.105,
    renderer.PHASE_GIF: 0.097,
    renderer.PHASE_VIDEO: 0.088,
    renderer.PHASE_STILLS: 0.221,
    renderer.PHASE_MP4: 0.471,
}

#: The phases that do not run when the caller skips the MP4.
VIDEO_PHASES = (renderer.PHASE_VIDEO, renderer.PHASE_STILLS, renderer.PHASE_MP4)

#: What each phase is called on the page. Here rather than in `app.js` so the
#: names and the constants they describe cannot drift apart in two files.
PHASE_LABELS = {
    renderer.PHASE_SPEC: "writing the spec",
    renderer.PHASE_PREVIEW: "composing the preview",
    renderer.PHASE_WEBP: "encoding the animation",
    renderer.PHASE_GIF: "encoding the GIF",
    renderer.PHASE_VIDEO: "decoding every shot at full size",
    renderer.PHASE_STILLS: "writing full-size frames",
    renderer.PHASE_MP4: "ffmpeg is encoding the film",
    renderer.PHASE_DONE: "done",
}


def _phase_fraction(phase: str, done: int, total: int, *, no_mp4: bool) -> float:
    """Where this phase's `done of total` lands on a 0..1 bar."""
    if phase == renderer.PHASE_DONE:
        return 1.0
    order = [p for p in PHASE_WEIGHTS if not (no_mp4 and p in VIDEO_PHASES)]
    scale = sum(PHASE_WEIGHTS[p] for p in order) or 1.0
    if phase not in order:
        return 1.0
    before = sum(PHASE_WEIGHTS[p] for p in order[: order.index(phase)])
    within = (done / total) if total else 0.0
    return min(1.0, (before + PHASE_WEIGHTS[phase] * within) / scale)


def render_progress(workshop: Workshop, session_id: str) -> dict:
    """How far the render for this session has got, right now.

    Polled by the page while the render POST is still open. Returns a
    not-started shape rather than raising when nothing has been rendered, so a
    poll that arrives a beat before the render begins is not an error the user
    has to see.
    """
    workshop.get(session_id)  # 404 for a session that is gone, as everywhere else
    live = workshop.progress_of(session_id)
    if live is None:
        return {"running": False, "phase": "", "label": "", "fraction": 0.0, "done": 0, "total": 0}
    return dict(live)


def render(workshop: Workshop, session_id: str, payload: dict | None = None) -> dict:
    payload = payload or {}
    session = workshop.get(session_id)
    draft = session.draft
    if not draft.order:
        raise ApiError("An empty memory cannot be rendered. Add a photo first.")

    folder = workshop.out_dir / (
        f"{date.today().isoformat()}-{draft.origin.recipe}-{safe_slug(draft.origin.key)}"
    )
    spec = draft.to_spec()
    options = RenderOptions(
        frame_ms=draft.pace.frame_ms,
        title_ms=draft.pace.title_ms,
        music=draft.pace.music,
        no_mp4=bool(payload.get("no_mp4", False)),
    )

    def report(phase: str, done: int, total: int) -> None:
        workshop.set_progress(
            session_id,
            {
                "running": phase != renderer.PHASE_DONE,
                "phase": phase,
                "label": PHASE_LABELS.get(phase, phase),
                "fraction": _phase_fraction(phase, done, total, no_mp4=options.no_mp4),
                "done": done,
                "total": total,
            },
        )

    report(renderer.PHASE_SPEC, 0, 0)
    try:
        result = render_spec(spec, _index(workshop), folder, options, progress=report)
    finally:
        # Whatever happened, the page must stop being told a render is in
        # flight - otherwise a failed render leaves a bar spinning forever.
        report(renderer.PHASE_DONE, 0, 0)
    session.last_render = result
    session.spec_path = folder / "memory.json"

    # Recorded for the same reason `rekindle memory` records it: the cooldown
    # is what stops a memory the user has already seen being offered again
    # next week, and a memory built in this window has certainly been seen.
    workshop.library.with_state(
        lambda state: state.record_surfaced(memory_id(spec.recipe, spec.key), title=spec.title)
    )
    return session_state(workshop, session)


def dismiss(workshop: Workshop, session_id: str) -> dict:
    """Never offer this memory again. The same permanent act as the CLI's."""
    session = workshop.get(session_id)
    draft = session.draft
    workshop.library.with_state(
        lambda state: state.dismiss(KIND_MEMORY, memory_id(draft.origin.recipe, draft.origin.key))
    )
    session.note = (
        f"Dismissed {draft.origin.recipe}:{draft.origin.key}. It will never be offered "
        "again. Undo with `rekindle undismiss`."
    )
    return session_state(workshop, session)


def _index(workshop: Workshop):
    try:
        return workshop.library.require_index()
    except LibraryError as exc:
        raise ApiError(str(exc), status=503) from exc
