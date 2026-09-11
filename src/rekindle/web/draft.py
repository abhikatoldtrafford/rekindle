"""A draft memory: what the engine chose, plus what the person changed.

This module is the whole point of the UI and it contains no HTTP, no HTML and
no rendering. A `Draft` is the engine's own output, the engine's own reports,
and a small ordered list of edits on top - so every behaviour the web layer
promises can be tested by calling functions here.

WHAT A DRAFT IS ALLOWED TO DO
-----------------------------
It may reorder, remove, restore, swap and add. It may **not** select: there is
no code here that decides a photo is good, no scoring, no filtering by date or
album, no dedup rule of its own. Those decisions arrive already made:

    selection  <- recipe.select(...)  or  prompt.build_selection(...)
    usable     <- composition.compose(selection.photos)
    groups     <- dedup.bursts(usable)
    chosen     <- engine.build(...).shots

`_reconcile` re-derives the per-photo reason the UI displays and checks the
resulting histogram against `CompositionReport.dropped`, which the engine
produced independently. A rule that drifts out of step with the engine turns
into a failing test rather than a UI that lies about why a photo is missing.

WHY `add` CANNOT BYPASS A GUARDRAIL
-----------------------------------
`add` resolves the hash through `MemoryIndex.get`. An archived, trashed or
excluded photo is not stored on the index at all - see `memory/index.py` - so
the lookup returns None and the edit is refused with a reason. There is no
code path in this package that reaches a photo any other way, which is why
"the UI can surface a photo the CLI would refuse" is a bug that requires
deleting a line rather than forgetting one.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

from rekindle.memory import captions as captions_mod
from rekindle.memory import engine
from rekindle.memory.composition import (
    DROP_MINORITY_ORIENTATION,
    CompositionReport,
    compose,
)
from rekindle.memory.dedup import bursts, collapse
from rekindle.memory.index import MemoryIndex
from rekindle.memory.recipes import Offer, Selection
from rekindle.memory.render.gif import DEFAULT_FRAME_MS, TITLE_MS
from rekindle.memory.spec import MemorySpec, Shot, build_fact_sheet
from rekindle.memory.strata import StratumReport
from rekindle.models import Photo

#: Why a photo is not in the memory. The composition reasons come from
#: `memory.composition` unchanged; these three are the stages after it.
CUT_BURST = "near_duplicate"
CUT_CAP = "not_enough_slots"
CUT_BY_YOU = "removed_by_you"

#: Where a photo in the memory came from.
FROM_ENGINE = "chosen_by_the_engine"
FROM_YOU = "added_by_you"

ORIGIN_RECIPE = "recipe"
ORIGIN_PROMPT = "prompt"


class DraftError(ValueError):
    """An edit that was refused, carrying the sentence to show the user."""


@dataclass(frozen=True)
class Origin:
    """Where a draft came from, and the command that produced it.

    Kept separate from `MemorySpec` because a spec records what a memory IS
    and this records how it was ASKED FOR - which is what the provenance line
    in the UI needs and what a spec deliberately does not carry.
    """

    kind: str
    recipe: str
    key: str
    title: str
    prompt: str = ""
    albums: tuple[str, ...] = ()
    title_substantiated: bool = True

    def command(self) -> list[str]:
        """The `rekindle` argv that produced this draft's starting point."""
        if self.kind == ORIGIN_PROMPT:
            return ["rekindle", "memory", self.prompt]
        return ["rekindle", "memory", "--recipe", self.recipe, "--key", self.key]


@dataclass(frozen=True)
class Pace:
    """How long each frame holds, and which track plays under the MP4.

    Per-SHOT holds are deliberately absent. `MemorySpec.Shot` has four fields
    and `SPEC_VERSION` is 1; a per-shot duration would have to live in the
    spec for `rekindle render` to reproduce it, and bumping the spec version
    invalidates every memory.json already on disk. A control whose result
    cannot be handed back as a command is worse than no control, so the pace
    is a property of the memory - which is also the only thing the GIF, WebP
    and MP4 writers all accept.
    """

    frame_ms: int = DEFAULT_FRAME_MS
    title_ms: int = TITLE_MS
    music: Path | None = None


@dataclass(frozen=True)
class CandidateView:
    """One photo as the UI shows it, with why it is where it is."""

    file_hash: str
    taken_at_local: str | None
    caption: str
    state: str
    reason: str
    name: str
    width: int | None
    height: int | None
    people: tuple[str, ...]
    albums: tuple[str, ...]
    has_gps: bool
    favorite: bool
    public_safe: bool
    #: For a burst member: the hash of the frame that was kept instead.
    instead_of: str = ""
    #: Other frames of the same burst, best-first as dedup ranked them.
    burst: tuple[str, ...] = ()

    def to_json(self) -> dict:
        return {
            "file_hash": self.file_hash,
            "taken_at_local": self.taken_at_local,
            "caption": self.caption,
            "state": self.state,
            "reason": self.reason,
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "people": list(self.people),
            "albums": list(self.albums),
            "has_gps": self.has_gps,
            "favorite": self.favorite,
            "public_safe": self.public_safe,
            "instead_of": self.instead_of,
            "burst": list(self.burst),
        }


@dataclass
class Draft:
    """One editable memory.

    `order` is the single source of truth for what the memory contains. Every
    other collection here is evidence for the UI to explain itself with.
    """

    origin: Origin
    index: MemoryIndex
    pool: tuple[Photo, ...]
    usable: tuple[Photo, ...]
    composition: CompositionReport
    groups: tuple[tuple[str, ...], ...]
    survivors: frozenset[str]
    engine_order: tuple[str, ...]
    captions: dict[str, str]
    report: engine.BuildReport
    stratum: StratumReport | None
    max_shots: int
    order: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    pace: Pace = field(default_factory=Pace)

    # ---- edits

    def remove(self, file_hash: str) -> None:
        if file_hash not in self.order:
            raise DraftError("That photo is not in the memory.")
        self.order.remove(file_hash)
        if file_hash in self.added:
            # Something the user added and then changed their mind about is
            # simply gone; recording it as "removed" would offer to restore a
            # photo the engine never proposed, under a heading that says the
            # engine did.
            self.added.remove(file_hash)
        elif file_hash not in self.removed:
            self.removed.append(file_hash)

    def add(self, file_hash: str, *, at: int | None = None) -> None:
        """Put a photo into the memory. THE guardrail seam.

        Resolution goes through `MemoryIndex.get`, which holds only photos the
        policy admitted. An archived or excluded hash therefore fails here
        with a message, and there is no second path into `order`.
        """
        if file_hash in self.order:
            raise DraftError("That photo is already in the memory.")
        if self.index.get(file_hash) is None:
            raise DraftError(
                "That photo is not available: the guardrails do not admit it, "
                "or it is not in this library."
            )
        position = len(self.order) if at is None else max(0, min(at, len(self.order)))
        self.order.insert(position, file_hash)
        if file_hash in self.removed:
            self.removed.remove(file_hash)
        elif file_hash not in self.engine_order and file_hash not in self.added:
            self.added.append(file_hash)

    def swap(self, out_hash: str, in_hash: str) -> None:
        """Replace one shot with another IN PLACE, keeping its position.

        This is the burst control: "use a different frame of this moment".
        Implemented as remove-then-add at the same index rather than as its
        own rule, so it cannot bypass the guardrail in `add`.
        """
        if out_hash not in self.order:
            raise DraftError("That photo is not in the memory.")
        if self.index.get(in_hash) is None:
            raise DraftError("That replacement is not available in this library.")
        if in_hash in self.order:
            raise DraftError("That replacement is already in the memory.")
        position = self.order.index(out_hash)
        self.remove(out_hash)
        self.add(in_hash, at=position)

    def reorder(self, order: list[str]) -> None:
        """Accept a new order, but only a PERMUTATION of the current one.

        A reorder that silently adds or drops a photo is an edit wearing a
        reorder's clothes; the drag handler in the browser can only ever send
        a permutation, so anything else is a bug and is refused loudly.
        """
        if sorted(order) != sorted(self.order):
            raise DraftError("A reorder must keep exactly the same photos.")
        self.order = list(order)

    def set_pace(
        self,
        *,
        frame_ms: int | None = None,
        title_ms: int | None = None,
        music: Path | None = None,
        clear_music: bool = False,
    ) -> None:
        if frame_ms is not None and not (100 <= frame_ms <= 20_000):
            raise DraftError("A hold must be between 100 and 20000 milliseconds.")
        if title_ms is not None and not (100 <= title_ms <= 20_000):
            raise DraftError("A title hold must be between 100 and 20000 milliseconds.")
        self.pace = replace(
            self.pace,
            frame_ms=self.pace.frame_ms if frame_ms is None else frame_ms,
            title_ms=self.pace.title_ms if title_ms is None else title_ms,
            music=None if clear_music else (music if music is not None else self.pace.music),
        )

    # ---- views

    def photos(self) -> list[Photo]:
        """The memory's photos, in the user's order.

        Resolved through the index every time rather than cached, so a photo
        that became excluded after this draft was built disappears from the
        memory instead of lingering in a stale list.
        """
        return self.index.resolve_many(self.order)

    def catalogue(self) -> list[CandidateView]:
        """Every photo this draft knows about, chosen and rejected alike."""
        anchor_of, alternates = self._burst_maps()
        chosen = set(self.order)
        # Hoisted: `_cut_reason` is called once per pool photo, and rebuilding
        # this set inside it makes a 510-candidate album quadratic for no
        # reason.
        usable = frozenset(p.file_hash for p in self.usable)
        out: list[CandidateView] = []

        for file_hash in self.order:
            photo = self.index.get(file_hash)
            if photo is None:
                continue
            out.append(
                self._view(
                    photo,
                    state="chosen",
                    reason=FROM_YOU if file_hash in self.added else FROM_ENGINE,
                    burst=alternates.get(anchor_of.get(file_hash, ""), ()),
                )
            )

        for photo in self.pool:
            file_hash = photo.file_hash
            if file_hash in chosen:
                continue
            state, reason = self._cut_reason(photo, usable)
            out.append(
                self._view(
                    photo,
                    state=state,
                    reason=reason,
                    instead_of=(anchor_of.get(file_hash, "") if reason == CUT_BURST else ""),
                    burst=alternates.get(anchor_of.get(file_hash, ""), ()),
                )
            )
        return out

    def _cut_reason(self, photo: Photo, usable: frozenset[str]) -> tuple[str, str]:
        file_hash = photo.file_hash
        if file_hash in self.removed:
            return "removed", CUT_BY_YOU
        if file_hash not in usable:
            return "rejected", composition_reason(photo)
        if file_hash not in self.survivors:
            return "rejected", CUT_BURST
        return "rejected", CUT_CAP

    def _burst_maps(self) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
        """hash -> its burst's survivor, and survivor -> the whole burst."""
        anchor_of: dict[str, str] = {}
        alternates: dict[str, tuple[str, ...]] = {}
        for group in self.groups:
            if len(group) < 2:
                continue
            survivor = next((h for h in group if h in self.survivors), group[0])
            alternates[survivor] = group
            for member in group:
                anchor_of[member] = survivor
        return anchor_of, alternates

    def _view(
        self,
        photo: Photo,
        *,
        state: str,
        reason: str,
        instead_of: str = "",
        burst: tuple[str, ...] = (),
    ) -> CandidateView:
        local = photo.meta.taken_at_local
        return CandidateView(
            file_hash=photo.file_hash,
            taken_at_local=local.isoformat() if local else None,
            caption=self.captions.get(photo.file_hash, ""),
            state=state,
            reason=reason,
            # The FILE NAME, never a path. A path carries a username and a
            # drive layout, and this string goes into a browser page and into
            # whatever that page is later screenshotted into.
            name=photo.paths[0].name if photo.paths else photo.file_hash[:12],
            width=photo.meta.width,
            height=photo.meta.height,
            people=tuple(p for p in photo.meta.people if p),
            albums=tuple(photo.albums),
            has_gps=photo.meta.gps is not None,
            favorite=photo.meta.favorite,
            public_safe=self.index.is_public_safe(photo),
            instead_of=instead_of,
            burst=burst,
        )

    # ---- the artefact

    def to_spec(self) -> MemorySpec:
        """The draft as a `MemorySpec`, byte-for-byte renderable.

        Everything derived here is derived by the same functions the engine
        uses - `captions.subtitle_for`, `build_fact_sheet`,
        `MemoryIndex.is_public_safe` - so a hand-edited memory is the same
        kind of object as a generated one and `rekindle render` cannot tell
        them apart.
        """
        photos = self.photos()
        shots = tuple(
            Shot(
                file_hash=p.file_hash,
                caption=self.captions.get(p.file_hash, ""),
                taken_at_local=(
                    p.meta.taken_at_local.isoformat() if p.meta.taken_at_local else None
                ),
                public_safe=self.index.is_public_safe(p),
            )
            for p in photos
        )
        return MemorySpec(
            recipe=self.origin.recipe,
            key=self.origin.key,
            title=self.origin.title,
            subtitle=captions_mod.subtitle_for(photos),
            shots=shots,
            facts=build_fact_sheet(
                photos,
                title=self.origin.title,
                recipe=self.origin.recipe,
                albums=self.origin.albums,
                title_substantiated=self.origin.title_substantiated,
            ),
            public_safe=bool(shots) and all(s.public_safe for s in shots),
        )

    def reproduce_command(self, spec_path: Path) -> list[str]:
        """The argv that rebuilds exactly this, from the spec on disk.

        Not `rekindle memory ...`: that re-runs selection, and selection is
        precisely what the user has just overruled. The spec IS the edit, so
        the command that reproduces the edit is the one that renders the spec.
        Flags are emitted only when they differ from the render defaults, so
        the line stays short enough to read.
        """
        argv = ["rekindle", "render", str(spec_path)]
        if self.pace.frame_ms != DEFAULT_FRAME_MS:
            argv += ["--frame-ms", str(self.pace.frame_ms)]
        if self.pace.title_ms != TITLE_MS:
            argv += ["--title-ms", str(self.pace.title_ms)]
        if self.pace.music is not None:
            argv += ["--music", str(self.pace.music)]
        return argv


# --------------------------------------------------------------------------
# building a draft from the engine


def composition_reason(photo: Photo) -> str:
    """Why `composition.compose` refused ONE photo.

    `compose` reports counts, not per-photo reasons, and the UI has to name a
    reason beside a thumbnail. Rather than copy the rule ladder - which would
    drift the first time a threshold moved - this asks `compose` itself about
    a one-photo list.

    `enforce_orientation=False` is what makes that valid: the orientation rule
    is a property of the SET, and a set of one is always its own majority, so
    with the rule disabled the report contains exactly the per-photo verdict.
    A photo that survives that call but was dropped by the real call can only
    have been the minority orientation.
    """
    kept, report = compose([photo], enforce_orientation=False)
    if kept:
        return DROP_MINORITY_ORIENTATION
    return next(iter(report.dropped), "unknown")


def draft_from_selection(
    index: MemoryIndex,
    origin: Origin,
    offer: Offer,
    selection: Selection,
    *,
    max_shots: int = engine.DEFAULT_MAX_SHOTS,
) -> Draft:
    """Run the real pipeline and keep every stage's evidence.

    `compose` and `collapse` run here AND inside `engine.build`. That is
    deliberate duplication of two pure functions with identical arguments, not
    a second implementation: it is the only way to hold the intermediate
    stages the UI has to explain, and `test_web_draft.py` asserts that the
    counts this produces equal the counts the engine's own report produces.
    """
    usable, comp = compose(selection.photos)
    groups = tuple(tuple(p.file_hash for p in g) for g in bursts(usable))
    kept, _ = collapse(usable)
    survivors = frozenset(p.file_hash for p in kept)

    report = engine.BuildReport(offered=1)
    spec = engine.build(
        index,
        offer,
        selection=selection,
        max_shots=max_shots,
        report=report,
    )
    engine_order = tuple(s.file_hash for s in spec.shots) if spec else ()

    return Draft(
        origin=origin,
        index=index,
        pool=tuple(selection.photos),
        usable=tuple(usable),
        composition=comp,
        groups=groups,
        survivors=survivors,
        engine_order=engine_order,
        captions=dict(selection.captions),
        report=report,
        stratum=report.strata[0] if report.strata else None,
        max_shots=max_shots,
        order=list(engine_order),
    )


def skip_explanation(report: engine.BuildReport) -> str:
    """A sentence for a build that produced nothing, or "" when it produced
    something. The UI still opens an editable draft in that case - an empty
    memory whose rejected pile is visible is exactly the situation where
    overruling a guardrail by hand is the right move."""
    if not report.skipped:
        return ""
    reasons = {
        engine.SKIP_EMPTY: "the recipe found no candidates at all",
        engine.SKIP_TOO_FEW: "too few photos survived the guardrails",
        engine.SKIP_TOO_NARROW: (
            "the surviving photos did not span enough periods for this kind of memory"
        ),
        engine.SKIP_NO_EMBEDDINGS: "this library has no embeddings",
    }
    named = [reasons.get(r, r.replace("_", " ")) for r in sorted(report.skipped)]
    return "Nothing was built: " + "; ".join(named) + "."


def reconcile(draft: Draft) -> tuple[Counter[str], dict[str, int]]:
    """(per-photo reasons this UI would display, the engine's own counts).

    Exported so the test can compare them rather than re-deriving either side.
    """
    shown: Counter[str] = Counter()
    usable = {p.file_hash for p in draft.usable}
    for photo in draft.pool:
        if photo.file_hash not in usable:
            shown[composition_reason(photo)] += 1
    return shown, dict(draft.composition.dropped)
