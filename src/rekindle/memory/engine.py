"""The pipeline. Recipes propose; the engine disposes.

    offer -> recipe.select -> compose -> dedup -> rank -> cap -> order -> spec

Everything after `select` happens HERE, once, for every recipe. That is the
point: a recipe cannot forget to dedup, cannot exceed the cap, cannot skip a
composition guardrail and cannot reach a photo the policy blocked. The only
thing a recipe controls is which photos it asks for and in what order they
should end up.

`test_engine.py` asserts the pipeline guarantees against EVERY REGISTERED
RECIPE via the registry, not against a hand-picked list, so a recipe added
later cannot opt out of them.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import date, datetime

from rekindle.config import CATALOGUE, active
from rekindle.memory import captions
from rekindle.memory.composition import CompositionReport, compose
from rekindle.memory.dedup import collapse
from rekindle.memory.diversity import DEFAULT_SIGNAL, CompositeSignal, DiversityReport
from rekindle.memory.diversity import SemanticSupport as SemanticSupport
from rekindle.memory.history import overlap
from rekindle.memory.index import MemoryIndex
from rekindle.memory.recipes import Offer, Selection, registered
from rekindle.memory.recipes.base import AS_GIVEN, chronological
from rekindle.memory.spec import MemorySpec, Shot, build_fact_sheet
from rekindle.memory.strata import StratumReport, stratify
from rekindle.models import Photo

# How many photos reach a memory. 24 is already long for a montage - at 2.5
# seconds a shot that is a minute - and the guide's own advice is "return
# fewer photos than you think".
DEFAULT_MAX_SHOTS = CATALOGUE["selection.max_shots"].default

# Reasons a candidate memory was not built. Counted and surfaced, never
# silently dropped.
SKIP_TOO_FEW = "too_few_photos"
SKIP_DISMISSED = "dismissed"
SKIP_COOLDOWN = "in_cooldown"
SKIP_OVERLAP = "overlaps_another"
#: Reached, and refused only because this recipe had already filled its share
#: of `--limit`. Distinct from every other reason: nothing is wrong with these
#: memories and raising the limit builds them.
SKIP_RECIPE_LIMIT = "recipe_limit_reached"
SKIP_EMPTY = "no_candidates"
# A recipe whose premise is spanning time could not span it: the quality or
# composition gates left too few distinct periods standing.
SKIP_TOO_NARROW = "too_few_periods"
# The library has no embeddings, so a prompt memory could not even be
# attempted. Counted separately from SKIP_EMPTY on purpose: an un-embedded
# library must not look like a query that found nothing.
SKIP_NO_EMBEDDINGS = "no_embeddings"


@dataclass
class BuildReport:
    """Why the memories you got are the memories you got."""

    offered: int = 0
    built: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    composition: CompositionReport = field(default_factory=CompositionReport)
    deduped: int = 0
    # One per memory built: how its shots were spread, and which periods could
    # not be represented. A bucket emptied by the quality gates is a correct
    # outcome, but a silent one would look exactly like the clustering bug
    # this reporting was added alongside.
    strata: list[StratumReport] = field(default_factory=list)
    diversity: DiversityReport = field(default_factory=DiversityReport)
    # Of `deduped`, the frames only the embedding could see were duplicates.
    deduped_semantically: int = 0

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    @property
    def total_skipped(self) -> int:
        return sum(self.skipped.values())

    @property
    def accounted(self) -> bool:
        return self.offered == self.built + self.total_skipped


def score(photo: Photo, sharp_rank: float) -> float:
    """How well a photo presents. Used ONLY to choose which of the allowed
    photos fit in the cap - never to decide what is allowed.

    A fixed, documented sum rather than a tuned model, so that the reason a
    photo did or did not appear can always be reconstructed by hand:

        +3  favourite          (3 rows in this library - nearly inert, but
                                free, and meaningful in other libraries)
        +2  has a description  (142 rows)
        +2  has face tags      (people are what makes a memory a memory)
        +1  has GPS
        +1  sidecar_match == "exact"
        +   sharpness percentile within this selection, in [0, 1]

    The sharpness term is a PERCENTILE within the selection, not the raw
    value: raw sharpness is not comparable between cameras, and a 2008 photo
    would lose every contest against a 2025 one on an absolute scale.
    """
    meta = photo.meta
    total = 0.0
    if meta.favorite:
        total += 3.0
    if meta.description:
        total += 2.0
    if any(meta.people):
        total += 2.0
    if meta.gps is not None:
        total += 1.0
    if photo.sidecar_match == "exact":
        total += 1.0
    return total + sharp_rank


def _ranked(photos: list[Photo]) -> list[Photo]:
    """Best first. Total order, so the result never depends on input order."""
    measured = sorted(
        (p.meta.sharpness for p in photos if p.meta.sharpness is not None),
    )

    def rank(photo: Photo) -> float:
        value = photo.meta.sharpness
        if value is None or not measured:
            # Never fingerprinted. Treated as mid-pack rather than worst:
            # penalising it would make an unfingerprinted library rank by
            # nothing but metadata, which is a different product.
            return 0.5
        # `bisect_left` over the sorted list, not a linear count over it.
        # Identical answer - the index of the first element not less than
        # `value` IS how many are strictly below it - and it turns the sort
        # from O(n^2) into O(n log n). Measured 0.45 s at n=4,000, against a
        # candidate pool that grows with the library.
        below = bisect_left(measured, value)
        return below / len(measured)

    return sorted(
        photos,
        key=lambda p: (-score(p, rank(p)), p.meta.taken_at_utc, p.file_hash),
    )


def build(
    index: MemoryIndex,
    offer: Offer,
    *,
    selection: Selection | None = None,
    max_shots: int | None = None,
    min_shots: int | None = None,
    report: BuildReport | None = None,
    semantic: SemanticSupport | None = None,
) -> MemorySpec | None:
    """One offer to one spec, or None with a counted reason.

    `semantic` is the optional embedding store, handed in rather than imported
    - see `memory.diversity.SemanticSupport`. None is a supported and fully
    tested configuration, not a degraded one: it is what CI runs and what any
    install without the extra runs.

    `selection` lets a caller that is not a registered recipe - the prompt
    path - hand the engine its own photos. It is safe because the pipeline's
    guarantee is about PHOTOS, not about who assembled them: everything from
    `compose` onward runs below this branch and cannot be skipped. The hole it
    opens is that `test_engine.py` sweeps the registry, so a Selection that
    never enters the registry is not swept - `test_engine.py` therefore pushes
    a synthetic, non-registry Selection through this function and asserts the
    same guarantees.
    """
    # `None` means "the active configuration, read now". A shipped number
    # bound as a default argument would be frozen at import time, so a user's
    # `rekindle.toml` could never reach it.
    cfg = active().selection
    max_shots = cfg.max_shots if max_shots is None else max_shots
    min_shots = cfg.min_shots if min_shots is None else min_shots

    report = report if report is not None else BuildReport()
    if selection is None:
        recipe = _recipe_for(offer)
        if recipe is None:
            report.skip(SKIP_EMPTY)
            return None
        selection = recipe.select(index, offer)
    if selection is None or not selection.photos:
        report.skip(SKIP_EMPTY)
        return None

    # A recipe with a fixed, smaller form declares its own floor.
    floor = selection.min_shots if selection.min_shots is not None else min_shots

    # 1. Composition guardrails. Before dedup, so that the orientation
    #    majority and the canvas are computed over photos that can be shown.
    usable, comp_report = compose(selection.photos)
    report.composition.merge(comp_report)
    if len(usable) < floor:
        report.skip(SKIP_TOO_FEW)
        return None

    # 2. Dedup, before the cap, so the cap is filled with 24 DISTINCT photos
    #    rather than 24 slots of which six are near-duplicates.
    kept, dedup_report = collapse(usable, cosine=semantic.cosine if semantic else None)
    report.deduped += dedup_report.collapsed
    report.deduped_semantically += dedup_report.semantic
    if len(kept) < floor:
        report.skip(SKIP_TOO_FEW)
        return None

    #    The diversity signal is calibrated against THIS memory's surviving
    #    candidates - after the gates, so the spread it measures is the spread
    #    the viewer could actually have been shown. `binding` stays the two
    #    pixel signals: an embedding may reorder a memory, never shorten one.
    #    See `memory.diversity.pick` and `semantic.diversity` for why.
    signal = DEFAULT_SIGNAL
    if semantic is not None:
        extra = semantic.signal_for(kept)
        if extra is not None:
            signal = CompositeSignal(signals=(*DEFAULT_SIGNAL.signals, extra))

    # 3. Spread, rank, cap, then restore the recipe's ordering.
    #
    #    Ranking first and re-sorting after is what lets a chronological
    #    memory still contain the best 24 of 500 photos rather than the first
    #    24. But ranking ALONE clustered every memory into whichever period
    #    scored highest - 16 of 37 confined to a single year on the real
    #    library - so slots are allocated across the dimension the recipe
    #    declared and each bucket is then filled best-first.
    chosen, stratum = stratify(
        kept,
        level=selection.stratify,
        slots=max_shots,
        rank=_ranked,
        offered=selection.photos,
        signal=signal,
        binding=DEFAULT_SIGNAL if signal is not DEFAULT_SIGNAL else None,
    )
    stratum.memory_id = offer.memory_id
    report.strata.append(stratum)
    report.diversity.merge(stratum.diversity)
    if stratum.dimension is not None and stratum.used < selection.min_strata:
        # "Over the years" across one year is not a weaker memory, it is a
        # different and misleading one. Refuse it and say so.
        report.skip(SKIP_TOO_NARROW)
        return None
    if selection.ordering != AS_GIVEN:
        chosen = chronological(chosen)
    else:
        # Preserve the recipe's own sequence among the survivors.
        order = {p.file_hash: i for i, p in enumerate(selection.photos)}
        chosen = sorted(chosen, key=lambda p: order.get(p.file_hash, 0))

    shots = tuple(
        Shot(
            file_hash=p.file_hash,
            caption=selection.captions.get(p.file_hash, ""),
            taken_at_local=p.meta.taken_at_local.isoformat() if p.meta.taken_at_local else None,
            public_safe=index.is_public_safe(p),
        )
        for p in chosen
    )
    report.built += 1
    return MemorySpec(
        recipe=offer.recipe,
        key=offer.key,
        title=offer.title,
        subtitle=captions.subtitle_for(chosen),
        shots=shots,
        # Facts describe the FINAL shots, not the candidate pool: a fact sheet
        # about 500 candidates while the memory shows 24 would let a narrator
        # assert things the viewer cannot see.
        facts=build_fact_sheet(
            chosen,
            title=offer.title,
            recipe=offer.recipe,
            albums=selection.facts.albums,
            # Carried from the Selection, not recomputed: whether a TITLE is a
            # fact is a property of who wrote it, and the engine cannot tell
            # an album name from a user's query by looking at the string.
            title_substantiated=selection.facts.title_substantiated,
        ),
        # A memory is publishable only when EVERY shot in it is. One
        # non-qualifying photo makes the whole thing unpublishable.
        public_safe=all(s.public_safe for s in shots),
    )


def _recipe_for(offer: Offer):
    for recipe in registered():
        if recipe.name == offer.recipe:
            return recipe
    return None


def all_offers(index: MemoryIndex) -> list[Offer]:
    """Every offer from every recipe, in registry order then recipe order."""
    out: list[Offer] = []
    for recipe in registered():
        out.extend(recipe.offers(index))
    return out


def build_all(
    index: MemoryIndex,
    offers: list[Offer],
    *,
    max_shots: int | None = None,
    min_shots: int | None = None,
    max_overlap: float | None = None,
    dismissed: frozenset[str] = frozenset(),
    cooling: frozenset[str] = frozenset(),
    limit: int | None = None,
    per_recipe_limit: int | None = None,
    semantic: SemanticSupport | None = None,
) -> tuple[list[MemorySpec], BuildReport]:
    """Build a batch, refusing dismissed, cooling and redundant memories.

    Overlap is checked against memories ALREADY ACCEPTED in this batch, in
    offer order. That makes the result depend on the order offers arrive in -
    which is exactly why `all_offers` is deterministic: the first of two
    redundant memories wins, and "first" must mean the same thing every run.

    `limit` caps the WHOLE batch and stops the loop. `per_recipe_limit` caps
    each recipe's share and does not - later recipes are still reached. That
    difference is the whole of `--all-recipes`: offers arrive in registry
    order and `on_this_day` alone contributes 192 of them, so a global cap of
    any sane size never gets past the second recipe.

    They compose, and `per_recipe_limit` is applied FIRST, inside one batch
    rather than by calling this function once per recipe. Per-recipe calls
    would lose the cross-recipe overlap check - `accepted` is what stops the
    same twelve photographs shipping as an album story and again as a year in
    review - and that check is the reason a batch is a batch.
    """
    overlap_cap = active().selection.max_overlap if max_overlap is None else max_overlap
    report = BuildReport(offered=len(offers))
    built: list[MemorySpec] = []
    accepted: list[list[str]] = []
    per_recipe: dict[str, int] = {}

    for offer in offers:
        if per_recipe_limit is not None and per_recipe.get(offer.recipe, 0) >= per_recipe_limit:
            report.skip(SKIP_RECIPE_LIMIT)
            continue
        if offer.memory_id in dismissed:
            report.skip(SKIP_DISMISSED)
            continue
        if offer.memory_id in cooling:
            report.skip(SKIP_COOLDOWN)
            continue
        spec = build(
            index,
            offer,
            max_shots=max_shots,
            min_shots=min_shots,
            report=report,
            semantic=semantic,
        )
        if spec is None:
            continue
        hashes = [s.file_hash for s in spec.shots]
        if any(overlap(hashes, other) >= overlap_cap for other in accepted):
            # `build` already counted this as built; undo that before
            # recording the real reason, or the accounting identity breaks.
            report.built -= 1
            report.skip(SKIP_OVERLAP)
            continue
        accepted.append(hashes)
        built.append(spec)
        per_recipe[offer.recipe] = per_recipe.get(offer.recipe, 0) + 1
        if limit is not None and len(built) >= limit:
            # The remaining offers were never considered. They are not
            # "skipped" - they were not reached - so the offered count is
            # corrected rather than a bogus reason being invented.
            report.offered = report.built + report.total_skipped
            break
    return built, report


def offers_for_today(index: MemoryIndex, today: date | datetime) -> list[Offer]:
    """Today's anniversary offers, best first.

    `on_this_day` for today's calendar date, falling back to `on_this_month`
    when the day is too thin to stand alone. The fallback is the reason
    `on_this_month` exists as a separate recipe.
    """
    moment = today.date() if isinstance(today, datetime) else today
    day_key = f"{moment.month:02d}-{moment.day:02d}"
    month_key = f"{moment.month:02d}"

    out = [o for o in _offers_of("on_this_day", index) if o.key == day_key]
    if not out:
        out = [o for o in _offers_of("on_this_month", index) if o.key == month_key]
    return out


def _offers_of(name: str, index: MemoryIndex) -> list[Offer]:
    for recipe in registered():
        if recipe.name == name:
            return recipe.offers(index)
    return []
