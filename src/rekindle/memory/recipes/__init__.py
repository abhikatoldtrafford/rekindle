"""The eight recipes, and the registry that holds them.

A recipe decides WHICH photos belong in a memory and IN WHAT ORDER. It decides
nothing else: not timing, not transitions, not the cap, not dedup, not
guardrails. Those belong to the engine, which every recipe passes through, so a
recipe cannot forget to apply one of them.

## The protocol

    class Recipe(Protocol):
        name: str     # "album_story" - stable, part of every memory id
        title: str    # "Album story" - human label for `rekindle memories`

        def offers(self, index) -> list[Offer]
            Every memory this recipe could build from this library.
            Deterministically ordered. Cheap: metadata only, no pixels.

        def select(self, index, offer) -> Selection | None
            The photos for ONE offer, generously and in order.
            None when the offer no longer yields enough.

Two methods, not one, because a recipe is a *generator of memories* rather
than one memory: `album_story` offers one per album, `person_years` one per
person. `offers()` being cheap and total is what makes `rekindle memories`
(list everything available) and `--auto` (pick today's) possible without
building anything.

This replaces the `candidates`/`order`/`fact_sheet` sketch in
docs/writing-recipes.md, which described `pydantic` params and an
`index.search(text=...)` - neither of which exists. That document is rewritten
alongside this code rather than left describing a fiction.
"""

from __future__ import annotations

# `builtin` is imported for its SIDE EFFECT: every @register in it populates
# REGISTRY, so `from rekindle.memory import recipes` is enough to have all
# eight available. It imports base and registry directly rather than through
# this module, so there is no import cycle despite appearances.
from rekindle.memory.recipes import builtin as _builtin  # noqa: F401
from rekindle.memory.recipes.base import MIN_SHOTS, Offer, Recipe, Selection
from rekindle.memory.recipes.registry import REGISTRY, get, register, registered

__all__ = [
    "MIN_SHOTS",
    "Offer",
    "Recipe",
    "Selection",
    "REGISTRY",
    "get",
    "register",
    "registered",
]
