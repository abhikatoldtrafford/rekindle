"""The recipe registry.

Deliberately a plain dict with a decorator, not an entry-point scan. Entry
points are the right mechanism for third-party recipes and the contributor
guide describes how to add one, but resolving them costs an
`importlib.metadata` walk on every CLI invocation and the eight built-ins
neither need nor benefit from it.

Order is insertion order and it is load-bearing: `rekindle memories` lists
recipes in this order, and `--auto` breaks ties between equally good offers by
it, so the same library always surfaces the same memory.
"""

from __future__ import annotations

from rekindle.memory.recipes.base import Recipe

REGISTRY: dict[str, Recipe] = {}


def register(recipe_cls):
    """Class decorator. Instantiates once - recipes are stateless."""
    instance = recipe_cls()
    if instance.name in REGISTRY:
        raise ValueError(f"duplicate recipe name: {instance.name!r}")
    REGISTRY[instance.name] = instance
    return recipe_cls


def get(name: str) -> Recipe | None:
    return REGISTRY.get(name)


def registered() -> list[Recipe]:
    """Every recipe, in registration order."""
    return list(REGISTRY.values())
