"""The recipe registry.

Deliberately a plain dict with a decorator, not an entry-point scan. Entry
points are the right mechanism for third-party recipes and the contributor
guide describes how to add one, but resolving them costs an
`importlib.metadata` walk on every CLI invocation and the nine built-ins
neither need nor benefit from it.

Order is insertion order and it is load-bearing: `rekindle memories` lists
recipes in this order, and `--auto` breaks ties between equally good offers by
it, so the same library always surfaces the same memory.

WHY THE DICT LOADS ITSELF
-------------------------
`REGISTRY` is populated by the `@register` decorators in
`recipes.builtin`, which means it is EMPTY until that module has been
imported. Today `recipes/__init__.py` imports it for that side effect, which
covers every path that goes through the package - but "covers it today" is
not the same as "cannot break", and the failure mode is the worst kind:

    from rekindle.memory.recipes.registry import REGISTRY
    for name in REGISTRY:            # zero iterations
        build(name)                  # never runs
    # exit code 0, empty output directory, no error

That was reported from a real script. Nothing raised, nothing warned, and the
caller had every reason to believe it had rebuilt the library. An empty
registry is never a legitimate state for this program - the nine built-ins are
not optional - so it is treated as "not loaded yet" rather than as an answer.

`_SelfLoadingRegistry` therefore imports the built-ins on first READ, whichever
read it is: iteration, `len`, `in`, `[]`, `.get`, `.keys`, `.values`,
`.items`, or truth-testing. `_loading` guards the window in which `builtin` is
halfway through importing itself, so a `@register` running inside that import
does not re-enter it.

`recipes/__init__.py` no longer imports `builtin` eagerly. Leaving that import
in place as a belt-and-braces would have meant the lazy path was never taken
on any real run and only a contrived test could reach it - a guard nothing
exercises is a guard nobody can trust. Now every enumeration in the program
and in the suite goes through `_ensure`.

`register` also calls `ensure_loaded` before it writes, so a third-party
recipe imported before anything reads the registry cannot end up ahead of the
built-ins. Registration order is load-bearing (see above) and must not depend
on which module a caller happened to import first.

The alternative - deleting `REGISTRY` from the public surface and exposing
only `registered()` - was rejected because `REGISTRY["album_story"]` is used
throughout the tests and is a reasonable thing for a caller to write. Making
the reasonable thing correct beats forbidding it.
"""

from __future__ import annotations

import importlib

from rekindle.memory.recipes.base import Recipe

#: The module whose import side effect fills the registry.
BUILTIN_MODULE = "rekindle.memory.recipes.builtin"


class _SelfLoadingRegistry(dict):
    """A dict that cannot be observed empty because nobody imported the built-ins.

    See WHY THE DICT LOADS ITSELF in the module docstring. Every read goes
    through `_ensure`; every write does not.
    """

    def __init__(self) -> None:
        super().__init__()
        self._loading = False

    def _ensure(self) -> None:
        # `dict.__len__`, not `len(self)` or `if self`: both would route back
        # through the overrides below and recurse forever.
        if dict.__len__(self) or self._loading:
            return
        self._loading = True
        try:
            importlib.import_module(BUILTIN_MODULE)
        finally:
            self._loading = False

    def __iter__(self):
        self._ensure()
        return dict.__iter__(self)

    def __len__(self) -> int:
        self._ensure()
        return dict.__len__(self)

    def __contains__(self, key: object) -> bool:
        self._ensure()
        return dict.__contains__(self, key)

    def __getitem__(self, key):
        self._ensure()
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        self._ensure()
        return dict.get(self, key, default)

    def keys(self):
        self._ensure()
        return dict.keys(self)

    def values(self):
        self._ensure()
        return dict.values(self)

    def items(self):
        self._ensure()
        return dict.items(self)


REGISTRY: dict[str, Recipe] = _SelfLoadingRegistry()


def ensure_loaded() -> None:
    """Import the built-ins if they are not in the registry yet. Idempotent."""
    REGISTRY._ensure()  # type: ignore[attr-defined]


def register(recipe_cls):
    """Class decorator. Instantiates once - recipes are stateless."""
    # Before the write, so the nine built-ins always occupy the first nine
    # slots however a third-party recipe module got imported. A no-op while
    # `builtin` is itself the thing being imported.
    ensure_loaded()
    instance = recipe_cls()
    # `dict.__contains__`, deliberately: a duplicate check is part of writing,
    # and routing it through the self-loading read would re-enter the import
    # that is running this very decorator.
    if dict.__contains__(REGISTRY, instance.name):
        raise ValueError(f"duplicate recipe name: {instance.name!r}")
    dict.__setitem__(REGISTRY, instance.name, instance)
    return recipe_cls


def get(name: str) -> Recipe | None:
    return REGISTRY.get(name)


def registered() -> list[Recipe]:
    """Every recipe, in registration order."""
    return list(REGISTRY.values())


def names() -> list[str]:
    """Every recipe name, in registration order.

    The thing a script that wants to "do all of them" actually needs, so that
    it does not have to reach into the dict and can never be handed an empty
    list by accident.
    """
    return list(REGISTRY)
