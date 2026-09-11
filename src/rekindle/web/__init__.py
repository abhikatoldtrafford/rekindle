"""The interactive memory builder: a local web UI that is a LENS onto the CLI.

Three rules shape everything in this package, and each is enforced by a test
rather than by convention:

**The UI selects nothing.** Every photo it can show came out of
`MemoryIndex`, every candidate list came out of `recipe.select` or
`prompt.build_selection`, every guardrail count came out of
`composition.compose` and `dedup.collapse`, and the chosen shots came out of
`engine.build`. This package re-runs those functions; it never re-implements
one. `tests/test_web_draft.py` reconciles the per-photo reasons it displays
against the engine's own aggregate report, so a divergence fails.

**The index is still the chokepoint.** Nothing here opens `PhotoStore` to
find a photo. A hash the policy refused is not in the index, so it cannot be
displayed, cannot be added to a draft, and has no thumbnail - the thumbnail
route resolves through `MemoryIndex.get` and returns 404 for anything else.

**Every edit ends as a command.** An afternoon of dragging thumbnails around
produces a `MemorySpec` on disk plus one `rekindle render` invocation that
rebuilds it byte for byte, so the browser is never the only way to get the
result back.
"""

from __future__ import annotations

__all__ = ["DEFAULT_PORT"]

#: Chosen once and kept, because a user bookmarks this. 8517 is unassigned by
#: IANA and sits below the ephemeral range on Windows, Linux and macOS, so it
#: cannot collide with an outbound socket the OS picked.
DEFAULT_PORT = 8517
