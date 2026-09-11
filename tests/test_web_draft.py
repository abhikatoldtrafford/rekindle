"""The editable draft: guardrails, edits, and agreement with the engine.

Every test here asserts an OUTCOME - which photos are in the memory, what the
spec says, what the reason beside a thumbnail is - rather than that a function
was called. The three that matter most:

* `test_catalogue_reasons_reconcile_with_the_engine` compares the per-photo
  reason this package displays against the counts `engine.build` produced by
  itself. Those are two independent derivations of the same fact, so a drift
  in either one fails.
* `test_archived_photo_cannot_be_added` and its excluded-person twin pin the
  only rule whose violation would be a critical defect.
* `test_removed_photo_stays_out_of_the_spec` is the core loop.
"""

from __future__ import annotations

from collections import Counter

import pytest

from rekindle.memory import engine
from rekindle.memory.composition import DROP_TOO_SMALL, compose
from rekindle.memory.recipes import registered
from rekindle.web.draft import (
    CUT_BURST,
    CUT_BY_YOU,
    CUT_CAP,
    FROM_YOU,
    ORIGIN_RECIPE,
    DraftError,
    Origin,
    composition_reason,
    draft_from_selection,
    reconcile,
)
from tests.fixtures.web import ALBUM, EXCLUDED_PERSON, exclude_person, make_library, open_library


def album_draft(tmp_path, *, data_dir=None, max_shots=24):
    data_dir = data_dir or make_library(tmp_path)[0]
    library = open_library(data_dir)
    index = library.require_index()
    recipe = next(r for r in registered() if r.name == "album_story")
    offer = next(o for o in recipe.offers(index) if o.key == ALBUM)
    selection = recipe.select(index, offer)
    origin = Origin(
        kind=ORIGIN_RECIPE,
        recipe=offer.recipe,
        key=offer.key,
        title=offer.title,
        albums=selection.facts.albums,
    )
    return draft_from_selection(index, origin, offer, selection, max_shots=max_shots)


# ---------------------------------------------------------------- guardrails


def test_archived_photo_is_invisible_everywhere(tmp_path):
    draft = album_draft(tmp_path)
    hashes = {c.file_hash for c in draft.catalogue()}
    assert "archived" not in hashes
    assert draft.index.get("archived") is None
    assert "archived" not in {p.file_hash for p in draft.pool}


def test_archived_photo_cannot_be_added(tmp_path):
    draft = album_draft(tmp_path)
    with pytest.raises(DraftError, match="guardrails do not admit"):
        draft.add("archived")
    assert "archived" not in draft.order


def test_excluded_person_cannot_be_added(tmp_path):
    data_dir, _ = make_library(tmp_path)
    exclude_person(data_dir)
    draft = album_draft(tmp_path, data_dir=data_dir)
    # The photo is in the album the recipe selected, so nothing but the policy
    # keeps it out.
    assert "excluded" not in {c.file_hash for c in draft.catalogue()}
    with pytest.raises(DraftError):
        draft.add("excluded")


def test_excluded_person_is_present_without_the_exclusion(tmp_path):
    """The mirror of the test above: without the rule, the photo IS reachable.

    Without this, the two tests above would pass on a library that simply has
    no such photo, which is the shape of a test that cannot fail.
    """
    draft = album_draft(tmp_path)
    assert draft.index.get("excluded") is not None
    assert "excluded" in draft.order
    assert "excluded" in {c.file_hash for c in draft.catalogue()}


def test_a_photo_excluded_after_the_draft_was_built_leaves_the_spec(tmp_path):
    """`to_spec` resolves through the index every time, not from a cache."""
    data_dir, _ = make_library(tmp_path)
    draft = album_draft(tmp_path, data_dir=data_dir)
    assert "excluded" in {s.file_hash for s in draft.to_spec().shots}

    exclude_person(data_dir)
    draft.index = open_library(data_dir).require_index()
    assert "excluded" not in {s.file_hash for s in draft.to_spec().shots}
    # The hash stays in `order` - the user did ask for it - but the memory
    # cannot contain it.
    assert "excluded" in draft.order


# --------------------------------------------------------------- review loop


def test_removed_photo_stays_out_of_the_spec(tmp_path):
    draft = album_draft(tmp_path)
    victim = draft.order[1]
    before = len(draft.order)

    draft.remove(victim)

    assert victim not in draft.order
    assert len(draft.order) == before - 1
    assert victim not in {s.file_hash for s in draft.to_spec().shots}
    view = next(c for c in draft.catalogue() if c.file_hash == victim)
    assert (view.state, view.reason) == ("removed", CUT_BY_YOU)


def test_removing_the_same_photo_twice_is_refused(tmp_path):
    draft = album_draft(tmp_path)
    victim = draft.order[0]
    draft.remove(victim)
    with pytest.raises(DraftError):
        draft.remove(victim)


def test_a_removed_photo_can_be_put_back(tmp_path):
    draft = album_draft(tmp_path)
    victim = draft.order[2]
    draft.remove(victim)
    draft.add(victim)
    assert victim in draft.order
    assert victim not in draft.removed
    # It came from the engine, so putting it back does not claim the user
    # added it.
    assert victim not in draft.added


def test_added_photo_is_marked_as_yours(tmp_path):
    draft = album_draft(tmp_path, max_shots=4)
    spare = next(c.file_hash for c in draft.catalogue() if c.state == "rejected")
    draft.add(spare)
    view = next(c for c in draft.catalogue() if c.file_hash == spare)
    assert (view.state, view.reason) == ("chosen", FROM_YOU)


# ------------------------------------------------------------------ ordering


def test_reorder_changes_the_spec_order(tmp_path):
    draft = album_draft(tmp_path)
    reversed_order = list(reversed(draft.order))
    draft.reorder(reversed_order)
    assert [s.file_hash for s in draft.to_spec().shots] == reversed_order


def test_reorder_must_be_a_permutation(tmp_path):
    draft = album_draft(tmp_path)
    with pytest.raises(DraftError, match="same photos"):
        draft.reorder(draft.order[:-1])
    with pytest.raises(DraftError):
        draft.reorder([*draft.order, draft.order[0]])


def test_reorder_does_not_admit_a_new_photo(tmp_path):
    """A reorder carrying an extra hash is an add in disguise, and refused."""
    draft = album_draft(tmp_path, max_shots=4)
    smuggled = next(c.file_hash for c in draft.catalogue() if c.state == "rejected")
    with pytest.raises(DraftError):
        draft.reorder([*draft.order[:-1], smuggled])
    assert smuggled not in draft.order


# ------------------------------------------------------------- burst control


def test_the_burst_is_collapsed_and_the_alternate_is_offered(tmp_path):
    draft = album_draft(tmp_path)
    views = {c.file_hash: c for c in draft.catalogue()}
    assert "burst_keep" in draft.order, "the sharper frame should be the survivor"
    assert views["burst_drop"].reason == CUT_BURST
    assert views["burst_drop"].instead_of == "burst_keep"
    assert set(views["burst_keep"].burst) == {"burst_keep", "burst_drop"}


def test_swap_replaces_a_frame_in_place(tmp_path):
    draft = album_draft(tmp_path)
    position = draft.order.index("burst_keep")
    draft.swap("burst_keep", "burst_drop")
    assert draft.order[position] == "burst_drop"
    assert "burst_keep" not in draft.order
    assert [s.file_hash for s in draft.to_spec().shots][position] == "burst_drop"


def test_swap_refuses_a_photo_the_guardrails_withhold(tmp_path):
    draft = album_draft(tmp_path)
    with pytest.raises(DraftError):
        draft.swap(draft.order[0], "archived")


# -------------------------------------------------- agreement with the engine


def test_catalogue_reasons_reconcile_with_the_engine(tmp_path):
    """The reasons shown beside thumbnails must equal the engine's own counts.

    Two independent derivations: `reconcile` asks `compose` about one photo at
    a time, while `draft.composition` is the report `compose` produced over
    the whole pool inside `engine.build`. If either drifts, this fails.
    """
    draft = album_draft(tmp_path)
    shown, engine_counts = reconcile(draft)
    assert dict(shown) == engine_counts
    assert engine_counts.get(DROP_TOO_SMALL) == 1, "the 400x300 photo must be rejected"


def test_every_pool_photo_has_exactly_one_state(tmp_path):
    draft = album_draft(tmp_path, max_shots=4)
    views = draft.catalogue()
    assert len(views) == len({v.file_hash for v in views}), "no photo appears twice"
    assert {v.file_hash for v in views} == {p.file_hash for p in draft.pool}
    counts = Counter(v.state for v in views)
    assert counts["chosen"] == len(draft.order)


def test_the_cap_is_the_reason_when_nothing_else_is(tmp_path):
    draft = album_draft(tmp_path, max_shots=3)
    reasons = Counter(c.reason for c in draft.catalogue())
    assert reasons[CUT_CAP] > 0
    # And a capped-out photo is genuinely usable - it survived composition and
    # dedup, which is what makes "not enough slots" the honest reason.
    capped = next(c for c in draft.catalogue() if c.reason == CUT_CAP)
    usable, _ = compose([draft.index.get(capped.file_hash)], enforce_orientation=False)
    assert usable


def test_composition_reason_matches_compose_for_a_rejected_photo(tmp_path):
    draft = album_draft(tmp_path)
    tiny = draft.index.get("tiny")
    assert composition_reason(tiny) == DROP_TOO_SMALL


def test_engine_order_is_the_engines_and_nothing_else(tmp_path):
    """The draft starts as exactly what `engine.build` returned, in its order."""
    data_dir, _ = make_library(tmp_path)
    draft = album_draft(tmp_path, data_dir=data_dir)
    library = open_library(data_dir)
    index = library.require_index()
    recipe = next(r for r in registered() if r.name == "album_story")
    offer = next(o for o in recipe.offers(index) if o.key == ALBUM)
    spec = engine.build(index, offer, max_shots=24)
    assert draft.order == [s.file_hash for s in spec.shots]


# --------------------------------------------------------------------- pace


def test_pace_bounds_are_enforced(tmp_path):
    draft = album_draft(tmp_path)
    with pytest.raises(DraftError):
        draft.set_pace(frame_ms=10)
    with pytest.raises(DraftError):
        draft.set_pace(title_ms=99_000)
    draft.set_pace(frame_ms=900)
    assert draft.pace.frame_ms == 900


def test_reproduce_command_names_the_spec_and_only_changed_flags(tmp_path):
    draft = album_draft(tmp_path)
    spec_path = tmp_path / "memories" / "m" / "memory.json"
    assert draft.reproduce_command(spec_path) == ["rekindle", "render", str(spec_path)]
    draft.set_pace(frame_ms=800)
    assert draft.reproduce_command(spec_path) == [
        "rekindle",
        "render",
        str(spec_path),
        "--frame-ms",
        "800",
    ]


def test_the_provenance_command_is_the_cli_one(tmp_path):
    draft = album_draft(tmp_path)
    assert draft.origin.command() == [
        "rekindle",
        "memory",
        "--recipe",
        "album_story",
        "--key",
        ALBUM,
    ]


def test_public_safe_is_computed_from_the_shots_not_asserted(tmp_path):
    """ "Safe to publish" is the one claim that must never be optimistic.

    Default deny: with no allow-list, a memory of tagged photos is NOT
    public-safe. It becomes so only when every name in every shot is allowed,
    which is `policy.is_public_safe` and not a flag this package sets.
    """
    data_dir, _ = make_library(tmp_path)
    draft = album_draft(tmp_path, data_dir=data_dir)
    assert draft.to_spec().public_safe is False
    assert all(not c.public_safe for c in draft.catalogue())

    (data_dir / "exclusions.toml").write_text(
        'public_safe_allow = ["Abhik Maiti", "Someone Excluded"]\n', encoding="utf-8"
    )
    allowed = album_draft(tmp_path, data_dir=data_dir)
    assert allowed.to_spec().public_safe is True

    # One shot that does not qualify makes the whole memory unpublishable.
    (data_dir / "exclusions.toml").write_text(
        'public_safe_allow = ["Abhik Maiti"]\n', encoding="utf-8"
    )
    partial = album_draft(tmp_path, data_dir=data_dir)
    assert any(c.public_safe for c in partial.catalogue())
    assert partial.to_spec().public_safe is False


def test_an_empty_memory_is_never_public_safe(tmp_path):
    """`all(...)` over an empty tuple is True, which would publish nothing as
    if it were everything."""
    data_dir, _ = make_library(tmp_path)
    (data_dir / "exclusions.toml").write_text(
        'public_safe_allow = ["Abhik Maiti", "Someone Excluded"]\n', encoding="utf-8"
    )
    draft = album_draft(tmp_path, data_dir=data_dir)
    for file_hash in list(draft.order):
        draft.remove(file_hash)
    assert draft.to_spec().public_safe is False


def test_excluded_person_is_actually_in_the_library_fixture(tmp_path):
    """Guards the fixture itself: the exclusion tests are only meaningful if
    the photo exists and carries that person."""
    data_dir, photos = make_library(tmp_path)
    row = next(p for p in photos if p.file_hash == "excluded")
    assert row.meta.people == [EXCLUDED_PERSON]
    assert ALBUM in row.albums
