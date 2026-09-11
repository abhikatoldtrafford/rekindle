"""Conformance against a REAL index. Skipped without REKINDLE_MEMORY_DB.

    REKINDLE_MEMORY_DB=/path/to/data/rekindle.sqlite uv run pytest \
        tests/test_memory_conformance.py -v

Every genuine bug in this project's previous milestone was found by running
against real data and none by reading. These assertions are the ones that a
synthetic fixture cannot make: they check invariants against whatever the
index actually contains, without hardcoding this library's numbers.

CI never needs a photo library - the whole module skips.
"""

import os
from pathlib import Path

import pytest

from rekindle.db import PhotoStore
from rekindle.memory import engine
from rekindle.memory.composition import compose
from rekindle.memory.dedup import collapse
from rekindle.memory.index import MemoryIndex
from rekindle.memory.policy import ExclusionPolicy
from rekindle.memory.recipes import registered

DB_ENV = "REKINDLE_MEMORY_DB"

pytestmark = pytest.mark.skipif(
    not os.environ.get(DB_ENV), reason=f"set {DB_ENV} to a real rekindle index"
)


@pytest.fixture(scope="module")
def store():
    path = Path(os.environ[DB_ENV])
    if not path.is_file():
        pytest.skip(f"{DB_ENV} does not point at a file")
    handle = PhotoStore(path)
    yield handle
    handle.close()


@pytest.fixture(scope="module")
def index(store):
    return MemoryIndex.open(store, ExclusionPolicy())


# --------------------------------------------------------------------------
# the guardrail, against real data


def test_no_archived_photo_survives_the_chokepoint(index):
    """162 rows on the reference export, flagged must-not-surface."""
    assert all(not p.meta.archived for p in index.all())
    assert all(not p.meta.trashed for p in index.all())


def test_the_exclusion_accounting_holds(index):
    assert index.report.accounted
    assert index.report.allowed + index.report.excluded == index.report.total


def test_every_allowed_photo_has_a_capture_date(index):
    """Every recipe is temporal, and the policy is supposed to guarantee it."""
    assert all(p.meta.taken_at_local is not None for p in index.all())


def test_excluding_a_real_person_actually_removes_photos(store, index):
    """A synthetic fixture cannot tell you the exclusion works against the
    name shapes Google actually writes - which here include parentheses and
    non-English words."""
    people = index.people_counts()
    if not people:
        pytest.skip("this index has no face tags")
    name, count = people.most_common(1)[0]

    narrowed = MemoryIndex.open(store, ExclusionPolicy(people=frozenset({name})))

    assert narrowed.count() == index.count() - count
    assert narrowed.by_person(name) == []


# --------------------------------------------------------------------------
# fingerprints and dedup


def test_every_image_was_fingerprinted_or_has_a_recorded_reason(index):
    """Silently dropping input is the bug the accounting identities exist to
    prevent. A photo with neither a hash nor a reason is an unrun pass."""
    from rekindle.models import MediaType

    unexplained = [
        p
        for p in index.all()
        if p.media_type is MediaType.IMAGE and p.meta.phash is None and not p.meta.phash_error
    ]
    if unexplained and len(unexplained) == len(index.images()):
        pytest.skip("this index has never been fingerprinted")
    assert unexplained == [], f"{len(unexplained)} images neither hashed nor explained"


def test_every_video_is_marked_as_never_fingerprinted(index):
    from rekindle.models import MediaType

    videos = [p for p in index.all() if p.media_type is MediaType.VIDEO]
    if not videos:
        pytest.skip("this index has no videos")
    assert all(v.meta.phash is None for v in videos)


def test_dedup_collapses_something_but_nothing_like_everything(index):
    """The measurement that shaped the design: time alone would collapse most
    of this library. With the pixel test, the rate must be modest.

    The bounds are deliberately wide - this is a REGRESSION guard against a
    catastrophic change, not a pin on one library's exact rate.
    """
    images = [p for p in index.images() if p.meta.phash is not None]
    if len(images) < 500:
        pytest.skip("not enough fingerprinted images to be meaningful")
    _, report = collapse(images)
    rate = report.collapsed / report.considered
    assert report.accounted
    assert rate < 0.35, f"dedup collapsed {rate:.1%} - the pixel test is not doing its job"


def test_photos_hours_apart_are_never_collapsed_together(index):
    """The brief's explicit requirement, checked against real timestamps."""
    from datetime import timedelta

    from rekindle.memory.dedup import bursts

    images = [p for p in index.images() if p.meta.phash is not None][:5000]
    if len(images) < 500:
        pytest.skip("not enough fingerprinted images")
    for group in bursts(images):
        if len(group) < 2:
            continue
        span = group[-1].meta.taken_at_utc - group[0].meta.taken_at_utc
        assert span < timedelta(hours=1), f"a burst spanned {span}"


# --------------------------------------------------------------------------
# composition


def test_composition_keeps_the_large_majority_of_a_real_library(index):
    """A guardrail that removes most of a library is a bug, not a filter."""
    kept, report = compose(index.all(), enforce_orientation=False)
    assert report.accounted
    assert len(kept) / max(report.considered, 1) > 0.75


def test_every_kept_photo_has_usable_dimensions(index):
    kept, _ = compose(index.all(), enforce_orientation=False)
    assert all(p.meta.width and p.meta.height for p in kept)


# --------------------------------------------------------------------------
# recipes and the engine


def test_every_recipe_offers_something_or_says_why(index):
    """A recipe that produces nothing on a 19,000-photo library is either
    broken or honestly limited - and the report should say which."""
    empty = [r.name for r in registered() if not r.offers(index)]
    assert empty == [], f"recipes with no offers on a real library: {empty}"


def test_every_offer_can_be_built(index):
    """`offers()` promises a memory exists; `select()` must agree. A mismatch
    means `rekindle memories` lists things `rekindle memory` cannot render."""
    failures = []
    for recipe in registered():
        for offer in recipe.offers(index)[:5]:
            if engine.build(index, offer) is None:
                failures.append(offer.memory_id)
    assert failures == [], f"offered but unbuildable: {failures[:10]}"


def test_no_memory_exceeds_the_cap_or_repeats_a_photo(index):
    for recipe in registered():
        for offer in recipe.offers(index)[:5]:
            spec = engine.build(index, offer, max_shots=24)
            if spec is None:
                continue
            hashes = [s.file_hash for s in spec.shots]
            assert len(hashes) <= 24
            assert len(hashes) == len(set(hashes))


def test_building_the_same_memory_twice_is_byte_identical(store, index):
    """Determinism, against real data rather than a fixture."""
    second = MemoryIndex.open(store, ExclusionPolicy())
    for recipe in registered():
        for offer in recipe.offers(index)[:3]:
            first_spec = engine.build(index, offer)
            again = engine.build(second, offer)
            if first_spec is None:
                assert again is None
                continue
            assert first_spec.dumps() == again.dumps(), offer.memory_id


def test_no_memory_claims_a_place_it_cannot_substantiate(index):
    """Never invent a fact. A memory with no GPS must carry no coordinates."""
    for recipe in registered():
        for offer in recipe.offers(index)[:5]:
            spec = engine.build(index, offer)
            if spec is None:
                continue
            photos = [index.get(s.file_hash) for s in spec.shots]
            has_gps = any(p and p.meta.gps for p in photos)
            if not has_gps:
                assert "places" not in spec.facts.to_json(), spec.key


def test_a_fact_sheet_never_carries_a_filesystem_path(index):
    """The fact sheet is the complete input to the optional GPT layer, and a
    path leaks a username and a drive layout. Real paths, real check."""
    for recipe in registered():
        for offer in recipe.offers(index)[:3]:
            spec = engine.build(index, offer)
            if spec is None:
                continue
            blob = str(spec.facts.to_json())
            assert ".jpg" not in blob.lower()
            assert "takeout" not in blob.lower()


# --------------------------------------------------------------------------
# public-safe, against the real face-tag coverage


def test_public_safe_never_admits_an_untagged_photo(store):
    """THE rule. Face tags are incomplete on any Takeout export, so an
    untagged photo may contain anyone."""
    plain = MemoryIndex.open(store, ExclusionPolicy())
    people = plain.people_counts()
    if not people:
        pytest.skip("this index has no face tags")
    allow = frozenset({people.most_common(1)[0][0]})

    safe = MemoryIndex.open(store, ExclusionPolicy(public_safe_allow=allow).with_public_safe(True))

    assert safe.count() > 0, "the allow-list admitted nothing at all"
    assert all(p.meta.people for p in safe.all()), "an untagged photo reached public-safe mode"
    assert all(set(p.meta.people) <= allow for p in safe.all())


def test_public_safe_is_a_small_fraction_of_a_real_library(store):
    """A sanity bound. If this ever approaches the whole library, the rule has
    been widened and strangers' faces are about to be published."""
    plain = MemoryIndex.open(store, ExclusionPolicy())
    people = plain.people_counts()
    if not people:
        pytest.skip("this index has no face tags")
    allow = frozenset({people.most_common(1)[0][0]})
    safe = MemoryIndex.open(store, ExclusionPolicy(public_safe_allow=allow).with_public_safe(True))
    assert safe.count() < plain.count() * 0.5


def test_a_public_safe_memory_has_only_public_safe_shots(store):
    plain = MemoryIndex.open(store, ExclusionPolicy())
    people = plain.people_counts()
    if not people:
        pytest.skip("this index has no face tags")
    allow = frozenset({people.most_common(1)[0][0]})
    index = MemoryIndex.open(store, ExclusionPolicy(public_safe_allow=allow))

    for recipe in registered():
        for offer in recipe.offers(index)[:5]:
            spec = engine.build(index, offer)
            if spec is not None and spec.public_safe:
                assert all(s.public_safe for s in spec.shots)


# --------------------------------------------------------------------------
# stratified spread
#
# The conformance suite did not catch the clustering defect: every assertion
# here was about correctness (no blocked photo, no duplicate, byte-identical)
# and none about whether the result was a good memory. These are the ones that
# would have caught it, added after the fact.


def test_a_multi_year_recipe_produces_a_multi_year_memory(index):
    """`on_this_day` and the over-the-years recipes are ABOUT a span. Before
    stratification, 16 of 37 rendered memories were confined to a single
    year - including an `on_this_day` showing only 2019."""
    from rekindle.memory.strata import bucket_key

    offenders = []
    for name in ("on_this_day", "on_this_month", "person_years", "pair_years"):
        recipe = next(r for r in registered() if r.name == name)
        for offer in recipe.offers(index)[:5]:
            spec = engine.build(index, offer)
            if spec is None:
                continue
            photos = [index.get(s.file_hash) for s in spec.shots]
            years = {p.meta.taken_at_local.year for p in photos if p}
            # A memory can legitimately have one year only if the gates left
            # one year standing. Ask the recipe what it had to work with.
            available = {bucket_key(p, "year") for p in recipe.select(index, offer).photos}
            if len(years) == 1 and len(available) > 1:
                offenders.append((offer.memory_id, sorted(years), len(available)))
    assert offenders == [], f"single-year memories from multi-year material: {offenders[:5]}"


def test_year_in_review_is_not_a_month_in_review(index):
    """Three of them were exactly that."""
    recipe = next(r for r in registered() if r.name == "year_in_review")
    offenders = []
    for offer in recipe.offers(index)[:6]:
        spec = engine.build(index, offer)
        if spec is None:
            continue
        photos = [index.get(s.file_hash) for s in spec.shots]
        months = {(p.meta.taken_at_local.year, p.meta.taken_at_local.month) for p in photos if p}
        available = {
            (p.meta.taken_at_local.year, p.meta.taken_at_local.month)
            for p in recipe.select(index, offer).photos
        }
        if len(months) == 1 and len(available) > 1:
            offenders.append((offer.key, len(available)))
    assert offenders == [], f"single-month years in review: {offenders}"


def test_every_period_that_survives_the_gates_gets_a_slot_when_there_is_room(index):
    """The floor, against real data: no period should be silently absent while
    slots remain unused."""
    from rekindle.memory.engine import BuildReport

    for recipe in registered():
        for offer in recipe.offers(index)[:3]:
            report = BuildReport(offered=1)
            spec = engine.build(index, offer, report=report)
            if spec is None or not report.strata:
                continue
            stratum = report.strata[0]
            if stratum.dimension is None:
                continue
            # Buckets are only left unslotted when there are more of them than
            # shots - never while the memory is under its cap.
            if len(spec.shots) < 24:
                assert stratum.unslotted == 0, (
                    f"{offer.memory_id}: {stratum.unslotted} periods unrepresented "
                    f"with only {len(spec.shots)} shots used"
                )


# --------------------------------------------------------------------------
# content diversity


def test_no_memory_contains_two_near_identical_shots(index):
    """The defect: selection ranked by quality and took the top N, so three
    good photos of the same child on the same afternoon all won on their own
    merits. Measured on the real library after the fix, the minimum pairwise
    dissimilarity within a memory is 0.33 and no pair is near-identical.

    The criterion is CONTENT, not the calendar - a single-day memory of
    genuinely different moments is fine, which is why this asserts on
    dissimilarity rather than on dates.
    """
    import itertools

    from rekindle.memory.diversity import HARD_FLOOR, CompositeSignal

    signal = CompositeSignal()
    offenders = []
    for recipe in registered():
        for offer in recipe.offers(index)[:4]:
            spec = engine.build(index, offer)
            if spec is None or len(spec.shots) < 2:
                continue
            photos = [index.get(s.file_hash) for s in spec.shots]
            for a, b in itertools.combinations([p for p in photos if p], 2):
                value = signal.between(a, b)
                if value is not None and value < HARD_FLOOR:
                    offenders.append((offer.memory_id, a.file_hash[:8], b.file_hash[:8], value))
    assert offenders == [], f"near-identical shots in one memory: {offenders[:5]}"


def test_the_diversity_signal_can_judge_most_of_a_real_library(index):
    """A signal that abstains everywhere is inert. This catches an index where
    the colour histogram was never backfilled, which would leave diversity
    running on the perceptual hash alone without saying so."""
    from rekindle.memory.diversity import ColourSignal

    images = index.images()[:400]
    if len(images) < 50:
        pytest.skip("not enough images")
    signal = ColourSignal()
    judged = sum(
        1 for a, b in zip(images, images[1:], strict=False) if signal.between(a, b) is not None
    )
    assert judged > len(images) * 0.9, "the colour histogram is missing from most of the index"
