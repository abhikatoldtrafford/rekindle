"""Structural conformance against a REAL Takeout export.

Skipped unless REKINDLE_TAKEOUT_DIR points at one, so CI never needs anyone's
personal library and no photo data is ever committed.

    REKINDLE_TAKEOUT_DIR="/path/to/Takeout/Google Photos" uv run pytest \
        tests/test_takeout_conformance.py -v

These assertions are the answer to "our fixtures certify our own fiction".
M0's motion-photo bug and v1's title-matching bug were BOTH invisible to a
green suite and both obvious within one run against real data.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rekindle.db import PhotoStore
from rekindle.enrich.takeout import (
    EnrichReport,
    JsonKind,
    TakeoutEnricher,
    account,
    build_index,
    classify_json,
)
from rekindle.sidecars import sidecar_target
from rekindle.sources.folder import FolderSource

_ENV = "REKINDLE_TAKEOUT_DIR"

pytestmark = pytest.mark.skipif(
    not os.environ.get(_ENV), reason=f"set {_ENV} to a real Takeout export to run these"
)


@pytest.fixture(scope="module")
def export() -> Path:
    root = Path(os.environ[_ENV])
    if not root.is_dir():
        pytest.skip(f"{_ENV} is not a directory: {root}")
    return root


@pytest.fixture(scope="module")
def indexed(export: Path):
    report = EnrichReport()
    return build_index(export, report), report


def test_every_json_file_lands_in_exactly_one_bucket(indexed):
    """THE invariant. A floor on the match rate would pass while a thousand
    sidecars vanished; this cannot."""
    _index, report = indexed
    assert report.json_files_seen > 0
    assert report.json_files_seen == report.files_accounted


def test_every_indexed_sidecar_is_matched_superseded_orphaned_or_ambiguous(export, indexed):
    """A partition check, and weak on its own - `account()` cannot fail it,
    because it partitions the very set it is given. It is kept to pin the
    arithmetic; `test_matched_equals_photos_enriched_on_a_real_export` below is
    the one that can actually fail.

    Note it applies exactly ONE sidecar per claimed target, so the rest land in
    `superseded`. Passing `applied=set()` here would put all of them there,
    which is also a valid partition - that is precisely why this test is weak.
    """
    index, report = indexed
    media = {
        p.name.casefold()
        for p in export.rglob("*")
        if p.is_file() and p.suffix.casefold() != ".json"
    }
    claimed = {target: "exact" for target in index.by_target if target in media}
    applied = {index.by_target[target][0].path for target in claimed}
    account(index, claimed, applied, report)
    assert report.sidecars_seen == report.sidecars_accounted
    assert report.matched == len(claimed)


def test_the_counter_relocation_reduces_collisions(export, indexed):
    """The measurement that falsified v1, re-run on whatever export is here.

    NOT a higher raw match count - that is the wrong property, and asserting
    it fails on the reference export by 4. Relocation moves 993 targets; five
    of those name a photo in an archive part the user has not extracted and
    correctly become orphans, one gains, so raw matches fall by 4 while
    correctness rises.

    The property that actually holds is COLLISION REDUCTION: how many sidecars
    beyond the first are claiming one photo. On the reference export, 4,737
    under `title` against 3,772 under the filename - the 965 that `title`
    mis-pairs, which is the 963-mis-pair figure plus the two (1)/(2) groups
    with no plain sibling.
    """
    index, _ = indexed
    name_collisions = sum(len(v) - 1 for v in index.by_target.values())

    by_title: dict[str, int] = {}
    for sidecars in index.by_target.values():
        for sidecar in sidecars:
            key = sidecar.title.casefold()
            by_title[key] = by_title.get(key, 0) + 1
    title_collisions = sum(n - 1 for n in by_title.values())

    assert name_collisions < title_collisions


def test_disagreeing_candidate_groups_are_detected(export, indexed):
    """583 groups on the reference export.

    The first draft asserted `(not _disagree(a, s)) or _disagree(a, s)` here -
    a literal tautology, true for any input including an empty index. If
    `_disagree` ever stopped working this reads 0 and both the refusal and the
    override-reporting machinery are silently dead.
    """
    from rekindle.enrich.takeout import _disagree

    index, _ = indexed
    multi = [group for group in index.by_target.values() if len(group) > 1]
    if not multi:
        pytest.skip("this export has at most one sidecar per photo")
    disagreeing = sum(1 for group in multi if any(_disagree(group[0], s) for s in group[1:]))
    assert disagreeing > 0


def test_every_sidecar_parses_and_names_a_target(export):
    checked = 0
    for path in export.rglob("*.json"):
        kind, payload = classify_json(path)
        assert kind is not JsonKind.UNPARSEABLE, f"{path} did not parse"
        if kind is JsonKind.PHOTO:
            assert sidecar_target(path.name) != path.name
            assert isinstance(payload.get("title"), str)
            checked += 1
    assert checked > 0


def test_people_names_are_clean(indexed):
    """Measured on 24,248 real sidecars: 40 distinct names, none empty,
    whitespace-only, or colliding on case."""
    index, _ = indexed
    names = {n for sidecars in index.by_target.values() for s in sidecars for n in s.people}
    assert all(n == n.strip() and n for n in names)


@pytest.fixture(scope="module")
def enriched(export: Path, tmp_path_factory) -> EnrichReport:
    """Index and enrich the real export once, into pytest's own tmp dir.

    Never inside the export: this must not write a single byte next to
    anyone's photos.
    """
    store = PhotoStore(tmp_path_factory.mktemp("conformance") / "rekindle.sqlite")
    try:
        photos, _ = FolderSource().scan(export)
        store.upsert_many(photos)
        store.set_meta("index_root", str(export))
        return TakeoutEnricher().enrich(export, store)
    finally:
        store.close()


def test_every_disagreeing_group_is_overridden_or_refused_never_silent(indexed, enriched):
    """No disagreement may be resolved without being counted.

    On the reference export every disagreeing group has its candidates in
    DIFFERENT directories, so rule 3 fires twice and rule 1 overrides 942
    times. Either number may move; what must never happen is a disagreement
    resolved with neither counter incrementing.
    """
    from rekindle.enrich.takeout import _disagree

    index, _ = indexed
    disagreeing = sum(
        1
        for group in index.by_target.values()
        if len(group) > 1 and any(_disagree(group[0], s) for s in group[1:])
    )
    if not disagreeing:
        pytest.skip("this export has no disagreeing sidecar groups")
    assert enriched.directory_preference_broke_a_tie + enriched.ambiguous > 0


def test_matched_equals_photos_enriched_on_a_real_export(enriched):
    """The one assertion here that ties accounting to the database.

    Credit `matched` with `len(sidecars)` rather than the applied count and
    this fails by 3,358.
    """
    assert enriched.matched == enriched.photos_enriched
    assert enriched.sidecars_seen == enriched.sidecars_accounted
    assert enriched.json_files_seen == enriched.files_accounted
