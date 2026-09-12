"""Opt-in: does the staleness detector fire on the REFERENCE library?

Skipped everywhere except a machine that has both the index and a real
`clip-vit-l14` store. CI has no photos and no embeddings and must stay green.

**Why this file exists at all.** A staleness detector that never fires looks
exactly like one with nothing to do, and the toy-encoder tests in
`test_semantic_embed.py` can only prove it fires on a fixture this file's
author also wrote. This asserts it against 18,201 real vectors and the real
orientation verdicts the `rekindle semantic orient` pass recorded - the
population the incident actually happened to.

**Nothing here writes to the live store.** The manifest is copied to
`tmp_path` first. `vectors.f32` is not needed at all: no vector is read, only
the manifest's provenance column, so the copy is 1 MB rather than 84 MB and
`_reconcile` is satisfied with a matching-size sparse file.

The assertion is a SET EQUALITY against the index, not a count. 212 is what
this library happens to hold today; re-running the orientation pass at a
different `--margin`, or indexing another folder, legitimately changes it. A
test pinned to 212 would fail for a correct reason and teach everyone to
ignore it. What must hold in every one of those worlds is the relationship:
*the vectors called stale are exactly the ones whose orientation verdict
disagrees with what they were embedded under.*
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path

import pytest

from rekindle.meta import orientation
from rekindle.semantic.embed import decode_key
from rekindle.semantic.photos import PhotoIndexReader, ReadFilter
from rekindle.semantic.store import EmbeddingStore

DATA = Path(os.environ.get("REKINDLE_REAL_DATA_DIR", r"D:\google_photos\data"))
DB = DATA / "rekindle.sqlite"
STORE = DATA / "semantic" / "clip-vit-l14"
DIM = 768
TARGET_PX = 224

pytestmark = pytest.mark.skipif(
    not (DB.is_file() and (STORE / "manifest.sqlite").is_file()),
    reason="needs the reference index and a clip-vit-l14 embedding store",
)


@pytest.fixture
def live(tmp_path):
    """A copy of the live manifest, and the live index open beside it."""
    root = tmp_path / "clip-vit-l14"
    root.mkdir()
    shutil.copy2(STORE / "manifest.sqlite", root / "manifest.sqlite")
    # A sparse stand-in for the 84 MB matrix: `_reconcile` compares its SIZE
    # against the manifest and nothing in this file reads a vector.
    conn = sqlite3.connect(root / "manifest.sqlite")
    held = int(conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0])
    conn.close()
    with (root / "vectors.f32").open("wb") as fh:
        fh.truncate(held * DIM * 4)

    reader = PhotoIndexReader(DB)  # installs this index's orientation verdicts
    store = EmbeddingStore(root, dim=DIM, model_key="clip-vit-l14")
    yield reader, store
    store.close()
    reader.close()
    orientation.clear_overrides()


def _ignoring_exif() -> set[str]:
    conn = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
    try:
        return {
            r[0] for r in conn.execute("SELECT file_hash FROM photos WHERE orient_ignore_exif = 1")
        }
    finally:
        conn.close()


def test_the_library_really_does_have_stale_tag_verdicts_to_detect(live):
    """If this is 0 the rest of the file proves nothing, so it is asserted."""
    reader, _ = live
    assert orientation.override_count() > 0, (
        "the index records no stale orientation tags, so nothing in this file "
        "can distinguish a working detector from a broken one"
    )
    assert _ignoring_exif(), "no photo in the index has a verdict of `tag is stale`"
    assert reader.count(ReadFilter(images_only=True)) > 1000


def test_the_detector_fires_on_exactly_the_photos_whose_verdict_changed(live):
    """The incident, replayed against the real store.

    Stamp every held vector with the decode key it would have had if it had
    been embedded BEFORE the orientation pass spoke - which is what actually
    happened, and is why 2,503 vectors stayed sideways through the fix. No
    file is touched, no hash moves, the model and its revision do not move.
    Only the verdict does.
    """
    reader, store = live
    photos = list(reader.iter_photos(ReadFilter(images_only=True)))
    now = {p.file_hash: decode_key(p.paths, target_px=TARGET_PX) for p in photos}

    orientation.clear_overrides()  # the world before `rekindle semantic orient`
    before = {p.file_hash: decode_key(p.paths, target_px=TARGET_PX) for p in photos}
    reader._load_orientation_overrides()  # and after it
    assert orientation.override_count() > 0

    held = set(store.decode_keys())
    store._conn.executemany(
        "UPDATE vectors SET decode_key = ? WHERE file_hash = ?",
        [(k, h) for h, k in before.items() if h in held],
    )
    store._conn.commit()

    plan = store.plan([(h, now[h]) for h in now])
    expected = _ignoring_exif() & held
    assert expected, "no corrected photo has a vector, so this proves nothing"
    assert set(plan.stale) == expected
    assert plan.fresh == len(held & set(now)) - len(expected)
    assert plan.unverified == []


def test_a_run_after_the_pass_finds_nothing_stale(live):
    """The other half. A detector that fires on everything is also broken, and
    on a store already embedded under the current verdicts the answer is 0."""
    reader, store = live
    photos = list(reader.iter_photos(ReadFilter(images_only=True)))
    now = {p.file_hash: decode_key(p.paths, target_px=TARGET_PX) for p in photos}
    held = set(store.decode_keys())
    store._conn.executemany(
        "UPDATE vectors SET decode_key = ? WHERE file_hash = ?",
        [(k, h) for h, k in now.items() if h in held],
    )
    store._conn.commit()
    plan = store.plan([(h, now[h]) for h in now])
    assert plan.stale == []
    assert plan.unverified == []
    assert plan.fresh == len(held & set(now))


def test_the_live_store_predates_provenance_and_says_so(live):
    """Nothing pretends the 18,201 existing vectors are verified. They are not,
    and the honest answer is `unverified`, which is what `--redo-unverified`
    is for."""
    reader, store = live
    photos = list(reader.iter_photos(ReadFilter(images_only=True)))
    plan = store.plan([(p.file_hash, decode_key(p.paths, target_px=TARGET_PX)) for p in photos])
    assert plan.fresh == 0
    assert plan.stale == []
    assert len(plan.unverified) == len(set(store.decode_keys()) & {p.file_hash for p in photos})
