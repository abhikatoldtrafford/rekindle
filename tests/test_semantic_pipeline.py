"""The pipelined embed loop: prefetch, off-thread preprocessing, ordering.

WHY THIS FILE EXISTS
--------------------
`embed_photos` used to call `pool.map` and immediately block on the result,
so the decode pool was idle for the whole encode and the encode was idle for
the whole decode - while the module docstring claimed the pool ran "one batch
ahead of the GPU". The claim was in prose and nothing tested it, which is the
"test that cannot fail" shape the M1 decision log warns about: a green suite
is compatible with a completely serial pipeline.

So these tests assert the three properties that make the new loop worth
having, each one chosen because a plausible mistake would break it silently:

  * the prepared vectors EQUAL the fallback vectors (a split encoder that
    drifts produces a store nothing downstream can detect as wrong);
  * preprocessing really happens on a worker thread (the whole point, and
    invisible to any test of the output alone);
  * results are consumed in submission order (a deque of futures is exactly
    where an off-by-one pairs a hash with its neighbour's vector, and every
    vector would still be a plausible unit vector).

Every test here runs with no model, no GPU, no network and no photos beyond
the JPEGs it writes itself.
"""

from __future__ import annotations

import threading

import pytest

from rekindle.semantic.embed import (
    DEFAULT_WORKERS,
    PREFETCH_BATCHES,
    _prepare_fn,
    embed_photos,
)
from rekindle.semantic.encoder import PreparingEncoder
from rekindle.semantic.photos import PhotoIndexReader
from rekindle.semantic.store import EmbeddingStore
from tests.fixtures.semantic import (
    TOY_DIM,
    HalfPreparedEncoder,
    PreparingToyEncoder,
    ToyEncoder,
    make_index,
    make_photo,
    write_photo,
)


def shade(i: int) -> tuple[int, int, int]:
    """A colour unique to `i`.

    Every photo in a run must have a DIFFERENT vector, or an off-by-one in the
    future queue would pair a hash with a neighbour's vector and the
    comparison would still pass. Spread over the cube rather than one ramp, so
    neighbours are far apart and JPEG quantisation cannot merge them.
    """
    return (20 + (i * 37) % 210, 20 + (i * 61) % 210, 20 + (i * 97) % 210)


def build(tmp_path, count=24):
    """`count` photos, no two alike, so a mis-paired vector is detectable."""
    root = tmp_path / "lib"
    photos = []
    for i in range(count):
        path = write_photo(root, f"p{i:03d}.jpg", shade(i))
        photos.append(make_photo(f"{i:03d}".ljust(32, "a"), path))
    db = tmp_path / "data" / "rekindle.sqlite"
    make_index(db, photos)
    return db


def vectors_from(db, encoder, tmp_path, name, **kw):
    store = EmbeddingStore(tmp_path / name, dim=TOY_DIM, model_key="toy")
    with PhotoIndexReader(db) as reader:
        report = embed_photos(reader.iter_photos(), encoder, store, **kw)
    out = {h: store.get(h) for h in store.hashes()}
    store.close()
    return report, out


# --------------------------------------------------------------- equivalence


def test_the_prepared_path_gives_the_same_vectors_as_the_fallback(tmp_path):
    """Not "close": the same numbers, for the same hashes.

    The split is a threading change, not an arithmetic one. If a future
    encoder's `prepare_images`/`encode_prepared` pair ever stops composing to
    `encode_images`, a store filled by one path and queried by the other is
    quietly wrong and nothing else in the system would notice.
    """
    db = build(tmp_path)
    _, plain = vectors_from(db, ToyEncoder(), tmp_path, "plain")
    _, split = vectors_from(db, PreparingToyEncoder(), tmp_path, "split")
    assert set(plain) == set(split)
    for file_hash in plain:
        assert plain[file_hash] == pytest.approx(split[file_hash], abs=1e-6)


def test_encode_images_and_the_split_pair_agree_on_the_encoder_itself(tmp_path):
    """The encoder's own contract, independent of `embed_photos`."""
    from tests.fixtures.semantic import solid_image

    encoder = PreparingToyEncoder()
    images = [solid_image(shade(i)) for i in range(8)]
    direct = encoder.encode_images(images)
    staged = encoder.encode_prepared(encoder.prepare_images(images))
    assert len(direct) == len(staged) == len(images)
    for a, b in zip(direct, staged, strict=True):
        assert a == pytest.approx(b, abs=1e-9)


# ------------------------------------------------------- the pool is real


def test_preprocessing_runs_off_the_main_thread(tmp_path):
    """The point of the change, and untestable from the output alone.

    `PreparingToyEncoder` records the thread that prepared each batch. If
    `embed_photos` ever regresses to preparing inline - which is what the old
    loop did, and what any "simplification" of the future/deque dance would do
    - this set contains exactly the main thread and the test fails.
    """
    db = build(tmp_path, count=48)
    encoder = PreparingToyEncoder()
    vectors_from(db, encoder, tmp_path, "s", batch_size=4, workers=4, prefetch=8)
    assert encoder.prepare_calls > 0
    assert threading.get_ident() not in encoder.prepare_threads
    assert len(encoder.prepare_threads) > 1, (
        f"only {len(encoder.prepare_threads)} thread(s) prepared; the pool is not being used"
    )


def test_the_prepared_path_is_the_one_actually_taken(tmp_path):
    db = build(tmp_path, count=12)
    encoder = PreparingToyEncoder()
    vectors_from(db, encoder, tmp_path, "s", batch_size=4)
    assert encoder.prepared_calls == encoder.prepare_calls > 0
    assert encoder.image_calls == 0, "encode_images was used despite a prepared batch"


def test_a_toy_encoder_without_the_pair_uses_the_fallback(tmp_path):
    db = build(tmp_path, count=12)
    encoder = ToyEncoder()
    report, out = vectors_from(db, encoder, tmp_path, "s", batch_size=4)
    assert encoder.image_calls > 0
    assert report.embedded == 12 and len(out) == 12


def test_half_a_migration_is_refused(tmp_path):
    """`prepare_images` without `encode_prepared` must not be used.

    Using it would hand un-normalised grids to `encode_images`, which expects
    PIL images - a crash if you are lucky and a garbage vector if you are not.
    """
    encoder = HalfPreparedEncoder()
    assert _prepare_fn(encoder) is None
    db = build(tmp_path, count=8)
    report, out = vectors_from(db, encoder, tmp_path, "s", batch_size=4)
    assert encoder.prepare_calls == 0
    assert encoder.image_calls > 0
    assert report.embedded == 8 and len(out) == 8


def test_the_protocol_recognises_a_split_encoder(tmp_path):
    assert isinstance(PreparingToyEncoder(), PreparingEncoder)


# ------------------------------------------------------------------ ordering


def test_every_hash_keeps_its_own_vector_across_prefetch_depths(tmp_path):
    """The bug a deque of futures invites, and the reason for 24 photos.

    With many batches in flight, pairing hash i with the vector of batch i+1
    still stores a valid-looking unit vector for every photo. Only comparing
    against the serial answer catches it - so the serial answer is the oracle,
    and the photos are all different.
    """
    db = build(tmp_path, count=24)
    _, oracle = vectors_from(db, ToyEncoder(), tmp_path, "oracle", prefetch=1, batch_size=1)
    for depth in (1, 2, 3, 12, 40):
        _, got = vectors_from(
            db, PreparingToyEncoder(), tmp_path, f"d{depth}", prefetch=depth, batch_size=3
        )
        assert set(got) == set(oracle), f"prefetch={depth} lost or invented hashes"
        for file_hash in oracle:
            assert got[file_hash] == pytest.approx(oracle[file_hash], abs=1e-6), (
                f"prefetch={depth} paired {file_hash} with the wrong vector"
            )


def test_a_prefetch_larger_than_the_whole_run_is_harmless(tmp_path):
    db = build(tmp_path, count=5)
    report, out = vectors_from(
        db, PreparingToyEncoder(), tmp_path, "s", batch_size=2, prefetch=1000
    )
    assert report.embedded == 5 and report.accounted and len(out) == 5


@pytest.mark.parametrize("prefetch", [0, -1])
def test_a_nonsense_prefetch_is_clamped_not_obeyed(tmp_path, prefetch):
    """0 would submit nothing and hang forever; it is clamped to 1."""
    db = build(tmp_path, count=6)
    report, out = vectors_from(
        db, PreparingToyEncoder(), tmp_path, f"s{prefetch}", batch_size=2, prefetch=prefetch
    )
    assert report.embedded == 6 and report.accounted and len(out) == 6


# ------------------------------------------------------------ robustness


def test_a_broken_prepare_falls_back_instead_of_ending_the_run(tmp_path):
    """Off-thread preprocessing is an optimisation, never a new way to fail.

    Asserts the VECTORS are still the right ones, not merely that nothing
    raised: a fallback that produced different numbers would be worse than a
    crash, because it would be silent.
    """
    db = build(tmp_path, count=12)
    _, oracle = vectors_from(db, ToyEncoder(), tmp_path, "oracle")
    encoder = PreparingToyEncoder(prepare_raises=True)
    report, out = vectors_from(db, encoder, tmp_path, "broken", batch_size=3)
    assert report.embedded == 12 and report.accounted
    assert encoder.image_calls > 0, "the fallback was not taken"
    for file_hash in oracle:
        assert out[file_hash] == pytest.approx(oracle[file_hash], abs=1e-6)


def test_accounting_holds_when_a_worker_meets_a_missing_and_a_broken_file(tmp_path):
    """The counts moved from the main loop into `_prepare_batch`.

    They are returned by the worker and added on the main thread, precisely so
    `report.unreadable += 1` is never executed by six threads at once. This
    test is what stops someone moving them back.
    """
    root = tmp_path / "lib"
    photos = []
    for i in range(9):
        path = write_photo(root, f"ok{i}.jpg", shade(i))
        photos.append(make_photo(f"g{i}".ljust(32, "a"), path))
    gone = root / "gone.jpg"
    photos.append(make_photo("m".ljust(32, "a"), gone))
    broken = root / "broken.jpg"
    broken.write_bytes(b"this is not a JPEG")
    photos.append(make_photo("b".ljust(32, "a"), broken))
    db = tmp_path / "data" / "rekindle.sqlite"
    make_index(db, photos)

    report, out = vectors_from(
        db, PreparingToyEncoder(), tmp_path, "s", batch_size=2, workers=4, prefetch=6
    )
    assert report.considered == 11
    assert report.embedded == 9
    assert report.missing_file == 1
    assert report.unreadable == 1
    assert report.accounted
    assert len(out) == 9
    assert len(report.failures) == 2
    assert {reason.split(":")[0] for _, reason in report.failures} >= {"file not found"}


def test_a_batch_that_is_entirely_unreadable_still_advances_the_progress_count(tmp_path):
    root = tmp_path / "lib"
    photos = []
    for i in range(4):
        bad = root / f"bad{i}.jpg"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_bytes(b"not a jpeg at all")
        photos.append(make_photo(f"b{i}".ljust(32, "a"), bad))
    db = tmp_path / "data" / "rekindle.sqlite"
    make_index(db, photos)
    seen: list[tuple[int, int]] = []
    store = EmbeddingStore(tmp_path / "store", dim=TOY_DIM, model_key="toy")
    with PhotoIndexReader(db) as reader:
        report = embed_photos(
            reader.iter_photos(),
            PreparingToyEncoder(),
            store,
            batch_size=2,
            progress=lambda d, t: seen.append((d, t)),
        )
    assert report.unreadable == 4 and report.embedded == 0 and report.accounted
    assert seen and seen[-1] == (4, 4), f"progress stopped at {seen[-1] if seen else None}"


# ----------------------------------------------------------------- defaults


def test_the_shipped_prefetch_is_deep_enough_to_keep_the_pool_busy():
    """Pins the measured conclusion, not a magic number.

    `PREFETCH_BATCHES = 1` is the old serial shape and measured 3-4x slower on
    real photos. Anything at or below `DEFAULT_WORKERS` leaves workers idle
    whenever the main thread is inside the device call.
    """
    assert PREFETCH_BATCHES >= 2 * DEFAULT_WORKERS
