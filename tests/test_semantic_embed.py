"""`rekindle semantic embed`, end to end with a real encoder and real JPEGs.

No model, no GPU, no network: `ToyEncoder` is a genuine embedding function and
the photos are Pillow-written JPEGs in tmp_path. What is under test is the
driver - resumption, accounting, decode failures, ordering - and every
assertion is about the vectors that came out, not about calls that went in.
"""

from __future__ import annotations

import pytest

from rekindle.meta import orientation
from rekindle.semantic.embed import (
    DECODE_REVISION,
    DEFAULT_BATCH,
    decode_key,
    embed_photos,
)
from rekindle.semantic.photos import PhotoIndexReader, ReadFilter
from rekindle.semantic.store import EmbeddingStore
from tests.fixtures.semantic import (
    TOY_DIM,
    ToyEncoder,
    make_index,
    make_photo,
    solid_image,
    write_photo,
)


@pytest.fixture
def setup(tmp_path):
    root = tmp_path / "lib"
    photos = []
    for i, colour in enumerate(("red", "green", "blue", "yellow", "white")):
        path = write_photo(root, f"{colour}.jpg", colour)
        photos.append(make_photo(f"{i}".ljust(32, "a"), path))
    db = tmp_path / "data" / "rekindle.sqlite"
    make_index(db, photos)
    store = EmbeddingStore(tmp_path / "store", dim=TOY_DIM, model_key="toy")
    return db, store, ToyEncoder()


def test_embeds_every_photo_once(setup):
    db, store, encoder = setup
    with PhotoIndexReader(db) as reader:
        report = embed_photos(reader.iter_photos(), encoder, store, batch_size=2)
    assert report.considered == 5
    assert report.embedded == 5
    assert report.already_embedded == 0
    assert store.count() == 5
    assert report.accounted


def test_the_stored_vector_is_the_encoder_s_answer(setup):
    """Not "a vector was stored" - THE vector, for the right photo."""
    db, store, encoder = setup
    with PhotoIndexReader(db) as reader:
        photos = {p.path.name: p for p in reader.iter_photos()}
        embed_photos(reader.iter_photos(), encoder, store)
    expected = encoder.encode_images([solid_image("red")])[0]
    stored = store.get(photos["red.jpg"].file_hash)
    assert stored == pytest.approx(expected, abs=2e-2)


def test_different_photos_get_different_vectors(setup):
    db, store, encoder = setup
    with PhotoIndexReader(db) as reader:
        embed_photos(reader.iter_photos(), encoder, store)
    vectors = {tuple(round(v, 3) for v in store.get(h)) for h in store.hashes()}
    assert len(vectors) == 5


def test_a_second_run_embeds_nothing_new(setup):
    db, store, encoder = setup
    with PhotoIndexReader(db) as reader:
        embed_photos(reader.iter_photos(), encoder, store)
        again = embed_photos(reader.iter_photos(), encoder, store)
    assert again.embedded == 0
    assert again.already_embedded == 5
    assert again.accounted
    assert store.count() == 5


def test_a_partial_run_resumes(setup):
    db, store, encoder = setup
    with PhotoIndexReader(db) as reader:
        first = embed_photos(reader.iter_photos(), encoder, store, limit=2)
        assert first.embedded == 2
        second = embed_photos(reader.iter_photos(), encoder, store)
    assert second.embedded == 3
    assert second.already_embedded == 2
    assert store.count() == 5


def test_a_missing_file_is_counted_not_dropped(tmp_path):
    root = tmp_path / "lib"
    real = write_photo(root, "there.jpg", "red")
    photos = [
        make_photo("1".ljust(32, "a"), real),
        make_photo("2".ljust(32, "a"), root / "gone.jpg"),
    ]
    db = tmp_path / "data" / "rekindle.sqlite"
    make_index(db, photos)
    store = EmbeddingStore(tmp_path / "store", dim=TOY_DIM, model_key="toy")
    with PhotoIndexReader(db) as reader:
        report = embed_photos(reader.iter_photos(), ToyEncoder(), store)
    assert report.considered == 2
    assert report.embedded == 1
    assert report.missing_file == 1
    assert report.accounted
    assert any("not found" in why for _, why in report.failures)


def test_an_undecodable_file_is_counted_not_fatal(tmp_path):
    """One corrupt file among 18,000 must not end the run - nor vanish."""
    root = tmp_path / "lib"
    good = write_photo(root, "good.jpg", "green")
    bad = root / "bad.jpg"
    bad.write_bytes(b"\xff\xd8\xff\xe0 this is not a JPEG body " * 8)
    db = tmp_path / "data" / "rekindle.sqlite"
    make_index(db, [make_photo("1".ljust(32, "a"), good), make_photo("2".ljust(32, "a"), bad)])
    store = EmbeddingStore(tmp_path / "store", dim=TOY_DIM, model_key="toy")
    with PhotoIndexReader(db) as reader:
        report = embed_photos(reader.iter_photos(), ToyEncoder(), store)
    assert report.embedded == 1
    assert report.unreadable == 1
    assert report.accounted
    assert store.count() == 1


def test_a_whole_bad_batch_does_not_stall_the_run(tmp_path):
    """Every photo in one batch unreadable: the run must continue AND account."""
    root = tmp_path / "lib"
    photos = []
    for i in range(2):
        bad = root / f"bad{i}.jpg"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_bytes(b"not an image")
        photos.append(make_photo(f"b{i}".ljust(32, "a"), bad))
    photos.append(make_photo("g0".ljust(32, "a"), write_photo(root, "good.jpg", "blue")))
    db = tmp_path / "data" / "rekindle.sqlite"
    make_index(db, photos)
    store = EmbeddingStore(tmp_path / "store", dim=TOY_DIM, model_key="toy")
    with PhotoIndexReader(db) as reader:
        report = embed_photos(reader.iter_photos(), ToyEncoder(), store, batch_size=2)
    assert report.considered == 3
    assert report.unreadable == 2
    assert report.embedded == 1
    assert report.accounted


def test_archived_photos_are_not_embedded_by_default(tmp_path):
    """A Google-archived photo is one the user hid. It must not resurface."""
    root = tmp_path / "lib"
    photos = [
        make_photo("1".ljust(32, "a"), write_photo(root, "ok.jpg", "red")),
        make_photo("2".ljust(32, "a"), write_photo(root, "hidden.jpg", "black"), archived=True),
    ]
    db = tmp_path / "data" / "rekindle.sqlite"
    make_index(db, photos)
    store = EmbeddingStore(tmp_path / "store", dim=TOY_DIM, model_key="toy")
    with PhotoIndexReader(db) as reader:
        report = embed_photos(reader.iter_photos(), ToyEncoder(), store)
    assert report.considered == 1
    assert store.hashes() == {"1".ljust(32, "a")}

    with PhotoIndexReader(db) as reader:
        widened = embed_photos(
            reader.iter_photos(ReadFilter(include_archived=True)), ToyEncoder(), store
        )
    assert widened.embedded == 1


def test_progress_reports_monotonically_up_to_total(setup):
    db, store, encoder = setup
    seen = []
    with PhotoIndexReader(db) as reader:
        embed_photos(
            reader.iter_photos(),
            encoder,
            store,
            batch_size=2,
            progress=lambda done, total: seen.append((done, total)),
        )
    assert seen
    assert [d for d, _ in seen] == sorted(d for d, _ in seen)
    assert seen[-1] == (5, 5)
    assert all(d <= t for d, t in seen)


def test_a_duplicate_hash_in_the_input_is_embedded_once(tmp_path):
    root = tmp_path / "lib"
    path = write_photo(root, "one.jpg", "red")
    photo = make_photo("1".ljust(32, "a"), path)
    db = tmp_path / "data" / "rekindle.sqlite"
    make_index(db, [photo])
    store = EmbeddingStore(tmp_path / "store", dim=TOY_DIM, model_key="toy")
    with PhotoIndexReader(db) as reader:
        found = next(iter(reader.iter_photos()))
    report = embed_photos([found, found], ToyEncoder(), store)
    assert report.considered == 2
    assert report.embedded == 1
    assert report.already_embedded == 1
    assert report.accounted


def test_nothing_to_do_returns_a_balanced_empty_report(tmp_path):
    store = EmbeddingStore(tmp_path / "store", dim=TOY_DIM, model_key="toy")
    report = embed_photos([], ToyEncoder(), store)
    assert report.considered == 0
    assert report.accounted
    assert report.images_per_s == 0.0


def test_report_carries_the_provenance_the_user_needs(setup):
    db, store, encoder = setup
    with PhotoIndexReader(db) as reader:
        report = embed_photos(reader.iter_photos(), encoder, store, runtime="onnx", device="cpu")
    assert report.model_key == "toy"
    assert report.runtime == "onnx"
    assert report.device == "cpu"
    assert report.elapsed_s > 0
    assert report.images_per_s > 0


def test_batch_size_does_not_change_the_answer(setup):
    db, store, encoder = setup
    with PhotoIndexReader(db) as reader:
        embed_photos(reader.iter_photos(), encoder, store, batch_size=1)
        one = {h: store.get(h) for h in store.hashes()}

    other = EmbeddingStore(store.root.parent / "store2", dim=TOY_DIM, model_key="toy")
    with PhotoIndexReader(db) as reader:
        embed_photos(reader.iter_photos(), encoder, other, batch_size=DEFAULT_BATCH)
    for digest, vector in one.items():
        assert other.get(digest) == pytest.approx(vector)


def test_decode_draft_does_not_change_the_embedding_much(tmp_path):
    """`draft` is a speed hack. If it changed the vector it would be a bug."""
    from rekindle.semantic.embed import _decode

    root = tmp_path / "lib"
    big = root / "big.jpg"
    big.parent.mkdir(parents=True, exist_ok=True)
    solid_image("green", size=1024).save(big, format="JPEG", quality=95)
    encoder = ToyEncoder()
    drafted = encoder.encode_images([_decode(big, 64)])[0]
    from PIL import Image

    with Image.open(big) as im:
        full = encoder.encode_images([im.convert("RGB")])[0]
    assert drafted == pytest.approx(full, abs=2e-2)


# ------------------------------------------------------- staleness detection
#
# THE TRAP THESE TESTS EXIST TO CATCH. The bug being fixed cost two full
# re-embeds because `embed_photos` skipped any hash the store already held.
# The obvious fix - "notice when the file changed" - does nothing here: the
# file does NOT change. Its bytes are identical before and after; what moves
# is the orientation verdict, and therefore the decode.
#
# So every test below keeps the bytes fixed and moves only the verdict, and
# asserts the VECTOR THAT CAME OUT, not that a counter was incremented.


@pytest.fixture
def rotated(tmp_path):
    """One real JPEG carrying EXIF orientation 6, indexed, with no vector yet.

    The four-colour quadrant marker is used rather than a solid colour on
    purpose: a solid square embeds to the same vector whichever way up it is,
    so a solid-colour fixture could not tell a working detector from a broken
    one.
    """
    from tests.fixtures.oriented import write_oriented

    path = tmp_path / "lib" / "rotated.jpg"
    write_oriented(path, 6)
    photo = make_photo("a".ljust(32, "a"), path)
    db = tmp_path / "data" / "rekindle.sqlite"
    make_index(db, [photo])
    store = EmbeddingStore(tmp_path / "store", dim=TOY_DIM, model_key="toy")
    orientation.clear_overrides()
    yield db, store, path
    orientation.clear_overrides()
    store.close()


def test_a_CHANGED_ORIENTATION_VERDICT_makes_the_vector_stale_and_redoes_it(rotated):
    """The incident, reproduced and then caught.

    Embed with the EXIF tag applied. Then let `meta.orientation` prove that
    tag stale, exactly as `rekindle semantic orient` does. The file on disk is
    untouched - same bytes, same hash, same model, same revision - and the
    vector in the store is now wrong. `embed` must notice and replace it.
    """
    db, store, path = rotated
    with PhotoIndexReader(db) as reader:
        first = embed_photos(reader.iter_photos(), ToyEncoder(), store)
    assert first.embedded == 1
    assert first.newly_embedded == 1
    tagged_vector = store.get("a".ljust(32, "a"))

    before = path.read_bytes()
    orientation.load_overrides([path])  # the verdict changes; the file does not
    assert path.read_bytes() == before, "the bytes must not move - that is the trap"

    with PhotoIndexReader(db) as reader:
        # PhotoIndexReader re-installs the index's overrides on open, and this
        # index has none, so re-apply after opening it.
        photos = list(reader.iter_photos())
    orientation.load_overrides([path])
    second = embed_photos(photos, ToyEncoder(), store)

    assert second.stale == 1, "the store failed to notice the decode moved"
    assert second.embedded == 1
    assert second.recomputed == 1
    assert second.newly_embedded == 0
    assert second.accounted

    raw_vector = store.get("a".ljust(32, "a"))
    assert raw_vector != pytest.approx(tagged_vector), (
        "the vector was rewritten but is unchanged - the decode did not really move"
    )
    # And it is the RIGHT vector: what the encoder gives for the raw pixels.
    from PIL import Image

    with Image.open(path) as im:
        expected = ToyEncoder().encode_images([im.convert("RGB")])[0]
    assert raw_vector == pytest.approx(expected, abs=3e-2)


def test_the_file_hash_alone_would_have_missed_it(rotated):
    """Guards the design, not the code: prove the key is NOT the file.

    If a future change keys staleness on file content, this fails - the hash
    and the bytes are provably identical across the verdict change.
    """
    db, store, path = rotated
    from rekindle.identity import file_hash as hash_file

    before_hash = hash_file(path)
    with PhotoIndexReader(db) as reader:
        embed_photos(reader.iter_photos(), ToyEncoder(), store)
        photos = list(reader.iter_photos())
    orientation.load_overrides([path])
    assert hash_file(path) == before_hash
    assert embed_photos(photos, ToyEncoder(), store).stale == 1


def test_an_UNCHANGED_verdict_leaves_the_vector_alone(rotated):
    """The other half: a detector that fires on everything is also broken."""
    db, store, _ = rotated
    with PhotoIndexReader(db) as reader:
        embed_photos(reader.iter_photos(), ToyEncoder(), store)
        again = embed_photos(reader.iter_photos(), ToyEncoder(), store)
    assert again.stale == 0
    assert again.fresh == 1
    assert again.embedded == 0
    assert again.already_embedded == 1
    assert again.accounted


def test_a_verdict_on_ANOTHER_file_does_not_make_this_one_stale(rotated):
    db, store, path = rotated
    with PhotoIndexReader(db) as reader:
        embed_photos(reader.iter_photos(), ToyEncoder(), store)
        photos = list(reader.iter_photos())
    orientation.load_overrides([path.with_name("someone-else.jpg")])
    report = embed_photos(photos, ToyEncoder(), store)
    assert report.stale == 0
    assert report.fresh == 1


def test_bumping_the_decode_revision_makes_every_vector_stale(setup):
    """The blunt instrument, for a decode change no other term captures."""
    db, store, encoder = setup
    with PhotoIndexReader(db) as reader:
        embed_photos(reader.iter_photos(), encoder, store)
        photos = list(reader.iter_photos())
    report = embed_photos(photos, encoder, store, decode_revision=DECODE_REVISION + 1)
    assert report.stale == 5
    assert report.embedded == 5
    assert report.recomputed == 5


def test_a_different_target_size_is_a_different_decode(setup):
    db, store, encoder = setup
    with PhotoIndexReader(db) as reader:
        embed_photos(reader.iter_photos(), encoder, store, target_px=224)
        photos = list(reader.iter_photos())
    assert embed_photos(photos, encoder, store, target_px=336).stale == 5


def test_decode_key_names_the_verdict_and_nothing_that_cannot_vary(tmp_path):
    """The EXIF TAG is deliberately absent: it lives in the bytes the hash
    already pins. What must be present is the verdict, which does not."""
    path = tmp_path / "x.jpg"
    orientation.clear_overrides()
    with_tag = decode_key([path], target_px=224)
    orientation.load_overrides([path])
    without = decode_key([path], target_px=224)
    orientation.clear_overrides()
    assert with_tag != without
    assert decode_key([path], target_px=224) != decode_key([path], target_px=336)
    assert decode_key([path], target_px=224, revision=1) != decode_key(
        [path], target_px=224, revision=2
    )


def test_an_unverified_vector_is_not_redone_unless_asked(setup):
    """Five hours on a contributor's CPU is not a default."""
    db, store, encoder = setup
    with PhotoIndexReader(db) as reader:
        photos = list(reader.iter_photos())
    # A store written by an older rekindle: vectors, no provenance.
    store.add_many([(p.file_hash, [0.0] * TOY_DIM) for p in photos])
    quiet = embed_photos(photos, encoder, store)
    assert quiet.unverified == 5
    assert quiet.embedded == 0
    assert quiet.already_embedded == 5
    assert quiet.accounted

    asked = embed_photos(photos, encoder, store, redo_unverified=True)
    assert asked.embedded == 5
    assert asked.recomputed == 5
    assert store.get(photos[0].file_hash) != pytest.approx([0.0] * TOY_DIM)
    # And once recomputed they are verified, so a third run does nothing.
    assert embed_photos(photos, encoder, store).fresh == 5


def test_redo_forces_a_fresh_vector_to_be_recomputed(setup):
    db, store, encoder = setup
    with PhotoIndexReader(db) as reader:
        embed_photos(reader.iter_photos(), encoder, store)
        photos = list(reader.iter_photos())
    target = photos[2].file_hash
    store.add_many([(target, [0.0] * TOY_DIM)], decode_keys={target: store.decode_key_of(target)})
    assert store.get(target) == pytest.approx([0.0] * TOY_DIM)

    report = embed_photos(photos, encoder, store, redo=[target])
    assert report.embedded == 1
    assert report.recomputed == 1
    assert report.fresh == 5, "redo must not pretend the plan found work"
    assert store.get(target) != pytest.approx([0.0] * TOY_DIM)
    assert report.accounted


def test_redo_reports_a_hash_the_index_never_offered(setup):
    """A typo in a hash is otherwise indistinguishable from a hash that was
    already fresh, and the user would believe the redo happened."""
    db, store, encoder = setup
    with PhotoIndexReader(db) as reader:
        photos = list(reader.iter_photos())
    report = embed_photos(photos, encoder, store, redo=["deadbeef"])
    assert report.redo_unknown == ["deadbeef"]
    assert report.accounted


def test_limit_DEFERS_the_rest_rather_than_never_looking(setup):
    """The count of stale vectors is the thing the incident needed to know,
    and it must survive a run that has time to fix only some of them."""
    db, store, encoder = setup
    with PhotoIndexReader(db) as reader:
        photos = list(reader.iter_photos())
    report = embed_photos(photos, encoder, store, limit=2)
    assert report.considered == 5
    assert report.embedded == 2
    assert report.deferred == 3
    assert report.accounted
    assert store.count() == 2
    rest = embed_photos(photos, encoder, store)
    assert rest.embedded == 3
    assert rest.deferred == 0
    assert store.count() == 5
