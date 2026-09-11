"""`rekindle semantic embed`, end to end with a real encoder and real JPEGs.

No model, no GPU, no network: `ToyEncoder` is a genuine embedding function and
the photos are Pillow-written JPEGs in tmp_path. What is under test is the
driver - resumption, accounting, decode failures, ordering - and every
assertion is about the vectors that came out, not about calls that went in.
"""

from __future__ import annotations

import pytest

from rekindle.semantic.embed import DEFAULT_BATCH, embed_photos
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
