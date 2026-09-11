"""The real CLIP encoder, when this machine happens to have it.

EVERY TEST HERE SKIPS unless torch, transformers AND the pinned weights are
all present in `./data/models`. CI has none of them and must stay green; these
run on a developer machine that has done `rekindle semantic setup`.

They exist because mutation testing found a hole that CI structurally cannot
cover: deleting the body of `TorchEncoder.encode_images` left the whole suite
green, since nothing in CI ever loads a real model. The two things that hole
hides are the two things it would be worst to get wrong -

  * `encode_images` and `prepare_images` + `encode_prepared` drifting apart,
    which fills a store with vectors that depend on which path wrote them;
  * `local_files_only` regressing, which turns an offline tool into one that
    fetches 1.7 GB the first time anyone touches it. That one has already
    happened once: an unreviewed commit flipped it to `False` and three tests
    caught it, and this file is the direct check underneath them.

Nothing here asserts a throughput number. Timings belong in the audit, where
they can be re-measured; a test that pins img/s fails on a busy machine and
teaches everyone to ignore it.
"""

from __future__ import annotations

import importlib.util
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from rekindle.semantic.availability import SemanticUnavailable
from rekindle.semantic.encoder import PreparingEncoder
from rekindle.semantic.registry import embed_model

CACHE = Path("data/models").resolve()
MODEL = "clip-vit-l14"


def _weights_present() -> bool:
    spec = embed_model(MODEL)
    repo = spec.pin("torch").repo_id.replace("/", "--")
    snaps = CACHE / f"models--{repo}" / "snapshots"
    return snaps.is_dir() and any(snaps.iterdir())


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("transformers") is None
    or not _weights_present(),
    reason="needs torch, transformers and the pinned CLIP weights in ./data/models",
)


@pytest.fixture(scope="module")
def encoder():
    """One load for the module. Loading CLIP costs about seven seconds."""
    from rekindle.semantic.encoder import TorchEncoder

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return TorchEncoder(embed_model(MODEL), device="auto", cache_dir=CACHE)


@pytest.fixture(scope="module")
def images():
    from PIL import Image, ImageDraw

    out = []
    for i in range(6):
        im = Image.new("RGB", (320, 240), (20 + 30 * i, 90, 200 - 25 * i))
        d = ImageDraw.Draw(im)
        d.ellipse((40 + 10 * i, 30, 180 + 10 * i, 170), fill=(240, 200 - 20 * i, 30))
        d.rectangle((10, 190, 300, 230), fill=(5, 5 + 20 * i, 5))
        out.append(im)
    return out


# ---------------------------------------------------------- the split is safe


def test_the_split_pair_composes_to_encode_images(encoder, images):
    """THE test this file exists for.

    `encode_images` is written as `encode_prepared(prepare_images(images))`
    precisely so the two cannot diverge, but that is an implementation choice
    a refactor could undo. If it is ever undone and the halves drift, a store
    written by `rekindle embed` (prepared path) and queried by anything
    calling `encode_images` holds two subtly different embeddings in one
    space, and NOTHING downstream can detect it - every vector is still unit
    length and still plausible.

    Bit-for-bit, not approximately: same tensors, same kernels, same order.
    """
    direct = encoder.encode_images(images)
    staged = encoder.encode_prepared(encoder.prepare_images(images))
    assert len(direct) == len(staged) == len(images)
    assert direct == staged


def test_preparing_does_not_touch_the_gpu(encoder, images):
    """`prepare_images` runs on six worker threads at once.

    It must stay on the CPU: CUDA work from several python threads serialises
    on the context and would hand back the concurrency the pool just bought.
    Checked by device, not by timing.
    """
    prepared = encoder.prepare_images(images)
    assert prepared.device.type == "cpu"


def test_the_real_encoder_satisfies_the_preparing_protocol(encoder):
    assert isinstance(encoder, PreparingEncoder)


def test_batching_does_not_change_a_vector_beyond_fp16_noise(encoder, images):
    """One at a time versus all at once.

    fp16 reduction order depends on batch shape, so these are NOT expected to
    be bit-identical - but they must be the same vector to well inside any
    margin that matters for search or clustering. Measured at 0.99998 on real
    photos; 1e-3 here is a floor, not a target.
    """
    together = encoder.encode_images(images)
    apart = [encoder.encode_images([im])[0] for im in images]
    for a, b in zip(together, apart, strict=True):
        cosine = sum(x * y for x, y in zip(a, b, strict=True))
        assert cosine > 1 - 1e-3, f"batching moved a vector: cosine {cosine}"


# ------------------------------------------------------------ vectors are real


def test_the_vectors_are_unit_length(encoder, images):
    for vec in encoder.encode_images(images):
        assert len(vec) == embed_model(MODEL).dim
        assert abs(sum(v * v for v in vec) ** 0.5 - 1.0) < 1e-3


def test_text_and_images_land_in_one_space_and_the_right_way_round(encoder):
    """Not "it returned 768 numbers": the RANKING has to be right.

    A yellow circle on blue must be nearer to "a yellow circle" than to "a
    page of dense text", and the reverse for the text image. A transposed
    projection or a dropped normalisation still yields unit vectors and would
    pass every shape assertion in this file.
    """
    from PIL import Image, ImageDraw

    circle = Image.new("RGB", (224, 224), (20, 60, 200))
    ImageDraw.Draw(circle).ellipse((40, 40, 184, 184), fill=(250, 220, 40))

    text = Image.new("RGB", (224, 224), (250, 250, 250))
    d = ImageDraw.Draw(text)
    for y in range(20, 200, 12):
        d.rectangle((20, y, 200, y + 5), fill=(20, 20, 20))

    img = encoder.encode_images([circle, text])
    txt = encoder.encode_texts(["a large yellow circle", "a page of dense printed text"])

    def cos(a, b):
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert cos(img[0], txt[0]) > cos(img[0], txt[1]), "the circle matched the text query"
    assert cos(img[1], txt[1]) > cos(img[1], txt[0]), "the text page matched the circle query"


def test_the_same_image_twice_gives_the_same_vector(encoder, images):
    """Determinism. A non-deterministic encoder makes a resumable store a lie."""
    assert encoder.encode_images(images[:3]) == encoder.encode_images(images[:3])


def test_an_empty_batch_is_an_empty_answer_not_a_crash(encoder):
    assert encoder.encode_images([]) == []
    assert encoder.encode_texts([]) == []
    assert encoder.encode_prepared(encoder.prepare_images([])) == []


# ---------------------------------------------------------------- offline


def test_an_empty_cache_refuses_instead_of_downloading(tmp_path):
    """The offline contract, exercised rather than assumed.

    An unreviewed commit changed `local_files_only` to a constant `False`,
    which makes `from_pretrained` consider the network no matter what the
    caller asked for. With that change in place this test does not raise - it
    downloads 1.7 GB into `tmp_path`. That is the whole point of it.
    """
    from rekindle.semantic.encoder import TorchEncoder

    empty = tmp_path / "no-models"
    empty.mkdir()
    with pytest.raises(SemanticUnavailable) as exc:
        TorchEncoder(embed_model(MODEL), device="cpu", cache_dir=empty)
    message = str(exc.value)
    assert "semantic setup" in message, "the error must say what to run"
    assert not any(empty.iterdir()), f"something was downloaded into {empty}"


def test_the_populated_cache_loads_with_the_network_switched_off(encoder):
    """The other half: offline must not mean broken."""
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert encoder.dim == embed_model(MODEL).dim
    assert encoder.key == MODEL


class _CountingHandler(BaseHTTPRequestHandler):
    """Answers 404 immediately and keeps score. Never serves a model."""

    def do_HEAD(self):  # noqa: N802 - BaseHTTPRequestHandler's naming
        self.server.hits.append(self.path)  # type: ignore[attr-defined]
        self.send_response(404)
        self.end_headers()

    do_GET = do_HEAD

    def log_message(self, *_args):  # keep pytest output clean
        return


def test_loading_from_the_cache_makes_no_network_request_at_all(monkeypatch):
    """The offline guarantee, measured rather than asserted.

    HF_HUB_OFFLINE is deliberately UNSET here. With it set, this test passes
    no matter what the code does - the environment does the work and the
    guarantee is untested, which is exactly the hole the inherited
    `local_files_only: False` commit slipped through. So instead: point the
    hub at a local socket that counts what arrives, and require the count to
    be zero.

    Measured on this machine with the guard removed, against an endpoint that
    refuses connections: the same load took **385.8 s instead of 15.0 s**,
    because huggingface_hub retries a HEAD five times per file before falling
    back to the cache. On a genuinely offline machine that is six and a half
    minutes of nothing, added to every single encoder load.
    """
    from rekindle.semantic.encoder import TorchEncoder

    server = HTTPServer(("127.0.0.1", 0), _CountingHandler)
    server.hits = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
        monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{server.server_port}")
        monkeypatch.setenv("HF_HUB_ETAG_TIMEOUT", "1")
        encoder = TorchEncoder(embed_model(MODEL), device="cpu", cache_dir=CACHE)
        assert encoder.key == MODEL, "the cached model did not load"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert server.hits == [], (  # type: ignore[attr-defined]
        f"the encoder reached for the network {len(server.hits)} time(s) "  # type: ignore[attr-defined]
        f"while loading from a populated cache: {server.hits[:3]}"  # type: ignore[attr-defined]
    )


def test_device_auto_picks_cuda_when_the_card_is_present(encoder):
    """A silent CPU fallback on a machine with a good GPU is a bug, not a
    graceful degradation - the whole point of `runtime.py`. Skipped rather
    than asserted when there is no CUDA, so it stays honest on a laptop."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA on this machine")
    assert encoder.report.device == "cuda"
    param = next(encoder._model.parameters())
    assert param.device.type == "cuda", "the model is not on the card"
    assert param.dtype is torch.float16, "CUDA should use fp16; fp32 halves throughput"
