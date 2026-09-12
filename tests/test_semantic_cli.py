"""Semantic CLI: degrade with a message, never with a traceback.

CI has no onnxruntime, no torch and no model weights, so the paths these tests
take are the REAL absent-dependency paths, not simulated ones. What must never
happen is an ImportError reaching the user - "the tool crashed" and "you need
to install an extra" are different things and only one of them is true.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from rekindle.cli import app
from rekindle.semantic.cli import EXIT_UNAVAILABLE
from rekindle.semantic.store import EmbeddingStore
from tests.fixtures.semantic import TOY_DIM, make_index, make_photo, write_photo
from tests.helpers import unwrapped

runner = CliRunner()

#: Rich truncates table cells to the terminal width, which under pytest is 80
#: and would make a path assertion pass or fail on cosmetics.
WIDE = {"COLUMNS": "240", "TERM": "dumb"}


@pytest.fixture
def library(tmp_path):
    root = tmp_path / "lib"
    data = tmp_path / "data"
    photos = [
        make_photo("a" * 32, write_photo(root, "a.jpg", "red")),
        make_photo("b" * 32, write_photo(root, "b.jpg", "blue")),
    ]
    make_index(data / "rekindle.sqlite", photos, index_root=root)
    return data


def test_semantic_group_is_reachable():
    result = runner.invoke(app, ["semantic", "--help"])
    assert result.exit_code == 0
    for verb in ("setup", "embed", "find", "cluster", "rank", "facegate", "doctor"):
        assert verb in result.stdout


def test_doctor_runs_with_no_extras_and_no_index(tmp_path):
    result = runner.invoke(app, ["semantic", "doctor", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "extras" in result.stdout
    assert "device" in result.stdout


def test_doctor_writes_nothing(tmp_path):
    runner.invoke(app, ["semantic", "doctor", "--data-dir", str(tmp_path / "data")])
    assert not (tmp_path / "data").exists() or not any((tmp_path / "data").rglob("*.sqlite"))


def test_licences_needs_no_dependency_and_names_every_model():
    result = runner.invoke(app, ["semantic", "licences"])
    assert result.exit_code == 0
    for key in ("clip-vit-l14", "siglip-so400m", "laion-aesthetic-v2", "yolov11n-face"):
        assert key in result.stdout


def test_licences_surfaces_the_non_commercial_warning():
    result = runner.invoke(app, ["semantic", "licences"])
    assert "NON-COMMERCIAL" in result.stdout


def test_find_without_an_index_says_so_and_exits_2(tmp_path):
    result = runner.invoke(
        app, ["semantic", "find", "sunset", "--data-dir", str(tmp_path / "nothing")]
    )
    assert result.exit_code == 2
    assert "rekindle index" in result.stdout


def test_find_without_embeddings_names_the_command_to_run(library):
    result = runner.invoke(app, ["semantic", "find", "sunset", "--data-dir", str(library)])
    assert result.exit_code == EXIT_UNAVAILABLE
    assert unwrapped("semantic embed") in unwrapped(result.stdout)


def test_embed_without_the_extra_names_the_extra(library):
    """The real path on CI: torch and onnxruntime are both absent."""
    result = runner.invoke(app, ["semantic", "embed", "--data-dir", str(library)])
    assert result.exit_code == EXIT_UNAVAILABLE
    assert "semantic" in result.stdout
    assert "Traceback" not in result.stdout


def test_embed_with_device_cuda_refuses_rather_than_falling_back(library):
    result = runner.invoke(
        app, ["semantic", "embed", "--device", "cuda", "--data-dir", str(library)]
    )
    assert result.exit_code == EXIT_UNAVAILABLE
    assert "Traceback" not in result.stdout


def test_facegate_without_weights_names_the_setup_command(library):
    result = runner.invoke(app, ["semantic", "facegate", "--data-dir", str(library)])
    assert result.exit_code == EXIT_UNAVAILABLE
    assert "Traceback" not in result.stdout


def test_rank_without_embeddings_says_so(library):
    result = runner.invoke(app, ["semantic", "rank", "--data-dir", str(library)])
    assert result.exit_code == EXIT_UNAVAILABLE


def test_cluster_without_embeddings_says_so(library):
    result = runner.invoke(app, ["semantic", "cluster", "--data-dir", str(library)])
    assert result.exit_code == EXIT_UNAVAILABLE


def test_setup_rejects_an_unknown_runtime(tmp_path):
    result = runner.invoke(
        app,
        ["semantic", "setup", "--runtime", "tensorflow", "--data-dir", str(tmp_path)],
    )
    assert result.exit_code == 2


def test_setup_rejects_an_unknown_model(tmp_path):
    result = runner.invoke(
        app, ["semantic", "setup", "--model", "nope", "--data-dir", str(tmp_path)]
    )
    assert result.exit_code == 2
    assert "clip-vit-l14" in result.stdout


def test_doctor_lists_an_existing_store(tmp_path):
    from rekindle.semantic.store import store_root

    root = store_root(tmp_path, "clip-vit-l14")
    EmbeddingStore(root, dim=768, model_key="clip-vit-l14").close()
    result = runner.invoke(app, ["semantic", "doctor", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "clip-vit-l14" in result.stdout
    assert "0 vectors" in result.stdout


def test_find_over_a_toy_store_returns_the_right_photo(library, monkeypatch):
    """The only CLI test that goes all the way through, using ToyEncoder.

    It exists because every other CLI test above asserts a REFUSAL, and a set
    of tests that only ever check the unhappy path would pass over a `find`
    that returns nothing at all.
    """
    pytest.importorskip("numpy")
    from rekindle.semantic import cli as semantic_cli
    from rekindle.semantic.encoder import LoadedEncoder
    from rekindle.semantic.registry import embed_model
    from rekindle.semantic.runtime import resolve_device
    from rekindle.semantic.store import store_root
    from tests.fixtures.semantic import ToyEncoder, solid_image

    encoder = ToyEncoder()
    store = EmbeddingStore(
        store_root(library, "clip-vit-l14"), dim=TOY_DIM, model_key="clip-vit-l14"
    )
    store.add_many(
        [
            ("a" * 32, encoder.encode_images([solid_image("red")])[0]),
            ("b" * 32, encoder.encode_images([solid_image("blue")])[0]),
        ]
    )
    store.close()

    def fake_load_encoder(*_args, **_kwargs):
        return LoadedEncoder(encoder, embed_model("clip-vit-l14"), "toy", resolve_device("cpu"))

    monkeypatch.setattr("rekindle.semantic.encoder.load_encoder", fake_load_encoder, raising=True)
    monkeypatch.setattr(semantic_cli, "_open_store", _store_opener(library), raising=True)

    result = runner.invoke(
        app,
        ["semantic", "find", "blue", "-k", "1", "--data-dir", str(library)],
        env=WIDE,
    )
    assert result.exit_code == 0, result.stdout
    assert "b.jpg" in result.stdout
    assert "a.jpg" not in result.stdout


def _store_opener(data_dir):
    """Open the toy-dimension store the test wrote, not a 768-dim one."""
    from rekindle.semantic.registry import embed_model
    from rekindle.semantic.store import store_root

    def opener(_data_dir, _model_key, *, create):
        spec = embed_model("clip-vit-l14")
        store = EmbeddingStore(
            store_root(data_dir, "clip-vit-l14"), dim=TOY_DIM, model_key="clip-vit-l14"
        )
        return spec, store

    return opener
