"""The import-weight contract, and the model registry's pins.

`rekindle --help` on a default install must not import numpy, torch,
onnxruntime or transformers. That is not a nicety: those imports cost seconds,
and on a machine where torch is half-installed they cost a traceback on a
command that has nothing to do with machine learning.

The check is a SUBPROCESS with a clean interpreter, because by the time this
test file runs, pytest has already imported numpy for the other test modules
and an in-process `sys.modules` check would pass no matter what.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from rekindle.semantic.registry import (
    AESTHETIC_MODELS,
    DEFAULT_AESTHETIC_MODEL,
    DEFAULT_EMBED_MODEL,
    DEFAULT_FACE_MODEL,
    EMBED_MODELS,
    FACE_MODELS,
    RepoPin,
    aesthetic_model,
    all_specs,
    embed_model,
    face_model,
)

HEAVY = ("numpy", "torch", "onnxruntime", "transformers", "tokenizers", "huggingface_hub")


def _import_check(module: str) -> list[str]:
    code = textwrap.dedent(f"""
        import sys
        import {module}  # noqa: F401
        print(",".join(m for m in {HEAVY!r} if m in sys.modules))
    """)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return [m for m in proc.stdout.strip().split(",") if m]


@pytest.mark.parametrize(
    "module",
    [
        "rekindle.cli",
        "rekindle.semantic",
        "rekindle.semantic.cli",
        "rekindle.semantic.registry",
        "rekindle.semantic.store",
        "rekindle.semantic.photos",
        "rekindle.semantic.availability",
        "rekindle.semantic.runtime",
    ],
)
def test_module_imports_nothing_heavy(module):
    assert _import_check(module) == []


def test_the_check_can_fail():
    """A test you have not seen fail is not a test - so fail it on purpose."""
    assert _import_check("numpy") == ["numpy"]


def test_help_works_and_lists_the_semantic_group():
    from typer.testing import CliRunner

    from rekindle.cli import app

    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "semantic" in result.stdout


# ------------------------------------------------------------------- registry


def test_every_revision_is_a_full_commit_sha():
    """`revision="main"` is not a pin, and two runs of it are two models."""
    for spec in all_specs():
        pins = [p for p in (spec.torch, spec.onnx) if p] if hasattr(spec, "torch") else [spec.pin]
        for pin in pins:
            assert len(pin.revision) == 40
            assert set(pin.revision) <= set("0123456789abcdef")


def test_a_branch_name_is_rejected_at_construction():
    with pytest.raises(ValueError, match="not a pin"):
        RepoPin("some/repo", "main", ("config.json",))


def test_a_short_sha_is_rejected():
    with pytest.raises(ValueError, match="40-hex"):
        RepoPin("some/repo", "32bd6428", ("config.json",))


def test_every_model_records_a_licence_with_a_url():
    for spec in all_specs():
        assert spec.licence.spdx
        assert spec.licence.url.startswith("https://")


def test_defaults_exist_and_are_permissively_licensed():
    assert embed_model().key == DEFAULT_EMBED_MODEL
    assert aesthetic_model().key == DEFAULT_AESTHETIC_MODEL
    assert face_model().key == DEFAULT_FACE_MODEL
    for spec in (embed_model(), aesthetic_model(), face_model()):
        assert spec.licence.spdx in {"MIT", "Apache-2.0"}, (
            f"{spec.key} defaults to a non-permissive licence ({spec.licence.spdx}); "
            "a default must not silently bind a user to one."
        )


def test_a_non_permissive_model_is_never_the_default():
    """scrfd-10g is non-commercial. It may be offered; it may not be assumed."""
    restrictive = [s for s in all_specs() if s.licence.spdx not in {"MIT", "Apache-2.0"}]
    assert restrictive, "this test is vacuous if no restrictive model is registered"
    for spec in restrictive:
        assert spec.key not in {
            DEFAULT_EMBED_MODEL,
            DEFAULT_AESTHETIC_MODEL,
            DEFAULT_FACE_MODEL,
        }
        assert spec.licence.note, f"{spec.key} must explain its restriction"


def test_unknown_keys_list_what_is_known():
    with pytest.raises(KeyError, match="clip-vit-l14"):
        embed_model("nope")
    with pytest.raises(KeyError):
        face_model("nope")
    with pytest.raises(KeyError):
        aesthetic_model("nope")


def test_the_default_model_has_both_runtimes():
    """The ONNX fallback is a requirement, not an aspiration."""
    assert set(embed_model().runtimes()) == {"torch", "onnx"}


def test_a_torch_only_model_says_so_rather_than_guessing():
    siglip = EMBED_MODELS["siglip-so400m"]
    assert siglip.runtimes() == ("torch",)
    with pytest.raises(KeyError, match="no onnx build"):
        siglip.pin("onnx")


def test_the_aesthetic_head_declares_the_space_it_scores():
    head = AESTHETIC_MODELS[DEFAULT_AESTHETIC_MODEL]
    assert head.embed_key in EMBED_MODELS
    assert head.in_dim == EMBED_MODELS[head.embed_key].dim


def test_face_models_are_onnx_files():
    for spec in FACE_MODELS.values():
        assert all(f.endswith(".onnx") for f in spec.pin.files)
        assert spec.family in {"yolo", "scrfd"}


def test_the_onnx_pin_names_the_single_tower_graphs_not_the_combined_one():
    """FOUND BY RUNNING the ONNX path against real photos.

    `Xenova/clip-vit-large-patch14` ships three graphs whose names differ by
    one word: `onnx/model.onnx` is the WHOLE CLIP model and requires
    input_ids, pixel_values AND attention_mask in a single call; only
    `onnx/vision_model.onnx` and `onnx/text_model.onnx` take one input each
    and return the 768-d projected embedding.

    Pinning the combined graph looked entirely correct in the file listing and
    passed every test, because no test loaded the real session. It failed at
    the first real image with "Required inputs (['pixel_values',
    'attention_mask']) are missing from input feed (['input_ids'])".
    """
    pin = embed_model("clip-vit-l14").pin("onnx")
    assert "onnx/vision_model.onnx" in pin.files
    assert "onnx/text_model.onnx" in pin.files
    assert "onnx/model.onnx" not in pin.files, (
        "onnx/model.onnx is the combined CLIP graph; it cannot be given images on their own"
    )
