"""Degrading cleanly when the optional extras are absent.

CI runs with numpy (a dev dependency) but with NO onnxruntime, torch or
transformers, so these tests exercise the real absent-dependency path rather
than a simulated one.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from rekindle.semantic import availability
from rekindle.semantic.availability import (
    EXTRA_CPU,
    EXTRA_GPU,
    Availability,
    SemanticUnavailable,
    probe,
    require,
)


def test_probe_reports_what_is_actually_missing():
    found = probe()
    assert isinstance(found, Availability)
    for name in found.missing_cpu:
        assert importlib.util.find_spec(name) is None
    for name in found.missing_gpu:
        assert importlib.util.find_spec(name) is None


def test_probe_imports_nothing(monkeypatch):
    """find_spec, not import: `rekindle semantic doctor` must stay instant."""
    for name in ("torch", "transformers", "onnxruntime"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    probe()
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


def test_require_gpu_names_the_extra_and_the_command(monkeypatch):
    monkeypatch.setattr(availability, "probe", lambda: Availability(True, False, (), ("torch",)))
    with pytest.raises(SemanticUnavailable) as exc:
        require(gpu=True, feature="Embedding on the GPU")
    message = str(exc.value)
    assert "Embedding on the GPU" in message
    assert EXTRA_GPU in message
    assert "uv sync --extra" in message
    assert "torch" in message


def test_require_cpu_accepts_either_runtime(monkeypatch):
    """An ONNX-only install can do everything the torch install can, slower."""
    monkeypatch.setattr(
        availability, "probe", lambda: Availability(False, True, ("onnxruntime",), ())
    )
    require(gpu=False)  # must not raise


def test_require_cpu_raises_when_nothing_is_installed(monkeypatch):
    monkeypatch.setattr(
        availability,
        "probe",
        lambda: Availability(False, False, ("numpy", "onnxruntime"), ("numpy", "torch")),
    )
    with pytest.raises(SemanticUnavailable) as exc:
        require(feature="Search")
    message = str(exc.value)
    assert "Search" in message
    assert EXTRA_CPU in message
    assert "numpy" in message


def test_require_gpu_passes_when_torch_is_present(monkeypatch):
    monkeypatch.setattr(availability, "probe", lambda: Availability(True, True, (), ()))
    require(gpu=True)


def test_a_module_whose_spec_raises_counts_as_missing(monkeypatch):
    """A half-uninstalled package raises from find_spec itself."""

    def boom(name):
        if name == "onnxruntime":
            raise ValueError("broken install")
        return object()

    monkeypatch.setattr(availability.importlib.util, "find_spec", boom)
    assert "onnxruntime" in probe().missing_cpu


def test_semantic_unavailable_is_a_runtime_error():
    """So a caller that wraps RuntimeError does not miss it."""
    assert issubclass(SemanticUnavailable, RuntimeError)


def test_any_is_true_when_either_runtime_is_present():
    assert Availability(True, False, (), ("torch",)).any
    assert Availability(False, True, ("onnxruntime",), ()).any
    assert not Availability(False, False, ("numpy",), ("numpy",)).any
