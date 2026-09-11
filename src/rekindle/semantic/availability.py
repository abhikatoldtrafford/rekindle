"""Is the optional `semantic` extra installed, and what is missing if not.

Every semantic feature must degrade to a clear sentence, never an ImportError
traceback and never a crash at startup. That means:

  * nothing in `rekindle.semantic` may import numpy / onnxruntime / torch /
    transformers at MODULE level - `tests/test_semantic_imports.py` pins this;
  * each feature calls `require(...)` at the top of its entry point, which
    raises `SemanticUnavailable` carrying the exact install command;
  * `rekindle.cli` catches `SemanticUnavailable` and prints it as a message
    with exit code 3.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

#: The light extra: everything except torch. Enough for CPU inference via ONNX.
EXTRA_CPU = "semantic"
#: The heavy extra: adds torch + transformers, and on Windows/Linux a CUDA build.
EXTRA_GPU = "semantic-gpu"

_CPU_MODULES = ("numpy", "onnxruntime", "huggingface_hub")
_GPU_MODULES = ("numpy", "torch", "transformers", "huggingface_hub")


class SemanticUnavailable(RuntimeError):
    """A semantic feature was asked for without the dependencies to run it.

    Carries a message that names the missing modules AND the command that
    installs them. A bare ImportError does neither.
    """


@dataclass(frozen=True)
class Availability:
    """What is importable right now. Cheap: uses find_spec, imports nothing."""

    cpu: bool
    gpu: bool
    missing_cpu: tuple[str, ...]
    missing_gpu: tuple[str, ...]

    @property
    def any(self) -> bool:
        return self.cpu or self.gpu


def _missing(modules: tuple[str, ...]) -> tuple[str, ...]:
    out = []
    for name in modules:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            # A half-installed package can raise from find_spec itself. Treat
            # it as missing: that is exactly what the user needs to hear.
            found = False
        if not found:
            out.append(name)
    return tuple(out)


def probe() -> Availability:
    missing_cpu = _missing(_CPU_MODULES)
    missing_gpu = _missing(_GPU_MODULES)
    return Availability(
        cpu=not missing_cpu,
        gpu=not missing_gpu,
        missing_cpu=missing_cpu,
        missing_gpu=missing_gpu,
    )


def require(*, gpu: bool = False, feature: str = "This feature") -> None:
    """Raise `SemanticUnavailable` unless the needed extra is importable.

    `gpu=True` demands torch; `gpu=False` accepts EITHER runtime, because an
    ONNX-only install can do everything the torch install can, only slower.
    """
    found = probe()
    if gpu:
        if found.gpu:
            return
        raise SemanticUnavailable(
            f"{feature} needs the '{EXTRA_GPU}' extra "
            f"(missing: {', '.join(found.missing_gpu)}). "
            f"Install it with:  uv sync --extra {EXTRA_GPU}"
        )
    if found.any:
        return
    raise SemanticUnavailable(
        f"{feature} needs the '{EXTRA_CPU}' extra "
        f"(missing: {', '.join(found.missing_cpu)}). "
        f"Install it with:  uv sync --extra {EXTRA_CPU}   "
        f"(or --extra {EXTRA_GPU} for the CUDA runtime)"
    )
