"""Semantic understanding: embeddings, search, scene clusters, aesthetics, faces.

Everything here is behind an OPTIONAL dependency extra. Importing this package
must stay free of numpy, torch, onnxruntime and transformers so that
`rekindle --help` works, and the test suite runs, on a machine that installed
none of them. Submodules that need those imports guard them at call time and
raise `SemanticUnavailable` with an actionable message; see `availability.py`.
"""

from __future__ import annotations

from rekindle.semantic.availability import SemanticUnavailable

__all__ = ["SemanticUnavailable"]
