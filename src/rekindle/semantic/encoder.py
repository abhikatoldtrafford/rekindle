"""Image/text encoders. One protocol, three implementations, no global state.

`ImageTextEncoder` is deliberately tiny - two methods over plain lists of
floats - because everything downstream (search, clustering, aesthetics) is
tested against a toy encoder that implements it in thirty lines. A test that
mocks a model and asserts the mock was called proves nothing; a test that runs
the real clustering over a real (if weak) embedding function proves the
clustering.

Heavy imports (`torch`, `transformers`, `onnxruntime`, `numpy`) happen INSIDE
the constructors, never at module level. `rekindle --help` must not pay for
them and CI must not need them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from rekindle.semantic.availability import SemanticUnavailable
from rekindle.semantic.registry import EmbedModel, embed_model
from rekindle.semantic.runtime import Device, DeviceReport, resolve_device

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image

Vector = list[float]


class EncodeError(RuntimeError):
    """The encoder could not produce a vector for an input."""


@runtime_checkable
class ImageTextEncoder(Protocol):
    """Maps images and text into ONE shared, L2-normalised vector space."""

    key: str
    dim: int

    def encode_images(self, images: Sequence[Image]) -> list[Vector]:
        """One unit vector per image, in order."""
        ...

    def encode_texts(self, texts: Sequence[str]) -> list[Vector]:
        """One unit vector per string, in order, comparable with images."""
        ...


def _normalise(rows: list[list[float]]) -> list[Vector]:
    out = []
    for row in rows:
        norm = sum(v * v for v in row) ** 0.5
        if norm == 0.0:
            raise EncodeError("encoder produced a zero vector; it cannot be normalised")
        out.append([v / norm for v in row])
    return out


@dataclass(frozen=True)
class LoadedEncoder:
    """An encoder plus how it was loaded, so the CLI can report both."""

    encoder: ImageTextEncoder
    spec: EmbedModel
    runtime: str
    device: DeviceReport


class TorchEncoder:
    """transformers + torch. CUDA in fp16 when available, else CPU in fp32.

    fp16 ON CPU IS A TRAP and is not used: torch's CPU fp16 kernels fall back
    to slow reference implementations, so a "faster" dtype makes the CPU path
    several times slower than fp32.
    """

    def __init__(
        self,
        spec: EmbedModel,
        *,
        device: Device = "auto",
        cache_dir: Path | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:  # pragma: no cover - covered by availability
            raise SemanticUnavailable(
                f"the torch runtime needs the 'semantic-gpu' extra ({exc})"
            ) from exc

        self.spec = spec
        self.key = spec.key
        self.dim = spec.dim
        self.report = resolve_device(device)
        self._torch = torch
        pin = spec.pin("torch")
        kwargs = {"revision": pin.revision}
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        self._proc = AutoProcessor.from_pretrained(pin.repo_id, **kwargs)
        dtype = torch.float16 if self.report.device == "cuda" else torch.float32
        self._model = (
            AutoModel.from_pretrained(pin.repo_id, dtype=dtype, **kwargs)
            .to(self.report.device)
            .eval()
        )
        self._dtype = dtype

    def encode_images(self, images: Sequence[Image]) -> list[Vector]:
        if not images:
            return []
        torch = self._torch
        inputs = self._proc(images=list(images), return_tensors="pt")
        px = inputs["pixel_values"].to(self.report.device, dtype=self._dtype)
        with torch.inference_mode():
            feats = self._model.get_image_features(pixel_values=px)
        return _normalise(feats.float().cpu().tolist())

    def encode_texts(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        torch = self._torch
        inputs = self._proc.tokenizer(
            list(texts),
            padding=self.spec.text_padding,
            truncation=True,
            return_tensors="pt",
        )
        moved = {k: v.to(self.report.device) for k, v in inputs.items()}
        with torch.inference_mode():
            feats = self._model.get_text_features(**moved)
        return _normalise(feats.float().cpu().tolist())


class OnnxEncoder:
    """onnxruntime + tokenizers + Pillow. No torch, no transformers.

    Preprocessing is reimplemented from the repo's `preprocessor_config.json`
    rather than assumed, because getting it wrong is silent: a wrong
    normalisation still yields plausible unit vectors and merely makes every
    result slightly worse. `tests/test_semantic_onnx_preprocess.py` pins the
    arithmetic against hand-computed values.
    """

    def __init__(self, spec: EmbedModel, *, model_dir: Path) -> None:
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise SemanticUnavailable(
                f"the ONNX runtime needs the 'semantic' extra ({exc})"
            ) from exc

        self.spec = spec
        self.key = spec.key
        self.dim = spec.dim
        self._np = np
        self.model_dir = model_dir
        vision = model_dir / "onnx" / "model.onnx"
        text = model_dir / "onnx" / "text_model.onnx"
        for path in (vision, text):
            if not path.is_file():
                raise SemanticUnavailable(
                    f"{path} is missing. Run `rekindle semantic setup "
                    f"--model {spec.key} --runtime onnx` first."
                )
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        self._vision = ort.InferenceSession(str(vision), opts, providers=providers)
        self._text = ort.InferenceSession(str(text), opts, providers=providers)
        self._tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._pre = _load_preprocess(model_dir / "preprocessor_config.json")

    def encode_images(self, images: Sequence[Image]) -> list[Vector]:
        if not images:
            return []
        np = self._np
        batch = np.stack([preprocess_image(im, self._pre) for im in images])
        name = self._vision.get_inputs()[0].name
        out = self._vision.run(None, {name: batch})
        return _normalise([list(map(float, row)) for row in _pick_embedding(out, self.dim)])

    def encode_texts(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        np = self._np
        encoded = [self._tok.encode(t) for t in texts]
        width = max(len(e.ids) for e in encoded)
        ids = np.zeros((len(encoded), width), dtype=np.int64)
        mask = np.zeros((len(encoded), width), dtype=np.int64)
        for i, enc in enumerate(encoded):
            ids[i, : len(enc.ids)] = enc.ids
            mask[i, : len(enc.ids)] = 1
        feed = {"input_ids": ids, "attention_mask": mask}
        wanted = {i.name for i in self._text.get_inputs()}
        out = self._text.run(None, {k: v for k, v in feed.items() if k in wanted})
        return _normalise([list(map(float, row)) for row in _pick_embedding(out, self.dim)])


def _pick_embedding(outputs: list, dim: int) -> list:
    """Choose the 2-D output whose width is the embedding dim.

    The Xenova exports return several heads (pooled output, last hidden
    state); indexing `[0]` happens to be right today and would silently start
    returning token states if the export changed. Selecting by shape is the
    check that cannot drift.
    """
    for arr in outputs:
        if getattr(arr, "ndim", 0) == 2 and arr.shape[1] == dim:
            return arr
    shapes = [getattr(a, "shape", None) for a in outputs]
    raise EncodeError(
        f"no ONNX output has shape (batch, {dim}); got {shapes}. "
        "The export does not match the registry's declared dim."
    )


@dataclass(frozen=True)
class Preprocess:
    size: int
    crop: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    rescale: float
    resample: int


def _load_preprocess(path: Path) -> Preprocess:
    import json

    from PIL import Image as PILImage

    cfg = json.loads(path.read_text(encoding="utf-8"))
    size = cfg.get("size", {})
    shortest = size.get("shortest_edge") or size.get("height") or 224
    crop = cfg.get("crop_size", {})
    crop_px = crop.get("height") or crop.get("shortest_edge") or shortest
    resample = {0: PILImage.NEAREST, 2: PILImage.BILINEAR, 3: PILImage.BICUBIC}.get(
        cfg.get("resample", 3), PILImage.BICUBIC
    )
    return Preprocess(
        size=int(shortest),
        crop=int(crop_px),
        mean=tuple(cfg.get("image_mean", [0.48145466, 0.4578275, 0.40821073])),
        std=tuple(cfg.get("image_std", [0.26862954, 0.26130258, 0.27577711])),
        rescale=float(cfg.get("rescale_factor", 1 / 255)),
        resample=resample,
    )


def preprocess_image(image: Image, pre: Preprocess):
    """Resize shortest edge -> centre crop -> rescale -> normalise -> CHW."""
    import numpy as np

    im = image.convert("RGB")
    w, h = im.size
    if w <= h:
        new = (pre.size, max(1, round(h * pre.size / w)))
    else:
        new = (max(1, round(w * pre.size / h)), pre.size)
    im = im.resize(new, pre.resample)
    left = (im.width - pre.crop) // 2
    top = (im.height - pre.crop) // 2
    im = im.crop((left, top, left + pre.crop, top + pre.crop))
    arr = np.asarray(im, dtype=np.float32) * pre.rescale
    arr = (arr - np.asarray(pre.mean, dtype=np.float32)) / np.asarray(pre.std, dtype=np.float32)
    return np.transpose(arr, (2, 0, 1))


def load_encoder(
    model_key: str | None = None,
    *,
    runtime: str = "auto",
    device: Device = "auto",
    cache_dir: Path | None = None,
    onnx_dir: Path | None = None,
) -> LoadedEncoder:
    """Build the best encoder available for `model_key`.

    `runtime="auto"` prefers torch when it is importable AND the model has a
    torch pin, because on this class of hardware torch+CUDA is two orders of
    magnitude faster. It falls back to ONNX rather than failing, and the
    returned `LoadedEncoder` says which was used - the caller prints it.
    """
    spec = embed_model(model_key)
    chosen = runtime
    if runtime == "auto":
        chosen = "torch" if _torch_importable() and spec.torch else "onnx"
    if chosen == "torch":
        enc = TorchEncoder(spec, device=device, cache_dir=cache_dir)
        return LoadedEncoder(enc, spec, "torch", enc.report)
    if spec.onnx is None:
        raise SemanticUnavailable(
            f"model '{spec.key}' has no ONNX build, and torch is not available. "
            f"Either install the 'semantic-gpu' extra, or use a model with an "
            f"ONNX build: "
            f"{', '.join(k for k, m in _onnx_capable().items() if m)}"
        )
    if onnx_dir is None:
        raise SemanticUnavailable("the ONNX runtime needs a model directory")
    enc = OnnxEncoder(spec, model_dir=onnx_dir)
    return LoadedEncoder(enc, spec, "onnx", resolve_device("cpu"))


def _torch_importable() -> bool:
    import importlib.util

    return importlib.util.find_spec("torch") is not None


def _onnx_capable() -> dict[str, bool]:
    from rekindle.semantic.registry import EMBED_MODELS

    return {k: m.onnx is not None for k, m in EMBED_MODELS.items()}
