"""Every model this milestone can fetch: pinned, checksummed, licence recorded.

THREE RULES, and they are the reason this file exists rather than a few
`from_pretrained("some/repo")` calls scattered through the code.

1. **Pin the revision.** `revision="main"` means "whatever that repo contains
   the day you run it". Embeddings from two revisions of the same model are
   not comparable, and nothing downstream would notice - a search would just
   get quietly worse. Every entry below pins a 40-hex commit.

2. **Verify what landed.** These are third-party binaries being written to
   someone's disk. `rekindle semantic setup` hashes every file it fetched and
   compares against `model-locks.json`, which is generated from a real
   download by `scripts/lock_models.py` and committed. A mismatch is an error,
   not a warning.

3. **Record the licence.** This repo is MIT and public. rekindle SHIPS no
   weights - it fetches them at the user's request - but a user is entitled to
   know what they just agreed to, so `rekindle semantic licences` prints this
   table, and `setup` prints the licence of anything it is about to fetch.

Nothing here imports torch, transformers or onnxruntime: the registry is
metadata, and `rekindle semantic licences` must work on a default install.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Runtime = Literal["torch", "onnx"]


@dataclass(frozen=True)
class RepoPin:
    """One Hugging Face repo at one commit, and the files wanted from it."""

    repo_id: str
    revision: str
    files: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.revision) != 40 or not all(c in "0123456789abcdef" for c in self.revision):
            raise ValueError(
                f"{self.repo_id}: revision must be a full 40-hex commit sha, "
                f"got {self.revision!r}. A branch name is not a pin."
            )


@dataclass(frozen=True)
class Licence:
    spdx: str
    url: str
    note: str = ""


@dataclass(frozen=True)
class EmbedModel:
    """An image/text embedding model, in one or both runtimes."""

    key: str
    dim: int
    image_size: int
    description: str
    licence: Licence
    torch: RepoPin | None = None
    onnx: RepoPin | None = None
    #: SigLIP tokenisers need `padding="max_length"`; CLIP's must not have it.
    text_padding: str = "max_length"
    kind: str = field(default="embed", init=False)

    def pin(self, runtime: Runtime) -> RepoPin:
        pin = self.torch if runtime == "torch" else self.onnx
        if pin is None:
            raise KeyError(
                f"model '{self.key}' has no {runtime} build. "
                f"Available: {', '.join(self.runtimes())}"
            )
        return pin

    def runtimes(self) -> tuple[str, ...]:
        return tuple(r for r, p in (("torch", self.torch), ("onnx", self.onnx)) if p)


@dataclass(frozen=True)
class HeadModel:
    """A small head that scores an embedding produced by `embed_key`."""

    key: str
    embed_key: str
    in_dim: int
    description: str
    licence: Licence
    pin: RepoPin
    kind: str = field(default="aesthetic", init=False)


@dataclass(frozen=True)
class FaceModel:
    """A purpose-built face DETECTOR. Not a recogniser, and never CLIP."""

    key: str
    description: str
    licence: Licence
    pin: RepoPin
    family: Literal["scrfd", "yolo"]
    input_size: int
    kind: str = field(default="face", init=False)


_APACHE_2 = Licence("Apache-2.0", "https://www.apache.org/licenses/LICENSE-2.0")

EMBED_MODELS: dict[str, EmbedModel] = {
    "clip-vit-l14": EmbedModel(
        key="clip-vit-l14",
        dim=768,
        image_size=224,
        description="OpenAI CLIP ViT-L/14. 304M params. The default: it is the "
        "only candidate with a public pre-built ONNX export AND a public "
        "aesthetic head trained on its embedding space.",
        licence=Licence(
            "MIT",
            "https://github.com/openai/CLIP/blob/main/LICENSE",
            "The HF model card carries no SPDX tag; MIT is the licence of the "
            "upstream openai/CLIP repository these weights come from.",
        ),
        torch=RepoPin(
            repo_id="openai/clip-vit-large-patch14",
            revision="32bd64288804d66eefd0ccbe215aa642df71cc41",
            files=(
                "config.json",
                "model.safetensors",
                "preprocessor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "special_tokens_map.json",
            ),
        ),
        onnx=RepoPin(
            repo_id="Xenova/clip-vit-large-patch14",
            revision="c307790166907339eed5a9a53a249af534102536",
            files=(
                "config.json",
                "preprocessor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                # vision_model / text_model, NOT the combined `onnx/model.onnx`.
                # The combined graph is the whole CLIP model: it requires
                # input_ids, pixel_values AND attention_mask in one call and
                # returns logits. Feeding it images alone fails with
                # "Required inputs (['pixel_values', 'attention_mask']) are
                # missing" - found by running the ONNX path against real
                # photos, not by reading the file list. These two each take
                # one input and return the 768-d projected embedding directly.
                "onnx/vision_model.onnx",
                "onnx/text_model.onnx",
            ),
        ),
        text_padding="max_length",
    ),
    "siglip-so400m": EmbedModel(
        key="siglip-so400m",
        dim=1152,
        image_size=384,
        description="Google SigLIP SO400M patch14-384. 878M params, stronger "
        "zero-shot scene understanding than CLIP ViT-L/14, but torch-only: no "
        "public ONNX export exists, so the CPU path cannot use it.",
        licence=_APACHE_2,
        torch=RepoPin(
            repo_id="google/siglip-so400m-patch14-384",
            revision="9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
            files=(
                "config.json",
                "model.safetensors",
                "preprocessor_config.json",
                "tokenizer_config.json",
                "spiece.model",
                "special_tokens_map.json",
            ),
        ),
        onnx=None,
        text_padding="max_length",
    ),
}

DEFAULT_EMBED_MODEL = "clip-vit-l14"

AESTHETIC_MODELS: dict[str, HeadModel] = {
    "laion-aesthetic-v2": HeadModel(
        key="laion-aesthetic-v2",
        embed_key="clip-vit-l14",
        in_dim=768,
        description="LAION improved-aesthetic-predictor V2: a 5-layer MLP "
        "regressing a 1-10 aesthetic rating from a CLIP ViT-L/14 image "
        "embedding. Trained on SAC + LAION-Logos + AVA.",
        licence=Licence(
            "MIT",
            "https://github.com/christophschuhmann/improved-aesthetic-predictor",
            "Mirrored on the Hub without an SPDX tag; upstream repository is MIT.",
        ),
        pin=RepoPin(
            repo_id="camenduru/improved-aesthetic-predictor",
            revision="7b2449be1264fcd9a1cf92e3d30dd29af989c836",
            files=("sac+logos+ava1-l14-linearMSE.pth",),
        ),
    ),
}

DEFAULT_AESTHETIC_MODEL = "laion-aesthetic-v2"

FACE_MODELS: dict[str, FaceModel] = {
    "yolov11n-face": FaceModel(
        key="yolov11n-face",
        description="YOLOv11-nano fine-tuned for face detection, 10 MB ONNX. "
        "The default: permissively licensed and fast enough to run on CPU.",
        licence=Licence(
            "Apache-2.0",
            "https://huggingface.co/AdamCodd/YOLOv11n-face-detection",
            "Declared Apache-2.0 by the publisher. The Ultralytics TRAINING "
            "framework is AGPL-3.0; only the exported ONNX graph is fetched "
            "and rekindle does not train.",
        ),
        pin=RepoPin(
            repo_id="AdamCodd/YOLOv11n-face-detection",
            revision="0e97fea5eacb1460b35725c929d813c7095841b5",
            files=("model.onnx",),
        ),
        family="yolo",
        input_size=640,
    ),
    "scrfd-10g": FaceModel(
        key="scrfd-10g",
        description="InsightFace SCRFD-10G (the detector half of buffalo_l), "
        "17 MB ONNX. Markedly better than YOLOv11n on small, side-on and "
        "partially occluded faces - which is what a publishing gate is for.",
        licence=Licence(
            "NON-COMMERCIAL",
            "https://github.com/deepinsight/insightface",
            "InsightFace pretrained models are released for NON-COMMERCIAL "
            "research use only. rekindle ships no weights; choosing this model "
            "means YOU fetch them under that licence. Not the default for "
            "exactly this reason.",
        ),
        pin=RepoPin(
            repo_id="immich-app/buffalo_l",
            revision="d09715916a0778919a770c343533641e250b8699",
            files=("detection/model.onnx",),
        ),
        family="scrfd",
        input_size=640,
    ),
}

DEFAULT_FACE_MODEL = "yolov11n-face"


def all_specs() -> list[EmbedModel | HeadModel | FaceModel]:
    return [*EMBED_MODELS.values(), *AESTHETIC_MODELS.values(), *FACE_MODELS.values()]


def embed_model(key: str | None = None) -> EmbedModel:
    return _lookup(EMBED_MODELS, key or DEFAULT_EMBED_MODEL, "embedding model")


def aesthetic_model(key: str | None = None) -> HeadModel:
    return _lookup(AESTHETIC_MODELS, key or DEFAULT_AESTHETIC_MODEL, "aesthetic model")


def face_model(key: str | None = None) -> FaceModel:
    return _lookup(FACE_MODELS, key or DEFAULT_FACE_MODEL, "face detector")


def _lookup(table: dict[str, object], key: str, what: str) -> object:  # type: ignore[misc]
    try:
        return table[key]
    except KeyError:
        raise KeyError(f"unknown {what} '{key}'. Known: {', '.join(sorted(table))}") from None
