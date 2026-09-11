"""ONNX-path preprocessing, pinned against hand-computed values.

This arithmetic is reimplemented from `preprocessor_config.json` rather than
borrowed from transformers, and getting it wrong is SILENT: a wrong
normalisation still produces plausible unit vectors and merely makes every
search result slightly worse for ever. So the numbers are computed by hand
here rather than snapshotted from the code's own output, which would pin
whatever it currently does.

The config file is a small JSON copied from the real `Xenova/clip-vit-large-
patch14` repository (7 keys, all values verbatim) - not an invented one.
"""

from __future__ import annotations

import json

import pytest

from rekindle.semantic.encoder import (
    Preprocess,
    _load_preprocess,
    _pick_embedding,
    features_of,
    preprocess_image,
)

np = pytest.importorskip("numpy")

# Verbatim from Xenova/clip-vit-large-patch14 @ c307790, preprocessor_config.json.
REAL_CONFIG = {
    "crop_size": {"height": 224, "width": 224},
    "do_center_crop": True,
    "do_convert_rgb": True,
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [0.48145466, 0.4578275, 0.40821073],
    "image_std": [0.26862954, 0.26130258, 0.27577711],
    "resample": 3,
    "rescale_factor": 0.00392156862745098,
    "size": {"shortest_edge": 224},
}


@pytest.fixture
def pre(tmp_path) -> Preprocess:
    path = tmp_path / "preprocessor_config.json"
    path.write_text(json.dumps(REAL_CONFIG), encoding="utf-8")
    return _load_preprocess(path)


def test_config_is_read_not_assumed(pre):
    assert pre.size == 224
    assert pre.crop == 224
    assert pre.mean == pytest.approx((0.48145466, 0.4578275, 0.40821073))
    assert pre.std == pytest.approx((0.26862954, 0.26130258, 0.27577711))
    assert pre.rescale == pytest.approx(1 / 255)


def test_missing_keys_fall_back_to_the_clip_defaults(tmp_path):
    path = tmp_path / "preprocessor_config.json"
    path.write_text("{}", encoding="utf-8")
    fallback = _load_preprocess(path)
    assert fallback.size == 224
    assert fallback.crop == 224
    assert fallback.mean == pytest.approx((0.48145466, 0.4578275, 0.40821073))


def test_output_is_chw_float32_at_the_crop_size(pre):
    from PIL import Image

    arr = preprocess_image(Image.new("RGB", (640, 480), (10, 20, 30)), pre)
    assert arr.shape == (3, 224, 224)
    assert arr.dtype == np.float32


def test_normalisation_is_exactly_x_over_255_minus_mean_over_std(pre):
    """Hand-computed, not snapshotted."""
    from PIL import Image

    arr = preprocess_image(Image.new("RGB", (300, 300), (255, 0, 128)), pre)
    expected_r = (1.0 - REAL_CONFIG["image_mean"][0]) / REAL_CONFIG["image_std"][0]
    expected_g = (0.0 - REAL_CONFIG["image_mean"][1]) / REAL_CONFIG["image_std"][1]
    expected_b = (128 / 255 - REAL_CONFIG["image_mean"][2]) / REAL_CONFIG["image_std"][2]
    assert arr[0].mean() == pytest.approx(expected_r, abs=1e-4)
    assert arr[1].mean() == pytest.approx(expected_g, abs=1e-4)
    assert arr[2].mean() == pytest.approx(expected_b, abs=1e-4)


def test_aspect_ratio_is_preserved_then_centre_cropped(pre):
    """A wide photo must not be squashed: CLIP was trained on centre crops.

    A red band down the middle of a 4:1 image survives a centre crop; it would
    be squeezed to a quarter of its width by a plain square resize.
    """
    from PIL import Image

    wide = Image.new("RGB", (896, 224), (0, 0, 0))
    for x in range(336, 560):  # the centre 224 columns
        for y in range(224):
            wide.putpixel((x, y), (255, 0, 0))
    arr = preprocess_image(wide, pre)
    # The crop takes the centre 224x224 of a 896x224 image - all red.
    red_channel_high = (1.0 - REAL_CONFIG["image_mean"][0]) / REAL_CONFIG["image_std"][0]
    assert arr[0].mean() == pytest.approx(red_channel_high, abs=1e-3)


def test_a_portrait_image_is_cropped_from_the_middle(pre):
    from PIL import Image

    tall = Image.new("RGB", (224, 896), (0, 255, 0))
    arr = preprocess_image(tall, pre)
    assert arr.shape == (3, 224, 224)
    expected_g = (1.0 - REAL_CONFIG["image_mean"][1]) / REAL_CONFIG["image_std"][1]
    assert arr[1].mean() == pytest.approx(expected_g, abs=1e-3)


def test_a_tiny_image_is_upscaled_to_the_crop(pre):
    from PIL import Image

    assert preprocess_image(Image.new("RGB", (8, 8), (0, 0, 255)), pre).shape == (3, 224, 224)


def test_greyscale_and_rgba_are_converted(pre):
    from PIL import Image

    assert preprocess_image(Image.new("L", (300, 300), 128), pre).shape == (3, 224, 224)
    assert preprocess_image(Image.new("RGBA", (300, 300)), pre).shape == (3, 224, 224)


# ------------------------------------------------------------ output selection


def test_embedding_output_is_chosen_by_shape_not_position():
    """Indexing [0] happens to work today and would silently start returning
    token states if the export changed."""
    tokens = np.zeros((2, 77, 768), dtype=np.float32)
    pooled = np.ones((2, 768), dtype=np.float32)
    assert _pick_embedding([tokens, pooled], 768) is pooled


def test_no_matching_output_is_an_error_not_a_guess():
    with pytest.raises(Exception, match="no ONNX output has shape"):
        _pick_embedding([np.zeros((2, 512), dtype=np.float32)], 768)


# -------------------------------------------------- transformers major version


class _V4Tensor:
    """transformers 4.x handed back a tensor, which has `.float`."""

    def float(self):  # noqa: A003
        return self


class _V5Output:
    """transformers 5.x hands back an output object with `pooler_output`."""

    def __init__(self, pooled):
        self.pooler_output = pooled


def test_features_of_unwraps_the_transformers_v5_output():
    pooled = object()
    assert features_of(_V5Output(pooled)) is pooled


def test_features_of_passes_a_v4_tensor_through():
    tensor = _V4Tensor()
    assert features_of(tensor) is tensor
