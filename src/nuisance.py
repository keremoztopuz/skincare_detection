"""Nuisance transforms: make acquisition uninformative, not merely equalized.

Canonicalization gives every stored image the same dimensions, format and
quality, and injects one random effective resolution per image. That fixes the
dataset. It does not, on its own, teach the model that resolution is
meaningless — a fixed per-image draw can still be memorized alongside the
image.

These transforms re-randomize the same nuisances on every epoch, so sharpness,
compression and noise vary for the *same* picture across steps. The model
therefore cannot use them to identify anything.

They are also used at evaluation time, as fixed named views. Selecting a
checkpoint on the worst view across {identity, down128, jpeg35, blur2.0} is
what makes a resolution detector unselectable: it scores well on identity and
collapses on down128, so the minimum destroys it. That is the direct fix for
macro AUROC hitting 1.0000 by epoch 4 on a dataset separable by image height.

Implemented on PIL with the standard library so the training environment needs
no new dependency.
"""

import io
import random
from typing import Callable, Dict

from PIL import Image, ImageFilter

# Down to 0.22 of the side, which on a 384 crop is about 85px effective —
# deliberately inside the regime where the old model's recall collapsed
# (measured 38/40 -> 8/40 at 200x200). The model has to work there.
DOWNSCALE_RANGE = (0.22, 1.0)
JPEG_QUALITY_RANGE = (28, 95)
BLUR_RADIUS_RANGE = (0.4, 2.2)

DOWNSCALE_PROBABILITY = 0.60
JPEG_PROBABILITY = 0.70
BLUR_PROBABILITY = 0.35
SHARPEN_PROBABILITY = 0.20


def _downscale(image: Image.Image, factor: float) -> Image.Image:
    width, height = image.size
    small = (max(8, int(width * factor)), max(8, int(height * factor)))
    return image.resize(small, Image.BILINEAR).resize((width, height), Image.BILINEAR)


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


class RandomNuisance:
    """Randomize resolution, compression and sharpness on every call."""

    def __init__(self, seed: int = None):
        self._rng = random.Random(seed)

    def __call__(self, image: Image.Image) -> Image.Image:
        rng = self._rng
        if rng.random() < DOWNSCALE_PROBABILITY:
            image = _downscale(image, rng.uniform(*DOWNSCALE_RANGE))
        if rng.random() < JPEG_PROBABILITY:
            image = _jpeg(image, rng.randint(*JPEG_QUALITY_RANGE))
        roll = rng.random()
        if roll < BLUR_PROBABILITY:
            image = image.filter(ImageFilter.GaussianBlur(rng.uniform(*BLUR_RADIUS_RANGE)))
        elif roll < BLUR_PROBABILITY + SHARPEN_PROBABILITY:
            # Sharpening as well as blurring, so sharpness is non-monotone and
            # "sharper than average" carries no information either.
            image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=rng.randint(40, 130)))
        return image


def identity(image: Image.Image) -> Image.Image:
    return image


def down_to(side: int) -> Callable[[Image.Image], Image.Image]:
    def apply(image: Image.Image) -> Image.Image:
        width, height = image.size
        return image.resize((side, side), Image.BILINEAR).resize((width, height), Image.BILINEAR)
    return apply


def jpeg_at(quality: int) -> Callable[[Image.Image], Image.Image]:
    def apply(image: Image.Image) -> Image.Image:
        return _jpeg(image, quality)
    return apply


def blur_at(radius: float) -> Callable[[Image.Image], Image.Image]:
    def apply(image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(radius))
    return apply


# The fixed evaluation views. `down128` is the assay for the resolution
# shortcut: a model that reads resolution rather than skin falls apart here
# while identity looks perfect.
EVAL_VIEWS: Dict[str, Callable[[Image.Image], Image.Image]] = {
    "identity": identity,
    "down128": down_to(128),
    "jpeg35": jpeg_at(35),
    "blur2.0": blur_at(2.0),
}
