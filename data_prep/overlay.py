"""Burned-in overlay detection and removal.

Three overlays were measured in the v3 pool, each of them a shortcut of the
kind Winkler et al. (JAMA Dermatology 2019) documented for surgical skin
markings — a consistent artifact that correlates with the label:

  1. "(c)Dermnet.com" text, centred, on essentially every DermNet image, i.e.
     on most Acne and Eczema samples and on nothing else.
  2. Roboflow letterbox bars, on the 640x640 Eye_Bags/Wrinkles exports.
  3. Black censor bars over the eyes on some DermNet portraits.

Letterbox is cropped away, the watermark is inpainted, and censor bars are a
hard reject: they sit exactly where Eye_Bags evidence lives.

Removing the watermark introduces its own risk. If only DermNet images carry a
smoothed inpainted patch, "has an inpainted region" simply replaces the
watermark as the cue. `apply_decoy_inpaint` exists so the caller can paint the
same kind of patch onto a matched share of other sources, making the operation
uninformative rather than merely one-sided.
"""

import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
WATERMARK_TEMPLATE = os.path.join(ASSETS, "dermnet_watermark.png")

# Measured on 141 averaged 720x472 DermNet images: the mark is a fixed-pixel
# overlay (~335x32) centred horizontally and sitting on the vertical midline,
# not a fraction of the frame. Confirmed against the 477x720 group, where it
# stayed ~343 px wide. Hence a narrow scale search rather than a size ratio.
TEMPLATE_SCALES = (0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3)
BAND_TOP = 0.38
BAND_BOTTOM = 0.64
MATCH_THRESHOLD = 0.34

# Tightened after measuring: the first pass stripped up to 28% off UTKFace
# crops by mistaking dark hair and background for padding. Cropping that much
# only from dark-haired or dark-skinned subjects is a demographic bias, so the
# bar for calling something padding is now deliberately high: near-black,
# near-constant, and symmetric on opposite sides.
LETTERBOX_MAX_FRACTION = 0.30
LETTERBOX_ROW_STD = 3.0
LETTERBOX_ROW_MEAN = 18.0
LETTERBOX_MIN_BAR = 2
LETTERBOX_SYMMETRY_TOLERANCE = 0.25

CENSOR_MIN_WIDTH_FRACTION = 0.25
CENSOR_MAX_SATURATION = 28.0
CENSOR_MAX_VALUE = 55.0

_template_cache: Optional[np.ndarray] = None


def _load_template() -> np.ndarray:
    global _template_cache
    if _template_cache is None:
        template = cv2.imread(WATERMARK_TEMPLATE, cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise FileNotFoundError(f"filigran sablonu yok: {WATERMARK_TEMPLATE}")
        _template_cache = template
    return _template_cache


def _high_pass(gray: np.ndarray, sigma: int = 6) -> np.ndarray:
    """Suppress content, keep the thin bright/dark structure of overlaid text."""
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.normalize(
        gray.astype(np.float32) - blurred.astype(np.float32),
        None, 0, 255, cv2.NORM_MINMAX,
    ).astype(np.uint8)


def detect_letterbox(image: np.ndarray) -> Dict[str, int]:
    """Find uniform dark bars padding a frame to a square.

    Returns the inset on each side, all zeros when there is no letterbox.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    height, width = gray.shape

    def scan(vector_means: np.ndarray, vector_stds: np.ndarray, limit: int) -> int:
        count = 0
        for index in range(limit):
            if vector_stds[index] < LETTERBOX_ROW_STD and vector_means[index] < LETTERBOX_ROW_MEAN:
                count += 1
            else:
                break
        return count

    row_means, row_stds = gray.mean(axis=1), gray.std(axis=1)
    col_means, col_stds = gray.mean(axis=0), gray.std(axis=0)
    max_rows = int(height * LETTERBOX_MAX_FRACTION)
    max_cols = int(width * LETTERBOX_MAX_FRACTION)

    top = scan(row_means, row_stds, max_rows)
    bottom = scan(row_means[::-1], row_stds[::-1], max_rows)
    left = scan(col_means, col_stds, max_cols)
    right = scan(col_means[::-1], col_stds[::-1], max_cols)

    def symmetric(first: int, second: int) -> Tuple[int, int]:
        # Padding is applied to both sides of an axis; a bar on one side only
        # is a dark edge in the photograph, not a letterbox.
        if first < LETTERBOX_MIN_BAR or second < LETTERBOX_MIN_BAR:
            return 0, 0
        larger = max(first, second)
        if abs(first - second) > larger * LETTERBOX_SYMMETRY_TOLERANCE:
            return 0, 0
        return first, second

    top, bottom = symmetric(top, bottom)
    left, right = symmetric(left, right)
    return {"top": top, "bottom": bottom, "left": left, "right": right}


def strip_letterbox(image: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    """Crop letterbox bars away. Returns the image unchanged when there are none."""
    inset = detect_letterbox(image)
    if not any(inset.values()):
        return image, inset
    height, width = image.shape[:2]
    top, bottom = inset["top"], height - inset["bottom"]
    left, right = inset["left"], width - inset["right"]
    if bottom - top < height * 0.4 or right - left < width * 0.4:
        # Refuse to strip more than we keep; that is a dark photo, not padding.
        return image, {"top": 0, "bottom": 0, "left": 0, "right": 0}
    return image[top:bottom, left:right], inset


def detect_watermark(image: np.ndarray) -> Optional[Dict[str, object]]:
    """Locate the DermNet mark. Returns None when it is not present.

    Matching runs on a high-pass view inside the central band, because the mark
    is semi-transparent: it has no constant pixel value, only a constant thin
    structure, so plain correlation against raw pixels does not find it.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    height, width = gray.shape
    band_top = int(height * BAND_TOP)
    band_bottom = int(height * BAND_BOTTOM)
    band = _high_pass(gray[band_top:band_bottom, :])
    template = _high_pass(_load_template())

    best = None
    for scale in TEMPLATE_SCALES:
        scaled_width = int(template.shape[1] * scale)
        scaled_height = int(template.shape[0] * scale)
        if scaled_width >= width or scaled_height >= band.shape[0]:
            continue
        resized = cv2.resize(template, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(band, resized, cv2.TM_CCOEFF_NORMED)
        _, max_value, _, max_location = cv2.minMaxLoc(result)
        if best is None or max_value > best["score"]:
            best = {
                "score": float(max_value),
                "x": int(max_location[0]),
                "y": int(max_location[1]) + band_top,
                "w": scaled_width,
                "h": scaled_height,
                "scale": scale,
            }

    if best is None or best["score"] < MATCH_THRESHOLD:
        return None
    return best


def _glyph_mask() -> np.ndarray:
    """Binary mask of the watermark's letter strokes, at template scale."""
    template = _load_template()
    high_pass = _high_pass(template)
    # The mark is light text with a dark drop shadow, so the strokes sit in
    # both tails of the high-pass distribution, not just the bright one.
    deviation = np.abs(high_pass.astype(np.int16) - int(np.median(high_pass)))
    threshold = np.percentile(deviation, 62)
    return (deviation >= threshold).astype(np.uint8) * 255


def _watermark_mask(image_shape: Tuple[int, int], hit: Dict[str, object]) -> np.ndarray:
    """Binary mask over the matched text.

    Masking the whole bounding box and inpainting it left a smooth rectangular
    smear — a cleaner, more obvious artifact than the watermark it replaced,
    and one only DermNet images carried. Masking the strokes themselves keeps
    the repair inside what cv2.inpaint is actually good at: thin scratches.
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    x, y, w, h = int(hit["x"]), int(hit["y"]), int(hit["w"]), int(hit["h"])
    if w <= 0 or h <= 0:
        return mask

    glyphs = cv2.resize(_glyph_mask(), (w, h), interpolation=cv2.INTER_NEAREST)
    # One dilation covers the anti-aliased stroke edges without merging
    # neighbouring letters into a solid block.
    glyphs = cv2.dilate(glyphs, np.ones((3, 3), np.uint8), iterations=1)

    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(image_shape[1], x + w), min(image_shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return mask
    mask[y0:y1, x0:x1] = glyphs[: y1 - y0, : x1 - x0]
    return mask


def remove_watermark(image: np.ndarray, hit: Optional[Dict[str, object]] = None) -> Tuple[np.ndarray, bool]:
    """Inpaint the mark. Returns (image, removed)."""
    if hit is None:
        hit = detect_watermark(image)
    if hit is None:
        return image, False
    mask = _watermark_mask(image.shape, hit)
    if not mask.any():
        return image, False
    repaired = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
    return repaired, True


def apply_decoy_inpaint(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inpaint a text-shaped patch at a random spot.

    Applied to a share of non-watermarked images so that the presence of an
    inpainted region carries no class information.
    """
    height, width = image.shape[:2]
    patch_width = int(width * rng.uniform(0.35, 0.55))
    patch_height = max(4, int(patch_width * 0.095))
    x = int(rng.integers(0, max(1, width - patch_width)))
    y = int(rng.integers(int(height * 0.25), max(int(height * 0.25) + 1, height - patch_height)))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y:y + patch_height, x:x + patch_width] = 255
    return cv2.inpaint(image, mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)


def detect_censor_bar(image: np.ndarray, eye_box: Optional[Tuple[int, int, int, int]] = None) -> bool:
    """Detect a black bar laid over the eyes.

    Unlike the watermark this is not repairable: it covers the periorbital
    region, which is where the Eye_Bags evidence lives. Callers should reject.
    """
    if image.ndim != 3 or eye_box is None:
        # Without a located eye region this fired on any dark background — 32%
        # of SCIN body-part photos on the first pass. A censor bar is only
        # meaningful where there are eyes to censor, so face detection is a
        # precondition rather than a fallback.
        return False
    x, y, w, h = eye_box
    region = image[max(0, y):y + h, max(0, x):x + w]
    if region.size == 0:
        return False

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    dark = (hsv[:, :, 1] < CENSOR_MAX_SATURATION) & (hsv[:, :, 2] < CENSOR_MAX_VALUE)
    if dark.mean() < 0.02:
        return False

    # A censor bar is a wide horizontal run, not scattered dark pixels.
    row_coverage = dark.mean(axis=1)
    wide_rows = row_coverage > CENSOR_MIN_WIDTH_FRACTION
    if not wide_rows.any():
        return False
    runs, current = [], 0
    for is_wide in wide_rows:
        if is_wide:
            current += 1
        else:
            runs.append(current)
            current = 0
    runs.append(current)
    return max(runs) >= max(3, int(region.shape[0] * 0.05))


def analyze(image: np.ndarray, eye_box: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, object]:
    """Full overlay report for one image, for the manifest."""
    stripped, letterbox = strip_letterbox(image)
    hit = detect_watermark(stripped)
    return {
        "letterbox": letterbox,
        "letterbox_stripped": any(letterbox.values()),
        "watermark": None if hit is None else {k: hit[k] for k in ("score", "x", "y", "w", "h", "scale")},
        "watermark_found": hit is not None,
        "censor_bar": detect_censor_bar(stripped, eye_box),
    }
