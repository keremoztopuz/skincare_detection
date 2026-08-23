"""Canonicalize every image so acquisition carries no class information.

This is the fix for the defect that sank the v3 model. Measured on that
dataset: Healthy was 200x200 in 100% of rows, Eye_Bags/Wrinkles 640x640 in
98-99%, Acne/Eczema native and never square. The model learned resolution.
Downscaling real Eye_Bags test images to 200x200 collapsed recall from 38/40
to 8/40, and the eight in-app guide photos went from 8/8 firing to 0/8 — same
pixels, same content, only the resolution changed.

Resizing everything to a common size is not enough. A 200px source upsampled
to 448 and a 640px source downsampled to 448 still differ in sharpness, and
that difference stays correlated with the class. So a random effective
resolution is injected per image *before* the common resize, which turns
sharpness into a per-image random variable rather than a class signature.
The train-time augmentation applies the same idea again, so the property is
learned rather than baked into one fixed draw.

Framing gets the same treatment. UTKFace is perfectly eye-aligned at a fixed
margin, Roboflow is letterboxed, SCIN is handheld. A single fixed crop rule
would just swap one systematic signature for another, so the crop margin and
the roll angle are randomized per image, seeded from the image id so a rebuild
reproduces them exactly.

Images without a detectable face are kept and centre-cropped: Acne and Eczema
are largely body close-ups, and excluding them would empty those classes.
"""

import hashlib
import io
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

import overlay as OV

# --- geometry --------------------------------------------------------------

CANON_SIDE = 448          # stored side; IMG_SIZE is 384, leaving crop headroom
MIN_CROP_SIDE = 224       # strictly above the 200 where UTKFace-aligned sits
MIN_FACE_SIDE = 160
JPEG_QUALITY = 92
JPEG_SUBSAMPLING = 2      # 4:2:0, one setting for every source

MARGIN_RANGE = (1.25, 1.60)
ROLL_RANGE = (-8.0, 8.0)
CENTER_JITTER = 0.03
# 0.35 * 224 is about 78px effective, deliberately below the 200x200 regime
# where recall was measured to collapse: the model has to work down there.
DEGRADE_RANGE = (0.35, 1.00)

MAX_YAW = 35.0
MAX_PITCH = 30.0
MIN_SATURATION_STD = 2.0
# Set to the pooled 1st percentile (measured 5.6), not guessed. Sharpness is
# itself a strong source signature here — median Laplacian variance runs 407
# for DermNet, 125 for Roboflow Eye_Bags, 27 for UTKFace and 10.5 for the
# Roboflow Wrinkles texture patches. Any threshold above the pooled floor
# deletes whole sources rather than degenerate frames, which manufactures the
# class-correlated deletion this filter exists to avoid. The spread itself is
# handled downstream by sharpness-quintile matching, not by a cutoff.
MIN_LAPLACIAN = 6.0

REJECT_UNREADABLE = "unreadable"
REJECT_TOO_SMALL = "crop_below_min_side"
REJECT_FACE_TOO_SMALL = "face_below_min_side"
REJECT_MULTI_FACE = "multiple_faces"
REJECT_POSE = "extreme_pose"
REJECT_GRAYSCALE = "grayscale_or_degenerate"
REJECT_BLUR = "blurred"
REJECT_CENSOR = "censor_bar"


@dataclass
class CanonResult:
    image: Optional[np.ndarray]
    status: str
    reject_reason: Optional[str]
    face: Dict[str, object]
    crop: Dict[str, object]
    quality: Dict[str, object]
    overlay: Dict[str, object]


_detectors = None


def _get_detectors():
    """Lazily build mediapipe detectors; they are expensive to construct."""
    global _detectors
    if _detectors is None:
        import mediapipe as mp
        _detectors = {
            "short": mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.6),
            "full": mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.6),
            "mesh": mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=2, refine_landmarks=False,
                min_detection_confidence=0.5),
        }
    return _detectors


def rng_for(image_id: str) -> random.Random:
    """Per-image RNG so a rebuild reproduces the same crop and degradation."""
    seed = int(hashlib.sha256(image_id.encode()).hexdigest()[:16], 16)
    return random.Random(seed)


def detect_faces(image_bgr: np.ndarray) -> Tuple[Optional[Dict[str, object]], int]:
    """Best face box plus how many faces were seen.

    Both the short-range and full-range models run: dermatology close-ups and
    wide portraits are far enough apart that neither alone covers the pool.
    """
    detectors = _get_detectors()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_bgr.shape[:2]

    boxes = []
    for key in ("short", "full"):
        result = detectors[key].process(rgb)
        for detection in (result.detections or []):
            box = detection.location_data.relative_bounding_box
            x = max(0.0, box.xmin) * width
            y = max(0.0, box.ymin) * height
            w = box.width * width
            h = box.height * height
            if w <= 0 or h <= 0:
                continue
            boxes.append({
                "x": float(x), "y": float(y), "w": float(w), "h": float(h),
                "score": float(detection.score[0]), "model": key,
            })

    if not boxes:
        return None, 0

    boxes.sort(key=lambda b: -(b["w"] * b["h"]))
    primary = boxes[0]
    primary_area = primary["w"] * primary["h"]
    # The two models see the same face twice; only count boxes that do not
    # overlap the primary as separate people.
    distinct = 1
    for box in boxes[1:]:
        if _iou(primary, box) < 0.3 and box["w"] * box["h"] >= 0.25 * primary_area:
            distinct += 1
    return primary, distinct


def _iou(a: Dict[str, object], b: Dict[str, object]) -> float:
    ax0, ay0, ax1, ay1 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx0, by0, bx1, by1 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    union = a["w"] * a["h"] + b["w"] * b["h"] - intersection
    return intersection / union if union > 0 else 0.0


def eye_roll_and_box(image_bgr: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int, int, int]]]:
    """Roll angle from the eye line, plus a box around the eyes.

    The eye box is what the censor-bar check needs; without it that check
    fired on any dark background.
    """
    detectors = _get_detectors()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = detectors["mesh"].process(rgb)
    if not result.multi_face_landmarks:
        return None, None
    height, width = image_bgr.shape[:2]
    landmarks = result.multi_face_landmarks[0].landmark
    # 33 and 263 are the outer eye corners in the mediapipe mesh topology.
    left = np.array([landmarks[33].x * width, landmarks[33].y * height])
    right = np.array([landmarks[263].x * width, landmarks[263].y * height])
    roll = math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))

    eye_width = float(np.linalg.norm(right - left))
    center = (left + right) / 2.0
    box_w = int(eye_width * 1.8)
    box_h = int(eye_width * 0.9)
    box = (int(center[0] - box_w / 2), int(center[1] - box_h / 2), box_w, box_h)
    return roll, box


def _rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    if abs(degrees) < 0.01:
        return image
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    return cv2.warpAffine(image, matrix, (width, height),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _square_crop(image: np.ndarray, cx: float, cy: float, side: float) -> Optional[np.ndarray]:
    half = side / 2.0
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    x1, y1 = int(round(cx + half)), int(round(cy + half))
    height, width = image.shape[:2]
    # Reflect rather than pad with a constant: a constant border would be a
    # new uniform-edge artifact, which is the family of cue being removed.
    pad_left, pad_top = max(0, -x0), max(0, -y0)
    pad_right, pad_bottom = max(0, x1 - width), max(0, y1 - height)
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right,
                                   cv2.BORDER_REFLECT_101)
        x0 += pad_left; x1 += pad_left; y0 += pad_top; y1 += pad_top
    crop = image[y0:y1, x0:x1]
    return crop if crop.size else None


def _degrade(image: np.ndarray, factor: float) -> np.ndarray:
    """Drop to `factor` of the current side, then restore. Sharpness jitter."""
    if factor >= 0.999:
        return image
    height, width = image.shape[:2]
    small = (max(8, int(width * factor)), max(8, int(height * factor)))
    reduced = cv2.resize(image, small, interpolation=cv2.INTER_AREA)
    return cv2.resize(reduced, (width, height), interpolation=cv2.INTER_LINEAR)


def quality_metrics(image_bgr: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return {
        "lapvar": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "saturation_std": float(hsv[:, :, 1].std()),
        "brightness_mean": float(gray.mean()),
    }


def canonicalize(path: str, image_id: str) -> CanonResult:
    """Run the full pipeline for one image."""
    empty = {"n": 0}
    try:
        with Image.open(path) as handle:
            pil = ImageOps.exif_transpose(handle).convert("RGB")
        image = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return CanonResult(None, "rejected", REJECT_UNREADABLE, empty, {}, {}, {})

    rng = rng_for(image_id)

    # Overlays first: a letterbox bar would distort the crop geometry, and the
    # watermark would survive into the stored image.
    image, letterbox = OV.strip_letterbox(image)
    watermark_hit = OV.detect_watermark(image)
    watermark_removed = False
    if watermark_hit is not None:
        image, watermark_removed = OV.remove_watermark(image, watermark_hit)

    face, face_count = detect_faces(image)
    roll, eye_box = eye_roll_and_box(image) if face is not None else (None, None)

    overlay_info = {
        "letterbox_stripped": any(letterbox.values()),
        "watermark_found": watermark_hit is not None,
        "watermark_removed": watermark_removed,
        "censor_bar": OV.detect_censor_bar(image, eye_box),
    }
    if overlay_info["censor_bar"]:
        return CanonResult(None, "rejected", REJECT_CENSOR, {"n": face_count}, {}, {}, overlay_info)

    face_info: Dict[str, object] = {"n": face_count}
    height, width = image.shape[:2]

    if face is not None:
        if face_count > 1:
            return CanonResult(None, "rejected", REJECT_MULTI_FACE, face_info, {}, {}, overlay_info)
        if max(face["w"], face["h"]) < MIN_FACE_SIDE:
            return CanonResult(None, "rejected", REJECT_FACE_TOO_SMALL, face_info, {}, {}, overlay_info)
        face_info.update({
            "det_score": face["score"], "model": face["model"],
            "box": [face["x"], face["y"], face["w"], face["h"]],
            "roll": roll,
        })
        # Level the eyes, then apply a fresh random roll. Derotating alone
        # would make every source share UTKFace's alignment convention.
        applied_roll = rng.uniform(*ROLL_RANGE)
        if roll is not None:
            if abs(roll) > MAX_YAW:
                return CanonResult(None, "rejected", REJECT_POSE, face_info, {}, {}, overlay_info)
            image = _rotate(image, roll)
        image = _rotate(image, -applied_roll)
        base_side = max(face["w"], face["h"])
        cx = face["x"] + face["w"] / 2.0
        cy = face["y"] + face["h"] / 2.0
        source = "face"
    else:
        applied_roll = 0.0
        base_side = min(width, height)
        cx, cy = width / 2.0, height / 2.0
        source = "center"

    margin = rng.uniform(*MARGIN_RANGE)
    side = base_side * margin
    cx += rng.uniform(-CENTER_JITTER, CENTER_JITTER) * side
    cy += rng.uniform(-CENTER_JITTER, CENTER_JITTER) * side

    if side < MIN_CROP_SIDE:
        return CanonResult(None, "rejected", REJECT_TOO_SMALL,
                           face_info, {"crop_side_px": float(side)}, {}, overlay_info)

    crop = _square_crop(image, cx, cy, side)
    if crop is None or crop.size == 0:
        return CanonResult(None, "rejected", REJECT_TOO_SMALL,
                           face_info, {"crop_side_px": float(side)}, {}, overlay_info)

    # Quality is judged on the crop as it arrived, BEFORE the deliberate
    # degradation. Measuring afterwards means measuring blur this pipeline
    # just added: the first version rejected 32/40 Wrinkles and 30/40 UTKFace
    # that way — the blur filter manufacturing exactly the source-correlated
    # deletion it exists to prevent.
    pre = cv2.resize(crop, (CANON_SIDE, CANON_SIDE), interpolation=cv2.INTER_AREA)
    quality = quality_metrics(pre)
    if quality["saturation_std"] < MIN_SATURATION_STD:
        return CanonResult(None, "rejected", REJECT_GRAYSCALE, face_info, {}, quality, overlay_info)
    # Only a floor for degenerate frames. A global sharpness threshold would
    # delete most from whichever source is softest, and that is class-
    # correlated.
    if quality["lapvar"] < MIN_LAPLACIAN:
        return CanonResult(None, "rejected", REJECT_BLUR, face_info, {}, quality, overlay_info)

    degrade = rng.uniform(*DEGRADE_RANGE)
    canonical = _degrade(pre, degrade)

    crop_info = {
        "source": source,
        "margin": round(margin, 4),
        "applied_roll": round(applied_roll, 3),
        "crop_side_px": round(float(side), 1),
        "degrade_scale": round(degrade, 4),
        "jpeg_quality": JPEG_QUALITY,
    }
    return CanonResult(canonical, "kept", None, face_info, crop_info, quality, overlay_info)


def save(image: np.ndarray, destination: str) -> str:
    """Write JPEG at the one quality every class shares. Returns sha256."""
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    pil.save(buffer, "JPEG", quality=JPEG_QUALITY, subsampling=JPEG_SUBSAMPLING, optimize=False)
    payload = buffer.getvalue()
    with open(destination, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()
