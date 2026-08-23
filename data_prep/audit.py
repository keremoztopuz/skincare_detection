"""Shortcut audit. Fails the build when acquisition predicts the label.

The v3 model reported val NegativeReject 1.0000 from epoch 4 and test 68/68,
and every one of those numbers was an artifact of image dimensions. Nothing in
the training loop could have caught it: the shortcut was present in the
holdout too, so an i.i.d. split validated it happily. The only thing that
catches this class of failure is a check that never looks at the pixels'
content — only at how the images were acquired.

The test: take metadata alone (width, height, aspect, bytes-per-pixel, JPEG
quantization, Laplacian variance, high-frequency energy, colour statistics)
and train a classifier to predict the class from it. If that succeeds
meaningfully above chance, the dataset is separable without looking at a
single lesion, and any model trained on it will find the same route.

Run it before training, and let a failure stop the run:

    python data_prep/audit.py --data orchestration_data_v2 --fail-on-shortcut
"""

import argparse
import collections
import glob
import json
import os
import struct
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Chance is 1/n_classes. These are deliberately a little loose, because
# re-encoding leaves faint class-correlated residue even after canonicalization
# — but far below the ~1.0 the old dataset scored.
MAX_BALANCED_ACCURACY_RATIO = 1.60      # 1.6x chance
MAX_MACRO_AUROC = 0.65
MAX_SINGLE_FEATURE_AUC = 0.72
PERMUTATION_ROUNDS = 100
PERMUTATION_ALPHA = 0.01


def jpeg_quantization_signature(path: str) -> Optional[int]:
    """Hash of the JPEG quantization tables.

    Encoder and quality settings leave a fingerprint here that survives
    resizing, which makes it one of the sharpest source cues available.
    """
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError:
        return None
    if not data.startswith(b"\xff\xd8"):
        return None
    offset, tables = 2, []
    while offset < len(data) - 1:
        if data[offset] != 0xFF:
            offset += 1
            continue
        marker = data[offset + 1]
        if marker == 0xDB:  # DQT
            length = struct.unpack(">H", data[offset + 2:offset + 4])[0]
            tables.append(data[offset + 4:offset + 2 + length])
            offset += 2 + length
        elif marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            offset += 2
        elif marker == 0xDA:  # start of scan, tables are all before this
            break
        else:
            if offset + 4 > len(data):
                break
            length = struct.unpack(">H", data[offset + 2:offset + 4])[0]
            offset += 2 + length
    if not tables:
        return None
    return hash(b"".join(tables)) & 0xFFFFFFFF


def extract_features(path: str) -> Optional[Dict[str, float]]:
    """Acquisition-only features. Nothing here describes the skin."""
    image = cv2.imread(path)
    if image is None:
        return None
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(cv2.resize(gray, (128, 128)))))
    centre = spectrum.shape[0] // 2
    yy, xx = np.ogrid[:spectrum.shape[0], :spectrum.shape[1]]
    radius = np.sqrt((yy - centre) ** 2 + (xx - centre) ** 2)
    high_frequency = float(spectrum[radius > centre * 0.5].sum() / (spectrum.sum() + 1e-9))

    blue, green, red = image[:, :, 0].astype(np.float32), image[:, :, 1].astype(np.float32), image[:, :, 2].astype(np.float32)
    rg, yb = red - green, 0.5 * (red + green) - blue
    colorfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    size_bytes = os.path.getsize(path)
    quantization = jpeg_quantization_signature(path)
    return {
        "width": float(width),
        "height": float(height),
        "log_pixels": float(np.log(width * height)),
        "aspect": float(width / height),
        "bytes_per_pixel": float(size_bytes / (width * height)),
        "lapvar": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "high_freq_ratio": high_frequency,
        "brightness_mean": float(gray.mean()),
        "brightness_std": float(gray.std()),
        "blue_mean": float(blue.mean()),
        "green_mean": float(green.mean()),
        "red_mean": float(red.mean()),
        "colorfulness": colorfulness,
        "quant_signature": float(quantization % 10007) if quantization is not None else -1.0,
    }


def collect(data_dir: str, splits=("train", "val")) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    rows, labels, paths = [], [], []
    class_names = sorted({
        os.path.basename(os.path.dirname(p))
        for split in splits
        for p in glob.glob(os.path.join(data_dir, split, "*", "*"))
        if p.lower().endswith((".jpg", ".jpeg", ".png"))
    })
    index = {name: i for i, name in enumerate(class_names)}
    for split in splits:
        for path in sorted(glob.glob(os.path.join(data_dir, split, "*", "*"))):
            if not path.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            features = extract_features(path)
            if features is None:
                continue
            rows.append(features)
            labels.append(index[os.path.basename(os.path.dirname(path))])
            paths.append(path)
    if not rows:
        raise SystemExit(f"goruntu bulunamadi: {data_dir}")
    names = sorted(rows[0].keys())
    matrix = np.array([[row[name] for name in names] for row in rows], dtype=np.float64)
    return matrix, np.array(labels), names, class_names


def balanced_accuracy(truth: np.ndarray, predicted: np.ndarray, n_classes: int) -> float:
    recalls = []
    for label in range(n_classes):
        mask = truth == label
        if mask.any():
            recalls.append(float((predicted[mask] == label).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def probe(matrix: np.ndarray, labels: np.ndarray, n_classes: int, seed: int = 42) -> Dict[str, float]:
    """5-fold stratified probe on metadata alone."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    predictions = np.zeros(len(labels), dtype=int)
    probabilities = np.zeros((len(labels), n_classes))
    for train_index, test_index in folds.split(matrix, labels):
        model = HistGradientBoostingClassifier(max_depth=3, max_iter=200, random_state=seed)
        model.fit(matrix[train_index], labels[train_index])
        predictions[test_index] = model.predict(matrix[test_index])
        proba = model.predict_proba(matrix[test_index])
        for column, label in enumerate(model.classes_):
            probabilities[test_index, label] = proba[:, column]

    accuracy = balanced_accuracy(labels, predictions, n_classes)
    try:
        auroc = float(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro"))
    except ValueError:
        auroc = float("nan")
    return {"balanced_accuracy": accuracy, "macro_auroc": auroc}


def permutation_test(matrix: np.ndarray, labels: np.ndarray, n_classes: int,
                     observed: float, rounds: int = PERMUTATION_ROUNDS) -> float:
    """How often shuffled labels reach the observed accuracy."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.default_rng(42)
    hits = 0
    for round_index in range(rounds):
        shuffled = rng.permutation(labels)
        folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=round_index)
        predictions = np.zeros(len(shuffled), dtype=int)
        for train_index, test_index in folds.split(matrix, shuffled):
            model = HistGradientBoostingClassifier(max_depth=3, max_iter=60, random_state=0)
            model.fit(matrix[train_index], shuffled[train_index])
            predictions[test_index] = model.predict(matrix[test_index])
        if balanced_accuracy(shuffled, predictions, n_classes) >= observed:
            hits += 1
    return (hits + 1) / (rounds + 1)


def univariate(matrix: np.ndarray, labels: np.ndarray, names: List[str],
               class_names: List[str]) -> List[Tuple[str, str, float]]:
    """One-vs-rest AUC for every single feature. Names the guilty cue."""
    from sklearn.metrics import roc_auc_score
    offenders = []
    for column, feature in enumerate(names):
        values = matrix[:, column]
        if np.allclose(values, values[0]):
            continue
        for label, class_name in enumerate(class_names):
            target = (labels == label).astype(int)
            if target.sum() in (0, len(target)):
                continue
            auc = float(roc_auc_score(target, values))
            offenders.append((class_name, feature, max(auc, 1.0 - auc)))
    offenders.sort(key=lambda item: -item[2])
    return offenders


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="orchestration_data_v2")
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--fail-on-shortcut", action="store_true")
    parser.add_argument("--skip-permutation", action="store_true")
    parser.add_argument("--report", default="outputs/audits/shortcut_report.json")
    arguments = parser.parse_args()

    splits = tuple(s.strip() for s in arguments.splits.split(","))
    matrix, labels, names, class_names = collect(arguments.data, splits)
    n_classes = len(class_names)
    chance = 1.0 / n_classes
    limit = chance * MAX_BALANCED_ACCURACY_RATIO

    counts = collections.Counter(labels.tolist())
    print(f"veri: {arguments.data}  {len(labels)} goruntu, {n_classes} sinif")
    for index, name in enumerate(class_names):
        print(f"   {name:<12} {counts.get(index, 0)}")
    print(f"ozellik: {len(names)} adet (yalnizca edinim, icerik yok)\n")

    scores = probe(matrix, labels, n_classes)
    print(f"metadata probu : balanced_accuracy={scores['balanced_accuracy']:.3f} "
          f"(sans {chance:.3f}, sinir {limit:.3f})")
    print(f"                 macro_auroc={scores['macro_auroc']:.3f} (sinir {MAX_MACRO_AUROC})")

    offenders = univariate(matrix, labels, names, class_names)
    print("\nen guclu tekil ozellikler:")
    for class_name, feature, auc in offenders[:6]:
        flag = "  <-- SINIR ASIMI" if auc >= MAX_SINGLE_FEATURE_AUC else ""
        print(f"   {class_name:<12} {feature:<18} AUC={auc:.3f}{flag}")

    failures = []
    if scores["balanced_accuracy"] > limit:
        failures.append(f"metadata {scores['balanced_accuracy']:.3f} > {limit:.3f}")
    if not np.isnan(scores["macro_auroc"]) and scores["macro_auroc"] > MAX_MACRO_AUROC:
        failures.append(f"macro_auroc {scores['macro_auroc']:.3f} > {MAX_MACRO_AUROC}")
    worst = [o for o in offenders if o[2] >= MAX_SINGLE_FEATURE_AUC]
    if worst:
        failures.append(f"{len(worst)} tekil ozellik AUC >= {MAX_SINGLE_FEATURE_AUC} "
                        f"(en kotu: {worst[0][0]}/{worst[0][1]} {worst[0][2]:.3f})")

    p_value = None
    if not arguments.skip_permutation and scores["balanced_accuracy"] > limit:
        p_value = permutation_test(matrix, labels, n_classes, scores["balanced_accuracy"])
        print(f"\npermutasyon testi: p={p_value:.4f} (alpha {PERMUTATION_ALPHA})")
        if p_value >= PERMUTATION_ALPHA:
            failures = [f for f in failures if not f.startswith("metadata")]

    os.makedirs(os.path.dirname(arguments.report) or ".", exist_ok=True)
    with open(arguments.report, "w", encoding="utf-8") as handle:
        json.dump({
            "data": arguments.data, "n": int(len(labels)), "classes": class_names,
            "scores": scores, "chance": chance, "limit": limit,
            "p_value": p_value,
            "top_features": [{"class": c, "feature": f, "auc": a} for c, f, a in offenders[:20]],
            "failures": failures,
        }, handle, indent=2, ensure_ascii=False)

    if failures:
        print("\nKISAYOL DENETIMI DUSTU:")
        for failure in failures:
            print(f"   - {failure}")
        print(f"\nrapor: {arguments.report}")
        return 1 if arguments.fail_on_shortcut else 0

    print("\nkisayol denetimi TEMIZ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
