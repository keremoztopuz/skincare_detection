import json
import os

import numpy as np
from sklearn.metrics import (
    fbeta_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config


def _known(labels, masks, class_idx):
    """Rows where this class was actually judged."""
    rows = masks[:, class_idx] > 0
    return rows, labels[rows, class_idx]


def calibrate_thresholds(labels, probabilities, masks):
    """Per-class thresholds, fitted only where the class is known.

    An unjudged entry is not a negative. Including it would push every
    threshold up, because the search would be rewarded for staying quiet on
    images nobody has actually checked.
    """
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    masks = np.asarray(masks)
    candidates = np.arange(
        config.THRESHOLD_SEARCH_MIN,
        config.THRESHOLD_SEARCH_MAX + config.THRESHOLD_SEARCH_STEP / 2,
        config.THRESHOLD_SEARCH_STEP,
    )
    thresholds = []
    for class_idx in range(labels.shape[1]):
        rows, truth = _known(labels, masks, class_idx)
        if truth.size == 0 or len(np.unique(truth)) < 2:
            thresholds.append(float(config.DETECTION_THRESHOLD))
            continue
        scores = [
            fbeta_score(truth, probabilities[rows, class_idx] >= threshold,
                        beta=0.5, zero_division=0)
            for threshold in candidates
        ]
        best_score = max(scores)
        tied = candidates[np.isclose(scores, best_score)]
        thresholds.append(float(tied[np.argmin(np.abs(tied - config.DETECTION_THRESHOLD))]))
    return np.asarray(thresholds, dtype=np.float32)


def threshold_predictions(probabilities, thresholds):
    return (np.asarray(probabilities) >= np.asarray(thresholds)).astype(np.float32)


def calculate_metrics(labels, probabilities, thresholds, masks):
    """Per-class metrics over judged entries, macro-averaged.

    Top-1 accuracy is gone. It asked which single class won, which has no
    meaning once an image can carry two conditions and can leave others
    unjudged — and as a checkpoint criterion it was the most shortcut-friendly
    number in the run, reaching 1.0000 by epoch four on a dataset where
    resolution alone separated the classes.
    """
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    masks = np.asarray(masks)
    predictions = threshold_predictions(probabilities, thresholds)

    aurocs, recalls, precisions, f1s, per_class_recall = [], [], [], [], []
    for class_idx in range(labels.shape[1]):
        rows, truth = _known(labels, masks, class_idx)
        predicted = predictions[rows, class_idx]
        if truth.size == 0 or len(np.unique(truth)) < 2:
            per_class_recall.append(float("nan"))
            continue
        aurocs.append(roc_auc_score(truth, probabilities[rows, class_idx]))
        recall = recall_score(truth, predicted, zero_division=0)
        recalls.append(recall)
        per_class_recall.append(recall)
        precisions.append(precision_score(truth, predicted, zero_division=0))
        f1s.append(f1_score(truth, predicted, zero_division=0))

    # NegativeReject is per head, not per image. No image in this dataset has
    # all four conditions judged — SCIN knows about acne and eczema, FFHQ
    # about eye bags and wrinkles — so "an image where nothing is present"
    # does not exist to be counted. What does exist, and is what actually
    # broke the previous model, is a head firing on skin where its own
    # condition was judged absent. That is specificity, averaged over heads.
    specificities = []
    for class_idx in range(labels.shape[1]):
        rows, truth = _known(labels, masks, class_idx)
        absent = truth == 0
        if absent.any():
            specificities.append(float((predictions[rows, class_idx][absent] == 0).mean()))
    negative_reject = float(np.mean(specificities)) if specificities else float("nan")

    mean = lambda values: float(np.mean(values)) if values else float("nan")
    return {
        "NegativeReject": negative_reject,
        "Precision": mean(precisions),
        "Recall": mean(recalls),
        "F1": mean(f1s),
        "AUROC": mean(aurocs),
        "PerClassRecall": np.asarray(per_class_recall),
        "LabelsPerImage": float(predictions.sum(axis=1).mean()),
        "KnownPerImage": float(masks.sum(axis=1).mean()),
    }, predictions


def save_thresholds(thresholds, path=config.THRESHOLDS_SAVE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = dict(zip(config.CLASS_NAMES, map(float, thresholds)))
    with open(path, "w", encoding="utf-8") as threshold_file:
        json.dump(payload, threshold_file, indent=2)


def load_thresholds(path=config.THRESHOLDS_SAVE_PATH):
    if not os.path.exists(path):
        return np.full(len(config.CLASS_NAMES), config.DETECTION_THRESHOLD, dtype=np.float32)
    with open(path, encoding="utf-8") as threshold_file:
        payload = json.load(threshold_file)
    return np.asarray(
        [payload.get(class_name, config.DETECTION_THRESHOLD) for class_name in config.CLASS_NAMES],
        dtype=np.float32,
    )
