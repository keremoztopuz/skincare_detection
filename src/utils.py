import json
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    fbeta_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config


def calibrate_thresholds(labels, probabilities):
    """Select precision-aware per-class thresholds for sparse multi-label output."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    candidates = np.arange(
        config.THRESHOLD_SEARCH_MIN,
        config.THRESHOLD_SEARCH_MAX + config.THRESHOLD_SEARCH_STEP / 2,
        config.THRESHOLD_SEARCH_STEP,
    )
    thresholds = []
    for class_idx in range(labels.shape[1]):
        scores = [
            fbeta_score(
                labels[:, class_idx],
                probabilities[:, class_idx] >= threshold,
                beta=0.5,
                zero_division=0,
            )
            for threshold in candidates
        ]
        best_score = max(scores)
        tied = candidates[np.isclose(scores, best_score)]
        best_threshold = tied[np.argmin(np.abs(tied - config.DETECTION_THRESHOLD))]
        thresholds.append(float(best_threshold))
    return np.asarray(thresholds, dtype=np.float32)


def threshold_predictions(probabilities, thresholds):
    return (np.asarray(probabilities) >= np.asarray(thresholds)).astype(np.float32)


def calculate_metrics(labels, probabilities, thresholds):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    predictions = threshold_predictions(probabilities, thresholds)
    try:
        auroc = roc_auc_score(labels, probabilities, average="macro")
    except ValueError:
        auroc = float("nan")

    return {
        "Accuracy": accuracy_score(labels, predictions),
        "Top1Accuracy": accuracy_score(labels.argmax(axis=1), probabilities.argmax(axis=1)),
        "Precision": precision_score(labels, predictions, average="macro", zero_division=0),
        "Recall": recall_score(labels, predictions, average="macro", zero_division=0),
        "F1": f1_score(labels, predictions, average="macro", zero_division=0),
        "AUROC": auroc,
        "PerClassRecall": recall_score(labels, predictions, average=None, zero_division=0),
        "LabelsPerImage": predictions.sum(axis=1).mean(),
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
