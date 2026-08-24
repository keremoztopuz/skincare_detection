"""Shortcut probe for multi-label data: one binary probe per condition.

The old probe asked a five-way question, which only made sense while each
image had exactly one class. Now an image can carry two conditions and leave
others unjudged, so the question is asked once per head: using nothing but how
the image was captured and framed, can a model tell the images where this
condition is present from the ones where it was judged absent?

Only rows where the condition is known take part. An unjudged entry is not a
negative, and feeding it in as one would let the probe separate "nobody
looked" from "looked and found nothing" — a difference in bookkeeping, not in
skin.

Gated features are acquisition and framing only. Colour and brightness are
measured and printed but never gated: acne really is red, and punishing that
would be punishing the signal.
"""

import argparse
import collections
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import audit
import provenance as PV

MAX_PROBE_AUROC = 0.65
MAX_SINGLE_FEATURE_AUC = 0.72
SEED = 42


def load_split(data_dir: str, splits=("train", "val")):
    manifest = PV.load_manifest()
    rows, conditions, names = [], [], None
    for split in splits:
        label_path = os.path.join(data_dir, f"{split}_labels.jsonl")
        if not os.path.exists(label_path):
            continue
        with open(label_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                path = os.path.join(data_dir, split, record["file"])
                features = audit.extract_features(path)
                if features is None:
                    continue
                features.update(audit.face_features(manifest, path))
                rows.append(features)
                conditions.append(record["conditions"])
    if not rows:
        raise SystemExit(f"goruntu bulunamadi: {data_dir}")
    names = sorted(rows[0].keys())
    matrix = np.array([[row[name] for name in names] for row in rows], dtype=np.float64)
    return matrix, conditions, names


def binary_auc(values: np.ndarray, truth: np.ndarray) -> float:
    """Rank AUC with ties averaged, folded to >= 0.5."""
    from scipy.stats import rankdata
    positive = values[truth == 1]
    negative = values[truth == 0]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([positive, negative]))
    area = ((ranks[:positive.size].sum() - positive.size * (positive.size + 1) / 2)
            / (positive.size * negative.size))
    return float(max(area, 1 - area))


def probe(matrix: np.ndarray, truth: np.ndarray) -> float:
    """Out-of-fold AUROC of a gradient boosting probe on the features."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    if len(np.unique(truth)) < 2 or min(np.bincount(truth.astype(int))) < 5:
        return float("nan")
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = np.zeros(len(truth))
    for train_index, test_index in folds.split(matrix, truth):
        model = HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                               random_state=SEED)
        model.fit(matrix[train_index], truth[train_index])
        scores[test_index] = model.predict_proba(matrix[test_index])[:, 1]
    return float(roc_auc_score(truth, scores))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="orchestration_data_v4")
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--fail-on-shortcut", action="store_true")
    parser.add_argument("--report", default="outputs/audits/shortcut_multilabel.json")
    arguments = parser.parse_args()

    splits = tuple(s.strip() for s in arguments.splits.split(","))
    matrix, conditions, names = load_split(arguments.data, splits)
    gated = [i for i, n in enumerate(names) if n in audit.ACQUISITION_FEATURES]
    reported = [i for i, n in enumerate(names) if n not in audit.ACQUISITION_FEATURES]
    print(f"veri: {arguments.data}  {len(conditions)} goruntu, "
          f"{len(gated)} kapili + {len(reported)} raporlanan ozellik\n")

    failures, report = [], {}
    print(f"{'kosul':<10}{'pos':>6}{'neg':>6}{'prob AUROC':>13}   en guclu edinim ozelligi")
    for name in sorted({k for row in conditions for k in row}):
        known = [i for i, row in enumerate(conditions) if row.get(name) is not None]
        if not known:
            continue
        truth = np.array([conditions[i][name] for i in known], dtype=int)
        subset = matrix[known]
        auroc = probe(subset[:, gated], truth)
        singles = sorted(
            ((binary_auc(subset[:, i], truth), names[i]) for i in gated),
            reverse=True)
        worst_auc, worst_name = singles[0]
        flag = ""
        if np.isfinite(auroc) and auroc > MAX_PROBE_AUROC:
            failures.append(f"{name}: prob AUROC {auroc:.3f} > {MAX_PROBE_AUROC}")
            flag = "  <-- SINIR ASIMI"
        if worst_auc >= MAX_SINGLE_FEATURE_AUC:
            failures.append(f"{name}: {worst_name} AUC {worst_auc:.3f} >= {MAX_SINGLE_FEATURE_AUC}")
            flag = "  <-- SINIR ASIMI"
        print(f"{name:<10}{int(truth.sum()):>6}{int((truth == 0).sum()):>6}"
              f"{auroc:>13.3f}   {worst_name} {worst_auc:.3f}{flag}")
        appearance = sorted(
            ((binary_auc(subset[:, i], truth), names[i]) for i in reported),
            reverse=True)[:2]
        report[name] = {
            "probe_auroc": auroc,
            "acquisition": [(n, round(a, 4)) for a, n in singles[:5]],
            "appearance": [(n, round(a, 4)) for a, n in appearance],
        }

    print("\ngorunum ozellikleri (yalnizca rapor, kapi degil):")
    for name, entry in sorted(report.items()):
        pairs = ", ".join(f"{n} {a:.3f}" for n, a in entry["appearance"])
        print(f"   {name:<10} {pairs}")

    os.makedirs(os.path.dirname(arguments.report) or ".", exist_ok=True)
    with open(arguments.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nrapor: {arguments.report}")

    if failures:
        print("\nKISAYOL DENETIMI DUSTU:")
        for line in failures:
            print(f"   - {line}")
        return 1 if arguments.fail_on_shortcut else 0
    print("\nkisayol denetimi TEMIZ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
