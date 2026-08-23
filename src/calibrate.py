import argparse
import json
import os

import torch
from tqdm import tqdm

from config import CLASS_NAMES, DEVICE, MODEL_SAVE_PATH, ROOT_DIR, THRESHOLDS_SAVE_PATH
from dataset import get_dataloaders
from model import build_model
from utils import calibrate_thresholds, calculate_metrics, save_thresholds

TEMPERATURE_SAVE_PATH = os.path.join(ROOT_DIR, "outputs", "model", "temperature.json")


def fit_temperature(logits, targets):
    """Single-scalar temperature minimizing validation BCE (Guo et al. 2017).

    The app currently hardcodes temperature 1.8; this measures the real value
    so the on-device sigmoid squashing can be grounded in validation data.
    """
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(logits / log_temperature.exp(), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp())


def calibrate_model(model_path=MODEL_SAVE_PATH, thresholds_path=THRESHOLDS_SAVE_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    _, val_loader, _ = get_dataloaders()
    labels = []
    probabilities = []
    raw_logits = []
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Calibrating"):
            outputs = model(images.to(DEVICE))
            raw_logits.append(outputs.cpu())
            probabilities.extend(torch.sigmoid(outputs).cpu().numpy())
            labels.extend(targets.numpy())

    temperature = fit_temperature(
        torch.cat(raw_logits), torch.tensor(labels, dtype=torch.float32)
    )
    with open(TEMPERATURE_SAVE_PATH, "w") as temperature_file:
        json.dump({"temperature": round(temperature, 4)}, temperature_file)
    print(f"Fitted temperature: {temperature:.3f} (saved to {TEMPERATURE_SAVE_PATH})")

    thresholds = calibrate_thresholds(labels, probabilities)
    metrics, _ = calculate_metrics(labels, probabilities, thresholds)
    save_thresholds(thresholds, thresholds_path)

    print("Saved thresholds:")
    for class_name, threshold in zip(CLASS_NAMES, thresholds):
        print(f"  {class_name}: {threshold:.2f}")
    print(
        f"Top-1 Acc: {metrics['Top1Accuracy']:.4f} | "
        f"NegativeReject: {metrics['NegativeReject']:.4f} | "
        f"Exact Match: {metrics['Accuracy']:.4f} | "
        f"Precision: {metrics['Precision']:.4f} | Recall: {metrics['Recall']:.4f} | "
        f"F1: {metrics['F1']:.4f} | AUROC: {metrics['AUROC']:.4f} | "
        f"Labels/Image: {metrics['LabelsPerImage']:.2f}"
    )
    return thresholds, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate per-class sigmoid thresholds")
    parser.add_argument("--model-path", default=MODEL_SAVE_PATH)
    parser.add_argument("--thresholds-path", default=THRESHOLDS_SAVE_PATH)
    args = parser.parse_args()
    calibrate_model(args.model_path, args.thresholds_path)
