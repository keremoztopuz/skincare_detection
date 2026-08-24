import os
import random
import shutil
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from config import (
    AUXILIARY_CE_WEIGHT,
    BACKBONE_LR_MULTIPLIER,
    HEAD_ONLY_EPOCHS,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    LOGS_DIR,
    NUM_EPOCHS,
    DEVICE,
    WEIGHT_DECAY,
    PATIENCE,
    CHECKPOINT_DIR,
    MODEL_SAVE_PATH,
    THRESHOLDS_SAVE_PATH,
    MODEL_NAME,
    MIN_CHECKPOINT_AUROC,
    POS_WEIGHT_POWER,
    DETECTION_THRESHOLD,
    WARMUP_EPOCHS,
    GRADIENT_CLIP,
    SEED,
)

from model import build_model, freeze_backbone, get_model_info
from dataset import calculate_pos_weights, get_dataloaders
from utils import calibrate_thresholds, calculate_metrics, save_thresholds


class MaskedMultiLabelLoss(nn.Module):
    """BCE scored only where a label exists.

    Most images know about some conditions and not others: a clinical close-up
    of a forearm says whether there is eczema and nothing at all about eye
    bags. The previous version had no way to say "unknown", so every unjudged
    condition arrived as a zero and trained the head on a false negative.

    The auxiliary cross-entropy is gone with it. It only ever fired on rows
    with exactly one positive, which excluded every clean image, and it asks
    the four conditions to compete for one answer — the opposite of what a
    multi-label head is for now that a face can carry two of them at once.
    """

    def __init__(self, pos_weight, label_smoothing=LABEL_SMOOTHING):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets, mask):
        # Smoothed BCE targets pull hard 0/1 labels slightly toward the center,
        # which regularizes the tiny dataset against overconfident logits.
        smoothed = targets * (1.0 - self.label_smoothing) + self.label_smoothing / 2.0
        per_element = nn.functional.binary_cross_entropy_with_logits(
            logits, smoothed, pos_weight=self.pos_weight, reduction="none")
        known = mask.sum()
        if known == 0:
            return logits.sum() * 0.0
        # Mean over known entries, not over the whole matrix: otherwise a
        # batch of mostly-unknown rows would look like a batch with a small
        # loss and the optimiser would coast through it.
        return (per_element * mask).sum() / known


def append_metrics_log(epoch, train_loss, val_loss, metrics, lr):
    """Append one row per epoch to a CSV so runs leave an inspectable history."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, "metrics_history.csv")
    write_header = not os.path.exists(log_path)
    with open(log_path, "a") as log_file:
        if write_header:
            log_file.write(
                "timestamp,epoch,train_loss,val_loss,top1_accuracy,"
                "negative_reject,f1,auroc,lr\n"
            )
        log_file.write(
            f"{datetime.now().isoformat(timespec='seconds')},{epoch},"
            f"{train_loss:.6f},{val_loss:.6f},{metrics['AUROC']:.6f},"
            f"{metrics['NegativeReject']:.6f},"
            f"{metrics['F1']:.6f},{metrics['AUROC']:.6f},{lr:.8f}\n"
        )


def backup_existing_artifacts(save_path):
    """Keep the previous best model recoverable before a new training run."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if os.path.exists(save_path):
        backup_path = os.path.join(CHECKPOINT_DIR, f"best_model-{timestamp}.pth")
        shutil.copy2(save_path, backup_path)
        print(f"Previous model backed up to: {backup_path}")
    if os.path.exists(THRESHOLDS_SAVE_PATH):
        threshold_backup = os.path.join(CHECKPOINT_DIR, f"thresholds-{timestamp}.json")
        shutil.copy2(THRESHOLDS_SAVE_PATH, threshold_backup)
        print(f"Previous thresholds backed up to: {threshold_backup}")

def validate_model(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    all_probabilities = []
    all_labels = []
    all_masks = []

    with torch.no_grad():
        for images, labels, masks in tqdm(val_loader, desc="Validating"):
            images = images.to(DEVICE)
            labels, masks = labels.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels, masks)
            running_loss += loss.item()
            probs = torch.sigmoid(outputs)
            all_probabilities.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_masks.extend(masks.cpu().numpy())

    avg_loss = running_loss / len(val_loader)
    thresholds = calibrate_thresholds(all_labels, all_probabilities, all_masks)
    metrics, _ = calculate_metrics(all_labels, all_probabilities, thresholds, all_masks)
    fixed_thresholds = np.full(len(thresholds), DETECTION_THRESHOLD, dtype=np.float32)
    fixed_metrics, _ = calculate_metrics(all_labels, all_probabilities, fixed_thresholds, all_masks)
    threshold_text = ", ".join(f"{value:.2f}" for value in thresholds)
    recall_text = ", ".join(f"{value:.3f}" for value in metrics["PerClassRecall"])
    print(
        f"Val Loss: {avg_loss:.4f} | "
        f"NegReject: {metrics['NegativeReject']:.4f} | "
        f"NegReject@{DETECTION_THRESHOLD:.2f}: {fixed_metrics['NegativeReject']:.4f} | "
        f"Prec: {metrics['Precision']:.4f} | Rec: {metrics['Recall']:.4f} | "
        f"F1: {metrics['F1']:.4f} | AUROC: {metrics['AUROC']:.4f} | "
        f"Labels/Image: {metrics['LabelsPerImage']:.2f} | "
        f"Known/Image: {metrics['KnownPerImage']:.2f}"
    )
    print(f"Thresholds: [{threshold_text}] | Per-class recall: [{recall_text}]")
    return avg_loss, metrics, thresholds

def train_model(model, model_name=None, save_path=None, epochs=None):
    # Leakage is checked by data_prep/leakage.py as a gate before training,
    # not by a byte-hash pass that missed augmented copies.
    train_loader, val_loader, _ = get_dataloaders()
    epochs = epochs or NUM_EPOCHS
    save_path = save_path or MODEL_SAVE_PATH

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    backup_existing_artifacts(save_path)

    raw_weights = calculate_pos_weights(*train_loader.dataset.labels).to(DEVICE)
    weights = raw_weights.pow(POS_WEIGHT_POWER)
    print(f"Raw BCE pos_weight: {[round(value, 4) for value in raw_weights.tolist()]}")
    print(f"Applied BCE pos_weight: {[round(value, 4) for value in weights.tolist()]}")
    criterion = MaskedMultiLabelLoss(pos_weight=weights).to(DEVICE)

    head_params = [p for p in model.head.parameters() if p.requires_grad]
    head_param_ids = {id(p) for p in head_params}
    backbone_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in head_param_ids
    ]
    trainable_params = backbone_params + head_params
    parameter_groups = []
    if backbone_params:
        parameter_groups.append({"params": backbone_params, "lr": LEARNING_RATE * BACKBONE_LR_MULTIPLIER})
    parameter_groups.append({"params": head_params, "lr": LEARNING_RATE})
    optimizer = optim.AdamW(parameter_groups, weight_decay=WEIGHT_DECAY)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - WARMUP_EPOCHS, eta_min=LEARNING_RATE/10)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS])
    # AMP is CUDA-only: GradScaler is a no-op on MPS and autocast is flaky there,
    # so Apple Silicon trains in plain FP32.
    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler(DEVICE, enabled=use_amp)

    for param in backbone_params:
        param.requires_grad = False

    best_val_auroc = MIN_CHECKPOINT_AUROC
    patience_counter = 0

    print(f"Training model: {model_name or MODEL_NAME}")
    print(f"Epochs 1-{HEAD_ONLY_EPOCHS}: classification head only")
    for epoch in range(epochs):
        if epoch == HEAD_ONLY_EPOCHS:
            for param in backbone_params:
                param.requires_grad = True
            print(
                f"Unfroze final ConvNeXt stage at epoch {epoch + 1}; "
                f"backbone LR={LEARNING_RATE * BACKBONE_LR_MULTIPLIER:.2e}"
            )

        model.train()
        # Keep frozen feature extractors deterministic during head-only warmup.
        if epoch < HEAD_ONLY_EPOCHS:
            for stage in model.stages:
                stage.eval()
        else:
            for stage in model.stages[:-1]:
                stage.eval()
        running_loss = 0.0
        for images, labels, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(DEVICE)
            labels, masks = labels.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            with torch.autocast(DEVICE, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels, masks)
            scaler.scale(loss).backward()
            if GRADIENT_CLIP > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        
        scheduler.step()
        val_loss, val_metrics, thresholds = validate_model(model, val_loader, criterion)
        train_loss = running_loss / len(train_loader)
        print(f"Train Loss: {train_loss:.4f}")
        append_metrics_log(epoch + 1, train_loss, val_loss, val_metrics, optimizer.param_groups[-1]["lr"])

        current_auroc = val_metrics["AUROC"]
        if np.isfinite(current_auroc) and current_auroc > best_val_auroc:
            best_val_auroc = current_auroc
            torch.save(model.state_dict() if hasattr(model, 'state_dict') else model, save_path)
            save_thresholds(thresholds)
            print(f"Saved new best model with validation AUROC={best_val_auroc:.4f}")
            patience_counter = 0
        elif epoch >= HEAD_ONLY_EPOCHS:
            patience_counter += 1

        # The Top-1 checkpoint is gone with the metric: it scored which
        # single class won, which has no meaning once an image can carry two
        # conditions, and it was the number that reached 1.0000 by epoch four
        # on a dataset separable by resolution alone.

        if patience_counter >= PATIENCE:
            break


if __name__ == "__main__":
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    model = build_model()
    model = freeze_backbone(model)
    model.to(DEVICE)
    print(f"Model info: {get_model_info(model)}")
    train_model(model)
