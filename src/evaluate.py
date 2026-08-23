import argparse
import os 
import torch
from tqdm import tqdm 
from sklearn.metrics import (
    precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    multilabel_confusion_matrix
)
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np

from config import DEVICE, MODEL_SAVE_PATH, CLASS_NAMES, IMAGES_DIR
from model import build_model
from dataset import get_dataloaders
from utils import calculate_metrics, load_thresholds

# evaluates trained model on test set
def evaluate_model(model_name=None, save_path=None, thresholds_path=None, tta=False):
    model_path = save_path or MODEL_SAVE_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_labels = []
    all_probabilities = []

    train_loader, val_loader, test_loader = get_dataloaders()

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            probs = torch.sigmoid(model(images))
            if tta:
                # Horizontal-flip test-time augmentation: average the two views.
                flipped_probs = torch.sigmoid(model(torch.flip(images, dims=[3])))
                probs = (probs + flipped_probs) / 2.0
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())

    thresholds = load_thresholds(thresholds_path) if thresholds_path else load_thresholds()
    metrics, all_predictions = calculate_metrics(all_labels, all_probabilities, thresholds)
    print("Thresholds:", dict(zip(CLASS_NAMES, map(float, thresholds))))

    return metrics, all_labels, all_predictions

# prints metrics and saves confusion matrix
def print_results(metrics, all_labels, all_predictions, save_plots=True):
    print(f"\n{'='*50}")
    print(f"Top-1 Accuracy: {metrics['Top1Accuracy']:.4f} ({metrics['Top1Accuracy']*100:.2f}%)")
    # Share of Healthy faces that fired no class at all. This is the metric
    # the Healthy negative set was added to move; nan means the split had none.
    print(f"NegativeReject: {metrics['NegativeReject']:.4f} ({metrics['NegativeReject']*100:.2f}%)")
    print(f"Exact Match:    {metrics['Accuracy']:.4f} ({metrics['Accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall:    {metrics['Recall']:.4f}")
    print(f"F1 Score:  {metrics['F1']:.4f}")
    print(f"AUROC:     {metrics['AUROC']:.4f}")
    print(f"Labels/Image: {metrics['LabelsPerImage']:.2f}")
    print(f"{'='*50}")
    
    print("\nclassification report:")
    print(classification_report(all_labels, all_predictions, target_names=CLASS_NAMES))
    
    if save_plots:
        os.makedirs(IMAGES_DIR, exist_ok=True)

        labels_array = np.asarray(all_labels)
        predictions_array = np.asarray(all_predictions)
        class_accuracy = (labels_array == predictions_array).mean(axis=0)
        class_precision = precision_score(labels_array, predictions_array, average=None, zero_division=0)
        class_recall = recall_score(labels_array, predictions_array, average=None, zero_division=0)
        class_f1 = f1_score(labels_array, predictions_array, average=None, zero_division=0)

        metric_names = ["Accuracy", "Precision", "Recall", "F1"]
        class_metrics = np.vstack([class_accuracy, class_precision, class_recall, class_f1])
        x = np.arange(len(CLASS_NAMES))
        width = 0.2

        fig_metrics, ax_metrics = plt.subplots(figsize=(12, 6))
        colors = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626"]
        for idx, (metric_name, metric_values) in enumerate(zip(metric_names, class_metrics)):
            offset = (idx - 1.5) * width
            bars = ax_metrics.bar(x + offset, metric_values, width, label=metric_name, color=colors[idx])
            ax_metrics.bar_label(bars, labels=[f"{value:.2f}" for value in metric_values], padding=2, fontsize=8)

        ax_metrics.set_xticks(x)
        ax_metrics.set_xticklabels(CLASS_NAMES, rotation=25, ha="right")
        ax_metrics.set_ylim(0, 1.08)
        ax_metrics.set_ylabel("score")
        ax_metrics.set_title("Class-based test metrics")
        ax_metrics.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
        ax_metrics.grid(axis="y", linestyle="--", alpha=0.35)
        plt.tight_layout()
        metrics_path = os.path.join(IMAGES_DIR, "test_metrics.png")
        plt.savefig(metrics_path, dpi=150)
        plt.close(fig_metrics)
        print(f"metrics plot saved to: {metrics_path}")

        mcm = multilabel_confusion_matrix(all_labels, all_predictions)
        num_cls = len(CLASS_NAMES)
        fig, axes = plt.subplots(1, num_cls, figsize=(4 * num_cls, 4))

        for i, (ax, class_name) in enumerate(zip(axes, CLASS_NAMES)):
            sns.heatmap(mcm[i], annot=True, fmt="d", cmap="Blues", 
                        xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"], ax=ax)
                    
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title(f"{class_name}")
        
        plt.suptitle(f"confusion matrix - accuracy: {metrics['Accuracy']*100:.2f}%")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        confusion_path = os.path.join(IMAGES_DIR, "confusion_matrix.png")
        plt.savefig(confusion_path, dpi=150)
        plt.close(fig)
        print(f"\nconfusion matrix saved to: {confusion_path}")

        for i, class_name in enumerate(CLASS_NAMES):
            print(f"{class_name}: TN={mcm[i,0,0]}, FP={mcm[i,0,1]}, FN={mcm[i,1,0]}, TP={mcm[i,1,1]}")

        true_classes = np.argmax(all_labels, axis=1)
        pred_classes = np.argmax(all_predictions, axis=1)
        cm = confusion_matrix(true_classes, pred_classes)

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax2)
        ax2.set_xlabel("predicted")
        ax2.set_ylabel("true")
        ax2.set_title(f"full confusion matrix - accuracy: {metrics['Accuracy']*100:.2f}%")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        full_confusion_path = os.path.join(IMAGES_DIR, "full_confusion_matrix.png")
        plt.savefig(full_confusion_path, dpi=150)
        plt.close(fig2)
        print(f"full confusion matrix saved to: {full_confusion_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained skin-condition model")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--thresholds-path", default=None)
    parser.add_argument("--tta", action="store_true", help="average predictions with a horizontal flip")
    args = parser.parse_args()
    metrics, all_labels, all_predictions = evaluate_model(
        save_path=args.model_path,
        thresholds_path=args.thresholds_path,
        tta=args.tta,
    )
    print_results(metrics, all_labels, all_predictions)
