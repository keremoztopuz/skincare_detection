import os 
import torch
from tqdm import tqdm 
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix, 
    hamming_loss, multilabel_confusion_matrix
)
import matplotlib.pyplot as plt 
import seaborn as sns 

from config import DEVICE, MODEL_SAVE_PATH, CLASS_NAMES, IMAGES_DIR
from model import build_model
from dataset import get_dataloaders

# evaluates trained model on test set
def evaluate_model(model_name=None, save_path=None):
    model_path = save_path or MODEL_SAVE_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_labels = []
    all_predictions = []

    train_loader, val_loader, test_loader = get_dataloaders()

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())

    metrics = {
        "Accuracy": accuracy_score(all_labels, all_predictions),
        "Precision": precision_score(all_labels, all_predictions, average="macro", zero_division=0),
        "Recall": recall_score(all_labels, all_predictions, average="macro", zero_division=0),
        "F1": f1_score(all_labels, all_predictions, average="macro", zero_division=0)
    }

    return metrics, all_labels, all_predictions

# prints metrics and saves confusion matrix
def print_results(metrics, all_labels, all_predictions, save_plots=True):
    print(f"\n{'='*50}")
    print(f"Accuracy:  {metrics['Accuracy']:.4f} ({metrics['Accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall:    {metrics['Recall']:.4f}")
    print(f"F1 Score:  {metrics['F1']:.4f}")
    print(f"{'='*50}")
    
    print("\nclassification report:")
    print(classification_report(all_labels, all_predictions, target_names=CLASS_NAMES))
    
    if save_plots:
        mcm = multilabel_confusion_matrix(all_labels, all_predictions)
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))

        for i, (ax, class_name) in enumerate(zip(axes, CLASS_NAMES)):
            sns.heatmap(mcm[i], annot=True, fmt="d", cmap="Blues", 
                        xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"], ax=ax)
                    
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title(f"{class_name}")
        
        plt.suptitle(f"confusion matrix - accuracy: {metrics['Accuracy']*100:.2f}%")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, "confusion_matrix.png"))
        print("\nconfusion matrix saved to: confusion_matrix.png")

        for i, class_name in enumerate(CLASS_NAMES):
            print(f"{class_name}: TN={mcm[i,0,0]}, FP={mcm[i,0,1]}, FN={mcm[i,1,0]}, TP={mcm[i,1,1]}")

if __name__ == "__main__":
    metrics, all_labels, all_predictions = evaluate_model()
    print_results(metrics, all_labels, all_predictions)