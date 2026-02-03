import os 
import torch
from tqdm import tqdm 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns 

from config import DEVICE, MODEL_SAVE_PATH, CLASS_NAMES
from model import build_model
from dataset import get_dataloaders

# evaluates trained model on test set
def evaluate_model(model_name=None, save_path=None):
    model_path = save_path or MODEL_SAVE_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(model_path))
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
            _, preds = torch.max(outputs, dim=1)

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
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(12, 10))    
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.xlabel("predicted")
        plt.ylabel("true")
        plt.title(f"confusion matrix - accuracy: {metrics['Accuracy']*100:.2f}%")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=150)
        print("\nconfusion matrix saved to: confusion_matrix.png")

if __name__ == "__main__":
    metrics, all_labels, all_predictions = evaluate_model()
    print_results(metrics, all_labels, all_predictions)