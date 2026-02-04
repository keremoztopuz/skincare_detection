import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import (
    LEARNING_RATE,
    NUM_EPOCHS,
    CLASS_WEIGHTS,
    DEVICE,
    WEIGHT_DECAY,
    PATIENCE,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    MODEL_SAVE_PATH,
)

from model import build_model
from dataset import get_dataloaders

def validate_model(model, val_loader, criterion):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
    avg_loss = running_loss / len(val_loader)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    
    print(f"Val Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

    return avg_loss

def train_model(model, model_name=None, save_path=None, epochs=None):
    train_loader, val_loader, test_loader = get_dataloaders()
    epochs = epochs or NUM_EPOCHS
    save_path = save_path or MODEL_SAVE_PATH

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=LEARNING_RATE/10)
    
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"Training model: {model_name or 'convnext-tiny'}")
    print(f"Device: {DEVICE}, Epochs: {epochs}, Batch Size: {BATCH_SIZE}, Learning Rate: {LEARNING_RATE}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_loss = validate_model(model, val_loader, criterion)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")
        else:
            patience_counter +=1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"Epoch_{epoch+1}.pth")
        torch.save({
            'Epoch': epoch,
            'Model_state_dict': model.state_dict(),
            'Optimizer_state_dict': optimizer.state_dict(),
            'Val_loss': val_loss,
        }, checkpoint_path)

    print("Training Completed")

if __name__ == "__main__":
    model = build_model().to(DEVICE)
    train_model(model)