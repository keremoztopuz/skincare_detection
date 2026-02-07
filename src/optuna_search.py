import os
import sys
import json
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import f1_score
import optuna
from optuna.trial import TrialState

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from model import build_model
from dataset import SkinDataset, load_data, val_transform

N_TRIALS = 30
EPOCHS_PER_TRIAL = 5
STUDY_NAME = "skincare_ai_optimization"
DEVICE = config.DEVICE
OUTPUT_DIR = os.path.join(config.ROOT_DIR, "outputs", "optuna_results")


def get_train_transform(trial):
    brightness = trial.suggest_float("brightness", 0.05, 0.3)
    contrast = trial.suggest_float("contrast", 0.05, 0.3)
    saturation = trial.suggest_float("saturation", 0.05, 0.2)
    erasing_prob = trial.suggest_float("erasing_prob", 0.1, 0.5)
    erasing_scale_min = trial.suggest_float("erasing_scale_min", 0.02, 0.1)
    erasing_scale_max = trial.suggest_float("erasing_scale_max", 0.15, 0.4)
    crop_scale_min = trial.suggest_float("crop_scale_min", 0.5, 0.8)
    
    transform = transforms.Compose([
        transforms.RandomResizedCrop(config.IMG_SIZE, scale=(crop_scale_min, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
        transforms.RandomErasing(p=erasing_prob, scale=(erasing_scale_min, erasing_scale_max), ratio=(0.3, 3.3)),
    ])
    
    return transform


def get_trial_dataloaders(trial, batch_size):
    train_images, train_labels = load_data(config.TRAIN_DIR)
    val_images, val_labels = load_data(config.VAL_DIR)
    
    train_transform = get_train_transform(trial)
    
    train_dataset = SkinDataset(train_images, train_labels, default_transform=train_transform)
    val_dataset = SkinDataset(val_images, val_labels, default_transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader


def validate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = running_loss / len(val_loader)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    
    return avg_loss, f1


def objective(trial):
    print(f"\n{'='*60}")
    print(f"TRIAL {trial.number + 1}/{N_TRIALS}")
    print(f"{'='*60}")
    
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])
    drop_rate = trial.suggest_float("drop_rate", 0.1, 0.5)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
    
    print(f"LR: {lr:.6f} | WD: {weight_decay:.6f} | BS: {batch_size} | Opt: {optimizer_name} | Drop: {drop_rate:.2f} | LS: {label_smoothing:.2f}")
    
    original_drop_rate = config.DROP_RATE
    config.DROP_RATE = drop_rate
    model = build_model().to(DEVICE)
    config.DROP_RATE = original_drop_rate
    
    train_loader, val_loader = get_trial_dataloaders(trial, batch_size)
    
    weights = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_PER_TRIAL, eta_min=lr/10)
    scaler = torch.amp.GradScaler(DEVICE)
    
    best_f1 = 0.0
    
    for epoch in range(EPOCHS_PER_TRIAL):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Trial {trial.number+1} Epoch {epoch+1}/{EPOCHS_PER_TRIAL}", 
                    position=0, leave=True, ncols=100)
        
        for images, labels in pbar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            if label_smoothing > 0:
                smooth_labels = labels * (1 - label_smoothing) + label_smoothing / config.NUM_CLASSES
            else:
                smooth_labels = labels
            
            optimizer.zero_grad()
            
            with torch.autocast(DEVICE):
                outputs = model(images)
                loss = criterion(outputs, smooth_labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        val_loss, f1 = validate(model, val_loader, criterion)
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Val Loss={val_loss:.4f}, F1={f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
        
        trial.report(f1, epoch)
        
        if trial.should_prune():
            print(f"Trial {trial.number + 1} pruned at epoch {epoch + 1}")
            raise optuna.exceptions.TrialPruned()
    
    print(f"Trial {trial.number + 1} completed. Best F1: {best_f1:.4f}")
    return best_f1


def save_results(study, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    best_params = study.best_params
    best_params["best_f1_score"] = study.best_value
    best_params["best_trial_number"] = study.best_trial.number
    
    with open(os.path.join(output_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=4)
    
    df = study.trials_dataframe()
    df.to_csv(os.path.join(output_dir, "all_trials.csv"), index=False)
    
    try:
        import optuna.visualization as vis
        figures_dir = os.path.join(output_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        
        vis.plot_optimization_history(study).write_image(os.path.join(figures_dir, "optimization_history.png"))
        vis.plot_param_importances(study).write_image(os.path.join(figures_dir, "param_importances.png"))
        vis.plot_parallel_coordinate(study).write_image(os.path.join(figures_dir, "parallel_coordinate.png"))
    except:
        pass
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Total Trials: {len(study.trials)}")
    print(f"Best F1: {study.best_value:.4f}")
    print(f"Best Trial: #{study.best_trial.number + 1}")
    print("\nBest Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


def main():
    print(f"Device: {DEVICE} | Trials: {N_TRIALS} | Epochs/Trial: {EPOCHS_PER_TRIAL}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    db_path = f"sqlite:///{os.path.join(OUTPUT_DIR, 'optuna_study.db')}"
    
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=db_path,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1, interval_steps=1)
    )
    
    completed = len([t for t in study.trials if t.state == TrialState.COMPLETE])
    if completed > 0:
        print(f"Resuming: {completed} trials already completed")
        if completed >= N_TRIALS:
            save_results(study, OUTPUT_DIR)
            return
    
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True, gc_after_trial=True)
    save_results(study, OUTPUT_DIR)
    print(f"\nDone! Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
