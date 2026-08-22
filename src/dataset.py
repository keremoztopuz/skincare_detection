import os 
import hashlib
import warnings
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import config

# Augmentation strengths follow the Optuna search results (trial #6): a much
# wider crop range plus random erasing beat the old conservative settings on
# this small dataset. ColorJitter is widened to cover phone-camera lighting
# and white-balance variance.
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.77, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(7),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN, std=config.STD),
    transforms.RandomErasing(p=0.45),
])

val_transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN, std=config.STD)
])

class SkinDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        target = torch.zeros(len(config.CLASS_NAMES))
        target[label] = 1

        return image, target

def load_data(DATA_DIR):
    images = []
    labels = []
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Directory not found: {DATA_DIR}")
    for class_name in config.CLASS_NAMES:
        CLASS_DIR = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(CLASS_DIR):
            continue
        label = config.CLASS_NAMES.index(class_name)
        for file in sorted(os.listdir(CLASS_DIR)):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                images.append(os.path.join(CLASS_DIR, file))
                labels.append(label) 
    return images, labels

def calculate_pos_weights(labels, num_classes=len(config.CLASS_NAMES)):
    """BCE pos_weight = negative sample count / positive sample count."""
    label_tensor = torch.as_tensor(labels, dtype=torch.long)
    positives = torch.bincount(label_tensor, minlength=num_classes).float()
    negatives = len(labels) - positives
    return negatives / positives.clamp_min(1.0)

def audit_split_leakage(split_dirs=None):
    """Find byte-identical images appearing in more than one data split."""
    split_dirs = split_dirs or {
        "train": config.TRAIN_DIR,
        "val": config.VAL_DIR,
        "test": config.TEST_DIR,
    }
    hashes = defaultdict(list)
    for split_name, split_dir in split_dirs.items():
        image_paths, _ = load_data(split_dir)
        for image_path in image_paths:
            with open(image_path, "rb") as image_file:
                digest = hashlib.sha256(image_file.read()).hexdigest()
            hashes[digest].append((split_name, image_path))

    leaks = [paths for paths in hashes.values() if len({p[0] for p in paths}) > 1]
    if leaks:
        warnings.warn(
            f"Found {len(leaks)} byte-identical image group(s) across data splits. "
            "Metrics may be optimistic; rebuild the splits after grouping duplicates.",
            stacklevel=2,
        )
    return leaks

def get_dataloaders(batch_size=config.BATCH_SIZE, shuffle=True):
    train_images, train_labels = load_data(config.TRAIN_DIR)
    val_images, val_labels = load_data(config.VAL_DIR)
    test_images, test_labels = load_data(config.TEST_DIR)

    train_dataset = SkinDataset(train_images, train_labels, transform=train_transform)
    val_dataset = SkinDataset(val_images, val_labels, transform=val_transform)
    test_dataset = SkinDataset(test_images, test_labels, transform=val_transform)

    # pin_memory only helps CUDA; on Apple Silicon the win comes from parallel
    # JPEG decoding in worker processes.
    workers = 4 if config.DEVICE != "cuda" else 2
    pin = config.DEVICE == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=pin, persistent_workers=workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin, persistent_workers=workers > 0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin, persistent_workers=workers > 0)

    return train_loader, val_loader, test_loader
