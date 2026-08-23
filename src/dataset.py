import os 
import hashlib
import warnings
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import config
import nuisance

# Augmentation strengths follow the Optuna search results (trial #6): a much
# wider crop range plus random erasing beat the old conservative settings on
# this small dataset. ColorJitter is widened to cover phone-camera lighting
# and white-balance variance.
#
# RandomNuisance comes first and is the reason this pipeline exists in its
# current form. Without it the previous dataset let the model separate classes
# on image resolution alone: every Healthy image was 200x200 and every
# Eye_Bags/Wrinkles 640x640, and downscaling real Eye_Bags photos to 200x200
# collapsed recall from 38/40 to 8/40. Note that RandomResizedCrop actively
# made that worse, upsampling a 154px crop of a 200px image to 384. Re-drawing
# the nuisance every epoch is what turns resolution into noise rather than a
# per-image constant the network can memorize.
train_transform = transforms.Compose([
    nuisance.RandomNuisance(),
    transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.55, 1.0), ratio=(0.85, 1.18)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.03),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN, std=config.STD),
    transforms.RandomErasing(p=0.45),
])


def eval_transform(view: str = "identity"):
    """Val/test transform under a named nuisance view.

    The squash to a square matches what the iOS app does to a Vision face crop
    (CameraViewModel.analyzeWithCoreML), and canonical images are already
    square, so it is a no-op on curated data. Any eval path the app cannot
    reproduce is a source of exactly the illusion this rebuild is undoing.
    """
    return transforms.Compose([
        transforms.Lambda(nuisance.EVAL_VIEWS[view]),
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
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
        if label >= 0:
            target[label] = 1
        # label < 0 (Healthy negatives) keeps the all-zero target.

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
    negative_dir = os.path.join(DATA_DIR, config.NEGATIVE_CLASS_NAME)
    if os.path.isdir(negative_dir):
        for file in sorted(os.listdir(negative_dir)):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                images.append(os.path.join(negative_dir, file))
                labels.append(config.NEGATIVE_LABEL)
    return images, labels

def calculate_pos_weights(labels, num_classes=len(config.CLASS_NAMES)):
    """BCE pos_weight = negative sample count / positive sample count.

    Healthy negatives (label < 0) contribute to every class's negative
    count but to no positive count, so they raise no pos_weight.
    """
    label_tensor = torch.as_tensor(labels, dtype=torch.long)
    positive_tensor = label_tensor[label_tensor >= 0]
    positives = torch.bincount(positive_tensor, minlength=num_classes).float()
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
