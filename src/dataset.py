import json
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
    """Images with a per-condition target and a mask of what is known.

    The mask is the whole point. Every row carries four values, but only the
    ones a person actually judged are scored; the rest are excluded from the
    loss and from the metrics rather than being called absent.
    """

    def __init__(self, records, image_dir, transform=None):
        self.records = records
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image = Image.open(os.path.join(self.image_dir, record["file"])).convert("RGB")
        if self.transform:
            image = self.transform(image)

        target = torch.zeros(len(config.CLASS_NAMES))
        mask = torch.zeros(len(config.CLASS_NAMES))
        for index, name in enumerate(config.CLASS_NAMES):
            value = record["conditions"].get(name)
            if value is not None:
                target[index] = float(value)
                mask[index] = 1.0
        return image, target, mask

    @property
    def labels(self):
        """Target and mask matrices, for pos_weight and for reporting."""
        targets, masks = [], []
        for record in self.records:
            row = [record["conditions"].get(name) for name in config.CLASS_NAMES]
            targets.append([float(v) if v is not None else 0.0 for v in row])
            masks.append([0.0 if v is None else 1.0 for v in row])
        return torch.tensor(targets), torch.tensor(masks)


def load_data(split_dir):
    """Read one split: its image directory and its label file.

    split_dir is <root>/<split>; the labels live beside it as
    <root>/<split>_labels.jsonl so the image directory stays a flat pile of
    files. Folder-per-class cannot express a face with two conditions, which
    is exactly what this dataset now contains.
    """
    root, split = os.path.dirname(split_dir), os.path.basename(split_dir)
    label_path = os.path.join(root, config.LABEL_FILENAME.format(split=split))
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")
    records = []
    with open(label_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records, split_dir


def calculate_pos_weights(targets, masks):
    """BCE pos_weight per class, counted over known entries only.

    Counting an unknown entry as a negative would inflate the weight of every
    head whose condition is rarely judged.
    """
    positives = (targets * masks).sum(dim=0)
    known = masks.sum(dim=0)
    negatives = known - positives
    return negatives / positives.clamp_min(1.0)


# audit_split_leakage lived here. It compared raw byte hashes, which is why
# it called the old dataset TEMIZ while 42 of 90 Eye_Bags test images were
# augmented copies of training images. data_prep/leakage.py replaces it and
# also matches derivation stems, group ids and mirrored perceptual hashes.


def get_dataloaders(batch_size=config.BATCH_SIZE, shuffle=True):
    splits = {}
    for name, directory in (("train", config.TRAIN_DIR),
                            ("val", config.VAL_DIR),
                            ("test", config.TEST_DIR)):
        records, image_dir = load_data(directory)
        transform = train_transform if name == "train" else val_transform
        splits[name] = SkinDataset(records, image_dir, transform=transform)

    # pin_memory only helps CUDA; on Apple Silicon the win comes from parallel
    # JPEG decoding in worker processes.
    workers = 4 if config.DEVICE != "cuda" else 2
    pin = config.DEVICE == "cuda"
    loaders = []
    for name in ("train", "val", "test"):
        loaders.append(DataLoader(
            splits[name], batch_size=batch_size,
            shuffle=shuffle and name == "train",
            num_workers=workers, pin_memory=pin,
            persistent_workers=workers > 0))
    return tuple(loaders)
