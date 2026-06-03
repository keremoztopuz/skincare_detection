import os 
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import config

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN, std=config.STD),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
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
        for file in os.listdir(CLASS_DIR):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                images.append(os.path.join(CLASS_DIR, file))
                labels.append(label) 
    return images, labels

def get_dataloaders(batch_size=config.BATCH_SIZE, shuffle=True):
    train_images, train_labels = load_data(config.TRAIN_DIR)
    val_images, val_labels = load_data(config.VAL_DIR)
    test_images, test_labels = load_data(config.TEST_DIR)

    train_dataset = SkinDataset(train_images, train_labels, transform=train_transform)
    val_dataset = SkinDataset(val_images, val_labels, transform=val_transform)
    test_dataset = SkinDataset(test_images, test_labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader
