import os 
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import config

#transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
])

healthy_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.15, 0.4)), 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN, std=config.STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])
# dataset class
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

        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]

        return image, torch.tensor(label, dtype=torch.long)

# gets all images and labels from the dataset
def load_data(DATA_DIR):
    images = []
    labels = []

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Directory not found: {DATA_DIR}")

    for class_name in config.CLASS_NAMES:
        CLASS_DIR = os.path.join(DATA_DIR, class_name)
        label = config.CLASS_NAMES.index(class_name)
        if not os.path.exists(CLASS_DIR):
            print(f"Warning: Directory not found: {CLASS_DIR}")
            continue

        for file in os.listdir(CLASS_DIR):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                images.append(os.path.join(CLASS_DIR, file))
                labels.append(label) 

    return images, labels

# dataloaders for training, validation and testing
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

if __name__ == "__main__":
    try:
        train_loader, val_loader, test_loader = get_dataloaders()
        print(f"Train: {len(train_loader.dataset)} images")
        print(f"Val: {len(val_loader.dataset)} images")
        print(f"Test: {len(test_loader.dataset)} images")
    except FileNotFoundError as e:
        train_loader, val_loader, test_loader = None, None, None
        print("Data loading failed. Please check the dataset paths.")