import os 
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import config

# disease transforms
disease_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN, std=config.STD),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
])

# healthy transforms
healthy_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.15, 0.45)), 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN, std=config.STD),
    transforms.RandomErasing(p=0.2),
])

# eye bagd transforms
eyebag_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.15, 0.45)), 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN, std=config.STD),
    transforms.RandomErasing(p=0.2),
])

val_transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN, std=config.STD)
])

# dataset class
class SkinDataset(Dataset):
    def __init__(self, image_paths, labels, transform_map=None, default_transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform_map = transform_map
        self.default_transform = default_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform_map and label in self.transform_map:
            image = self.transform_map[label](image)
        elif self.default_transform:
            image = self.default_transform(image)

        one_hot_encoding = torch.zeros(len(config.CLASS_NAMES))
        one_hot_encoding[label] = 1

        return image, one_hot_encoding

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

    healthy_idx = config.CLASS_NAMES.index("Healthy")
    train_transform_map[healthy_idx] = healthy_train_transform

    eyebag_idx = config.CLASS_NAMES.index("Eye_Bags")
    train_transform_map[eyebag_idx] = eyebag_train_transform

    train_dataset = SkinDataset(
        train_images, train_labels, 
        transform_map=train_transform_map, 
        default_transform=disease_train_transform
    )
    val_dataset = SkinDataset(val_images, val_labels, default_transform=val_transform)
    test_dataset = SkinDataset(test_images, test_labels, default_transform=val_transform)

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