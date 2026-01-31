import os
import shutil
import cv2
from tqdm import tqdm

# --- paths ---
base_path = "/Users/keremoztopuz/Desktop/dataset prep"
target_root = os.path.join(base_path, "MASTER_DATASET")

# --- sources ---
celeba_src = os.path.join(base_path, "celeba", "img_align_celeba")
isic_src = os.path.join(base_path, "isic-2024-challenge", "train-image", "image")
dermnet_root = os.path.join(base_path, "dermnet")

# --- targets ---
categories = ["Acne", "Eczema", "Psoriasis", "Ben_Lezyon", "Healthy"]
for cat in categories:
    os.makedirs(os.path.join(target_root, cat), exist_ok=True)

# --- 1. DERMNET ---
dermnet_map = {
    "Acne and Rosacea Photos": "Acne",
    "Eczema Photos": "Eczema",
    "Atopic Dermatitis Photos": "Eczema",  
    "Psoriasis pictures Lichen Planus and related diseases": "Psoriasis"
}

print("Dermnet datas merging...")
for folder_name, cat in dermnet_map.items():
    for sub in ["train", "test"]:
        src = os.path.join(dermnet_root, sub, folder_name)
        if os.path.exists(src):
            files = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
            for f in tqdm(files, desc=f"{cat} <- {folder_name[:15]}"):
                new_name = f"{sub}_{folder_name[:5]}_{f}"
                shutil.copy(os.path.join(src, f), os.path.join(target_root, cat, new_name))

# --- 2. ISIC 2024 ---
print("\nISIC 2024 Ben_Lezyon copying...")
if os.path.exists(isic_src):
    isic_files = [f for f in os.listdir(isic_src) if os.path.isfile(os.path.join(isic_src, f))]
    for f in tqdm(isic_files[:4000], desc="Ben_Lezyon"):
        shutil.copy(os.path.join(isic_src, f), os.path.join(target_root, "Ben_Lezyon", f))

# --- 3. CELEBA ---
print("\nCelebA healthy parts copying...")
if os.path.exists(celeba_src):
    celeba_files = [f for f in os.listdir(celeba_src) if f.lower().endswith(('.jpg', '.png'))]
    for i, f in enumerate(tqdm(celeba_files[:4000], desc="Healthy")):
        img = cv2.imread(os.path.join(celeba_src, f))
        if img is not None:
            patch = img[100:180, 50:130] 
            patch = cv2.resize(patch, (224, 224))
            cv2.imwrite(os.path.join(target_root, "Healthy", f"healthy_{i}.jpg"), patch)

print(f"\nProcess completed. Data is here: {target_root}")