import os
import shutil
import random
from tqdm import tqdm

# --- paths ---
src_root = "/Users/keremoztopuz/Desktop/dataset prep/MASTER_DATASET"
dest_root = "/Users/keremoztopuz/Desktop/dataset prep/FINAL_SPLIT"

# --- limits ---
LIMIT_PER_CLASS = 3700 

# --- ratios ---
split_ratio = {'train': 0.8, 'val': 0.1, 'test': 0.1}

# --- categories ---
categories = ["Acne", "Healthy", "Eczema", "Psoriasis", "Ben_Lezyon"]
for split in ['train', 'val', 'test']:
    for cat in categories:
        os.makedirs(os.path.join(dest_root, split, cat), exist_ok=True)

print("Data is being balanced and copied to FINAL_SPLIT folder...")

for cat in categories:
    cat_path = os.path.join(src_root, cat)
    if not os.path.exists(cat_path):
        continue
        
    files = [f for f in os.listdir(cat_path) if os.path.isfile(os.path.join(cat_path, f))]
    random.shuffle(files)
    
    # Balance: Only LIMIT images are selected
    selected_files = files[:LIMIT_PER_CLASS]
    
    n = len(selected_files)
    train_end = int(n * split_ratio['train'])
    val_end = train_end + int(n * split_ratio['val'])
    
    split_data = {
        'train': selected_files[:train_end],
        'val': selected_files[train_end:val_end],
        'test': selected_files[val_end:]
    }
    
    for split_name, split_files in split_data.items():
        for f in tqdm(split_files, desc=f"{cat} -> {split_name}"):
            src_file = os.path.join(cat_path, f)
            dst_file = os.path.join(dest_root, split_name, cat, f)
            shutil.copy(src_file, dst_file)

print(f"\nProcess completed. {len(categories) * LIMIT_PER_CLASS} images distributed.")
print(f"Location: {dest_root}")