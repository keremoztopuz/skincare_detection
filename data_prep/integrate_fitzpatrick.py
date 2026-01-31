import pandas as pd
import os
import shutil
from tqdm import tqdm

# --- paths ---
csv_path = "/Users/keremoztopuz/Downloads/fitzpatrick17k.csv"
images_src = "/Users/keremoztopuz/Downloads/data/finalfitz17k" # Resimlerin olduğu klasörün içi (örn: .jpg'lerin olduğu yer)
target_root = "/Users/keremoztopuz/Desktop/dataset prep/MASTER_DATASET"

# --- read ---
df = pd.read_csv(csv_path)

# --- mapping ---
mapping = {
    "acne": "Acne",
    "rosacea": "Acne",
    "acne vulgaris": "Acne",
    "eczema": "Eczema",
    "seborrheic dermatitis": "Eczema",
    "atopic dermatitis": "Eczema",
    "allergic contact dermatitis": "Eczema",
    "dyshidrotic eczema": "Eczema",
    "neurodermatitis": "Eczema",
    "psoriasis": "Psoriasis",
    "pustular psoriasis": "Psoriasis",
    "nevus": "Ben_Lezyon"
}

print("Fitzpatrick data analyzing...")

stats = {"Acne": 0, "Eczema": 0, "Psoriasis": 0, "Ben_Lezyon": 0}
skin_types = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, -1: 0}

for _, row in tqdm(df.iterrows(), total=len(df)):
    label_val = str(row['label']).lower()
    
    target_folder = None
    for key, folder in mapping.items():
        if key in label_val:
            target_folder = folder
            break
            
    if target_folder:
        img_id = str(row['md5hash'])
        possible_names = [img_id, f"{img_id}.jpg", f"{img_id}.png", f"{img_id}.jpeg"]
        
        found = False
        for name in possible_names:
            src_path = os.path.join(images_src, name)
            if os.path.exists(src_path):
                #save as jpg always
                dst_path = os.path.join(target_root, target_folder, f"fitz_{img_id}.jpg")
                shutil.copy(src_path, dst_path)
                
                stats[target_folder] += 1
                scale = row['fitzpatrick_scale']
                skin_types[scale] = skin_types.get(scale, 0) + 1
                found = True
                break
        
        # If not found, check images_src for files

print("\n--- FITZPATRICK INTEGRATION REPORT ---")
for cat, count in stats.items():
    print(f"{cat:<12}: {count} image added.")

print("\n--- SKIN TYPE DISTRIBUTION (Scale 1-6) ---")
for scale in sorted(skin_types.keys()):
    if scale != -1:
        print(f"Scale {scale}: {skin_types[scale]} image")