import os
import cv2
import numpy as np
from tqdm import tqdm

data_path = "/Users/keremoztopuz/Desktop/dataset prep/FINAL_SPLIT/train"

def get_mean_std(path):
    channels_sum, channels_squared_sum, num_pixels = 0, 0, 0
    
    for subdir in os.listdir(path):
        subdir_path = os.path.join(path, subdir)
        if not os.path.isdir(subdir_path): continue
        
        files = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for file in tqdm(files, desc=f"Computing: {subdir}"):
            img_path = os.path.join(subdir_path, file)
            img = cv2.imread(img_path)
            if img is None: continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            
            channels_sum += np.mean(img, axis=(0, 1))
            channels_squared_sum += np.mean(img**2, axis=(0, 1))
            num_pixels += 1

    mean = channels_sum / num_pixels
    std = np.sqrt((channels_squared_sum / num_pixels) - mean**2)
    
    return mean, std

mean, std = get_mean_std(data_path)

print(f"\nMean: {mean}")
print(f"Std: {std}")