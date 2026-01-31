import os

target_root = "/Users/keremoztopuz/Desktop/dataset prep/FINAL_SPLIT/train"

print(f"{'Category':<20} | {'Image Count':<15}")
print("-" * 40)

total_images = 0
for cat in os.listdir(target_root):
    cat_path = os.path.join(target_root, cat)
    
    if os.path.isdir(cat_path):
        count = len([f for f in os.listdir(cat_path) if os.path.isfile(os.path.join(cat_path, f))])
        print(f"{cat:<20} | {count:<15}")
        total_images += count

print("-" * 40)
print(f"{'Total':<20} | {total_images:<15}")