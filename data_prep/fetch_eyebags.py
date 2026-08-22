"""Stage new Eye_Bags / Wrinkles positives and Healthy negatives.

Sources (already downloaded under data_prep/downloads/):
  eyebags_yolo/   Roboflow "skin-condition-detection_merged" v3, CC BY 4.0,
                  YOLOv8 boxes. Every image carries exactly one class; we take
                  eyebag-only images as Eye_Bags and wrinkle-only as Wrinkles.
  skin_defects/   TrainingData.pro free sample. Only front.jpg per subject is
                  used so the three views of one person cannot straddle splits.
  utkface/        UTKFace aligned crops named age_gender_race_date.jpg.
                  Ages 18-35 sampled as clean-face negatives (label: none).

Output: data_prep/downloads/staged/{Eye_Bags,Wrinkles,Healthy}/ with prefixed
file names, ready for clean_and_split.py to pool, dedupe and split.
"""

import os
import random
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOADS = os.path.join(ROOT, "data_prep", "downloads")
STAGED = os.path.join(DOWNLOADS, "staged")

HEALTHY_SAMPLE = 450
HEALTHY_AGE_RANGE = range(18, 36)
SEED = 42

# YOLO class ids in eyebags_yolo/data.yaml: 0 Acne, 1 Eyebags, 2 Wrinkles.
YOLO_TO_CLASS = {"1": "Eye_Bags", "2": "Wrinkles"}


def stage(source_path, class_name, prefix):
    destination_dir = os.path.join(STAGED, class_name)
    os.makedirs(destination_dir, exist_ok=True)
    target = os.path.join(destination_dir, prefix + os.path.basename(source_path))
    if not os.path.exists(target):
        shutil.copy2(source_path, target)


def stage_yolo_singles():
    counts = {name: 0 for name in YOLO_TO_CLASS.values()}
    base = os.path.join(DOWNLOADS, "eyebags_yolo")
    for split in ("train", "valid", "test"):
        labels_dir = os.path.join(base, split, "labels")
        images_dir = os.path.join(base, split, "images")
        if not os.path.isdir(labels_dir):
            continue
        for label_file in sorted(os.listdir(labels_dir)):
            with open(os.path.join(labels_dir, label_file)) as handle:
                classes = {line.split()[0] for line in handle if line.strip()}
            if len(classes) != 1:
                continue
            class_name = YOLO_TO_CLASS.get(classes.pop())
            if class_name is None:
                continue
            image_name = label_file.rsplit(".", 1)[0] + ".jpg"
            image_path = os.path.join(images_dir, image_name)
            if os.path.exists(image_path):
                stage(image_path, class_name, "rf3_")
                counts[class_name] += 1
    return counts


def stage_tdpro_fronts():
    count = 0
    bags_dir = os.path.join(DOWNLOADS, "skin_defects", "files", "bags")
    if not os.path.isdir(bags_dir):
        return count
    for subject in sorted(os.listdir(bags_dir)):
        front = os.path.join(bags_dir, subject, "front.jpg")
        if os.path.exists(front):
            stage(front, "Eye_Bags", f"tdpro_{subject}_")
            count += 1
    return count


def stage_utkface_negatives():
    rng = random.Random(SEED)
    utk_root = os.path.join(DOWNLOADS, "utkface")
    candidates = []
    for current_dir, _, files in os.walk(utk_root):
        for file_name in files:
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            try:
                age = int(file_name.split("_")[0])
            except ValueError:
                continue
            if age in HEALTHY_AGE_RANGE:
                candidates.append(os.path.join(current_dir, file_name))
    # os.walk order is stable after sorting; dedupe same-named files across
    # the duplicate UTKFace folder copies before sampling.
    unique = {}
    for path in sorted(candidates):
        unique.setdefault(os.path.basename(path), path)
    pool = list(unique.values())
    picked = rng.sample(pool, min(HEALTHY_SAMPLE, len(pool)))
    for path in picked:
        stage(path, "Healthy", "utk_")
    return len(picked), len(pool)


def main():
    yolo_counts = stage_yolo_singles()
    tdpro_count = stage_tdpro_fronts()
    healthy_count, healthy_pool = stage_utkface_negatives()
    print(f"Eye_Bags : roboflow {yolo_counts['Eye_Bags']} + tdpro {tdpro_count}")
    print(f"Wrinkles : roboflow {yolo_counts['Wrinkles']}")
    print(f"Healthy  : {healthy_count} (havuz {healthy_pool}, yas 18-35)")
    for class_name in ("Eye_Bags", "Wrinkles", "Healthy"):
        class_dir = os.path.join(STAGED, class_name)
        total = len(os.listdir(class_dir)) if os.path.isdir(class_dir) else 0
        print(f"staged/{class_name}: {total}")


if __name__ == "__main__":
    main()
