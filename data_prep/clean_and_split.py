"""Merge all image sources, deduplicate, and rebuild leak-free splits.

Pools the existing orchestration_data (all splits) with everything under
data_prep/downloads/{scin,dermnet}/, then:
  1. drops unreadable or tiny (<200 px shorter side) images,
  2. groups duplicates by MD5 and perceptual dhash,
  3. caps each class at MAX_PER_CLASS (seeded random sample of groups),
  4. splits 70/15/15 at the GROUP level, so no duplicate pair can ever
     straddle two splits (this removes the train-test leakage the old
     split shipped with),
  5. rewrites orchestration_data/ after backing the old tree up to
     orchestration_data_v1_backup/.

Optional --face-priority flag reports (does not filter) how many images per
class contain a detectable face, since the app analyzes face crops.
"""

import argparse
import hashlib
import os
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import config  # noqa: E402

DOWNLOADS = os.path.join(ROOT, "data_prep", "downloads")
BACKUP_DIR = os.path.join(ROOT, "orchestration_data_v1_backup")

SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
MIN_SIDE = 200
MAX_PER_CLASS = 600
SEED = 42


def dhash(image, hash_size=8):
    gray = image.convert("L").resize((hash_size + 1, hash_size))
    pixels = list(gray.getdata())
    bits = []
    for row in range(hash_size):
        for col in range(hash_size):
            left = pixels[row * (hash_size + 1) + col]
            right = pixels[row * (hash_size + 1) + col + 1]
            bits.append("1" if left > right else "0")
    return "".join(bits)


def collect_sources():
    # The previous dataset lives in the backup tree by the time we read it,
    # so pool from the backup splits plus the freshly downloaded providers.
    sources = defaultdict(list)
    pools = [os.path.join(BACKUP_DIR, split) for split in SPLITS]
    for provider in ("scin", "dermnet", "staged"):
        pools.append(os.path.join(DOWNLOADS, provider))
    for pool in pools:
        if not os.path.isdir(pool):
            continue
        for class_name in list(config.CLASS_NAMES) + [config.NEGATIVE_CLASS_NAME]:
            class_dir = os.path.join(pool, class_name)
            if not os.path.isdir(class_dir):
                continue
            for file_name in sorted(os.listdir(class_dir)):
                if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    sources[class_name].append(os.path.join(class_dir, file_name))
    return sources


def group_images(paths):
    """Return duplicate groups: images sharing an MD5 or a dhash stay together."""
    groups = defaultdict(list)
    dropped = 0
    for path in paths:
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                if min(image.size) < MIN_SIDE:
                    dropped += 1
                    continue
                perceptual = dhash(image)
        except Exception:
            dropped += 1
            continue
        with open(path, "rb") as image_file:
            digest = hashlib.md5(image_file.read()).hexdigest()
        groups[(perceptual, digest[:8])].append(path)
    # Merge groups that share the perceptual hash alone (near-duplicates).
    merged = defaultdict(list)
    for (perceptual, _), members in groups.items():
        merged[perceptual].extend(members)
    return list(merged.values()), dropped


def split_groups(groups, rng):
    rng.shuffle(groups)
    total = len(groups)
    train_end = int(total * SPLITS["train"])
    val_end = train_end + int(total * SPLITS["val"])
    return {
        "train": groups[:train_end],
        "val": groups[train_end:val_end],
        "test": groups[val_end:],
    }


def count_faces(paths):
    try:
        import cv2
    except ImportError:
        return None
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    hits = 0
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(cascade.detectMultiScale(gray, 1.2, 4)) > 0:
            hits += 1
    return hits


DEPRECATION_NOTICE = """
clean_and_split.py artik kullanim disi. Yerine data_prep/build_dataset.py kullanin.

Nedeni: bu betik uc olcumlu kusur uretti.
  1. Cozunurluk kisayolu  - hicbir yeniden boyutlandirma yapmadigi icin Healthy
     200x200, Eye_Bags/Wrinkles 640x640 kaldi (shutil.copy2, asagida).
  2. Split sizintisi      - gruplama tam-dhash esitligine dayaniyor, Hamming
     yaricapi yok, ayna hash'i yok. Roboflow augment kopyalari train ve test'e
     dagildi (Eye_Bags 42/90, Wrinkles 34/90).
  3. Provenans kaybi      - kaynak/lisans bilgisi havuzlamada siliniyor.

Yine de calistirmak zorundaysaniz --force verin; hedef dizin zaman damgali
olarak yedeklenir, asla silinmez.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--face-priority", action="store_true", help="report face counts per class")
    parser.add_argument(
        "--force",
        action="store_true",
        help="deprecated betigi yine de calistir; DATA_DIR zaman damgali yedeklenir",
    )
    arguments = parser.parse_args()

    if not arguments.force:
        print(DEPRECATION_NOTICE)
        return 1

    rng = random.Random(SEED)

    # Back up the current split BEFORE collecting paths, so every pooled path
    # stays readable while images are hashed and copied.
    #
    # The previous version deleted DATA_DIR outright whenever BACKUP_DIR already
    # existed. Since both trees are gitignored that was an unrecoverable wipe of
    # the only copy, and BACKUP_DIR held a stale 434-image tree. Never delete;
    # always move to a fresh timestamped directory.
    if os.path.isdir(config.DATA_DIR):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = f"{BACKUP_DIR}_{stamp}"
        shutil.move(config.DATA_DIR, backup_path)
        print(f"eski veri yedeklendi: {backup_path}")

    sources = collect_sources()

    print(f"{'sinif':<10} {'ham':>5} {'grup':>5} {'elenen':>6} {'train':>6} {'val':>4} {'test':>5}")
    for class_name, paths in sources.items():
        groups, dropped = group_images(paths)
        if len(groups) > MAX_PER_CLASS:
            groups = rng.sample(groups, MAX_PER_CLASS)
        split = split_groups(groups, rng)
        counts = {}
        for split_name, split_group_list in split.items():
            destination = os.path.join(config.DATA_DIR, split_name, class_name)
            os.makedirs(destination, exist_ok=True)
            written = 0
            for group in split_group_list:
                # One representative per duplicate group is enough.
                source_path = group[0]
                target = os.path.join(destination, os.path.basename(source_path))
                if not os.path.exists(target):
                    shutil.copy2(source_path, target)
                written += 1
            counts[split_name] = written
        print(f"{class_name:<10} {len(paths):>5} {len(groups):>5} {dropped:>6} "
              f"{counts['train']:>6} {counts['val']:>4} {counts['test']:>5}")
        if arguments.face_priority:
            face_hits = count_faces([g[0] for g in split["train"]])
            if face_hits is not None:
                print(f"  {class_name}: {face_hits}/{counts['train']} train goruntusunde yuz bulundu")

    print("\nleakage denetimi...")
    from dataset import audit_split_leakage  # noqa: E402
    leaks = audit_split_leakage()
    print("TEMIZ" if not leaks else f"HALA {len(leaks)} sizinti var!")


if __name__ == "__main__":
    sys.exit(main() or 0)
