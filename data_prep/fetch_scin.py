"""Download Acne and Eczema cases from the SCIN dataset (Google/Stanford).

SCIN is served from the public GCS bucket dx-scin-public-data (CC-BY 4.0),
so no credentials are required. Cases are selected by their dermatologist
weighted label: a case is taken when the target condition is either the
dominant label or carries at least MIN_WEIGHT of the label mass. The
Eczema mapping mirrors the original DermNet prep (Eczema + Atopic
Dermatitis); broader dermatitis labels are deliberately excluded to keep
the class clean.

Output: data_prep/downloads/scin/{Acne,Eczema}/scin_<case>_<n>.png
"""

import csv
import json
import os
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE_URL = "https://storage.googleapis.com/dx-scin-public-data/"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data_prep", "downloads", "scin")
CSV_DIR = os.path.join(ROOT, "data_prep", "downloads")

MIN_WEIGHT = 0.4
MAX_CASES_PER_CLASS = 400
MAX_IMAGES_PER_CASE = 2
SEED = 42

TARGETS = {
    "Acne": lambda name: "acne" in name,
    "Eczema": lambda name: "eczema" in name or "atopic dermatitis" in name,
}


def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as error:
        print(f"  indirilemedi: {url} ({error})")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def parse_weighted(raw):
    if not raw:
        return {}
    try:
        parsed = json.loads(raw.replace("'", '"'))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def select_cases(labels_path):
    selected = {name: [] for name in TARGETS}
    with open(labels_path) as labels_file:
        for row in csv.DictReader(labels_file):
            weights = parse_weighted(row.get("weighted_skin_condition_label", ""))
            if not weights:
                continue
            total = sum(weights.values()) or 1.0
            dominant = max(weights, key=weights.get).lower()
            for class_name, matcher in TARGETS.items():
                mass = sum(v for k, v in weights.items() if matcher(k.lower())) / total
                if matcher(dominant) or mass >= MIN_WEIGHT:
                    selected[class_name].append(row["case_id"])
                    break
    return selected


def main():
    os.makedirs(CSV_DIR, exist_ok=True)
    labels_path = os.path.join(CSV_DIR, "scin_labels.csv")
    cases_path = os.path.join(CSV_DIR, "scin_cases.csv")
    download(BASE_URL + "dataset/scin_labels.csv", labels_path)
    download(BASE_URL + "dataset/scin_cases.csv", cases_path)

    selected = select_cases(labels_path)
    rng = random.Random(SEED)
    for class_name, case_ids in selected.items():
        if len(case_ids) > MAX_CASES_PER_CLASS:
            selected[class_name] = rng.sample(case_ids, MAX_CASES_PER_CLASS)
        print(f"{class_name}: {len(case_ids)} aday case, {len(selected[class_name])} secildi")

    image_paths = {}
    with open(cases_path) as cases_file:
        for row in csv.DictReader(cases_file):
            paths = [row.get(f"image_{i}_path", "") for i in (1, 2, 3)]
            image_paths[row["case_id"]] = [p for p in paths if p][:MAX_IMAGES_PER_CASE]

    jobs = []
    for class_name, case_ids in selected.items():
        class_dir = os.path.join(OUT_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for case_id in case_ids:
            for index, path in enumerate(image_paths.get(case_id, [])):
                dest = os.path.join(class_dir, f"scin_{case_id}_{index}.png")
                jobs.append((BASE_URL + path, dest))

    print(f"{len(jobs)} goruntu indiriliyor...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda job: download(*job), jobs))
    print(f"tamam: {sum(results)}/{len(jobs)} basarili")
    for class_name in TARGETS:
        class_dir = os.path.join(OUT_DIR, class_name)
        count = len(os.listdir(class_dir)) if os.path.isdir(class_dir) else 0
        print(f"  {class_name}: {count} dosya")


if __name__ == "__main__":
    main()
