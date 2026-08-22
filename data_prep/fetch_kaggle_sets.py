"""Download the DermNet Kaggle mirror and extract Acne and Eczema folders.

Requires ~/.kaggle/kaggle.json (Kaggle API credentials). The full archive is
about 1.7 GB; it is cached under data_prep/downloads/ so reruns skip the
download. Only the folders that map onto the project's classes are extracted,
mirroring the original data_prep mapping:
  "Acne and Rosacea Photos"  -> Acne
  "Eczema Photos"            -> Eczema

Output: data_prep/downloads/dermnet/{Acne,Eczema}/
"""

import base64
import json
import os
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOADS = os.path.join(ROOT, "data_prep", "downloads")
ZIP_PATH = os.path.join(DOWNLOADS, "dermnet.zip")
OUT_DIR = os.path.join(DOWNLOADS, "dermnet")

DATASET = "shubhamgoel27/dermnet"
FOLDER_MAP = {
    "Acne and Rosacea Photos": "Acne",
    "Eczema Photos": "Eczema",
}
MAX_PER_CLASS = 600


def kaggle_auth_header():
    credentials_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(credentials_path):
        return None
    with open(credentials_path) as credentials_file:
        credentials = json.load(credentials_file)
    token = f"{credentials['username']}:{credentials['key']}".encode()
    return "Basic " + base64.b64encode(token).decode()


def download_archive(auth_header):
    if os.path.exists(ZIP_PATH) and os.path.getsize(ZIP_PATH) > 100_000_000:
        print("arsiv zaten indirilmis, atlaniyor")
        return True
    url = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}"
    request = urllib.request.Request(url, headers={"Authorization": auth_header})
    print(f"{DATASET} indiriliyor (~1.7 GB, birkac dakika surebilir)...")
    try:
        with urllib.request.urlopen(request) as response, open(ZIP_PATH, "wb") as archive:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                archive.write(chunk)
        return True
    except Exception as error:
        print(f"indirme basarisiz: {error}")
        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)
        return False


def extract_classes():
    counts = {}
    with zipfile.ZipFile(ZIP_PATH) as archive:
        for source_folder, class_name in FOLDER_MAP.items():
            class_dir = os.path.join(OUT_DIR, class_name)
            os.makedirs(class_dir, exist_ok=True)
            members = sorted(
                name for name in archive.namelist()
                if name.startswith(f"train/{source_folder}/")
                and name.lower().endswith((".jpg", ".jpeg", ".png"))
            )[:MAX_PER_CLASS]
            for member in members:
                dest = os.path.join(class_dir, os.path.basename(member))
                if os.path.exists(dest):
                    continue
                with archive.open(member) as source, open(dest, "wb") as target:
                    target.write(source.read())
            counts[class_name] = len(os.listdir(class_dir))
    return counts


def main():
    os.makedirs(DOWNLOADS, exist_ok=True)
    auth_header = kaggle_auth_header()
    if auth_header is None:
        print("~/.kaggle/kaggle.json bulunamadi; DermNet takviyesi atlaniyor.")
        return
    if not download_archive(auth_header):
        return
    counts = extract_classes()
    for class_name, count in counts.items():
        print(f"  {class_name}: {count} dosya")


if __name__ == "__main__":
    main()
