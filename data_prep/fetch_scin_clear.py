"""Fetch the SCIN cases a dermatologist read as showing no pathology.

Every other class has negatives that come from somewhere else, and for Acne
and Eczema that is a problem with a number attached: their positives are
clinical close-ups where only a quarter contain a face, while a portrait pool
is 100% faces. Adding FFHQ portraits as the clean-skin negative would let
"is there a face in shot" answer the question instead of the skin — measured
as has_face AUC 0.567 today, and it would rise toward 0.85.

These images avoid that. They come from the same submission flow as the SCIN
positives — same phones, same lighting, same framing of an affected area —
so they differ from the positives in the skin and almost nothing else.

The label is explicit rather than inferred: `dermatologist_gradable_for_
skin_condition_N == YES_IMAGE_QUALITY_SUFFICIENT_NO_DISCERNIBLE_PATHOLOGY`
means someone qualified looked at a usable image and found nothing. An empty
weighted_skin_condition_label is NOT the same thing and is not used here —
1815 of those are simply "image quality insufficient".
"""

import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fetch_scin import BASE_URL, CSV_DIR, OUT_DIR, download

CLEAR_MARKER = "YES_IMAGE_QUALITY_SUFFICIENT_NO_DISCERNIBLE_PATHOLOGY"
CLASS_DIR = "Clear"
MAX_IMAGES_PER_CASE = 2


def clear_cases(labels_path):
    cases = []
    with open(labels_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            verdicts = [row.get(f"dermatologist_gradable_for_skin_condition_{i}", "")
                        for i in (1, 2, 3)]
            graded = [v for v in verdicts if v]
            # Every dermatologist who could grade it has to agree there is
            # nothing there. One "no pathology" beside a real diagnosis is a
            # disagreement, not a clean image.
            if graded and all(CLEAR_MARKER in v for v in graded):
                cases.append(row["case_id"])
    return cases


def main() -> int:
    labels_path = os.path.join(CSV_DIR, "scin_labels.csv")
    cases_path = os.path.join(CSV_DIR, "scin_cases.csv")
    for name, path in (("scin_labels.csv", labels_path), ("scin_cases.csv", cases_path)):
        if not os.path.exists(path):
            download(BASE_URL + "dataset/" + name, path)

    selected = set(clear_cases(labels_path))
    print(f"patolojisiz vaka: {len(selected)}")

    jobs = []
    destination_dir = os.path.join(OUT_DIR, CLASS_DIR)
    os.makedirs(destination_dir, exist_ok=True)
    with open(cases_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["case_id"] not in selected:
                continue
            paths = [row.get(f"image_{i}_path", "") for i in (1, 2, 3)]
            for index, path in enumerate([p for p in paths if p][:MAX_IMAGES_PER_CASE]):
                jobs.append((BASE_URL + path,
                             os.path.join(destination_dir,
                                          f"scin_{row['case_id']}_{index}.png")))

    print(f"{len(jobs)} goruntu indiriliyor...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda job: download(*job), jobs))
    print(f"tamam: {sum(results)}/{len(jobs)} basarili")
    return 0


if __name__ == "__main__":
    sys.exit(main())
