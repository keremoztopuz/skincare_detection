"""Write the FFHQ-Aging glasses label onto the manifest.

Ingest read this column to drop dark glasses but never stored it, so every
later question about eyewear meant re-joining a 70k-row CSV. Frames sit
directly over the periorbital region — the one part of the face both the
wrinkle and the eye-bag call depend on — so the field is worth keeping.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import provenance as PV

LABELS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "downloads", "ffhq_aging_labels.csv")


def main() -> int:
    glasses = {}
    with open(LABELS, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            glasses[row["image_number"].zfill(5)] = row["glasses"]

    updates, missing = {}, 0
    for record in PV.load_manifest().values():
        if record.get("source") != "ffhq":
            continue
        value = glasses.get(str(record.get("source_record_id")).zfill(5))
        if value is None:
            missing += 1
            continue
        if record.get("glasses") != value:
            updates[record["id"]] = {"glasses": value}

    written = PV.update_records(updates)
    print(f"glasses yazildi: {written} kayit, {missing} eslesmedi")
    return 0


if __name__ == "__main__":
    sys.exit(main())
