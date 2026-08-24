"""Turn contact-sheet verdicts into decision records.

The sheets are judged in bulk by cell number, so this maps (sheet, cell) back
to the manifest id through the index the sheet builder wrote, and appends one
decision per judged face. The reviewer field carries who decided: these are
model judgements, not a person's, and the distinction has to survive into the
audit trail so an agreement check can be run against them later.
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import provenance as PV

HERE = os.path.dirname(os.path.abspath(__file__))
VALID = {"eyebags", "clean", "skip"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=os.path.join(HERE, "claude_labels"))
    parser.add_argument("--index", default=os.path.join(HERE, "sheets", "index.json"))
    parser.add_argument("--job", default="ffhq-eyebags")
    parser.add_argument("--reviewer", default="claude")
    arguments = parser.parse_args()

    with open(arguments.index, encoding="utf-8") as handle:
        index = {sheet["sheet"]: sheet for sheet in json.load(handle)}

    rows, unknown = [], 0
    stamp = PV.utcnow()
    for path in sorted(glob.glob(os.path.join(arguments.labels, "*.json"))):
        with open(path, encoding="utf-8") as handle:
            batch = json.load(handle)
        for sheet_number, cells in batch.items():
            sheet = index.get(int(sheet_number))
            if sheet is None:
                unknown += len(cells)
                continue
            for cell, verdict in cells.items():
                if verdict not in VALID:
                    raise ValueError(f"gecersiz karar: {verdict}")
                position = int(cell) - 1
                if position >= len(sheet["cells"]):
                    unknown += 1
                    continue
                rows.append({
                    "id": sheet["cells"][position]["manifest_id"],
                    "decision": verdict,
                    "reviewer": arguments.reviewer,
                    "reviewed_at": stamp,
                    "tool_version": "sheets-1.0.0",
                    "job": arguments.job,
                })

    with open(PV.DECISIONS_PATH, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(f"{len(rows)} karar yazildi ({unknown} eslesmedi), reviewer={arguments.reviewer}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
