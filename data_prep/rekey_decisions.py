"""Re-key review decisions from the canonical hash to the manifest id.

The browser sheet keys its state by a hash of the canonical file, because
that is all it has in the page. But canonicalization is not frozen: fixing a
crop or a padding rule rewrites every canonical file, changes every hash, and
orphans every decision keyed to one. Human review is the most expensive thing
in this pipeline and it must survive a rebuild.

The manifest id is a hash of the *original* bytes, so it is stable across any
change to how those bytes are processed. This rewrites past decisions onto
that key, keeping the old id in `review_id` so the trail stays followable.
Append-only: nothing is deleted, and load_decisions takes the latest per id.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import provenance as PV


def main() -> int:
    manifest = PV.load_manifest()
    by_review_id = {}
    for record in manifest.values():
        digest = record.get("canonical_sha256")
        if digest:
            by_review_id[PV.make_id(digest)] = record["id"]

    rows = list(PV.iter_records(PV.DECISIONS_PATH))
    rekeyed, already, orphan = [], 0, 0
    for row in rows:
        if row["id"] in manifest:
            already += 1
            continue
        target = by_review_id.get(row["id"])
        if target is None:
            orphan += 1
            continue
        revised = dict(row)
        revised["review_id"] = row["id"]
        revised["id"] = target
        rekeyed.append(revised)

    if rekeyed:
        with open(PV.DECISIONS_PATH, "a", encoding="utf-8") as handle:
            for row in rekeyed:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    print(f"yeniden anahtarlandi {len(rekeyed)}, zaten manifest id {already}, "
          f"eslesmeyen {orphan}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
