"""Turn the human FFHQ review into Wrinkles / Healthy labels, age-matched.

Two things happen here and they are separate on purpose.

Labelling: the reviewer judged visible wrinkles, not age. The decisions land
on the manifest as labels with label_source="human_review". Nothing infers a
label from the age band — that is the confound this whole pool exists to
remove.

Age matching: wrinkles and age really are correlated, so a straight import
would hand the model "old face" as a shortcut for Wrinkles. Within every age
band we keep min(wrinkles, healthy) of each, so the two classes end up with
an identical age histogram and the band carries no information about the
label. The surplus is held out, not deleted; the manifest keeps it findable.

Also retires the two sources the shortcut audit ruled out:
  utkface_aligned/Healthy   every file 200x200; upsampling to 448 leaves a
                            blur signature that separates the class perfectly
  roboflow_v3/Wrinkles      not faces at all — extreme close-up skin texture,
                            0% face detection, lapvar median 10.5 vs 407 on
                            DermNet. The app sends face crops; this modality
                            does not transfer.
"""

import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import provenance as PV

SEED = 42
DECISION_TO_LABEL = {"wrinkles": "Wrinkles", "healthy": "Healthy"}
RETIRED_SOURCES = {
    ("utkface_aligned", "Healthy"): "source_retired:resolution_shortcut",
    ("roboflow_v3", "Wrinkles"): "source_retired:not_faces",
}


def load_ffhq_decisions() -> dict:
    """Latest FFHQ decision per review id."""
    latest = {}
    for record in PV.iter_records(PV.DECISIONS_PATH):
        if record.get("job") == "ffhq":
            latest[record["id"]] = record["decision"]
    return latest


def main() -> int:
    manifest = PV.load_manifest()
    decisions = load_ffhq_decisions()

    # The review sheet keys state by a hash of the canonical file, while the
    # manifest is keyed by a hash of the original. Both are already recorded,
    # so the join is a lookup rather than a re-hash of 2725 files.
    # held_out is a reserve, not a verdict: if dedup later drops a matched
    # image its partner would be left unbalanced, so a rerun has to be able
    # to draw a replacement back out. Only "rejected" is final.
    by_review_id = {}
    for record in manifest.values():
        if record.get("source") == "ffhq" and record["status"] in ("canonical", "held_out"):
            digest = record.get("canonical_sha256")
            if digest:
                by_review_id[PV.make_id(digest)] = record

    unknown = [i for i in decisions if i not in by_review_id]
    if unknown:
        print(f"UYARI: {len(unknown)} karar hicbir kanonik dosyayla eslesmedi")

    # Bucket by (age band, decision) before matching, so the trim is per band.
    buckets = collections.defaultdict(lambda: collections.defaultdict(list))
    judged = set()
    for review_id, decision in decisions.items():
        record = by_review_id.get(review_id)
        if record is None:
            continue
        judged.add(record["id"])
        if decision in DECISION_TO_LABEL:
            buckets[record.get("age_band") or "?"][decision].append(record["id"])

    rng = random.Random(SEED)
    updates, kept = {}, collections.Counter()
    print(f"{'band':<10}{'wrink':>7}{'healthy':>9}{'esit':>7}{'fazla':>7}")
    for band in sorted(buckets, key=lambda b: (b == "?", b)):
        pair = buckets[band]
        wrinkles, healthy = sorted(pair["wrinkles"]), sorted(pair["healthy"])
        rng.shuffle(wrinkles)
        rng.shuffle(healthy)
        take = min(len(wrinkles), len(healthy))
        surplus = 0
        for decision, ids in (("wrinkles", wrinkles), ("healthy", healthy)):
            label = DECISION_TO_LABEL[decision]
            for index, image_id in enumerate(ids):
                if index < take:
                    updates[image_id] = {
                        "status": "canonical",
                        "label": label,
                        "label_source": "human_review",
                        "reject_reason": None,
                        "review_id": PV.make_id(manifest[image_id]["canonical_sha256"]),
                    }
                    kept[label] += 1
                else:
                    updates[image_id] = {
                        "status": "held_out",
                        "reject_reason": f"age_match_surplus:{label}",
                    }
                    surplus += 1
        print(f"{band:<10}{len(wrinkles):>7}{len(healthy):>9}{take:>7}{surplus:>7}")

    # Skipped and never-reached images stay in the manifest as a labelled
    # reserve. "Not judged" is not "healthy" — that is why the sheet has a
    # skip key at all, and holding them out is the same principle at the
    # dataset level.
    reserve = collections.Counter()
    for record in by_review_id.values():
        if record["id"] in updates:
            continue
        reason = "review_skip" if record["id"] in judged else "unreviewed"
        updates[record["id"]] = {"status": "held_out", "reject_reason": reason}
        reserve[reason] += 1

    retired = collections.Counter()
    for record in manifest.values():
        key = (record.get("source"), record.get("label"))
        if key in RETIRED_SOURCES and record["status"] == "canonical":
            updates[record["id"]] = {
                "status": "held_out", "reject_reason": RETIRED_SOURCES[key],
            }
            retired[RETIRED_SOURCES[key]] += 1

    PV.update_records(updates)
    print()
    for label, count in sorted(kept.items()):
        print(f"etiketlendi {label}: {count}")
    for reason, count in reserve.most_common():
        print(f"beklemede {reason}: {count}")
    for reason, count in retired.most_common():
        print(f"emekli {reason}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
