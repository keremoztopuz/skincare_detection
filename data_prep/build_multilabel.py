"""Derive per-condition label vectors, with unknown as a first-class state.

A face can have wrinkles and eye bags at once, and the folder-per-class layout
could not say so: it forced one label per image and dropped whatever else was
true. Measured on the FFHQ pool, 30% of the faces labelled Healthy had visible
eye bags, and enforcing a single label would have cut Healthy and Wrinkles
from 345 to 190 each just to keep the classes pure.

So each image carries a vector over CONDITIONS with three states:

    1     the condition is present
    0     the condition is absent
    None  nobody has looked

None is the point. A DermNet acne photo of a forearm says nothing about that
person's wrinkles, and scoring it as "no wrinkles" would train the Wrinkles
head on a false negative. The loss masks unknown entries instead of guessing,
so a partially labelled image still contributes everything it does know.
"""

import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import provenance as PV

CONDITIONS = ("Acne", "Eczema", "Eye_Bags", "Wrinkles")

# A diagnosis of one skin condition rules out the other, but says nothing
# about the periorbital conditions — and these images are mostly not faces.
DIAGNOSED = {
    "Acne": {"Acne": 1, "Eczema": 0},
    "Eczema": {"Eczema": 1, "Acne": 0},
    # Read as showing no pathology by every dermatologist who could grade it.
    # Periorbital conditions stay unknown: these are close-ups of an arm or a
    # hand, and nothing in them speaks to eye bags or wrinkles.
    "Clear": {"Acne": 0, "Eczema": 0},
}

DECISION_TO_VALUE = {
    "ffhq": {"wrinkles": ("Wrinkles", 1), "healthy": ("Wrinkles", 0)},
    "ffhq-eyebags": {"eyebags": ("Eye_Bags", 1), "clean": ("Eye_Bags", 0)},
}

# Retired because the class was measured, not assumed, to be unusable: of 36
# random samples about 15-20% actually showed eye bags. The rest were clean
# young faces, dark circles (a different condition), single-eye beauty stock
# shots, crops with no eye in frame, a "Before/After EES Serum" advert with
# burned-in text, and an image with white arrows drawn onto the eye bags. It
# is a web search dump for the phrase, and it explains the false alarms on
# clean young faces directly: the positive class was full of them.
RETIRED = {("roboflow_v3", "Eye_Bags"): "source_retired:label_noise"}


def decisions_by_job():
    jobs = collections.defaultdict(dict)
    for record in PV.iter_records(PV.DECISIONS_PATH):
        job = record.get("job")
        if job in DECISION_TO_VALUE:
            jobs[job][record["id"]] = record["decision"]
    return jobs


def main() -> int:
    manifest = PV.load_manifest()
    jobs = decisions_by_job()

    updates = {}
    known = collections.Counter()
    positive = collections.Counter()
    retired = collections.Counter()

    for record in manifest.values():
        key = (record.get("source"), record.get("label"))
        if key in RETIRED:
            if record["status"] == "canonical":
                updates[record["id"]] = {"status": "held_out",
                                         "reject_reason": RETIRED[key]}
                retired[RETIRED[key]] += 1
            continue
        if record["status"] not in ("canonical", "held_out"):
            continue

        conditions = {name: None for name in CONDITIONS}
        conditions.update(DIAGNOSED.get(record.get("label"), {}))
        for job, decisions in jobs.items():
            verdict = decisions.get(record["id"])
            mapping = DECISION_TO_VALUE[job].get(verdict) if verdict else None
            if mapping:
                conditions[mapping[0]] = mapping[1]

        if all(value is None for value in conditions.values()):
            continue
        updates[record["id"]] = {"conditions": conditions}
        for name, value in conditions.items():
            if value is not None:
                known[name] += 1
                positive[name] += value

    PV.update_records(updates)
    print(f"{'kosul':<10}{'bilinen':>9}{'pozitif':>9}{'negatif':>9}")
    for name in CONDITIONS:
        print(f"{name:<10}{known[name]:>9}{positive[name]:>9}"
              f"{known[name]-positive[name]:>9}")
    for reason, count in retired.most_common():
        print(f"emekli {reason}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
