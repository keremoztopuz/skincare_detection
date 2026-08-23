"""Extract FFHQ faces as Healthy and Wrinkles candidates.

Both classes come from the same pool on purpose. The v3 dataset drew Healthy
from UTKFace and Wrinkles from a Roboflow export, so "which dataset is this
from" separated them perfectly — and the model read that instead of the skin.
When Healthy and Wrinkles share a source, capture characteristics cannot carry
class information at all, because they are drawn from one distribution.

Age is the other trap. Wrinkled faces are mostly older, so if Healthy is drawn
young the model learns age, not wrinkles — the same failure in a new coat. The
FFHQ-Aging labels give a real age group per image, so Healthy is stratified to
match the Wrinkles age histogram rather than being sampled from whatever is
convenient. Faces aged 50+ with clear skin are the ones that make the class
mean anything.

The age label decides who is a *candidate*. Whether wrinkles are actually
visible is a human call, made in the review sheet — an age threshold used as a
label would just rebuild the confound.

    python data_prep/ingest_ffhq.py --shards 00000,01000 --min-age 18
"""

import argparse
import collections
import csv
import hashlib
import io
import os
import sys
import tarfile
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image

import provenance as PV

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFHQ_DIR = os.path.join(ROOT, "data_prep", "downloads", "ffhq")
AGING_LABELS = os.path.join(ROOT, "data_prep", "downloads", "ffhq_aging_labels.csv")
STAGE_DIR = os.path.join(ROOT, "data_prep", "downloads", "ffhq_staged")

# Age groups as FFHQ-Aging spells them, ordered youngest first.
ADULT_GROUPS = ("20-29", "30-39", "40-49", "50-69", "70-120")
OLDER_GROUPS = ("50-69", "70-120")

# Dark glasses hide the periorbital region, which is exactly where both
# wrinkles and eye bags live. Normal glasses are kept: excluding every
# spectacle wearer would bias the classes by age all over again.
EXCLUDED_GLASSES = ("Dark",)

MAX_YAW = 35.0
MAX_PITCH = 30.0


def load_age_labels() -> Dict[int, Dict[str, str]]:
    if not os.path.exists(AGING_LABELS):
        raise SystemExit(f"yas etiketleri yok: {AGING_LABELS}")
    labels = {}
    with open(AGING_LABELS, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            labels[int(row["image_number"])] = row
    return labels


def eligible(row: Dict[str, str], min_age_group: str = "20-29") -> Tuple[bool, str]:
    if row["age_group"] not in ADULT_GROUPS:
        return False, "under_age"
    if row["glasses"] in EXCLUDED_GLASSES:
        return False, "dark_glasses"
    if row["left_eye_occluded"] == "1" or row["right_eye_occluded"] == "1":
        return False, "eye_occluded"
    try:
        if abs(float(row["head_yaw"])) > MAX_YAW or abs(float(row["head_pitch"])) > MAX_PITCH:
            return False, "extreme_pose"
    except (TypeError, ValueError):
        pass
    return True, ""


def band_of(age_group: str) -> str:
    return {"20-29": "20_29", "30-39": "30_39", "40-49": "40_49",
            "50-69": "50_69", "70-120": "70_plus"}.get(age_group, "unknown")


def extract(shards: List[str], per_shard: Optional[int] = None) -> List[Dict[str, object]]:
    """Pull eligible images out of the webdataset tars onto disk."""
    labels = load_age_labels()
    staged: List[Dict[str, object]] = []
    skipped = collections.Counter()

    for shard in shards:
        path = os.path.join(FFHQ_DIR, f"{shard}.tar")
        if not os.path.exists(path):
            print(f"  atlandi (yok): {path}")
            continue
        taken = 0
        with tarfile.open(path) as archive:
            for member in archive:
                if not member.isfile() or not member.name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    continue
                stem = os.path.splitext(os.path.basename(member.name))[0]
                try:
                    index = int(stem)
                except ValueError:
                    skipped["unparsable_name"] += 1
                    continue
                row = labels.get(index)
                if row is None:
                    skipped["no_age_label"] += 1
                    continue
                ok, reason = eligible(row)
                if not ok:
                    skipped[reason] += 1
                    continue

                payload = archive.extractfile(member)
                if payload is None:
                    continue
                data = payload.read()
                band = band_of(row["age_group"])
                # Candidates only. The Wrinkles/Healthy call is made by a human
                # in the review sheet; an age cutoff used as a label would be
                # the age confound wearing a different hat.
                bucket = "older" if row["age_group"] in OLDER_GROUPS else "adult"
                destination = os.path.join(STAGE_DIR, bucket, band, f"ffhq_{index:05d}.jpg")
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                if not os.path.exists(destination):
                    # The shards ship WebP. Writing JPEG here keeps the source
                    # format uniform across every class, so "which codec" is
                    # one less thing that could correlate with a label.
                    with Image.open(io.BytesIO(data)) as image:
                        image.convert("RGB").save(destination, "JPEG", quality=95)
                staged.append({
                    "path": destination, "index": index, "band": band,
                    "bucket": bucket, "gender": row["gender"],
                    "glasses": row["glasses"],
                })
                taken += 1
                if per_shard and taken >= per_shard:
                    break
        print(f"  {shard}.tar -> {taken} goruntu")

    for reason, count in skipped.most_common():
        print(f"  elenen {reason}: {count}")
    return staged


def write_manifest(staged: List[Dict[str, object]]) -> int:
    existing = PV.load_manifest()
    seen = {record["sha256_orig"] for record in existing.values()}
    records = []
    for item in staged:
        with open(item["path"], "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        with Image.open(item["path"]) as image:
            width, height, fmt = image.width, image.height, image.format
        records.append(PV.ImageRecord(
            id=PV.make_id(digest), sha256_orig=digest, source="ffhq",
            # Deliberately unlabelled until a human decides. Writing "Healthy"
            # here because the subject is young is the confound this module
            # exists to avoid.
            label="UNLABELLED",
            source_record_id=f"{item['index']:05d}",
            local_path=os.path.relpath(item["path"], ROOT),
            bytes=os.path.getsize(item["path"]),
            orig_width=width, orig_height=height, orig_format=fmt,
            age_band=item["band"], age_source="ffhq_aging",
            label_source="pending_review",
            status="pending",
        ))
    written = PV.append_records(records)
    print(f"manifest: {written} yeni FFHQ kaydi")
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", default="00000,01000,02000,03000")
    parser.add_argument("--per-shard", type=int)
    parser.add_argument("--no-manifest", action="store_true")
    arguments = parser.parse_args()

    shards = [s.strip() for s in arguments.shards.split(",") if s.strip()]
    staged = extract(shards, arguments.per_shard)

    bands = collections.Counter(item["band"] for item in staged)
    buckets = collections.Counter(item["bucket"] for item in staged)
    print(f"\ntoplam {len(staged)} goruntu")
    print("  yas bandi:", dict(sorted(bands.items())))
    print("  kova:", dict(buckets))

    if not arguments.no_manifest:
        write_manifest(staged)
    return 0


if __name__ == "__main__":
    sys.exit(main())
