"""Split-leakage audit that actually catches augmented copies.

The old audit (src/dataset.py:audit_split_leakage) compared SHA-256 of raw
bytes and reported TEMIZ on a dataset where 42 of 90 Eye_Bags test images and
34 of 90 Wrinkles test images were Roboflow augmentations of training images.
Byte hashing cannot see a re-encode, a rotation or a flip, so the test metrics
it blessed were inflated: honest test Top-1 was 94.0%, not the 95.0% reported,
and Wrinkles scored 1.000 on the leaked subset against 0.911 on the clean one.

Four checks, each of which the byte hash misses:

  bytes      identical files, as before. Cheap, keep it.
  stem       Roboflow names an augmented copy <stem>_jpg.rf.<md5>.jpg, so the
             derivation is written on the file. This alone catches the 42+34.
  phash      perceptual hash within a Hamming radius, compared against the
             mirror too, since the augmentations include horizontal flips that
             plain pHash does not match.
  group      whatever the build assigned as group_id, which already unions the
             above plus identity clustering when available.

Any hit is a failure. A leak is not a warning: it silently invalidates every
number downstream of it.
"""

import argparse
import collections
import glob
import hashlib
import json
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASH_RADIUS = 8
SPLITS = ("train", "val", "test")

DERIVATION_PATTERNS = (
    re.compile(r"^(?:rf3_)?(?P<stem>.+?)_(?:jpg|png)\.rf\.[0-9a-f]{32}\.(?:jpg|jpeg|png)$", re.I),
    re.compile(r"^(?P<stem>.+?)[-_](?:aug|rot|flip|copy)\d*\.(?:jpg|jpeg|png)$", re.I),
)


class LeakageError(RuntimeError):
    pass


def derivation_stem(name: str) -> Optional[str]:
    for pattern in DERIVATION_PATTERNS:
        match = pattern.match(name)
        if match:
            return match.group("stem")
    return None


def _sha256(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _phash_pair(path: str) -> Optional[Tuple[int, int]]:
    import imagehash
    from PIL import Image
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            straight = imagehash.phash(rgb, hash_size=8)
            mirrored = imagehash.phash(rgb.transpose(Image.FLIP_LEFT_RIGHT), hash_size=8)
    except Exception:
        return None
    to_int = lambda h: int("".join("1" if b else "0" for b in h.hash.flatten()), 2)
    return to_int(straight), to_int(mirrored)


def collect(data_dir: str) -> List[Dict[str, str]]:
    rows = []
    for split in SPLITS:
        for path in sorted(glob.glob(os.path.join(data_dir, split, "*", "*"))):
            if path.lower().endswith((".jpg", ".jpeg", ".png")):
                rows.append({
                    "path": path, "split": split,
                    "label": os.path.basename(os.path.dirname(path)),
                    "name": os.path.basename(path),
                })
    return rows


def _report(kind: str, groups: Dict[str, List[Dict[str, str]]]) -> List[Dict[str, object]]:
    leaks = []
    for key, members in groups.items():
        splits = {m["split"] for m in members}
        if len(splits) > 1:
            leaks.append({
                "kind": kind, "key": str(key), "splits": sorted(splits),
                "paths": [os.path.relpath(m["path"], ROOT) for m in members[:6]],
                "count": len(members),
            })
    return leaks


def audit(data_dir: str, manifest_path: Optional[str] = None) -> List[Dict[str, object]]:
    rows = collect(data_dir)
    if not rows:
        raise SystemExit(f"goruntu bulunamadi: {data_dir}")
    leaks: List[Dict[str, object]] = []

    by_bytes = collections.defaultdict(list)
    for row in rows:
        by_bytes[_sha256(row["path"])].append(row)
    leaks += _report("bytes", by_bytes)

    by_stem = collections.defaultdict(list)
    for row in rows:
        stem = derivation_stem(row["name"])
        if stem:
            by_stem[(row["label"], stem)].append(row)
    leaks += _report("derivation_stem", by_stem)

    if manifest_path and os.path.exists(manifest_path):
        # The build's group_id already unions stems, perceptual duplicates and
        # identity clusters, so when a manifest is available this is the
        # strongest of the four.
        group_of = {}
        with open(manifest_path, encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("group_id"):
                    group_of[record["id"]] = record["group_id"]
        by_group = collections.defaultdict(list)
        for row in rows:
            record_id = os.path.splitext(row["name"])[0]
            if record_id in group_of:
                by_group[group_of[record_id]].append(row)
        leaks += _report("group_id", by_group)

    hashes = {}
    for row in rows:
        pair = _phash_pair(row["path"])
        if pair is not None:
            hashes[row["path"]] = pair
    paths = list(hashes.keys())
    index = {row["path"]: row for row in rows}
    seen_pairs = set()
    for i in range(len(paths)):
        a_straight, a_mirror = hashes[paths[i]]
        for j in range(i + 1, len(paths)):
            b_straight, _ = hashes[paths[j]]
            close = (bin(a_straight ^ b_straight).count("1") <= PHASH_RADIUS
                     or bin(a_mirror ^ b_straight).count("1") <= PHASH_RADIUS)
            if not close:
                continue
            left, right = index[paths[i]], index[paths[j]]
            if left["split"] == right["split"]:
                continue
            key = tuple(sorted((paths[i], paths[j])))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            leaks.append({
                "kind": "phash", "key": "", "count": 2,
                "splits": sorted({left["split"], right["split"]}),
                "paths": [os.path.relpath(p, ROOT) for p in key],
            })
    return leaks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="orchestration_data_v2")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--report", default="outputs/audits/leakage_report.json")
    arguments = parser.parse_args()

    manifest = arguments.manifest or os.path.join(arguments.data, "manifest.jsonl")
    leaks = audit(arguments.data, manifest)

    os.makedirs(os.path.dirname(arguments.report) or ".", exist_ok=True)
    with open(arguments.report, "w", encoding="utf-8") as handle:
        json.dump({"data": arguments.data, "leaks": leaks}, handle, indent=2, ensure_ascii=False)

    if not leaks:
        print("sizinti denetimi TEMIZ")
        return 0

    by_kind = collections.Counter(leak["kind"] for leak in leaks)
    print(f"SIZINTI: {len(leaks)} grup")
    for kind, count in by_kind.most_common():
        print(f"   {kind:<18} {count}")
    for leak in leaks[:5]:
        print(f"   ornek [{leak['kind']}] {leak['splits']}: {leak['paths'][:2]}")
    print(f"\nrapor: {arguments.report}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
