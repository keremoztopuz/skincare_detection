"""Build the canonicalized dataset. Replaces clean_and_split.py.

Pipeline, in order:

  ingest      walk the local pools, apply filename screening, write manifest
              records. Nothing is copied yet.
  canonicalize decode -> overlays -> face -> randomized crop -> resolution
              jitter -> 448/q92. Resumable: an id already canonicalized is
              skipped.
  dedup       perceptual hash with a Hamming radius, mirror-aware, across all
              classes at once. The old pipeline compared exact dhash strings
              within a class, which is why 42/90 Eye_Bags test images turned
              out to be augmented copies of training images.
  split       cut on group ids, never on files, stratified by class, source
              and sharpness quintile so no split is source- or sharpness-
              skewed.

Every stage appends to the manifest rather than editing it, so the log stays
an audit trail and a rebuild replays human decisions instead of asking again.
"""

import argparse
import collections
import glob
import json
import hashlib
import os
import random
import sys
from typing import Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image

import body_region as BR
import canonicalize as C
import provenance as PV

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOADS = os.path.join(ROOT, "data_prep", "downloads")
OUT_DIR = os.path.join(ROOT, "orchestration_data_v2")
CANON_DIR = os.path.join(ROOT, "data_prep", "canonical")

SPLITS = (("train", 0.70), ("val", 0.15), ("test", 0.15))
PHASH_RADIUS = 8
SEED = 42

# Where each class may come from. Kept explicit so adding a source is a
# visible edit rather than a directory appearing on disk.
POOLS: List[Tuple[str, str, str]] = [
    ("dermnet", "Acne", os.path.join(DOWNLOADS, "dermnet", "Acne")),
    ("dermnet", "Eczema", os.path.join(DOWNLOADS, "dermnet", "Eczema")),
    ("scin", "Acne", os.path.join(DOWNLOADS, "scin", "Acne")),
    ("scin", "Eczema", os.path.join(DOWNLOADS, "scin", "Eczema")),
    ("roboflow_v3", "Eye_Bags", os.path.join(DOWNLOADS, "staged", "Eye_Bags")),
    ("roboflow_v3", "Wrinkles", os.path.join(DOWNLOADS, "staged", "Wrinkles")),
    ("utkface_aligned", "Healthy", os.path.join(DOWNLOADS, "staged", "Healthy")),
    # Dermatologist-read clean skin from the same submission flow as the SCIN
    # positives. The only clean-skin negative Acne and Eczema can have without
    # letting "is there a face in shot" stand in for the diagnosis.
    ("scin", "Clear", os.path.join(DOWNLOADS, "scin", "Clear")),
]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def sha256_of(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def roboflow_stem(name: str) -> Optional[str]:
    """Source stem behind a Roboflow augmented copy, if this is one."""
    import re
    match = re.match(r"(?:rf3_)?(.+?)_(?:jpg|png)\.rf\.[0-9a-f]{32}\.(?:jpg|png)$", name)
    return match.group(1) if match else None


def stage_ingest() -> int:
    """Walk the pools and write one manifest record per unique image."""
    existing = PV.load_manifest()
    seen_hashes = {record["sha256_orig"] for record in existing.values()}
    decisions = PV.load_decisions()

    records, skipped, rejected = [], 0, collections.Counter()
    for source, label, directory in POOLS:
        if not os.path.isdir(directory):
            print(f"  atlandi (yok): {directory}")
            continue
        for path in sorted(glob.glob(os.path.join(directory, "*"))):
            if not path.lower().endswith(IMAGE_EXTENSIONS):
                continue
            digest = sha256_of(path)
            if digest in seen_hashes:
                skipped += 1
                continue
            seen_hashes.add(digest)

            record_id = PV.make_id(digest)
            verdict, detail = BR.classify_filename(path)
            status, reason = "pending", None
            if verdict in (BR.INTIMATE, BR.NON_PHOTOGRAPH, BR.MISLABELLED):
                status, reason = "rejected", f"{verdict}:{detail}"
                rejected[verdict] += 1
            elif decisions.get(record_id, {}).get("decision") == "reject":
                status, reason = "rejected", "human_review"
                rejected["human_review"] += 1

            try:
                with Image.open(path) as image:
                    width, height, fmt = image.width, image.height, image.format
            except Exception:
                width = height = None
                fmt = None

            record = PV.ImageRecord(
                id=record_id, sha256_orig=digest, source=source, label=label,
                source_record_id=roboflow_stem(os.path.basename(path)),
                local_path=os.path.relpath(path, ROOT),
                bytes=os.path.getsize(path),
                orig_width=width, orig_height=height, orig_format=fmt,
                status=status, reject_reason=reason,
            )
            records.append(record)

    written = PV.append_records(records)
    print(f"ingest: {written} yeni kayit, {skipped} zaten vardi")
    for reason, count in rejected.most_common():
        print(f"   red {reason}: {count}")
    return written


# Rejections that describe the source file itself, not how it was processed.
# Re-running canonicalization cannot change these, and a human screen is far
# too expensive to discard.
STICKY_REJECTIONS = ("intimate", "non_photograph", "mislabelled", "human_review")


def stage_reset() -> int:
    """Send processed records back to pending so a rule change can re-run.

    Canonicalization is not frozen — a crop or padding rule can change — and
    every derived field goes stale with it. Labels do not: they describe the
    face, not the processing, so they stay. So does anything rejected by
    filename or by a person.
    """
    updates = {}
    for record in PV.load_manifest().values():
        reason = record.get("reject_reason") or ""
        if reason.startswith(STICKY_REJECTIONS):
            continue
        if record["status"] == "pending" and not record.get("canonical_path"):
            continue
        updates[record["id"]] = {
            "status": "pending", "reject_reason": None,
            "canonical_path": None, "canonical_sha256": None,
            "face": None, "crop": None, "quality": None, "overlay": None,
            "group_id": None, "split": None, "split_scheme": None,
        }
    PV.update_records(updates)
    print(f"reset: {len(updates)} kayit yeniden islenecek")
    return len(updates)


def stage_canonicalize(limit: Optional[int] = None) -> int:
    manifest = PV.load_manifest()
    todo = [r for r in manifest.values()
            if r["status"] == "pending" and not r.get("canonical_path")]
    if limit:
        todo = todo[:limit]
    print(f"canonicalize: {len(todo)} goruntu")

    updates, rejected = {}, collections.Counter()
    for index, record in enumerate(todo, 1):
        path = os.path.join(ROOT, record["local_path"])
        result = C.canonicalize(path, record["id"])
        if result.status != "kept":
            rejected[result.reject_reason] += 1
            updates[record["id"]] = {
                "status": "rejected", "reject_reason": result.reject_reason,
                "face": result.face, "quality": result.quality, "overlay": result.overlay,
            }
            continue
        destination = os.path.join(CANON_DIR, record["label"], f"{record['id']}.jpg")
        canonical_sha = C.save(result.image, destination)
        updates[record["id"]] = {
            "status": "canonical",
            "canonical_path": os.path.relpath(destination, ROOT),
            "canonical_sha256": canonical_sha,
            "face": result.face, "crop": result.crop,
            "quality": result.quality, "overlay": result.overlay,
        }
        if index % 250 == 0:
            PV.update_records(updates); updates = {}
            print(f"   {index}/{len(todo)}", flush=True)

    PV.update_records(updates)
    for reason, count in rejected.most_common():
        print(f"   red {reason}: {count}")
    kept = len(todo) - sum(rejected.values())
    print(f"canonicalize: {kept} kabul, {sum(rejected.values())} red")
    return kept


def _phash(path: str) -> Optional[Tuple[int, int]]:
    import imagehash
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            straight = imagehash.phash(rgb, hash_size=8)
            mirrored = imagehash.phash(rgb.transpose(Image.FLIP_LEFT_RIGHT), hash_size=8)
    except Exception:
        return None
    to_int = lambda h: int("".join("1" if b else "0" for b in h.hash.flatten()), 2)
    return to_int(straight), to_int(mirrored)


def stage_dedup() -> int:
    """Group duplicates across every class at once, mirror-aware."""
    manifest = PV.load_manifest()
    rows = [r for r in manifest.values() if r["status"] == "canonical"]
    print(f"dedup: {len(rows)} goruntu")

    hashes = {}
    for record in rows:
        result = _phash(os.path.join(ROOT, record["canonical_path"]))
        if result is not None:
            hashes[record["id"]] = result

    parent = {record["id"]: record["id"] for record in rows}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Deterministic keys first: a Roboflow stem is ground truth about
    # derivation and costs nothing to use.
    by_stem = collections.defaultdict(list)
    for record in rows:
        if record.get("source_record_id"):
            by_stem[(record["source"], record["source_record_id"])].append(record["id"])
    for members in by_stem.values():
        for other in members[1:]:
            union(members[0], other)

    # Then perceptual, with a radius. The mirror term is required: Roboflow
    # augmentations include horizontal flips, which plain pHash does not match
    # — that is precisely how the old audit missed 42/90 Eye_Bags test images.
    ids = list(hashes.keys())
    for i in range(len(ids)):
        hash_a, mirror_a = hashes[ids[i]]
        for j in range(i + 1, len(ids)):
            hash_b, mirror_b = hashes[ids[j]]
            # Both mirror directions. Comparing only mirror(a) against b
            # makes the match order-dependent: a pair can sit at 8 bits one
            # way and 10 the other, so whether a flipped duplicate is caught
            # depended on which image the loop reached first.
            if min(bin(hash_a ^ hash_b).count("1"),
                   bin(mirror_a ^ hash_b).count("1"),
                   bin(hash_a ^ mirror_b).count("1")) <= PHASH_RADIUS:
                union(ids[i], ids[j])

    groups = collections.defaultdict(list)
    for record in rows:
        groups[find(record["id"])].append(record)

    updates = {}
    cross_class = 0
    for group_id, members in groups.items():
        labels = {m["label"] for m in members}
        if len(labels) > 1:
            # The same picture under two labels is a labelling error. Dropping
            # the whole group is the safe reading; logging it makes the pair
            # findable.
            cross_class += 1
            for member in members:
                updates[member["id"]] = {
                    "group_id": group_id, "status": "rejected",
                    "reject_reason": f"cross_class_conflict:{'|'.join(sorted(labels))}",
                }
            continue
        for member in members:
            updates[member["id"]] = {"group_id": group_id}
    PV.update_records(updates)

    sizes = collections.Counter(len(m) for m in groups.values())
    print(f"dedup: {len(groups)} grup ({len(rows)} goruntuden), "
          f"tekil {sizes.get(1,0)}, en buyuk grup {max(sizes) if sizes else 0}")
    print(f"   siniflar arasi cakisma: {cross_class} grup atildi")
    return len(groups)


def group_source(group: List[dict]) -> str:
    """The source a group belongs to. Groups are near-duplicates, so the
    representative's source describes the whole group."""
    return min(group, key=lambda item: item["id"])["source"]


def source_quotas(by_class: Dict[str, List[dict]]) -> Dict[str, Dict[str, int]]:
    """Per-source group budgets so classes sharing a pool draw the same mix.

    Acne and Eczema both come from DermNet and SCIN, but in different
    proportions: 81/19 against 32/68. That difference is a shortcut on its
    own — the source, not the skin, tells the two apart, and the audit
    measured it as Eczema/bytes_per_pixel AUC 0.728 while the same feature
    separated Eczema's own two sources at 0.746. Capping every class in a
    family at the smallest per-source supply makes the mix identical, so
    source carries no information about the label.

    Classes that do not share their pool with another class are left alone;
    there is nothing to equalise against.
    """
    available: Dict[str, collections.Counter] = {}
    for label, records in by_class.items():
        groups = collections.defaultdict(list)
        for record in records:
            groups[record["group_id"]].append(record)
        available[label] = collections.Counter(
            group_source(group) for group in groups.values())

    families = collections.defaultdict(list)
    for label, counter in available.items():
        families[frozenset(counter)].append(label)

    quotas: Dict[str, Dict[str, int]] = {}
    for sources, labels in families.items():
        if len(labels) < 2:
            continue
        quota = {source: min(available[label][source] for label in labels)
                 for source in sources}
        for label in labels:
            quotas[label] = quota
    return quotas


def apply_source_quota(group_list: List[List[dict]],
                       quota: Optional[Dict[str, int]]) -> List[List[dict]]:
    """Keep at most quota[source] groups per source, order preserved."""
    if not quota:
        return group_list
    taken = collections.Counter()
    kept = []
    for group in group_list:
        source = group_source(group)
        if taken[source] >= quota.get(source, 0):
            continue
        taken[source] += 1
        kept.append(group)
    return kept


def stage_split(target_per_class: int) -> Dict[str, int]:
    """Group-disjoint split for multi-label data.

    There is no longer one class per image, so there is nothing to stratify a
    class on. What has to stay even across the splits is the combination that
    would otherwise let a split differ systematically: which source the image
    came from, which conditions are known about it, which of those are
    positive, and how sharp it is. Groups, never files, are what get cut, so a
    near-duplicate cannot straddle the boundary.
    """
    manifest = PV.load_manifest()
    rows = [r for r in manifest.values()
            if r["status"] == "canonical" and r.get("group_id")
            and r.get("conditions")]
    if not rows:
        print("split: conditions yazilmis kayit yok — once build_multilabel.py")
        return collections.Counter()

    values = [r["quality"]["lapvar"] for r in rows if r.get("quality")]
    edges = np.percentile(values, [20, 40, 60, 80]) if values else []

    def quintile(record) -> int:
        value = (record.get("quality") or {}).get("lapvar", 0.0)
        return int(np.searchsorted(edges, value)) if len(edges) else 0

    def stratum(record) -> Tuple:
        conditions = record["conditions"]
        known = tuple(sorted(k for k, v in conditions.items() if v is not None))
        positive = tuple(sorted(k for k, v in conditions.items() if v == 1))
        return (record["source"], known, positive, quintile(record))

    groups = collections.defaultdict(list)
    for record in rows:
        groups[record["group_id"]].append(record)

    # One representative per group; see the note in the old class-based split.
    # Roboflow shipped its own augmented copies and keeping them inflated a
    # class threefold while adding nothing the train-time nuisance does not.
    representatives = [min(members, key=lambda item: item["id"])
                       for members in groups.values()]

    by_stratum = collections.defaultdict(list)
    for record in representatives:
        by_stratum[stratum(record)].append(record)

    rng = random.Random(SEED)
    assignments = {r["id"]: {"split": None, "split_scheme": None} for r in rows}
    counts = collections.Counter()
    for key in sorted(by_stratum, key=repr):
        members = by_stratum[key]
        rng.shuffle(members)
        if target_per_class:
            members = members[:target_per_class]
        total = len(members)
        train_end = int(total * SPLITS[0][1])
        val_end = train_end + int(total * SPLITS[1][1])
        # A stratum with one or two members would otherwise put everything in
        # train and leave val and test blind to it.
        for index, record in enumerate(members):
            split = "train" if index < train_end else ("val" if index < val_end else "test")
            assignments[record["id"]] = {"split": split, "split_scheme": "multilabel_v1"}
            counts[split] += 1
            for name, value in record["conditions"].items():
                if value is not None:
                    counts[(split, name, "pos" if value else "neg")] += 1

    PV.update_records(assignments)
    print(f"{'kosul':<10}" + "".join(f"{s:>18}" for s, _ in SPLITS))
    for name in sorted({n for r in rows for n in r["conditions"]}):
        cells = []
        for split, _ in SPLITS:
            cells.append(f"{counts[(split, name, 'pos')]}+/{counts[(split, name, 'neg')]}-")
        print(f"{name:<10}" + "".join(f"{c:>18}" for c in cells))
    print(f"{'toplam':<10}" + "".join(f"{counts[s]:>18}" for s, _ in SPLITS))
    return counts


def stage_materialize(out_dir: str) -> int:
    """Hard-link the split into a flat tree and write the label file.

    Folder-per-class cannot express a face that has both wrinkles and eye
    bags, so the layout is flat and the labels live beside it. Each row gives
    the four conditions with null for "nobody looked" — the loss masks those
    rather than scoring them as absent.
    """
    manifest = PV.load_manifest()
    rows = [r for r in manifest.values()
            if r["status"] == "canonical" and r.get("split") and r.get("conditions")]
    written = 0
    by_split = collections.defaultdict(list)
    for record in rows:
        by_split[record["split"]].append(record)

    for split, records in by_split.items():
        image_dir = os.path.join(out_dir, split)
        os.makedirs(image_dir, exist_ok=True)
        label_path = os.path.join(out_dir, f"{split}_labels.jsonl")
        with open(label_path, "w", encoding="utf-8") as handle:
            for record in sorted(records, key=lambda item: item["id"]):
                destination = os.path.join(image_dir, f"{record['id']}.jpg")
                if not os.path.exists(destination):
                    source = os.path.join(ROOT, record["canonical_path"])
                    try:
                        os.link(source, destination)
                    except OSError:
                        import shutil
                        shutil.copy2(source, destination)
                    written += 1
                handle.write(json.dumps({
                    "id": record["id"],
                    "file": f"{record['id']}.jpg",
                    "conditions": record["conditions"],
                    "source": record["source"],
                    "age_band": record.get("age_band"),
                    "group_id": record.get("group_id"),
                }, ensure_ascii=False) + "\n")

    PV.write_snapshot(os.path.join(out_dir, "manifest.jsonl"))
    print(f"materialize: {written} yeni dosya -> {out_dir}")
    for split in sorted(by_split):
        print(f"   {split}: {len(by_split[split])} kayit")
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=("ingest", "reset", "canonicalize", "dedup", "split", "materialize", "all"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--target", type=int, default=0,
                        help="sinif basina hedef grup sayisi; 0 = sinirsiz")
    # materialize only ever adds links and never removes them, so pointing a
    # rebuild at a directory that already holds an older build silently mixes
    # the two. A new build gets a new directory; the old one stays readable.
    parser.add_argument("--out", default=OUT_DIR,
                        help="cikis agaci (varsayilan: orchestration_data_v2)")
    arguments = parser.parse_args()

    stages = (("ingest", "canonicalize", "dedup", "split", "materialize")
              if arguments.stage == "all" else (arguments.stage,))
    for stage in stages:
        print(f"\n=== {stage} ===")
        if stage == "ingest":
            stage_ingest()
        elif stage == "reset":
            stage_reset()
        elif stage == "canonicalize":
            stage_canonicalize(arguments.limit)
        elif stage == "dedup":
            stage_dedup()
        elif stage == "split":
            stage_split(arguments.target)
        elif stage == "materialize":
            stage_materialize(arguments.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
