"""Per-image provenance manifest.

Every image that enters the pipeline gets one append-only record here. The
built dataset tree is a derivative: filenames in it are manifest ids, so
provenance survives pooling. The previous pipeline lost it, because
clean_and_split.py copied files under arbitrary basenames and the only
surviving trace of a source was a filename prefix.

Two files, both append-only, never rewritten in place:

    manifest/images.jsonl     one record per original image
    manifest/decisions.jsonl  one record per human review decision

Append-only matters: a rebuild replays the log rather than re-deriving it, so
a decision made once survives every future rebuild.

`commercial_ok` is derived from the license, never hand-set. Selecting the
App-Store-safe subset later is then one filter over this field.
"""

import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_DIR = os.path.join(ROOT, "data_prep", "manifest")
IMAGES_PATH = os.path.join(MANIFEST_DIR, "images.jsonl")
DECISIONS_PATH = os.path.join(MANIFEST_DIR, "decisions.jsonl")

# Stable namespace so an id is a pure function of the original bytes; the same
# image re-ingested from a different path keeps its id and its review decision.
ID_NAMESPACE = uuid.UUID("6f9b2d18-6c1a-5f4e-9f2c-0d3a7b5e1c44")

PIPELINE_VERSION = "canon-1.0.0"
SEED = 42


# --- licensing -------------------------------------------------------------
#
# commercial_ok reflects the *dataset* license as published. It is not legal
# advice, and it deliberately says nothing about whether model weights trained
# on non-commercial data may be shipped — that question is unsettled and needs
# a lawyer, not a lookup table.

LICENSE_COMMERCIAL: Dict[str, bool] = {
    "CC0-1.0": True,
    "PDM": True,
    "CC-BY-2.0": True,
    "CC-BY-4.0": True,
    "CC-BY-SA-2.0": True,
    "CC-BY-SA-4.0": True,
    "MIT": True,
    "US-Gov-Work": True,
    "SCIN-Public-License": True,   # CC-BY derived, attribution + no re-identification
    "CC-BY-NC-2.0": False,
    "CC-BY-NC-4.0": False,
    "CC-BY-NC-SA-3.0": False,
    "CC-BY-NC-SA-4.0": False,
    "research-only": False,
    "unknown-scraped": False,
    "unknown": False,
}

# Default license per source, used when a source carries no per-image license.
SOURCE_DEFAULT_LICENSE: Dict[str, str] = {
    "scin": "SCIN-Public-License",
    "roboflow_v3": "CC-BY-4.0",       # data.yaml declares CC BY 4.0
    "dermnet": "unknown",             # Kaggle mirror of a scraped atlas
    "tdpro": "research-only",         # TrainingData.pro free sample
    "utkface_wild": "research-only",
    "ffhq": "research-only",          # aggregate license is CC BY-NC-SA 4.0
    "fairface": "CC-BY-4.0",
    "flickr_cc": "unknown",           # always set per photo from the API
    "openverse": "unknown",
    "wikimedia": "unknown",
    "websearch_raw": "unknown-scraped",
}


def commercial_ok(license_id: str) -> bool:
    """Whether the license permits commercial use. Unknown means no."""
    return LICENSE_COMMERCIAL.get(license_id, False)


def make_id(sha256_orig: str) -> str:
    """Deterministic id from the original bytes."""
    return str(uuid.uuid5(ID_NAMESPACE, sha256_orig))


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ImageRecord:
    """One original image. Written once, then only annotated by later stages."""

    id: str
    sha256_orig: str
    source: str
    label: str

    # provenance
    source_record_id: Optional[str] = None
    license: str = "unknown"
    license_url: Optional[str] = None
    attribution: Optional[str] = None
    original_url: Optional[str] = None
    query: Optional[str] = None
    retrieved_at: str = field(default_factory=utcnow)
    local_path: Optional[str] = None

    # original file
    bytes: Optional[int] = None
    orig_width: Optional[int] = None
    orig_height: Optional[int] = None
    orig_format: Optional[str] = None
    exif_orientation: Optional[int] = None

    # stage outputs, filled in later
    face: Optional[Dict[str, Any]] = None
    crop: Optional[Dict[str, Any]] = None
    quality: Optional[Dict[str, Any]] = None
    overlay: Optional[Dict[str, Any]] = None

    label_source: str = "dataset"        # dataset | derived | human
    age_band: Optional[str] = None
    age_source: Optional[str] = None     # filename | survey | estimator | unknown
    fitzpatrick: Optional[str] = None
    monk: Optional[str] = None

    phash: Optional[str] = None
    phash_mirror: Optional[str] = None
    group_id: Optional[str] = None
    identity_cluster_id: Optional[str] = None

    status: str = "pending"              # pending | kept | rejected
    reject_reason: Optional[str] = None

    split: Optional[str] = None
    split_scheme: Optional[str] = None
    canonical_path: Optional[str] = None
    canonical_sha256: Optional[str] = None

    pipeline_version: str = PIPELINE_VERSION
    seed: int = SEED

    def __post_init__(self) -> None:
        if self.license == "unknown" and self.source in SOURCE_DEFAULT_LICENSE:
            self.license = SOURCE_DEFAULT_LICENSE[self.source]

    @property
    def commercial_ok(self) -> bool:
        return commercial_ok(self.license)

    def to_json(self) -> Dict[str, Any]:
        payload = asdict(self)
        # Derived, so it is stored for downstream filtering but never read back
        # into the dataclass — the license stays the single source of truth.
        payload["commercial_ok"] = self.commercial_ok
        return payload


def append_records(records: List[ImageRecord], path: str = IMAGES_PATH) -> int:
    """Append records atomically. Returns how many were written."""
    if not records:
        return 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = "".join(json.dumps(r.to_json(), ensure_ascii=False) + "\n" for r in records)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(lines)
        handle.flush()
        os.fsync(handle.fileno())
    return len(records)


def iter_records(path: str = IMAGES_PATH) -> Iterator[Dict[str, Any]]:
    """Stream raw records in write order, including superseded ones."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} bozuk JSON: {error}") from error


def load_manifest(path: str = IMAGES_PATH) -> Dict[str, Dict[str, Any]]:
    """Collapse the log to current state: last record wins per id."""
    latest: Dict[str, Dict[str, Any]] = {}
    for record in iter_records(path):
        latest[record["id"]] = record
    return latest


def update_records(updates: Dict[str, Dict[str, Any]], path: str = IMAGES_PATH) -> int:
    """Append revised copies of existing records.

    Later stages annotate a record (crop geometry, group_id, split) by
    appending a full revised copy rather than editing in place, so the log
    stays an audit trail.
    """
    current = load_manifest(path)
    revised = []
    for record_id, changes in updates.items():
        if record_id not in current:
            raise KeyError(f"manifest'te yok: {record_id}")
        merged = dict(current[record_id])
        merged.update(changes)
        revised.append(merged)
    if not revised:
        return 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in revised)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(lines)
        handle.flush()
        os.fsync(handle.fileno())
    return len(revised)


def append_decision(
    image_id: str,
    decision: str,
    reviewer: str,
    tool_version: str = "review-1.0.0",
    path: str = DECISIONS_PATH,
) -> None:
    """Record one human review decision. approve | reject | unsure."""
    if decision not in {"approve", "reject", "unsure"}:
        raise ValueError(f"gecersiz karar: {decision}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "id": image_id,
        "decision": decision,
        "reviewer": reviewer,
        "reviewed_at": utcnow(),
        "tool_version": tool_version,
    }
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_decisions(path: str = DECISIONS_PATH) -> Dict[str, Dict[str, Any]]:
    """Latest decision per image id."""
    latest: Dict[str, Dict[str, Any]] = {}
    for record in iter_records(path):
        latest[record["id"]] = record
    return latest


def write_snapshot(destination: str, path: str = IMAGES_PATH) -> int:
    """Write the collapsed manifest next to a built dataset tree."""
    manifest = load_manifest(path)
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    directory = os.path.dirname(destination) or "."
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=directory, delete=False, suffix=".tmp"
    )
    try:
        for record in manifest.values():
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(handle.name, destination)
    except BaseException:
        handle.close()
        if os.path.exists(handle.name):
            os.unlink(handle.name)
        raise
    return len(manifest)
