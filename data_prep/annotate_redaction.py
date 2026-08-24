"""Record how much of each canonical image is a solid black rectangle.

SCIN masks identifying features with a filled black box. It is burned into
the pixels, so it is the same family of cue as the DermNet watermark — and it
is not spread evenly: 45% of the acne images carry one against 16% of the
eczema ones, which the shortcut probe read as its strongest single feature at
AUC 0.631.

Stored rather than recomputed so the balancing step can treat it as a
stratum, the way it already treats source and age band.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image

import provenance as PV

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Anything under this is a dark corner or a shadow, not a mask.
PRESENT_THRESHOLD = 0.02


def main() -> int:
    updates = {}
    for record in PV.load_manifest().values():
        if record["status"] not in ("canonical", "held_out"):
            continue
        path = record.get("canonical_path")
        if not path or not os.path.exists(os.path.join(ROOT, path)):
            continue
        with Image.open(os.path.join(ROOT, path)) as handle:
            gray = np.asarray(handle.convert("L").resize((96, 96)))
        fraction = float((gray < 8).mean())
        bucket = "yes" if fraction >= PRESENT_THRESHOLD else "no"
        if record.get("redaction") != bucket:
            updates[record["id"]] = {"redaction": bucket,
                                     "redaction_fraction": round(fraction, 4)}
    written = PV.update_records(updates)
    print(f"redaksiyon yazildi: {written} kayit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
