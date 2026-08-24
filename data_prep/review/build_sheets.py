"""Contact sheets of the periorbital band, for the eye-bag call.

The browser sheet shows whole faces, which is right when the question is
"which condition is this". For eye bags the answer lives entirely in the strip
under the eyes, so cropping to that band spends every pixel on the decision
and fits twenty faces on a legible page instead of six.

Order matches the browser tool: the faces currently labelled Healthy come
first, because a face with eye bags sitting in the negative class teaches the
Eye_Bags head that its own condition is normal. The reserve follows,
oldest-first — age orders the queue, never the label.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from PIL import Image, ImageDraw

import provenance as PV

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sheets")

PER_SHEET = 20
COLUMNS = 5
CELL_WIDTH = 330
CELL_HEIGHT = int(CELL_WIDTH * 0.45)
PAD = 5
LABEL_HEIGHT = 16
# Vertical slice of the canonical crop holding brow to mid-cheek. Wider than
# the lid alone: the cheek gives the contrast that separates a bag from a
# shadow, and the brow keeps the face recognisable enough to spot a squint.
BAND = (0.10, 0.24, 0.90, 0.60)
AGE_ORDER = {"70_plus": 0, "50_69": 1, "40_49": 2, "30_39": 3, "20_29": 4}
RESERVE_REASONS = {"review_skip", "unreviewed",
                   "age_match_surplus:Healthy", "age_match_surplus:Wrinkles"}


def candidates():
    items = []
    for record in PV.load_manifest().values():
        if record.get("source") != "ffhq" or not record.get("canonical_path"):
            continue
        if record.get("glasses") not in (None, "None"):
            continue
        if record["status"] == "canonical" and record.get("label") == "Healthy":
            priority = 0
        elif record["status"] == "held_out" and record.get("reject_reason") in RESERVE_REASONS:
            priority = 1
        else:
            continue
        if not os.path.exists(os.path.join(ROOT, record["canonical_path"])):
            continue
        items.append((priority, AGE_ORDER.get(record.get("age_band"), 9),
                      record["id"], record))
    items.sort()
    return items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=OUT_DIR)
    arguments = parser.parse_args()

    items = candidates()
    if arguments.limit:
        items = items[:arguments.limit]
    os.makedirs(arguments.out, exist_ok=True)
    for stale in os.listdir(arguments.out):
        os.remove(os.path.join(arguments.out, stale))

    index = []
    for number in range((len(items) + PER_SHEET - 1) // PER_SHEET):
        chunk = items[number * PER_SHEET:(number + 1) * PER_SHEET]
        rows = (len(chunk) + COLUMNS - 1) // COLUMNS
        canvas = Image.new(
            "RGB",
            (COLUMNS * (CELL_WIDTH + PAD) + PAD,
             rows * (CELL_HEIGHT + PAD + LABEL_HEIGHT) + PAD),
            (18, 18, 22))
        draw = ImageDraw.Draw(canvas)
        cells = []
        for position, (priority, _, image_id, record) in enumerate(chunk):
            with Image.open(os.path.join(ROOT, record["canonical_path"])) as handle:
                image = handle.convert("RGB")
            width, height = image.size
            box = (int(width * BAND[0]), int(height * BAND[1]),
                   int(width * BAND[2]), int(height * BAND[3]))
            image = image.crop(box).resize((CELL_WIDTH, CELL_HEIGHT), Image.LANCZOS)
            x = PAD + (position % COLUMNS) * (CELL_WIDTH + PAD)
            y = PAD + (position // COLUMNS) * (CELL_HEIGHT + PAD + LABEL_HEIGHT)
            canvas.paste(image, (x, y))
            draw.text((x + 2, y + CELL_HEIGHT + 2), str(position + 1),
                      fill=(190, 190, 200))
            cells.append({
                "cell": position + 1,
                "manifest_id": image_id,
                "review_id": PV.make_id(record["canonical_sha256"]),
                "age_band": record.get("age_band"),
                "now_healthy": priority == 0,
            })
        path = os.path.join(arguments.out, f"sheet_{number:03d}.jpg")
        canvas.save(path, quality=93)
        index.append({"sheet": number, "path": path, "cells": cells})

    with open(os.path.join(arguments.out, "index.json"), "w", encoding="utf-8") as handle:
        json.dump(index, handle)
    healthy = sum(1 for item in items if item[0] == 0)
    print(f"{len(items)} goruntu ({healthy} mevcut Healthy), {len(index)} sayfa -> {arguments.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
