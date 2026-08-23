"""Build a self-contained HTML review sheet.

Two review jobs need a human in the loop, for the same reason: no available
model is trustworthy on clinical close-ups. Body-region screening needs it
because both a nudity detector and CLIP zero-shot failed (see
data_prep/body_region.py), and Eye_Bags labels need it because the Roboflow
community labels are noisy.

The sheet is one HTML file with the thumbnails inlined as base64. No server,
no dependencies, works offline, and the decisions land in localStorage until
exported. Keeping is the default and rejecting is a click, because the
overwhelming majority of any queue is fine — the reviewer should only have to
act on the exceptions.

    python data_prep/review/build_review.py --job body-region --out review.html
    open review.html            # click the bad ones, then Export
    # drop the downloaded decisions.jsonl into data_prep/manifest/
"""

import argparse
import base64
import glob
import io
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "data_prep"))

import body_region as BR  # noqa: E402
import provenance as PV  # noqa: E402

THUMB_SIDE = 240
THUMB_QUALITY = 72


def thumbnail(path: str) -> Optional[str]:
    """Square-padded base64 JPEG thumbnail, or None if unreadable."""
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((THUMB_SIDE, THUMB_SIDE), Image.LANCZOS)
            canvas = Image.new("RGB", (THUMB_SIDE, THUMB_SIDE), (24, 24, 27))
            canvas.paste(image, ((THUMB_SIDE - image.width) // 2, (THUMB_SIDE - image.height) // 2))
            buffer = io.BytesIO()
            canvas.save(buffer, "JPEG", quality=THUMB_QUALITY, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        return None


def collect_body_region() -> List[Dict[str, object]]:
    """Queue for body-region review, riskiest first.

    Filename-decided images are excluded: they are already settled, and
    padding the queue with them would bury the cases that need a person.
    """
    items = []
    for class_name in ("Acne", "Eczema"):
        for path in sorted(glob.glob(os.path.join(ROOT, "data_prep/downloads/dermnet", class_name, "*"))):
            if not path.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            verdict, detail = BR.classify_filename(path)
            if verdict in (BR.INTIMATE, BR.MISLABELLED, BR.NON_PHOTOGRAPH):
                continue
            items.append({
                "path": path,
                "label": class_name,
                "note": detail or "",
                "priority": 0 if verdict == BR.UNKNOWN else 1,
            })
    items.sort(key=lambda item: (item["priority"], item["path"]))
    return items


def collect_ffhq() -> List[Dict[str, object]]:
    """FFHQ candidates for the Wrinkles / Healthy call, oldest first.

    Age decides the order, never the label. Sorting oldest-first puts the
    likely Wrinkles cases early so the reviewer finds them without hunting,
    but every image still has to be judged on whether wrinkles are visible —
    using the age band as the label is exactly the confound this avoids.
    """
    import provenance as PV
    order = {"70_plus": 0, "50_69": 1, "40_49": 2, "30_39": 3, "20_29": 4}
    items = []
    for record in PV.load_manifest().values():
        if record.get("source") != "ffhq" or record.get("status") != "canonical":
            continue
        path = os.path.join(ROOT, record["canonical_path"])
        if not os.path.exists(path):
            continue
        items.append({
            "path": path, "label": "UNLABELLED",
            "note": record.get("age_band") or "?",
            "priority": order.get(record.get("age_band"), 9),
        })
    items.sort(key=lambda item: (item["priority"], item["path"]))
    return items


def collect_glob(pattern: str, label: str) -> List[Dict[str, object]]:
    items = []
    for path in sorted(glob.glob(pattern)):
        if path.lower().endswith((".jpg", ".jpeg", ".png")):
            items.append({"path": path, "label": label, "note": "", "priority": 0})
    return items


PAGE = """<!doctype html>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
 :root { color-scheme: dark; }
 body { margin:0; background:#111114; color:#e7e7ea;
        font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
 header { position:sticky; top:0; z-index:5; background:#191920; border-bottom:1px solid #2c2c36;
          padding:12px 18px; display:flex; gap:18px; align-items:center; flex-wrap:wrap; }
 h1 { font-size:15px; margin:0; font-weight:600; }
 .muted { color:#9a9aa6; }
 .bar { flex:1; min-width:180px; height:6px; background:#2c2c36; border-radius:3px; overflow:hidden; }
 .bar > i { display:block; height:100%; background:#7c5cff; width:0; }
 button { background:#26262f; color:#e7e7ea; border:1px solid #3a3a46; border-radius:7px;
          padding:7px 13px; font-size:13px; cursor:pointer; }
 button:hover { background:#32323d; }
 button.primary { background:#7c5cff; border-color:#7c5cff; color:#fff; font-weight:600; }
 .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr));
         gap:10px; padding:16px 18px 90px; }
 figure { margin:0; position:relative; border-radius:9px; overflow:hidden; cursor:pointer;
          border:2px solid transparent; background:#1c1c23; }
 figure img { display:block; width:100%; aspect-ratio:1; object-fit:cover; }
 figure figcaption { font-size:10px; color:#8a8a96; padding:4px 6px; overflow:hidden;
                     text-overflow:ellipsis; white-space:nowrap; }
 figure.reject { border-color:#ff4d6a; }
 figure.wrinkles { border-color:#ffa726; }
 figure.wrinkles::after { content:"KIRISIK"; position:absolute; inset:auto 0 0 0; background:#ffa726;
   color:#111; font-size:10px; font-weight:700; text-align:center; padding:2px; }
 figure.healthy { border-color:#26c281; }
 figure.healthy::after { content:"TEMIZ"; position:absolute; inset:auto 0 0 0; background:#26c281;
   color:#111; font-size:10px; font-weight:700; text-align:center; padding:2px; }
 figure.skipped { opacity:.3; }
 figure.reject img { opacity:.26; }
 figure.reject::after { content:"CIKAR"; position:absolute; inset:0; display:flex;
   align-items:center; justify-content:center; font-weight:700; color:#ff4d6a; letter-spacing:1px; }
 figure.cursor { outline:2px solid #7c5cff; outline-offset:1px; }
 footer { position:fixed; bottom:0; left:0; right:0; background:#191920;
          border-top:1px solid #2c2c36; padding:10px 18px; display:flex; gap:14px; align-items:center; }
</style>
<header>
  <h1>__TITLE__</h1>
  <span class="muted" id="stat"></span>
  <span class="bar"><i id="bar"></i></span>
  <button id="clear">Secimleri sifirla</button>
  <button class="primary" id="export">decisions.jsonl indir</button>
</header>
<div class="grid" id="grid"></div>
<footer>
  <span class="muted" id="help"></span>
</footer>
<script>
const ITEMS = __ITEMS__;
const JOB = "__JOB__";
const KEY = "review:" + JOB;
let state = JSON.parse(localStorage.getItem(KEY) || "{}");
let cursor = 0;

const grid = document.getElementById("grid");
ITEMS.forEach((item, index) => {
  const figure = document.createElement("figure");
  figure.dataset.index = index;
  figure.innerHTML =
    '<img loading="lazy" src="data:image/jpeg;base64,' + item.thumb + '">' +
    '<figcaption>' + item.name + '</figcaption>';
  figure.addEventListener("click", () => { toggle(index); move(index); });
  grid.appendChild(figure);
});

// Two shapes of review. A reject/keep queue is a single toggle; sorting into
// classes needs three outcomes, and "skip" has to be one of them — forcing a
// call on an ambiguous face is how label noise gets in.
const THREE_WAY = JOB === "ffhq";
const CHOICES = THREE_WAY ? ["wrinkles", "healthy", "skip"] : ["reject"];

function setChoice(index, choice) {
  const id = ITEMS[index].id;
  if (!choice || state[id] === choice) { delete state[id]; }
  else { state[id] = choice; }
  localStorage.setItem(KEY, JSON.stringify(state));
  render();
}
function toggle(index) {
  if (THREE_WAY) { return; }
  const id = ITEMS[index].id;
  setChoice(index, state[id] ? null : "reject");
}
function move(index) {
  cursor = Math.max(0, Math.min(ITEMS.length - 1, index));
  render();
}
function render() {
  const figures = grid.children;
  for (let index = 0; index < figures.length; index++) {
    const choice = state[ITEMS[index].id];
    figures[index].classList.toggle("reject", choice === "reject");
    figures[index].classList.toggle("wrinkles", choice === "wrinkles");
    figures[index].classList.toggle("healthy", choice === "healthy");
    figures[index].classList.toggle("skipped", choice === "skip");
    figures[index].classList.toggle("cursor", index === cursor);
  }
  const tally = {};
  for (const value of Object.values(state)) { tally[value] = (tally[value] || 0) + 1; }
  const summary = THREE_WAY
    ? ["wrinkles", "healthy", "skip"].map(k => (tally[k] || 0) + " " + k).join(" \u00b7 ")
    : (tally.reject || 0) + " cikarildi";
  document.getElementById("stat").textContent =
    ITEMS.length + " gorsel \u00b7 " + summary;
  document.getElementById("bar").style.width = (100 * cursor / ITEMS.length) + "%";
}
addEventListener("keydown", (event) => {
  const columns = Math.max(1, Math.floor(grid.clientWidth / 160));
  if (event.key === "ArrowRight") { move(cursor + 1); event.preventDefault(); }
  else if (event.key === "ArrowLeft") { move(cursor - 1); event.preventDefault(); }
  else if (event.key === "ArrowDown") { move(cursor + columns); event.preventDefault(); }
  else if (event.key === "ArrowUp") { move(cursor - columns); event.preventDefault(); }
  else if (THREE_WAY && event.key === "a") { setChoice(cursor, "wrinkles"); move(cursor + 1); event.preventDefault(); }
  else if (THREE_WAY && event.key === "s") { setChoice(cursor, "healthy"); move(cursor + 1); event.preventDefault(); }
  else if (THREE_WAY && (event.key === "d" || event.key === " ")) { setChoice(cursor, "skip"); move(cursor + 1); event.preventDefault(); }
  else if (!THREE_WAY && (event.key === "x" || event.key === " ")) { toggle(cursor); event.preventDefault(); }
  grid.children[cursor] && grid.children[cursor].scrollIntoView({block: "nearest"});
});
document.getElementById("clear").addEventListener("click", () => {
  if (confirm("Tum secimler silinsin mi?")) { state = {}; localStorage.setItem(KEY, "{}"); render(); }
});
document.getElementById("export").addEventListener("click", () => {
  const stamp = new Date().toISOString();
  // Every reviewed item is written, not only the rejects: "looked at and kept"
  // is a decision, and without it a rebuild cannot tell it from "never seen".
  // Only judged items are written. In the three-way job an unvisited face is
  // not "healthy by default" — silence is not a label.
  const source = THREE_WAY ? ITEMS.filter(i => state[i.id]) : ITEMS;
  const lines = source.map(item => JSON.stringify({
    id: item.id,
    decision: THREE_WAY ? state[item.id] : (state[item.id] ? "reject" : "approve"),
    reviewer: "human",
    reviewed_at: stamp,
    tool_version: "review-1.0.0",
    job: JOB
  })).join("\\n") + "\\n";
  const url = URL.createObjectURL(new Blob([lines], {type: "application/x-ndjson"}));
  const link = document.createElement("a");
  link.href = url; link.download = "decisions.jsonl"; link.click();
  URL.revokeObjectURL(url);
});
document.getElementById("help").innerHTML = THREE_WAY
  ? "Klavye: <b>a</b> kirisik &middot; <b>s</b> temiz &middot; <b>d</b> atla &middot; oklar gez. Karar verilmeyen yazilmaz."
  : "Varsayilan: <b>tut</b>. Yalnizca <b>cikarilacaklara</b> tikla. <b>x</b>/space cikar.";
render();
</script>
"""


def build(items: List[Dict[str, object]], job: str, title: str, out_path: str, limit: int) -> Tuple[int, int, int]:
    import hashlib

    payload, skipped, duplicates = [], 0, 0
    seen = set()
    for item in items:
        if len(payload) >= limit:
            break
        with open(item["path"], "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        record_id = PV.make_id(digest)
        # The pool holds byte-identical copies under different names. Showing
        # them twice wastes review time and makes the counter disagree with
        # the state, since state is keyed by id and one click would light up
        # every copy.
        if record_id in seen:
            duplicates += 1
            continue
        seen.add(record_id)
        thumb = thumbnail(item["path"])
        if thumb is None:
            skipped += 1
            continue
        payload.append({
            "id": record_id,
            # Canonical files are named by manifest id, so the filename tells a
            # reviewer nothing. Show the note (age band) instead — useful
            # context for the call, and it keeps the id out of the way.
            "name": (item.get("note") or os.path.basename(item["path"]))[:40],
            "label": item["label"],
            "thumb": thumb,
        })
    html = (PAGE
            .replace("__ITEMS__", json.dumps(payload))
            .replace("__TITLE__", title)
            .replace("__JOB__", job))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(html)
    return len(payload), skipped, duplicates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", default="body-region", choices=("body-region", "eyebags", "ffhq", "glob"))
    parser.add_argument("--pattern", help="--job glob icin dosya deseni")
    parser.add_argument("--label", default="", help="--job glob icin sinif adi")
    parser.add_argument("--limit", type=int, default=1200)
    parser.add_argument("--out", default=os.path.join(HERE, "review.html"))
    arguments = parser.parse_args()

    if arguments.job == "ffhq":
        items = collect_ffhq()
        title = "FFHQ: kirisik mi, temiz mi?  (a=kirisik  s=temiz  d=atla)"
    elif arguments.job == "body-region":
        items = collect_body_region()
        title = "Vucut bolgesi taramasi - mahrem bolgeleri cikar"
    elif arguments.job == "eyebags":
        items = collect_glob(os.path.join(ROOT, "data_prep/downloads/staged/Eye_Bags", "*"), "Eye_Bags")
        title = "Eye_Bags etiket dogrulama"
    else:
        if not arguments.pattern:
            parser.error("--job glob icin --pattern gerekli")
        items = collect_glob(arguments.pattern, arguments.label)
        title = f"Inceleme: {arguments.label or arguments.pattern}"

    if not items:
        print("inceleneceek gorsel yok")
        return 1

    written, skipped, duplicates = build(items, arguments.job, title, arguments.out, arguments.limit)
    size_mb = os.path.getsize(arguments.out) / 1e6
    print(f"{written} benzersiz gorsel yazildi "
          f"({duplicates} kopya elendi, {skipped} okunamadi), {size_mb:.1f} MB")
    print(f"ac: {arguments.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
