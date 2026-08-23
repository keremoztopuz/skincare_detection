"""Body-region screening.

Two jobs, both driven by an explicit rule that intimate regions are excluded
from the dataset:

  1. Reject images of genital, perianal, buttock and breast regions. Those
     areas differ enough in skin structure that they teach the wrong thing to
     a face-facing model, and they have no place in a consumer app's dataset.
  2. Reject conditions that were filed under the wrong label. DermNet's
     "Acne and Rosacea Photos" folder is only ~65% acne: 94 files are
     hidradenitis suppurativa, 21 rosacea, 59 perioral dermatitis, 8 drug
     eruptions and 7 Fordyce spots.
  3. Reject images that are not photographs of skin at all — histology slides
     and lesion diagrams.

Three signals, in order of trustworthiness:

  * SCIN carries per-case body-part columns, so no inference is needed at all.
  * DermNet filenames are descriptive for ~40% of files ("eczema-foot-81",
    "hidradenitis-suppurativa-59"). Where present they are ground truth.
  * Everything else needs a model, and the choice of model matters. A nudity
    detector was tried first and is unusable here: on clinical close-ups it
    called a foot "male genitalia" at 0.91, fingertips at 0.83, and a rosacea
    face "female genitalia" at 0.57. It is trained on pornography and inflamed
    skin lies far outside that distribution. CLIP zero-shot over body-part
    prompts replaces it, and is validated against the filename-labelled subset
    before being trusted on the rest.
"""

import os
import re
from typing import Dict, List, Optional, Tuple

# --- filename rules --------------------------------------------------------

# Intimate regions. Matched as substrings against a lowercased filename.
INTIMATE_TOKENS = (
    "genital", "genitalia", "penis", "penile", "scrotum", "scrotal", "foreskin",
    "glans", "balanitis", "vulva", "vulval", "labia", "vaginal", "perineal",
    "perianal", "anal", "anus", "buttock", "gluteal", "groin", "inguinal",
    "pubic", "crotch", "napkin", "diaper", "areola", "nipple", "breast",
)

# Conditions that are not the class they were filed under.
MISLABELLED_CONDITIONS = {
    "hidradenitis": "hidradenitis_suppurativa",
    "suppurativa": "hidradenitis_suppurativa",
    "rosacea": "rosacea",
    "rhinophyma": "rosacea",
    # Perioral dermatitis is not acne. It is treated differently — topical
    # steroids make it worse, and some acne regimens are wrong for it — so a
    # product-recommending app labelling it "Acne" is a real error, not a
    # rounding one. Dropping it costs 22 of Acne's 136 face-containing images
    # and leaves 114, so the "we need facial images" argument does not hold.
    # The manifest keeps them with a reason, so a future facial-dermatitis
    # class can recover them.
    "perioral": "perioral_dermatitis",
    # Found during human review, then generalised from the filenames.
    "fordyce": "fordyce_spots",
    "minocycline": "drug_induced_pigmentation",
    "drug-eruption": "drug_induced_pigmentation",
    "drug-induced": "drug_induced_pigmentation",
}

# Not photographs of skin at all. The reviewer caught six histology slides and
# six lesion diagrams by hand; generalising their filenames covers the rest of
# the pool, including the part nobody has looked at. A microscope slide or a
# line drawing shares no statistics with a phone photo of a face.
NON_PHOTOGRAPH_TOKENS = (
    "histology", "histopath", "micrograph", "pathology", "biopsy", "h&e",
    "primary-lesion", "diagram", "illustration", "schematic", "drawing",
)

# Regions that are safe to keep. Used to auto-accept without a model.
SAFE_TOKENS = (
    "face", "facial", "cheek", "forehead", "chin", "nose", "lip",
    "lids", "eyelid", "periorbital", "scalp", "ear", "neck",  # perioral: MISLABELLED_CONDITIONS
    "hand", "finger", "fingertips", "palm", "wrist", "nail", "knuckle",
    "arm", "forearm", "elbow", "shoulder",
    "leg", "knee", "thigh", "shin", "calf", "ankle",
    "foot", "feet", "toe", "heel", "sole",
    "back", "trunk", "torso", "abdomen",
)

INTIMATE = "intimate"
MISLABELLED = "mislabelled_condition"
NON_PHOTOGRAPH = "not_a_photograph"
SAFE = "safe"
UNKNOWN = "unknown"


def classify_filename(filename: str) -> Tuple[str, Optional[str]]:
    """Classify by filename alone.

    Returns (verdict, detail). Intimate wins over everything else, then
    not-a-photograph, then mislabelled, then safe.
    """
    stem = os.path.splitext(os.path.basename(filename))[0].lower()
    # Split CamelCase so "AnalExcoriation" yields "anal".
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", os.path.splitext(os.path.basename(filename))[0]).lower()
    haystack = f"{stem} {spaced}"

    for token in INTIMATE_TOKENS:
        if token in haystack:
            return INTIMATE, token
    for token in NON_PHOTOGRAPH_TOKENS:
        if token in haystack:
            return NON_PHOTOGRAPH, token
    for token, condition in MISLABELLED_CONDITIONS.items():
        if token in haystack:
            return MISLABELLED, condition
    for token in SAFE_TOKENS:
        if token in haystack:
            return SAFE, token
    return UNKNOWN, None


# --- SCIN metadata ---------------------------------------------------------

SCIN_INTIMATE_COLUMNS = (
    "body_parts_genitalia_or_groin",
    "body_parts_buttocks",
)


def classify_scin_case(case_row: Dict[str, str]) -> Tuple[str, Optional[str]]:
    """Classify a SCIN case from its own body-part columns. No inference."""
    for column in SCIN_INTIMATE_COLUMNS:
        if (case_row.get(column) or "").strip().upper() == "YES":
            return INTIMATE, column
    return SAFE, None


# --- CLIP zero-shot --------------------------------------------------------

# MEASURED RESULT: CLIP is not usable as a gate here either.
#
# Validated against the 434 filename-labelled DermNet images (34 intimate,
# 400 safe). Best operating point was threshold 0.90 at recall 0.941 /
# precision 0.800 — and its confident mistakes are disqualifying:
#
#   07PerioralDermEye.jpg   0.972  (a face, around the eye)
#   eczema-foot-21.jpg      0.967  (a foot)
#   fordyce-spots-lip.jpg   0.964  (a lip)
#   eczema-face-11.jpg      0.901  (a face)
#
# It fails for the same reason the nudity detector did: a close-up of
# inflamed skin is texture-dominated, and both models key on "red inflamed
# skin close-up" rather than on anatomy. Two independent models failing the
# same way is evidence about the task, not about the models.
#
# So CLIP is kept only as a REVIEW ORDERING signal — a weak prior that puts
# likely-intimate images at the front of the human review queue, where a
# reviewer finds them early. It never auto-rejects and never auto-accepts.
#
# Prompts are phrased as clinical photographs because that is the actual
# domain; "a photo of a foot" competes badly against "a photo of genitals"
# when the foot is covered in eczema.
CLIP_PROMPTS: List[Tuple[str, str]] = [
    (SAFE, "a clinical photograph of a person's face"),
    (SAFE, "a clinical photograph of a hand or fingers"),
    (SAFE, "a clinical photograph of a foot or toes"),
    (SAFE, "a clinical photograph of an arm or elbow"),
    (SAFE, "a clinical photograph of a leg or knee"),
    (SAFE, "a clinical photograph of a person's back or shoulder"),
    (SAFE, "a clinical photograph of a neck or chest"),
    (SAFE, "a close-up clinical photograph of skin on a limb"),
    (INTIMATE, "a clinical photograph of the genital area"),
    (INTIMATE, "a clinical photograph of the groin or inner thigh crease"),
    (INTIMATE, "a clinical photograph of the buttocks or perianal area"),
    (INTIMATE, "a clinical photograph of a breast or nipple"),
]

# Deliberately absent: there is no CLIP threshold that gates anything. See the
# measurement above. The score is an ordering key for review, nothing more.
CLIP_IS_A_GATE = False


def summarize(verdicts: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for verdict in verdicts:
        counts[verdict] = counts.get(verdict, 0) + 1
    return counts
