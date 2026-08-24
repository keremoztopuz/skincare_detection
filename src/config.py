import os 
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "orchestration_data_v4")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")
IMAGES_DIR = os.path.join(ROOT_DIR, "outputs", "images")

MODEL_NAME = "convnext_tiny"
NUM_CLASSES = 4
DROP_RATE = 0.2
DROP_PATH_RATE = 0.1

NUM_EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 4.25e-5
LABEL_SMOOTHING = 0.03
BACKBONE_LR_MULTIPLIER = 0.05
WEIGHT_DECAY = 0.05
DETECTION_THRESHOLD = 0.45
THRESHOLD_SEARCH_MIN = 0.10
THRESHOLD_SEARCH_MAX = 0.90
THRESHOLD_SEARCH_STEP = 0.01
PATIENCE = 8
WARMUP_EPOCHS = 2
HEAD_ONLY_EPOCHS = 5
GRADIENT_CLIP = 1.0
UNFREEZE_LAST_N_STAGES = 1
POS_WEIGHT_POWER = 0.25
AUXILIARY_CE_WEIGHT = 0.25
MIN_CHECKPOINT_AUROC = 0.50
CLASS_NAMES = ["Acne", "Eczema", "Eye_Bags", "Wrinkles"]
# Labels are per-condition and may be unknown. A DermNet acne photo of a
# forearm says nothing about that person's wrinkles, and scoring it as "no
# wrinkles" would be a false negative, so the loss and the metrics both mask
# whatever nobody has judged. There is no Healthy class any more: a clean
# image is simply one whose conditions are all known and all zero.
LABEL_FILENAME = "{split}_labels.jsonl"
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

IMG_SIZE = 384
MEAN = [0.5942, 0.4433, 0.3871]
STD = [0.2427, 0.2027, 0.1930]

CHECKPOINT_DIR = os.path.join(ROOT_DIR, "outputs", "checkpoints")
MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "outputs", "model", "best_model.pth")
THRESHOLDS_SAVE_PATH = os.path.join(ROOT_DIR, "outputs", "model", "thresholds.json")
LOGS_DIR = os.path.join(ROOT_DIR, "outputs", "logs")
