import os 
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "FINAL_SPLIT")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")
IMAGES_DIR = os.path.join(ROOT_DIR, "outputs", "images")

MODEL_NAME = "convnext_tiny"
NUM_CLASSES = 5
DROP_RATE = 0.3
DROP_PATH_RATE = 0.2

NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.1
DETECTION_THRESHOLD = 0.5
PATIENCE = 10
WARMUP_EPOCHS = 5
GRADIENT_CLIP = 1.0
LABEL_SMOOTHING = 0.1
CLASS_NAMES = ["Acne", "Eczema", "Psoriasis", "Eye_Bags", "Wrinkles"]
CLASS_WEIGHTS = [1.0, 1.0, 1.0, 2.03, 4.23]
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

IMG_SIZE = 384
MEAN = [0.5942, 0.4433, 0.3871]
STD = [0.2427, 0.2027, 0.1930]

CHECKPOINT_DIR = os.path.join(ROOT_DIR, "outputs", "checkpoints")
MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "outputs", "model", "best_model.pth")
LOGS_DIR = os.path.join(ROOT_DIR, "outputs", "logs")
