import os 
import torch

# dataset paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "FINAL_SPLIT")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")
IMAGES_DIR = os.path.join(ROOT_DIR, "outputs", "images")

# model parameters 
MODEL_NAME = "convnext_tiny"
NUM_CLASSES = 6
DROP_RATE = 0.2

# training parameters
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
DETECTION_THRESHOLD = 0.3
PATIENCE = 5
CLASS_NAMES = ["Acne", "Eczema", "Psoriasis", "Ben_Lezyon", "Healthy", "Eye_Bags"]
CLASS_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 0.7, 2.5]
SEED = 42

# device
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# augmentations

IMG_SIZE = 224
MEAN = [0.5942, 0.4433, 0.3871]
STD = [0.2427, 0.2027, 0.1930]

# saving paths

CHECKPOINT_DIR = os.path.join(ROOT_DIR, "outputs", "checkpoints")
MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "outputs", "model", "best_model.pth")
LOGS_DIR = os.path.join(ROOT_DIR, "outputs", "logs")