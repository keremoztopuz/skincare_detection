import os
import torch
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

from config import DEVICE, MODEL_SAVE_PATH, CLASS_NAMES, DATA_DIR, IMG_SIZE, MEAN, STD
from model import build_model

TEST_IMAGE_PATH = os.path.join(DATA_DIR, "test", "Acne", "acne-cystic-21.jpg")

def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

def predict(image_path):
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    transform = get_transform()
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad(): 
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = CLASS_NAMES[predicted_idx.item()]

        return {
            "Class": predicted_class,
            "Confidence": confidence.item(),
            "Probabilities": dict(zip(CLASS_NAMES, probabilities[0].tolist()))
        }

if __name__ == "__main__":

    result = predict(TEST_IMAGE_PATH)   
    print(result)