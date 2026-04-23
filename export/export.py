import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import coremltools as ct
from src.model import build_model
from src.config import IMG_SIZE, MODEL_SAVE_PATH, CLASS_NAMES, ROOT_DIR

def export_to_coreml(model_path=None, output_path=None):
    model_path = model_path or MODEL_SAVE_PATH
    output_path = os.path.join(ROOT_DIR, "outputs", "coreml", "skin_disease.mlpackage")
    
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    example_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    traced_model = torch.jit.trace(model, example_input)

    class_labels = CLASS_NAMES
    
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=example_input.shape, name="image")],
        minimum_deployment_target=ct.target.iOS18
    )
    
    mlmodel.author = "Berat Kerem Öztopuz, Zeynep Aslan"
    mlmodel.license = "MIT"
    mlmodel.short_description = f"Skin Disease Classifier with {len(CLASS_NAMES)} classes"
    mlmodel.version = "1.0"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mlmodel.save(output_path)
    
    print(f"coreML model saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    export_to_coreml()