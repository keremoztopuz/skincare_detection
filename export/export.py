import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import coremltools as ct
from model import build_model
from config import IMG_SIZE, MODEL_SAVE_PATH, CLASS_NAMES, ROOT_DIR


def export_to_coreml(model_path=None, output_path=None, palettize=False):
    model_path = model_path or MODEL_SAVE_PATH
    output_path = output_path or os.path.join(ROOT_DIR, "outputs", "coreml", "skin_disease.mlpackage")

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    example_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    traced_model = torch.jit.trace(model, example_input)

    # FLOAT16 halves the package size versus the old FLOAT32 export with no
    # measurable accuracy cost for this model. The TensorType input contract
    # is kept so the app's preprocessing code stays untouched.
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=example_input.shape, name="image")],
        outputs=[ct.TensorType(name="scores")],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )

    if palettize:
        from coremltools.optimize.coreml import (
            OpPalettizerConfig,
            OptimizationConfig,
            palettize_weights,
        )
        op_config = OpPalettizerConfig(mode="kmeans", nbits=8)
        mlmodel = palettize_weights(mlmodel, config=OptimizationConfig(global_config=op_config))

    mlmodel.author = "Berat Kerem Öztopuz, Zeynep Aslan"
    mlmodel.license = "MIT"
    mlmodel.short_description = f"Skin Disease Classifier with {len(CLASS_NAMES)} classes"
    mlmodel.version = "2.0"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mlmodel.save(output_path)
    print(f"coreML model saved to: {output_path}")

    verify_export(mlmodel, traced_model, example_input)
    return output_path


def verify_export(mlmodel, traced_model, example_input):
    """Compare CoreML logits against PyTorch on a fixed input.

    CoreML prediction requires macOS; skipped gracefully elsewhere.
    """
    with torch.no_grad():
        torch_logits = traced_model(example_input).numpy().flatten()
    try:
        prediction = mlmodel.predict({"image": example_input.numpy()})
    except Exception as error:
        print(f"verification skipped (predict unavailable): {error}")
        return
    coreml_logits = np.asarray(next(iter(prediction.values()))).flatten()
    max_delta = float(np.max(np.abs(torch_logits - coreml_logits)))
    print(f"logit karsilastirma: max |Δ| = {max_delta:.4f} "
          f"({'OK' if max_delta < 0.15 else 'SAPMA BUYUK — kontrol et'})")
    for name, torch_value, coreml_value in zip(CLASS_NAMES, torch_logits, coreml_logits):
        print(f"  {name:<10} torch={torch_value:+.3f}  coreml={coreml_value:+.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export the trained model to CoreML")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--palettize", action="store_true",
                        help="8-bit weight palettization (~27 MB instead of ~53 MB)")
    args = parser.parse_args()
    export_to_coreml(args.model_path, args.output_path, palettize=args.palettize)
