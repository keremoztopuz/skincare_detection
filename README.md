# Skin Condition Orchestration Model

A ConvNeXt-Tiny based multi-label model for detecting acne, eczema, eye bags, and wrinkles from skin/face images.

## Authors
- Berat Kerem Öztopuz

## Model Architecture
- **Base Model:** ConvNeXt-Tiny (pretrained on ImageNet)
- **Input Size:** 384×384
- **Output:** 4 independent logits interpreted with Sigmoid
- **Fine-tuning:** Head-only warmup, then final ConvNeXt stage at a reduced learning rate
- **Loss Function:** BCEWithLogitsLoss with training-set-derived positive weights
- **Optimizer:** AdamW (weight_decay=0.05) with Warmup + CosineAnnealing scheduler
- **Thresholds:** Precision-aware per-class thresholds calibrated on the validation set
- **Regularization:** Dropout (0.2), Drop Path (0.1), Gradient Clipping (1.0)
- **Reproducibility:** Seed fixed (42) across all random generators

## Classes
| Class | Description |
|-------|-------------|
| Acne | Inflammatory skin condition with pimples and lesions |
| Eczema | Chronic skin condition causing itchy, inflamed patches |
| Eye_Bags | Under-eye bag appearance |
| Wrinkles | Visible facial wrinkles |

## Results

### Current Checkpoint Baseline

Measured on the current 56-image test split before retraining with the latest pipeline changes:

| Metric | Score |
|--------|-------|
| Exact-match Accuracy | 58.93% |
| Macro Precision | 70.16% |
| Macro Recall | 71.82% |
| Macro F1 | 68.52% |

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Image Size | 384×384 |
| Batch Size | 16 |
| Learning Rate | 1e-4 |
| Weight Decay | 0.05 |
| Warmup Epochs | 2 |
| Gradient Clip | 1.0 |
| Drop Rate | 0.2 |
| Drop Path Rate | 0.1 |
| Early Stopping | Patience 8, monitored with validation AUROC |

### Confusion Matrix
![Confusion Matrix](outputs/images/confusion_matrix.png)

### Grad-CAM Visualization
Model's attention areas for disease detection:

![Grad-CAM Result](outputs/images/gradcam_result.png)

## Project Structure
```
├── src/
│   ├── config.py       # Configuration parameters
│   ├── dataset.py      # Dataset loading and transforms
│   ├── model.py        # Model architecture
│   ├── train.py        # Training script
│   ├── evaluate.py     # Evaluation script
│   ├── predict.py      # Prediction script
│   └── gradcam.py      # Grad-CAM visualization
├── export/
│   └── export.py       # CoreML export
├── outputs/
│   ├── model/          # Saved model weights
│   ├── checkpoints/    # Training checkpoints
│   └── images/         # Generated visualizations
└── orchestration_data/
    ├── train/
    ├── val/
    └── test/
```

## Usage

### Training
```bash
cd src
python train.py
```

### Evaluation
```bash
python calibrate.py
python evaluate.py
```

To evaluate the checkpoint selected specifically for Top-1 accuracy:

```bash
python calibrate.py --model-path ../outputs/model/best_top1_model.pth --thresholds-path ../outputs/model/top1_thresholds.json
python evaluate.py --model-path ../outputs/model/best_top1_model.pth --thresholds-path ../outputs/model/top1_thresholds.json
```

### Prediction
```python
from predict import predict

result = predict("path/to/image.jpg")
print(result)
# {'Class': 'Acne', 'Confidence': 0.99, 'Detected': {'Acne': 0.99}}
```

### CoreML Export (for iOS)
```bash
cd export
python export.py
```

## Requirements
- Python 3.9+
- PyTorch 2.0+
- timm
- torchvision
- scikit-learn
- matplotlib
- seaborn
- coremltools (for iOS export)

## Future Work

### 1. Multi-Label Dataset
Current dataset is single-label. Training with true multi-label annotated data would enable simultaneous detection of multiple conditions (e.g., Acne + Eczema).

### 2. On-Device Personalization
Implement user-specific "Healthy" baseline calibration on iOS devices to address domain shift from training dataset.

### 3. Severity Prediction
Add severity levels (mild, moderate, severe) for each detected condition to provide more detailed diagnosis information.

### 4. Temporal Tracking
Enable users to track disease progression over time with photo comparisons.

### 5. Hyperparameter Optimization
Integrate Optuna for automated hyperparameter tuning to potentially improve model performance.

## License
MIT
