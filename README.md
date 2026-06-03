# Skin Disease Classification Model

A deep learning model for classifying skin diseases using ConvNeXt-Tiny architecture with multi-label prediction support.

## Authors
- Berat Kerem Öztopuz

## Model Architecture
- **Base Model:** ConvNeXt-Tiny (pretrained on ImageNet)
- **Input Size:** 384×384
- **Output:** 5 classes with Sigmoid activation (multi-label support)
- **Loss Function:** BCEWithLogitsLoss with label smoothing (0.05)
- **Optimizer:** AdamW (weight_decay=0.05) with Warmup + CosineAnnealing scheduler
- **Regularization:** Dropout (0.2), Drop Path (0.1), Gradient Clipping (1.0)
- **Reproducibility:** Seed fixed (42) across all random generators

## Classes
| Class | Description |
|-------|-------------|
| Acne | Inflammatory skin condition with pimples and lesions |
| Eczema | Chronic skin condition causing itchy, inflamed patches |
| Psoriasis | Autoimmune disease causing scaly, red skin patches |
| Ben_Lezyon | Benign skin lesions and moles |
| Healthy | Normal, healthy skin without conditions |

## Results

### Performance Metrics
| Metric | Score |
|--------|-------|
| Accuracy | 94.49% |
| Precision | 94.56% |
| Recall | 96.16% |
| F1 Score | 95.33% |

### Per-Class Performance
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Acne | 93% | 99% | 96% |
| Eczema | 93% | 90% | 92% |
| Psoriasis | 90% | 94% | 92% |
| Ben_Lezyon | 96% | 99% | 97% |
| Healthy | 100% | 100% | 100% |

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Image Size | 384×384 |
| Batch Size | 32 |
| Learning Rate | 1e-4 |
| Weight Decay | 0.05 |
| Warmup Epochs | 5 |
| Gradient Clip | 1.0 |
| Label Smoothing | 0.05 |
| Drop Rate | 0.2 |
| Drop Path Rate | 0.1 |
| Early Stopping | Patience 5 |

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
└── FINAL_SPLIT/
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
python evaluate.py
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
