# Skin Disease Classification Model

A deep learning model for classifying skin diseases using ConvNeXt-Tiny architecture with multi-label prediction support.

## Authors
- Berat Kerem Öztopuz
- Zeynep Aslan

## Model Architecture
- **Base Model:** ConvNeXt-Tiny (pretrained on ImageNet)
- **Output:** 5 classes with Sigmoid activation (multi-label support)
- **Loss Function:** BCEWithLogitsLoss
- **Optimizer:** AdamW with CosineAnnealingLR

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
| Accuracy | 94.00% |
| Precision | 95.06% |
| Recall | 94.54% |
| F1 Score | 94.78% |

### Per-Class Performance
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Acne | 93% | 97% | 95% |
| Eczema | 89% | 87% | 88% |
| Psoriasis | 93% | 91% | 92% |
| Ben_Lezyon | 99% | 97% | 98% |
| Healthy | 100% | 99% | 100% |

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

## Acknowledgments
- CelebA Dataset for healthy skin images
- DermNet for disease images
- ISIC Archive for dermoscopy images
- Fitzpatrick17k for diverse skin tone representation
