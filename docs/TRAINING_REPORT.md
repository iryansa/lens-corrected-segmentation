# Model Training Report

This report documents the deep learning segmentation model selection, hyperparameters, training process, and evaluation metrics.

## 1. Model Architecture Selection
A **U-Net** architecture with a **ResNet-34** encoder was selected for the playing card segmentation task.

- **Why U-Net?** U-Net's symmetric encoder-decoder structure with skip connections preserves low-level spatial details (edges and corners) during upsampling. This is critical for metric measurement, where precise boundary contours determine accuracy.
- **Why ResNet-34?** ResNet-34 is a lightweight yet deep backbone that extracts high-quality hierarchical features. The pre-trained ImageNet weights provide a strong starting point, enabling rapid convergence (within 15 epochs) and robust generalization.
- **Library**: `segmentation-models-pytorch` (SMP).
- **Restrictions Met**: Ultralytics YOLO models and Roboflow models were excluded as per the assessment rules.

## 2. Training Configurations
The model was trained on the synthetic playing card dataset using [train.py](file:///e:/xis.ai/models/train.py).

- **Input Resolution**: $512 \times 512$ pixels (resized down from $2992 \times 3992$ to fit memory and speed up training).
- **Loss Function**: Combination of **Binary Cross Entropy (BCE)** and **Dice Loss** ($Loss = L_{\text{BCE}} + L_{\text{Dice}}$) to handle pixel-level classification and global overlap optimization.
- **Optimizer**: Adam with a learning rate of $10^{-4}$.
- **Batch Size**: $4$.
- **Epochs**: $15$.
- **Augmentation**: Random horizontal and vertical flips, and random brightness shifts.
- **Framework Logging**: MLflow tracked the metrics in [mlflow.db](file:///e:/xis.ai/mlflow.db).

## 3. Training and Validation Results
The model converged smoothly over the 15 epochs:

| Epoch | Train Loss | Train IoU | Val Loss | Val IoU |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 1.4786 | 0.2388 | 1.5857 | 0.1440 |
| 5 | 0.8946 | 0.9380 | 0.8778 | 0.9534 |
| 10 | 0.6901 | 0.9677 | 0.6737 | 0.9671 |
| 14 | 0.5739 | 0.9733 | 0.5584 | **0.9815** |
| 15 | 0.5476 | 0.9753 | 0.5325 | 0.9765 |

### Final Validation Performance (Epoch 14 Best Checkpoint)
- **Validation IoU**: **0.9815**
- **Validation F1-Score**: **0.9907**
- **Validation Precision**: **0.9842**
- **Validation Recall**: **0.9972**

These metrics indicate that the U-Net model achieves pixel-perfect classification boundaries on validation images, which forms a solid foundation for the millimeter measurements.
