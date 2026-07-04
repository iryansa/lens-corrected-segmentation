# Setup and Installation Guide

This document describes the environment setup, package requirements, and run commands for executing all parts of the calibration, training, and measurement pipeline.

## 1. Environment & Setup
The pipeline is designed to run in a Python 3.13 virtual environment on Windows (and is compatible with Unix-like systems).

### Installation Steps
1. **Clone/Open the repository**:
   Ensure you are in the project root directory: `e:\xis.ai`.

2. **Initialize the Virtual Environment**:
   If not already created, initialize the environment:
   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment**:
   - **PowerShell**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Command Prompt**:
     ```cmd
     .\venv\Scripts\activate.bat
     ```
   - **Bash/Git Bash**:
     ```bash
     source venv/Scripts/activate
     ```

4. **Install Dependencies**:
   Install all package requirements listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

---

## 2. Command Reference

Follow this sequence to run the entire pipeline end-to-end:

### Step 1: Camera Calibration
Processes the 29 unique calibration images, resolves the lens distortion parameters, and outputs `camera_params.json`.
```bash
python calibration/calibrate.py
```
- **Output file**: [camera_params.json](file:///e:/xis.ai/calibration/camera_params.json)
- **Report generated**: [CALIBRATION_REPORT.md](file:///e:/xis.ai/docs/CALIBRATION_REPORT.md)

### Step 2: Synthetic Dataset Generation
Generates the synthetic training, validation, and test datasets of standard playing cards superimposed on the calibration scenes.
```bash
python dataset/generate_dataset.py
```
- **Output directory**: `dataset/train/`, `dataset/val/`, `dataset/test/`
- **Report generated**: [DATASET_CARD.md](file:///e:/xis.ai/docs/DATASET_CARD.md)

### Step 3: Model Training
Trains the PyTorch U-Net ResNet-34 segmentation model on the generated dataset and logs stats to MLflow database.
```bash
python models/train.py
```
- **Output weight file**: `models/best_model.pth` (best checkpoint based on validation loss)
- **MLflow database**: [mlflow.db](file:///e:/xis.ai/mlflow.db)
- **Report generated**: [TRAINING_REPORT.md](file:///e:/xis.ai/docs/TRAINING_REPORT.md)

### Step 4: Segmentation Inference (Demo)
Runs inference on a raw image, applies camera undistortion, runs the trained model, and outputs a visual mask overlay.
```bash
python inference/predict.py --image dataset/test/images/sample_0000.png --output inference/output.png
```
- **Output image**: `inference/output.png`

### Step 5: Metric Measurement (Single Image)
Runs the full measurement pipeline on a specific raw image. Fits a 3D-aligned physical bounding box and displays metrics in millimeters.
```bash
python measurements/measure.py --image dataset/test/images/sample_0000.png
```
- **Output annotated image**: Saved inside `measurements/outputs/`

### Step 6: Measurement Validation (Test Split)
Evaluates the metric measurement pipeline across all 12 test split images and computes MAE/MPE tables.
```bash
python measurements/validate.py
```
- **Report generated**: [accuracy_report.md](file:///e:/xis.ai/measurements/accuracy_report.md) and [MEASUREMENT_REPORT.md](file:///e:/xis.ai/docs/MEASUREMENT_REPORT.md)
