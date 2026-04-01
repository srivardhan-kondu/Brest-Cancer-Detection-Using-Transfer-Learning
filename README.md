# Breast Cancer Image Classification Based on Deep Transfer Learning

> A full-stack web application that classifies breast histopathology image patches as **Benign (IDC−)** or **Malignant (IDC+)** using EfficientNet deep transfer learning, Grad-CAM visual explainability, and LLM-assisted hospital recommendations.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Dataset Information](#2-dataset-information)
3. [Model Architecture](#3-model-architecture)
4. [Training Pipeline (Step by Step)](#4-training-pipeline-step-by-step)
5. [Model Parameters & Hyperparameters](#5-model-parameters--hyperparameters)
6. [Training / Validation / Test Split](#6-training--validation--test-split)
7. [Evaluation & Metrics](#7-evaluation--metrics)
8. [Inference Pipeline](#8-inference-pipeline)
9. [Grad-CAM Explainability](#9-grad-cam-explainability)
10. [Hospital Recommendation Engine](#10-hospital-recommendation-engine)
11. [Project Directory Structure](#11-project-directory-structure)
12. [How to Clone & Run the Project](#12-how-to-clone--run-the-project)
13. [API Endpoints](#13-api-endpoints)
14. [Frontend Overview](#14-frontend-overview)
15. [Retraining the Model (Optional)](#15-retraining-the-model-optional)
16. [Training Script CLI Arguments](#16-training-script-cli-arguments)
17. [Troubleshooting](#17-troubleshooting)
18. [Disclaimer](#18-disclaimer)

---

## 1. What This Project Does

This project is a complete end-to-end system for detecting **Invasive Ductal Carcinoma (IDC)** in breast tissue from histopathology images. It combines three capabilities:

1. **AI-Powered Classification** — An ensemble of EfficientNet models (B0 and B7) fine-tuned on the IDC Histopathology dataset classifies 50×50 pixel tissue patches as benign or malignant and returns calibrated probability scores.

2. **Visual Explainability (Grad-CAM)** — Gradient-weighted Class Activation Maps highlight exactly which tissue regions influenced the model's decision, making the prediction interpretable.

3. **Hospital Recommendation Engine** — After prediction, users can enter their location to receive a ranked list of nearby hospitals (from OpenStreetMap) with optional LLM-generated follow-up guidance.

The system is delivered as a **FastAPI backend** serving a **vanilla JavaScript frontend** with a glassmorphism UI. Pre-trained models are included in the repository — **no retraining is required** to use the application.

---

## 2. Dataset Information

| Property | Details |
|---|---|
| **Name** | IDC (Invasive Ductal Carcinoma) Histopathology Dataset |
| **Source** | [Kaggle — Breast Histopathology Images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) |
| **Total Images** | 277,524 image patches |
| **Image Size** | 50 × 50 pixels (RGB) |
| **Source Slides** | Extracted from 162 whole-mount slide images of breast cancer specimens |
| **Classes** | **Benign (IDC−)** labeled `0` — 198,738 patches |
| | **Malignant (IDC+)** labeled `1` — 78,786 patches |
| **Class Imbalance** | ~71.6% Benign vs ~28.4% Malignant |
| **Format** | PNG files organized in folders: `/<patient_id>/<class>/` where class is `0` or `1` |

### Directory Structure of the Dataset

```
dataset/
├── <patient_id_1>/
│   ├── 0/          ← Benign patches
│   │   ├── img1.png
│   │   ├── img2.png
│   │   └── ...
│   └── 1/          ← Malignant patches
│       ├── img1.png
│       └── ...
├── <patient_id_2>/
│   ├── 0/
│   └── 1/
└── ...
```

The training script automatically downloads this dataset from Kaggle on first run (requires Kaggle API credentials). Once downloaded, it is cached locally.

---

## 3. Model Architecture

The model uses **EfficientNet** as the backbone via **transfer learning** from ImageNet pre-trained weights.

### Base Model: EfficientNet (B0 / B3 / B7)

EfficientNet is a family of convolutional neural networks that uniformly scales depth, width, and resolution using a compound scaling method. The project supports three variants:

| Variant | Parameters | Top-1 ImageNet Acc | Use Case |
|---|---|---|---|
| **EfficientNetB0** | ~5.3M | 77.1% | Fast training, lower memory |
| **EfficientNetB3** | ~12M | 81.6% | Balanced speed/accuracy |
| **EfficientNetB7** | ~66M | 84.3% | Highest accuracy, more compute |

### Custom Classification Head

On top of the frozen/unfrozen EfficientNet backbone, a custom classification head is added:

```
Input Image (224 × 224 × 3)
        ↓
EfficientNet Backbone (ImageNet pre-trained)
        ↓
GlobalAveragePooling2D
        ↓
BatchNormalization
        ↓
Dense(512, activation='swish')    ← 320 for B0/B3, 512 for B7
        ↓
Dropout(0.35)
        ↓
Dense(192, activation='swish')
        ↓
Dropout(0.30)
        ↓
Dense(1, activation='sigmoid')    ← Binary output: probability of malignancy
```

- **Input Preprocessing**: Images are resized to 224×224 and normalized to [0, 1] range.
- **Output**: A single sigmoid value representing P(malignant).

### Ensemble Inference

At inference time, all available models (B0, B3, B7) are loaded and their predictions are **averaged** to produce the final probability:

```
Final Probability = (P_B0 + P_B3 + P_B7) / N_models
```

If a model file is missing, it is simply excluded from the ensemble.

---

## 4. Training Pipeline (Step by Step)

Training follows a **3-phase progressive unfreezing** strategy:

### Phase 1: Head Training (Frozen Backbone)

| Setting | Value |
|---|---|
| Backbone layers | **All frozen** (0 layers trainable) |
| Learning rate | `1e-3` |
| Loss function | Binary Cross-Entropy (label smoothing = 0.02) |
| Epochs | 2 (default) |
| Early stopping patience | 4 epochs |
| Purpose | Train only the classification head to learn task-specific features without disturbing ImageNet weights |

### Phase 2: Partial Fine-Tuning

| Setting | Value |
|---|---|
| Backbone layers | **Top 40 layers unfrozen** |
| Learning rate | `2e-4` |
| Loss function | **Binary Focal Loss** (γ=2.0, α=0.35) |
| Epochs | 3 (default) |
| Early stopping patience | 5 epochs |
| Purpose | Fine-tune upper backbone layers to adapt high-level features to histopathology domain |

### Phase 3: Deep Fine-Tuning

| Setting | Value |
|---|---|
| Backbone layers | **Top 80 layers unfrozen** |
| Learning rate | `7e-5` |
| Loss function | **Binary Focal Loss** (γ=2.0, α=0.35) |
| Epochs | 4 (default) |
| Early stopping patience | 6 epochs |
| Purpose | Deep fine-tuning of mid-level features for maximum accuracy |

### Why 3 Phases?

1. **Phase 1** prevents catastrophic forgetting — the backbone retains ImageNet features while the head learns the new task.
2. **Phase 2** allows gradual adaptation of high-level features using a smaller learning rate.
3. **Phase 3** pushes accuracy further by fine-tuning deeper layers with an even smaller learning rate.

### Data Augmentation

During training, the following augmentations are applied:

| Augmentation | Details |
|---|---|
| Horizontal Flip | Random left-right flip |
| Vertical Flip | Random up-down flip |
| Brightness | ±0.25 max delta |
| Contrast | Range [0.75, 1.30] |
| Saturation | Range [0.75, 1.30] |
| Hue | ±0.04 max delta |
| Rotation | Random 90° rotations (0°, 90°, 180°, 270°) |
| MixUp | Optional (α=0.2) — blends pairs of images and labels |

### Loss Functions Used

**Binary Cross-Entropy with Label Smoothing** (Phase 1):

Standard BCE loss with label smoothing of 0.02 to prevent overconfident predictions.

**Binary Focal Loss** (Phases 2 & 3):

Focal Loss down-weights well-classified examples and focuses on hard negatives:
- γ = 2.0 — focuses learning on hard-to-classify samples
- α = 0.35 — balances importance between classes
- This helps the model pay more attention to difficult malignant cases

### Class Weighting

To handle the 71.6% / 28.4% class imbalance, class weights are computed as:

```
weight_class = N_total / (2 × N_class)
```

This gives higher weight to the minority class (malignant) during training.

---

## 5. Model Parameters & Hyperparameters

### Architecture Parameters

| Parameter | Value |
|---|---|
| Input size | 224 × 224 × 3 |
| Backbone | EfficientNetB0 / B3 / B7 |
| Pre-trained weights | ImageNet |
| Hidden layer 1 | 512 units (B7) or 320 units (B0/B3), Swish activation |
| Hidden layer 2 | 192 units, Swish activation |
| Dropout rate (Phase 1-2) | 0.35 |
| Dropout rate (Phase 3) | 0.30 |
| Output | 1 unit, Sigmoid activation |

### Training Hyperparameters

| Hyperparameter | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| Learning rate | 1e-3 | 2e-4 | 7e-5 |
| Optimizer | Adam | Adam | Adam |
| Batch size | 64 | 64 | 64 |
| Epochs | 2 | 3 | 4 |
| Loss | BCE (smooth=0.02) | Focal (γ=2, α=0.35) | Focal (γ=2, α=0.35) |
| Unfrozen layers | 0 | 40 | 80 |
| Early stopping | val_auc, patience=4 | val_auc, patience=5 | val_auc, patience=6 |
| LR reduction | val_loss, factor=0.3 | val_loss, factor=0.3 | val_loss, factor=0.3 |

### Other Settings

| Setting | Value |
|---|---|
| Random seed | 42 |
| Data fraction (default) | 35% of full dataset |
| Mixed precision | Enabled automatically on GPU |
| Checkpoint metric | Validation AUC (save best only) |

---

## 6. Training / Validation / Test Split

The dataset is split using stratified sampling to maintain class proportions:

```
Full Dataset (277,524 images or fraction thereof)
        ↓
┌───────────────────────────────────────┐
│          80% Training Set             │
└───────────────────────────────────────┘
        ↓
┌───────────────────┬───────────────────┐
│   10% Validation  │    10% Test       │
└───────────────────┴───────────────────┘
```

| Split | Percentage | Purpose |
|---|---|---|
| **Training** | 80% | Model learns from this data (with augmentation) |
| **Validation** | 10% | Used for early stopping, LR scheduling, and threshold tuning |
| **Test** | 10% | Final held-out evaluation — never seen during training |

- All splits are **stratified** — the benign/malignant ratio is preserved in each split.
- The `--data-fraction` flag controls what percentage of the full dataset is used (default: 35%).

---

## 7. Evaluation & Metrics

### Test Set Results (EfficientNet-B7)

Model evaluated on **9,714 held-out test samples** (35% data fraction, 80/10/10 split). Optimal classification threshold (**0.4627**) selected by maximizing F1 on the validation set.

| Metric | Score |
|---|---|
| **Accuracy** | **95.64%** |
| **AUC-ROC** | **0.9912** |
| **Macro F1** | **0.9533** |
| **Weighted F1** | **0.9563** |
| **Optimal Threshold** | 0.4627 |

### Per-Class Breakdown

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Benign (IDC−)** | 0.9641 | 0.9663 | 0.9652 | 6,083 |
| **Malignant (IDC+)** | 0.9433 | 0.9397 | 0.9415 | 3,631 |

### Confusion Matrix

|  | Predicted Benign | Predicted Malignant |
|---|---|---|
| **Actual Benign** | 5,878 (96.6%) | 205 (3.4%) |
| **Actual Malignant** | 219 (6.0%) | 3,412 (94.0%) |

<p align="center">
  <img src="backend/saved_model/confusion_matrix.png" alt="Confusion Matrix" width="480">
</p>

### Training History

Accuracy, Loss, and AUC-ROC curves across the 3-phase training pipeline (9 epochs total):

<p align="center">
  <img src="backend/saved_model/training_history.png" alt="Training History" width="900">
</p>

### Per-Class Metrics Comparison

<p align="center">
  <img src="backend/saved_model/class_metrics.png" alt="Per-Class Metrics" width="600">
</p>

### Metrics Tracked During Training

| Metric | Description |
|---|---|
| **Loss** | BCE or Focal Loss value |
| **Accuracy** | Percentage of correct predictions |
| **AUC** | Area Under the ROC Curve — main checkpoint metric |
| **Precision** | True Positives / (True Positives + False Positives) |
| **Recall** | True Positives / (True Positives + False Negatives) |

### Post-Training Evaluation Pipeline

After all three training phases:

1. **Optimal Threshold Selection** — The validation set is used to find the probability threshold that maximizes F1 score (instead of using the default 0.5).

2. **Test Set Evaluation** — The model is evaluated on the held-out test set using the optimal threshold.

3. **Outputs Generated**:
   - `saved_model/breast_cancer_model.keras` — Best model checkpoint
   - `saved_model/metrics_report.json` — JSON with all metrics (F1, accuracy, precision/recall per class, confusion matrix)
   - `saved_model/confusion_matrix.png` — Visual confusion matrix heatmap
   - `saved_model/training_history.png` — Accuracy, Loss, AUC curves over epochs
   - `saved_model/class_metrics.png` — Per-class precision/recall/F1 bar chart

### Callbacks Used

| Callback | Configuration |
|---|---|
| **EarlyStopping** | Monitor `val_auc`, restore best weights |
| **ReduceLROnPlateau** | Monitor `val_loss`, reduce by factor 0.3, min LR = 1e-7 |
| **ModelCheckpoint** | Save best model by `val_auc` |
| **TensorBoard** | Logs to `backend/logs/<phase_name>/` |
| **CSVLogger** | Training history to `backend/logs/<phase_name>_history.csv` |

---

## 8. Inference Pipeline

When a user uploads an image, this is what happens step by step:

```
User uploads image (JPEG/PNG/BMP/TIFF/WebP, max 20MB)
        ↓
FastAPI validates file type and size
        ↓
Image bytes → PIL Image → Resize to 224×224 → Normalize to [0,1]
        ↓
Run through each loaded model (B0, B7)
        ↓
Average the sigmoid outputs → Final probability
        ↓
Apply threshold (default 0.5) → "Benign" or "Malignant"
        ↓
Optionally generate Grad-CAM heatmap (uses B7 > B3 > B0 preference)
        ↓
Return JSON response with prediction, confidence, probabilities, heatmap
```

### Response Format

```json
{
  "prediction": "Malignant",
  "confidence": 0.9234,
  "idc_positive_prob": 0.9234,
  "idc_negative_prob": 0.0766,
  "is_malignant": true,
  "model_predictions": {
    "B0": 0.9112,
    "B7": 0.9356
  },
  "ensemble_method": "average_probability",
  "img_size_used": "224x224",
  "threshold_used": 0.5,
  "gradcam_overlay_base64": "<base64 PNG>",
  "gradcam_heatmap_base64": "<base64 PNG>",
  "filename": "slide_001.png"
}
```

---

## 9. Grad-CAM Explainability

**Gradient-weighted Class Activation Mapping (Grad-CAM)** provides visual explanations for what the model "sees."

### How It Works

1. A forward pass computes the output prediction and the feature maps of the last convolutional layer.
2. Gradients of the output with respect to those feature maps are computed via backpropagation.
3. The gradients are globally average-pooled to get importance weights for each feature map channel.
4. A weighted combination of the feature maps is computed and passed through ReLU.
5. The resulting heatmap is resized to the input image dimensions and overlaid.

### Output

- **Heatmap Overlay** — The Grad-CAM heatmap blended (50% opacity) on top of the original image, showing which regions drove the prediction.
- **Standalone Heatmap** — A jet-colormap visualization of the raw activation intensities.

Red/warm areas = high model attention (most influential for prediction).
Blue/cool areas = low model attention.

---

## 10. Hospital Recommendation Engine

After receiving a prediction, users can enter their location to get nearby hospital recommendations.

### How It Works

1. **Geocoding** — The user's text location (e.g., "Hyderabad, Telangana") is resolved to latitude/longitude using the Nominatim geocoding API.

2. **Hospital Search** — The Overpass API queries OpenStreetMap for hospitals within a configurable radius (default: 35 km, max: 200 km). Up to 8 hospitals are returned, sorted by distance.

3. **LLM Summary** — The hospital list and diagnosis are sent to an LLM for a concise next-step recommendation:
   - **Primary**: OpenAI-compatible API (configured via `LLM_API_KEY`, `LLM_API_BASE`, `LLM_MODEL` env vars)
   - **Fallback 1**: Local Ollama (configured via `OLLAMA_MODEL` env var, default: `llama3.1:8b`)
   - **Fallback 2**: Deterministic template-based summary (always works, no API needed)

---

## 11. Project Directory Structure

```
Breast Cancer Detection/
├── backend/
│   ├── main.py                    # FastAPI server, API routes, static file serving
│   ├── model.py                   # Model loading, preprocessing, ensemble inference, Grad-CAM
│   ├── train.py                   # Complete 3-phase training pipeline
│   ├── hospital_recommender.py    # Geocoding, hospital search, LLM summarization
│   ├── dummy_model_gen.py         # Utility to generate lightweight dummy models for testing
│   ├── requirements.txt           # Python dependencies
│   ├── generate_metrics.py         # Generate evaluation metrics, plots, and confusion matrix
│   └── saved_model/               # Pre-trained models & evaluation outputs (included in repo)
│       ├── model_B0.keras         # Trained EfficientNetB0 weights
│       ├── model_B7.keras         # Trained EfficientNetB7 weights
│       ├── metrics_report.json    # Full evaluation metrics in JSON format
│       ├── confusion_matrix.png   # Confusion matrix heatmap
│       ├── training_history.png   # Training curves (accuracy, loss, AUC)
│       └── class_metrics.png      # Per-class precision/recall/F1 chart
├── frontend/
│   ├── index.html                 # Main UI — upload, results, info sections
│   ├── styles.css                 # Complete design system (glassmorphism, animations, responsive)
│   └── app.js                     # Client-side logic (file handling, API calls, rendering)
├── run.sh                         # Shell script to activate venv and start the server
├── .gitignore                     # Excludes venv, __pycache__, logs, .env
└── README.md                      # This file
```

### File Responsibilities

| File | Purpose |
|---|---|
| `main.py` | FastAPI application with `/predict`, `/health`, `/model-info`, `/recommend-hospitals` endpoints. Serves the frontend as static files. |
| `model.py` | Loads all `.keras` models into memory, runs ensemble prediction, generates Grad-CAM heatmaps. |
| `train.py` | Downloads dataset from Kaggle, runs 3-phase transfer learning, evaluates on test set, saves model + metrics. |
| `hospital_recommender.py` | Geocodes location via Nominatim, queries OpenStreetMap for hospitals, generates LLM summary. |
| `dummy_model_gen.py` | Creates small dummy `.keras` files for testing without real training. |
| `generate_metrics.py` | Generates evaluation charts (confusion matrix, training curves, per-class metrics) and `metrics_report.json`. |
| `app.js` | Handles drag-drop upload, calls `/predict` API, renders results with animations. |
| `index.html` | Full UI with upload zone, results panel, Grad-CAM display, hospital recommendation form. |
| `styles.css` | Glassmorphism design system with dark theme, animations, and responsive layout. |

---

## 12. How to Clone & Run the Project

### Prerequisites

- **Python 3.11+** (recommended)
- **pip** package manager
- **Git**
- **Internet connection** on first run (downloads model weights automatically, ~15 MB)

### Step 1: Clone the Repository

```bash
git clone https://github.com/srivardhan-kondu/Brest-Cancer-Detection-Using-Transfer-Learning.git
cd Brest-Cancer-Detection-Using-Transfer-Learning
```

### Step 2: Create a Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r backend/requirements.txt
```

> **macOS (Apple Silicon):** also run `pip install tensorflow-metal` for GPU acceleration.

#### Key Dependencies

| Package | Purpose |
|---|---|
| `fastapi` + `uvicorn` | Backend web framework & ASGI server |
| `tensorflow ≥2.13` | Deep learning inference & Grad-CAM |
| `pillow`, `numpy` | Image processing |
| `scikit-learn` | Metrics & thresholding |
| `huggingface_hub` | Auto-download model weights on first run |
| `python-multipart` | File upload handling |

### Step 4: Run the Application

**Option A — Using the run script (recommended):**

```bash
bash run.sh
```

**Option B — Manually:**

```bash
source venv/bin/activate
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Step 5: Open in Browser

- **Application**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### What Happens on First Run

**No model files need to be downloaded manually.** When the server starts for the first time:

1. **Grad-CAM model** — A DenseNet121 model is automatically generated locally (uses pretrained ImageNet weights, takes ~30 seconds on first start only). Saved to `backend/saved_model/model_DenseNet121.keras`.
2. **Analysis model** — The prediction engine weights (~15 MB) are automatically downloaded from HuggingFace and cached to `backend/saved_model/_hf_cache.keras`. Requires a one-time internet connection.

Subsequent startups load both from local cache and are fast.

> **To retrain on the full IDC dataset yourself**, see [Section 15](#15-retraining-the-model-optional).

---

## 13. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check — returns server status and whether models are loaded |
| `GET` | `/model-info` | Returns model metadata (architecture, parameters, input shape) |
| `POST` | `/predict` | Upload an image for classification. Query param: `include_heatmap=true` for Grad-CAM |
| `POST` | `/recommend-hospitals` | Send location + diagnosis to get nearby hospitals and LLM recommendation |
| `GET` | `/docs` | Interactive Swagger API documentation |
| `GET` | `/` | Serves the frontend UI |

### Example: Prediction API Call

```bash
curl -X POST "http://localhost:8000/predict?include_heatmap=true" \
  -F "file=@sample_patch.png"
```

### Example: Hospital Recommendation

```bash
curl -X POST "http://localhost:8000/recommend-hospitals" \
  -H "Content-Type: application/json" \
  -d '{"location": "Hyderabad, Telangana", "diagnosis": "Malignant", "radius_km": 35}'
```

---

## 14. Frontend Overview

The frontend is a single-page application built with vanilla HTML, CSS, and JavaScript — no build tools or framework dependencies.

### Features

- **Drag & Drop Upload** — Drag histopathology images or click to browse
- **Real-time Preview** — See the uploaded image with file details before analysis
- **Animated Results** — Confidence bars, probability breakdowns with smooth animations
- **Grad-CAM Display** — Side-by-side overlay and standalone heatmap visualization
- **Hospital Finder** — Enter location, get nearby hospitals with LLM-generated guidance
- **Model Status Badge** — Live indicator showing if the backend model is loaded
- **Responsive Design** — Works on desktop, tablet, and mobile
- **Accessibility** — ARIA labels, keyboard navigation, semantic HTML

### Supported Image Formats

JPEG, PNG, BMP, TIFF, WebP (max 20 MB)

---

## 15. Retraining the Model (Optional)

You **do not need** to retrain — pre-trained models are included. But if you want to train from scratch or with different settings:

### Setup Kaggle Credentials

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → Create New Token
2. Place the downloaded `kaggle.json` at `~/.kaggle/kaggle.json`
3. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Run Training

```bash
cd backend
python train.py
```

This will:
1. Download the IDC dataset (~1.6 GB) from Kaggle
2. Run 3-phase transfer learning (Phase 1 → Phase 2 → Phase 3)
3. Evaluate on the test set and find the optimal threshold
4. Save the trained model to `saved_model/breast_cancer_model.keras`
5. Save metrics to `saved_model/metrics_report.json`
6. Save confusion matrix to `saved_model/confusion_matrix.png`

### Train with a Specific Backbone

```bash
python train.py --backbone EfficientNetB7 --batch-size 32
python train.py --backbone EfficientNetB0 --batch-size 64
```

### Train on Full Dataset

```bash
python train.py --data-fraction 1.0
```

---

## 16. Training Script CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `""` | Path to extracted IDC dataset root (auto-downloads if empty) |
| `--backbone` | `EfficientNetB0` | Backbone architecture: `EfficientNetB0`, `EfficientNetB3`, or `EfficientNetB7` |
| `--data-fraction` | `0.35` | Fraction of dataset to use (0.1 to 1.0). Lower = faster training. |
| `--batch-size` | `64` | Training batch size. Reduce if running out of memory. |
| `--epochs-head` | `2` | Epochs for Phase 1 (frozen backbone) |
| `--epochs-finetune-1` | `3` | Epochs for Phase 2 (partial fine-tuning) |
| `--epochs-finetune-2` | `4` | Epochs for Phase 3 (deep fine-tuning) |
| `--unfreeze-phase2` | `40` | Number of backbone layers to unfreeze in Phase 2 |
| `--unfreeze-phase3` | `80` | Number of backbone layers to unfreeze in Phase 3 |
| `--enable-mixup` | `false` | Enable MixUp regularization (slower but can improve generalization) |

### Examples

```bash
# Quick test run with small subset
python train.py --backbone EfficientNetB0 --data-fraction 0.1 --batch-size 128

# Full high-accuracy training
python train.py --backbone EfficientNetB7 --data-fraction 1.0 --batch-size 32 --enable-mixup

# Custom phase epochs
python train.py --epochs-head 5 --epochs-finetune-1 8 --epochs-finetune-2 10
```

---

## 17. Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` on import | Make sure virtual environment is activated: `source venv/bin/activate` |
| TensorFlow installation fails | Use Python 3.11. On macOS Apple Silicon, `tensorflow-metal` is installed automatically. |
| "No trained models found" at startup | Pre-trained models should be in `backend/saved_model/`. If missing, run `python train.py` to regenerate. |
| Training is too slow | Use `--backbone EfficientNetB0`, increase `--batch-size`, or reduce `--data-fraction` |
| Out of memory during training | Reduce `--batch-size` (try 16 or 32). Ensure mixed precision is enabled (automatic on GPU). |
| Hospital recommendations fail | LLM features need `LLM_API_KEY` env var or local Ollama. The deterministic fallback always works without any API. |
| Frontend not loading | Ensure the server is started from the `backend/` directory (or use `run.sh`). |
| GPU not detected | Verify CUDA/cuDNN installation. Check with: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"` |
| Port 8000 already in use | Kill the existing process or use a different port: `uvicorn main:app --port 8080` |

---

## 18. Disclaimer

**This tool is for research and educational purposes only.**

- It is **not** FDA-approved and is **not** a medical device.
- It is **not** a substitute for professional medical advice, diagnosis, or treatment.
- Always consult a qualified pathologist or healthcare provider for clinical decisions.
- If symptoms are severe or rapidly worsening, seek emergency care immediately.

---

Built with EfficientNet · FastAPI · TensorFlow · Grad-CAM