# 🧬 SkinSegAI

### Few-Shot Skin Lesion Segmentation Using Transfer Learning with MobileNetV3-UNet

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-ISIC%202018%20Task%201-green.svg)](https://challenge.isic-archive.com/)

---

## 📌 Project Overview

**SkinSegAI** is a clinical AI research dashboard investigating **low-resource and few-shot medical image segmentation**. Pixel-level annotation of dermoscopic images requires certified dermatologists, making large labeled datasets prohibitively expensive to obtain.

This project examines how well skin lesion segmentation performs when trained on small, annotated subsets:
**50, 100, 250, and 500 images**, utilizing an ImageNet-pretrained **MobileNetV3-Small** encoder fused into a **U-Net** decoder architecture with multi-scale skip connections.

All models were trained on Google Colab with NVIDIA T4 GPUs using a hybrid **Dice Loss + BCEWithLogitsLoss**, and evaluated dynamically inside an interactive **Streamlit dashboard**.

---

## 🔬 Benchmark Results (Validation Metrics)

Extracted directly from the trained checkpoints:

| Configuration | Training Set Size | Best Epoch | Best Val Dice | Best Val IoU (Jaccard) | Performance Rating |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Exp 1** | **50 images** | 15 | `0.5104` | `0.3621` | Baseline Few-Shot |
| **Exp 2** | **100 images** | 15 | `0.5765` | `0.4106` | +6.6% Dice Gain |
| **Exp 3** | **250 images** | 15 | `0.8246` | `0.7081` | Critical Data Threshold (+24.8%) |
| **Exp 4** | **500 images** | 9 | `0.8707` | `0.7752` | Near-Supervised SOTA (`87.07%`) |

> **Key Research Finding:** An exponential performance jump occurs between **100 and 250 images** (Dice jumps from **57.65%** to **82.46%**), proving that with transfer learning, 250 labeled images provide a strong viability threshold for clinical lesion boundary segmentation.

---

## 🏗️ Architecture & Pipeline

```
Input Image (224 × 224 × 3)
         │
         ▼
MobileNetV3-Small Encoder (ImageNet Pretrained)
  ├─ features[0] (112×112, 16 ch) ───────────────► Skip 4 ──┐
  ├─ features[1] (56×56, 16 ch)   ───────────────► Skip 3 ──┼─┐
  ├─ features[2] (28×28, 24 ch)   ───────────────► Skip 2 ──┼─┼─┐
  ├─ features[8] (14×14, 48 ch)   ───────────────► Skip 1 ──┼─┼─┼─┐
  └─ features[12] Bridge (7×7, 576 ch)                      │ │ │ │
         │                                                  │ │ │ │
         ▼                                                  │ │ │ │
ConvTranspose2d (576 → 48, 14×14) + Concat ◄────────────────┘ │ │ │
  └─ ConvBlock1 (96 → 48)                                     │ │ │
         │                                                    │ │ │
         ▼                                                    │ │ │
ConvTranspose2d (48 → 24, 28×28) + Concat ◄───────────────────┘ │ │
  └─ ConvBlock2 (48 → 24)                                       │ │
         │                                                      │ │
         ▼                                                      │ │
ConvTranspose2d (24 → 16, 56×56) + Concat ◄─────────────────────┘ │
  └─ ConvBlock3 (32 → 16)                                         │
         │                                                        │
         ▼                                                        │
ConvTranspose2d (16 → 16, 112×112) + Concat ◄─────────────────────┘
  └─ ConvBlock4 (32 → 16)
         │
         ▼
ConvTranspose2d (16 → 16, 224×224)
         │
         ▼
Final Conv (1×1, 16 → 1 logit)
         │
         ▼
Sigmoid Function (Threshold = 0.5)
         │
         ▼
Binary Lesion Mask (224 × 224 × 1)
```

---

## 📂 Project Structure

```
skin_lesion_dashboard/
│
├── app.py                      # Interactive 6-page Streamlit Dashboard
├── requirements.txt            # Dependency specifications
├── README.md                   # Project documentation
│
├── src/
│   ├── model.py                # MobileNetV3UNet exact PyTorch architecture
│   ├── inference.py            # Preprocessing, forward pass, heatmaps & overlays
│   ├── metrics.py              # Dice, IoU, Precision, Recall calculation
│   ├── preprocessing.py        # ImageNet normalization and 224×224 resizing
│   └── utils.py                # Multi-folder checkpoint discovery and data loader
│
├── models/                     # Trained checkpoint storage (.pth)
│   ├── mobilenetv3_unet_50_best.pth
│   ├── mobilenetv3_unet_100_best.pth
│   ├── mobilenetv3_unet_250_best.pth
│   └── mobilenetv3_unet_500_best.pth
│
├── results/                    # Validation and evaluation CSV
│   └── few_shot_experiment_results.csv
│
└── sample_data/                # Ground truth test pairs (ISIC convention)
    ├── images/                 # e.g., ISIC_0000001.jpg
    └── masks/                  # e.g., ISIC_0000001_segmentation.png
```

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup

```bash
# Clone the repository
git clone https://github.com/Gowthami1214/SkinSegAi.git
cd SkinSegAi

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

The application will launch in your browser at `http://localhost:8501`.

---

## 🖥️ Dashboard Features

1. **🏠 Executive Dashboard**:
   - System status indicators showing active/loaded models.
   - Experiment progression pipeline (50 → 100 → 250 → 500).
   - Live Validation Score Cards displaying real Dice and IoU coefficients.
   - Progress bar comparisons with threshold color coding.

2. **🔬 Experiment Lab**:
   - Per-configuration inspection (hyperparameters, loss, early stopping).
   - Interactive Plotly curves: Data Size vs. Dice and Data Size vs. IoU.
   - Empirical research insights dynamically derived from checkpoint results.

3. **🎯 Segmentation Studio**:
   - Upload any dermoscopic image (PNG/JPG).
   - Select between any of the 4 trained checkpoints (50, 100, 250, 500).
   - 4-panel visualizer: Original Image, Probability Heatmap (turbo/jet), Binary Mask (0.5 threshold), and Lesion Overlay.
   - Prediction statistics: Lesion Area % and inference latency in milliseconds.

4. **📊 Model Comparison**:
   - Validation Score Cards with exact 4-decimal precision.
   - **🔥 Dice Coefficient Matrix Heatmap**: Heatmap displaying metric profiles per model.
   - Side-by-side grouped bar charts and trend curves.
   - Configuration radar fingerprints.

5. **🧪 Test Evaluation**:
   - Pairwise ground-truth evaluation against ISIC test images.
   - Computes empirical Dice, IoU, Precision, and Recall when ground truth masks are provided.

6. **ℹ️ About Project**:
   - Clinical context, architecture diagrams, and medical AI disclaimers.

---

## ⚖️ Clinical Disclaimer

> **Research Prototype:** This application and its models are built solely for academic research and educational exploration of few-shot transfer learning in medical imaging. The segmentation outputs are not clinically certified and must **never** be used for primary diagnosis or direct clinical decision-making. Always consult a licensed dermatologist for skin lesion evaluations.