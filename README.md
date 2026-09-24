# 🧬 SkinSeg AI
### Few-Shot Skin Lesion Segmentation Dashboard
**MobileNetV3-Small + U-Net | Research Prototype**

---

## Project Overview

SkinSeg AI is a Streamlit-based AI research dashboard that visualizes and evaluates a
**few-shot skin lesion segmentation** study. The project investigates how model performance
changes as the number of labeled training images increases from **50 to 1000**, using a
MobileNetV3-Small encoder combined with a U-Net decoder.

Training is performed separately on Google Colab (NVIDIA T4 GPU). This dashboard is designed
to be built **before training finishes** and automatically activates when model files and
results are placed in the correct locations.

---

## Architecture

```
Input Image (224×224×3)
        │
        ▼
MobileNetV3-Small Encoder  ──── Skip Connections (4 levels) ────┐
        │                                                         │
        ▼                                                         │
Bottleneck (576→256 ch)                                          │
        │                                                         │
        ▼                                                         │
U-Net Decoder ◄───────────────────────────────────────────────────┘
        │
        ▼
Sigmoid → Threshold 0.5
        │
        ▼
Binary Lesion Mask (224×224×1)
```

| Component        | Detail                                    |
|------------------|-------------------------------------------|
| Encoder          | MobileNetV3-Small (ImageNet pretrained)   |
| Decoder          | U-Net with 4 skip-connection stages       |
| Loss             | Dice Loss + BCEWithLogitsLoss             |
| Input resolution | 224 × 224                                 |
| Threshold        | 0.5 on sigmoid output                     |
| Output           | 1-channel binary segmentation mask        |

---

## Folder Structure

```
skin_lesion_dashboard/
│
├── app.py                          ← Main Streamlit entry point
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── model.py                    ← MobileNetV3UNet class (exact training arch)
│   ├── inference.py                ← Prediction + visualization helpers
│   ├── metrics.py                  ← Dice, IoU, Precision, Recall
│   ├── preprocessing.py            ← Training-identical transforms
│   └── utils.py                    ← Model discovery, CSV loading, formatting
│
├── models/                         ← Place trained .pth files here
│   ├── mobilenetv3_unet_50_best.pth
│   ├── mobilenetv3_unet_100_best.pth
│   ├── mobilenetv3_unet_250_best.pth
│   ├── mobilenetv3_unet_500_best.pth
│   └── mobilenetv3_unet_1000_best.pth
│
├── results/                        ← Place experiment CSV here
│   └── few_shot_experiment_results.csv
│
└── sample_data/
    ├── images/                     ← ISIC_XXXXXXX.jpg
    └── masks/                      ← ISIC_XXXXXXX_segmentation.png
```

---

## Installation

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

> **Note:** If you have a CUDA-enabled GPU, install the CUDA build of PyTorch from
> https://pytorch.org/get-started/locally/ before running `pip install -r requirements.txt`.

---

## Running the Dashboard

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501` and works immediately — even without any
trained model files. Missing models are shown with a friendly placeholder message.

---

## Adding Trained Models

After training completes on Colab, download the `.pth` checkpoint files and place them in
the `models/` folder using the **exact filenames**:

| Training Size | Expected Filename                        |
|---------------|------------------------------------------|
| 50 images     | `mobilenetv3_unet_50_best.pth`           |
| 100 images    | `mobilenetv3_unet_100_best.pth`          |
| 250 images    | `mobilenetv3_unet_250_best.pth`          |
| 500 images    | `mobilenetv3_unet_500_best.pth`          |
| 1000 images   | `mobilenetv3_unet_1000_best.pth`         |

The dashboard automatically detects which files are present on every page load.

### Supported checkpoint formats

```python
# Format A — state dict only
torch.save(model.state_dict(), path)

# Format B — dict with metadata
torch.save({
    "model_state_dict": model.state_dict(),
    "best_val_dice": ...,
    "best_epoch": ...,
}, path)
```

Both formats are handled automatically. If Format B is used, checkpoint metadata
(best validation Dice, best epoch) is displayed inside the Segmentation Studio.

---

## Adding Experiment Results

Place the training results CSV at:

```
results/few_shot_experiment_results.csv
```

Expected columns (any subset is acceptable; the dashboard auto-detects available columns):

| Column           | Description                              |
|------------------|------------------------------------------|
| `train_size`     | Number of training images (required)     |
| `best_epoch`     | Epoch at which best validation Dice occurred |
| `best_val_dice`  | Best validation Dice coefficient         |
| `test_dice`      | Test-set Dice coefficient                |
| `test_iou`       | Test-set IoU (Jaccard index)             |
| `test_precision` | Test-set Precision                       |
| `test_recall`    | Test-set Recall                          |

Example CSV:

```csv
train_size,best_epoch,best_val_dice,test_dice,test_iou,test_precision,test_recall
50,8,0.623,0.601,0.441,0.675,0.553
100,12,0.712,0.698,0.541,0.743,0.662
250,18,0.781,0.769,0.628,0.812,0.731
500,23,0.824,0.817,0.697,0.849,0.789
1000,31,0.858,0.851,0.742,0.877,0.828
```

---

## Adding Test Images and Masks

Place dermoscopic images in `sample_data/images/` and their ground-truth masks in
`sample_data/masks/` using ISIC naming convention:

```
sample_data/
├── images/
│   ├── ISIC_0000001.jpg
│   └── ISIC_0000002.jpg
└── masks/
    ├── ISIC_0000001_segmentation.png
    └── ISIC_0000002_segmentation.png
```

> Images and masks are matched by ISIC ID, **not** by directory order.

Masks must be grayscale PNG files where:
- **White (255)** = lesion region
- **Black (0)** = background

---

## Dashboard Pages

| Page                  | Purpose                                                         |
|-----------------------|-----------------------------------------------------------------|
| **Dashboard**         | Overview, KPIs, experiment progression, results summary         |
| **Experiment Lab**    | Per-experiment config, performance charts, research insights    |
| **Segmentation Studio** | Upload any image, run prediction, 4-panel visualization       |
| **Model Comparison**  | Multi-metric line and radar charts across all configurations    |
| **Test Evaluation**   | Ground-truth comparison: Dice, IoU, Precision, Recall          |
| **About Project**     | Research background, architecture diagram, component guide     |

---

## Missing Models — Graceful Degradation

The dashboard never crashes due to a missing file. Instead:

- **Missing model**: shows the expected filename and a warning
- **Missing CSV**: shows a placeholder explaining where to place the file
- **No test pairs**: explains the naming convention required
- **Invalid upload**: shows a clear error message

Once files are added, simply refresh the page — no code changes required.

---

## Test Evaluation vs Arbitrary Upload

| Feature                    | Test Evaluation (sample_data/) | Segmentation Studio (upload) |
|----------------------------|---------------------------------|------------------------------|
| Ground-truth mask          | ✅ Yes                          | ❌ No                         |
| Dice, IoU, Precision, Recall | ✅ Computed from GT            | ❌ Not calculated             |
| Lesion pixel percentage    | ✅ Yes                          | ✅ Yes                        |
| Probability heatmap        | ❌ Not shown                    | ✅ Yes                        |
| Inference time             | ✅ Yes                          | ✅ Yes                        |

Metrics are **never fabricated** for images without ground truth.

---

## Disclaimer

> This is a **research prototype** built for academic demonstration purposes.
> Model predictions are not clinically validated and must not be used for medical diagnosis.
> Always consult a qualified healthcare professional for skin lesion concerns.
#   S k i n G e n A I  
 