"""
Utility functions shared across the dashboard pages.

Includes:
  - model file discovery
  - experiment CSV loading
  - test-image/mask pairing
  - device detection
  - numeric formatting helpers
"""

import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR      = Path(__file__).resolve().parent.parent
MODELS_DIR    = ROOT_DIR / "models"
RESULTS_DIR   = ROOT_DIR / "results"
SAMPLE_IMAGES = ROOT_DIR / "sample_data" / "images"
SAMPLE_MASKS  = ROOT_DIR / "sample_data" / "masks"
RESULTS_CSV   = RESULTS_DIR / "few_shot_experiment_results.csv"

# Search these folders for .pth files (models/ takes priority over results/)
MODEL_SEARCH_DIRS = [MODELS_DIR, RESULTS_DIR]

TRAIN_SIZES = [50, 100, 250, 500]

EXPERIMENT_CONFIG = {
    50:   {"encoder": "MobileNetV3-Small", "decoder": "U-Net", "input": "224×224",
           "loss": "Dice + BCE", "max_epochs": 15, "optimizer": "Adam",
           "early_stopping": "Yes (patience=5)"},
    100:  {"encoder": "MobileNetV3-Small", "decoder": "U-Net", "input": "224×224",
           "loss": "Dice + BCE", "max_epochs": 20, "optimizer": "Adam",
           "early_stopping": "Yes (patience=5)"},
    250:  {"encoder": "MobileNetV3-Small", "decoder": "U-Net", "input": "224×224",
           "loss": "Dice + BCE", "max_epochs": 30, "optimizer": "Adam",
           "early_stopping": "Yes (patience=5)"},
    500:  {"encoder": "MobileNetV3-Small", "decoder": "U-Net", "input": "224×224",
           "loss": "Dice + BCE", "max_epochs": 40, "optimizer": "Adam",
           "early_stopping": "Yes (patience=5)"},
}


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def discover_models() -> dict[int, Path]:
    """
    Scan models/ and results/ directories for *_best.pth files.

    Returns a dict mapping train_size (int) to Path.
    models/ takes priority over results/ if both contain the same size.
    """
    found: dict[int, Path] = {}
    pattern = re.compile(r"mobilenetv3_unet_(\d+)_best\.pth", re.IGNORECASE)

    for search_dir in MODEL_SEARCH_DIRS:
        if not search_dir.exists():
            continue
        for f in search_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                size = int(m.group(1))
                if size in TRAIN_SIZES and size not in found:
                    found[size] = f

    return dict(sorted(found.items()))


def model_path(train_size: int) -> Path:
    """
    Return the path to a model checkpoint, searching models/ then results/.
    Falls back to the canonical models/ path if the file doesn't exist anywhere.
    """
    pattern = f"mobilenetv3_unet_{train_size}_best.pth"
    for search_dir in MODEL_SEARCH_DIRS:
        candidate = search_dir / pattern
        if candidate.exists():
            return candidate
    # Default (for display of expected path)
    return MODELS_DIR / pattern


def model_exists(train_size: int) -> bool:
    return model_path(train_size).exists()


# ---------------------------------------------------------------------------
# Experiment results
# ---------------------------------------------------------------------------

@torch.no_grad()
def load_experiment_results() -> Optional[pd.DataFrame]:
    """
    Load the experiment results CSV.

    Returns a DataFrame or None if the file doesn't exist.

    Expected columns (subset is fine):
        train_size, best_epoch, best_val_dice,
        test_dice, test_iou, test_precision, test_recall
    """
    if not RESULTS_CSV.exists():
        return None

    df = pd.read_csv(RESULTS_CSV)
    df = df.sort_values("train_size").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Test image / mask pairing
# ---------------------------------------------------------------------------

def discover_test_pairs() -> list[dict]:
    """
    Scan sample_data/images/ for images and match them to corresponding
    masks in sample_data/masks/ using the ISIC image ID.

    Naming convention:
        image : ISIC_XXXXXXX.jpg  (or .png / .jpeg)
        mask  : ISIC_XXXXXXX_segmentation.png

    Returns a list of dicts:
        [{"id": "ISIC_0000001", "image": Path, "mask": Path}, ...]
    """
    pairs = []
    if not SAMPLE_IMAGES.exists():
        return pairs

    img_exts = {".jpg", ".jpeg", ".png"}
    img_pattern = re.compile(r"(ISIC_\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)

    for img_file in sorted(SAMPLE_IMAGES.iterdir()):
        if img_file.suffix.lower() not in img_exts:
            continue
        m = img_pattern.match(img_file.name)
        if not m:
            continue

        isic_id  = m.group(1)
        mask_file = SAMPLE_MASKS / f"{isic_id}_segmentation.png"

        if mask_file.exists():
            pairs.append({"id": isic_id, "image": img_file, "mask": mask_file})

    return pairs


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def fmt3(value) -> str:
    """Format a float to 3 decimal places."""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "—"


def fmt_pct(value: float) -> str:
    """Format a float as a percentage string."""
    return f"{value:.1f}%"


def status_summary() -> dict:
    """
    Return a summary dict for the dashboard status indicator:
        available_models  — count of existing .pth files
        total_models      — 5
        results_ready     — bool
    """
    available = discover_models()
    return {
        "available_models": len(available),
        "total_models":     len(TRAIN_SIZES),
        "results_ready":    RESULTS_CSV.exists(),
        "model_sizes":      sorted(available.keys()),
    }
