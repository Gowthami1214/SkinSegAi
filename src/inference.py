"""
Inference helpers — model loading, prediction, and visualization arrays.

All public functions are pure (no Streamlit imports) so they can be
unit-tested independently.
"""

import time
import numpy as np
import torch
from PIL import Image

from src.preprocessing import preprocess_image, INPUT_SIZE
from src.model import load_checkpoint

THRESHOLD = 0.5  # Must match the threshold used during training evaluation


# ---------------------------------------------------------------------------
# Model loading (called inside @st.cache_resource in app.py)
# ---------------------------------------------------------------------------

def load_model(model_path: str, device: torch.device) -> dict:
    """
    Load a MobileNetV3UNet from a .pth checkpoint file.

    Returns:
        {
          "model": MobileNetV3UNet,
          "meta":  dict with optional keys (best_val_dice, best_epoch, …)
        }

    Raises FileNotFoundError if the file doesn't exist.
    """
    return load_checkpoint(model_path, device)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_mask(
    model: torch.nn.Module,
    img: Image.Image,
    device: torch.device,
) -> dict:
    """
    Run segmentation inference on a PIL Image.

    Returns a dict with:
        input_tensor   — (1, 3, H, W) preprocessed tensor
        prob_map       — (H, W) float32 numpy array, sigmoid probability
        binary_mask    — (H, W) uint8 numpy array, values in {0, 1}
        inference_ms   — wall-clock inference time in milliseconds
        lesion_pct     — predicted lesion percentage (0–100)
    """
    tensor = preprocess_image(img).to(device)

    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)                     # [1, 1, H, W]
        probs  = torch.sigmoid(logits)             # [1, 1, H, W]
    t1 = time.perf_counter()

    prob_map    = probs.squeeze().cpu().numpy()           # [H, W]
    binary_mask = (prob_map > THRESHOLD).astype(np.uint8) # {0,1}
    lesion_pct  = float(binary_mask.mean() * 100.0)
    inference_ms = (t1 - t0) * 1000.0

    return {
        "input_tensor":  tensor,
        "prob_map":      prob_map.astype(np.float32),
        "binary_mask":   binary_mask,
        "inference_ms":  inference_ms,
        "lesion_pct":    lesion_pct,
    }


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def prob_map_to_heatmap_rgb(prob_map: np.ndarray) -> np.ndarray:
    """
    Convert a (H, W) float32 probability map to an (H, W, 3) uint8 RGB
    image using a blue-red colormap (low prob = blue, high = red).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    norm = prob_map.clip(0.0, 1.0)
    colored = cm.RdBu_r(norm)          # RGBA in [0, 1]
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    return rgb


def overlay_mask_on_image(
    original_img: Image.Image,
    binary_mask: np.ndarray,
    color: tuple = (255, 70, 70),
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Overlay the binary mask on the original image.

    Returns an (H, W, 3) uint8 numpy array at INPUT_SIZE resolution.
    """
    img_resized = original_img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    img_arr = np.array(img_resized, dtype=np.uint8)

    overlay = img_arr.copy().astype(np.float32)
    mask_region = binary_mask.astype(bool)

    for ch, c in enumerate(color):
        overlay[mask_region, ch] = (
            alpha * c + (1 - alpha) * overlay[mask_region, ch]
        )

    return overlay.astype(np.uint8)


def binary_mask_to_rgb(binary_mask: np.ndarray) -> np.ndarray:
    """
    Convert a {0,1} binary mask to a 3-channel uint8 image.
    Background = near-black, lesion = bright teal.
    """
    h, w = binary_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[binary_mask == 1] = [0, 220, 180]   # teal for lesion
    rgb[binary_mask == 0] = [18,  18, 30]   # dark background
    return rgb
