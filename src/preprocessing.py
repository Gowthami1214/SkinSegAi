"""
Preprocessing utilities — identical to the transforms applied during training.

Do NOT add augmentation here.  Inference must use the exact same
normalization and resize that training used.
"""

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF


# ImageNet statistics used during training
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
INPUT_SIZE    = 224


def load_image_rgb(source) -> Image.Image:
    """
    Load an image and return it as an RGB PIL Image.

    source can be:
        • A file path (str)
        • A file-like object (e.g. Streamlit UploadedFile)
    """
    img = Image.open(source)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def preprocess_image(img: Image.Image) -> torch.Tensor:
    """
    Apply the exact inference pre-processing pipeline:

        1. Resize to INPUT_SIZE × INPUT_SIZE
        2. Convert to float tensor  [0, 1]
        3. Normalize with ImageNet mean/std

    Returns a (1, 3, H, W) tensor ready for the model.
    """
    img_resized = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    tensor = TF.to_tensor(img_resized)                         # [3, H, W] in [0,1]
    tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD) # normalize
    return tensor.unsqueeze(0)                                 # [1, 3, H, W]


def load_mask_binary(source, threshold: int = 127) -> np.ndarray:
    """
    Load a ground-truth mask and return a binary uint8 numpy array
    (values 0 or 1) at INPUT_SIZE × INPUT_SIZE.

    Works with ISIC-style masks that are grayscale PNGs where
    the lesion region is white (255) and background is black (0).
    """
    mask = Image.open(source).convert("L")
    mask = mask.resize((INPUT_SIZE, INPUT_SIZE), Image.NEAREST)
    arr  = np.array(mask, dtype=np.uint8)
    return (arr > threshold).astype(np.uint8)
