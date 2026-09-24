"""
Segmentation metrics computed from numpy binary arrays.

All functions accept arrays of shape (H, W) with values in {0, 1}.
Smoothing epsilon prevents division-by-zero on degenerate predictions.
"""

import numpy as np

EPSILON = 1e-7


def _validate(pred: np.ndarray, gt: np.ndarray):
    """Ensure arrays are binary and the same shape."""
    assert pred.shape == gt.shape, (
        f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}"
    )


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Dice similarity coefficient.
        2 * |pred ∩ gt|
       ─────────────────
        |pred| + |gt|
    """
    _validate(pred, gt)
    intersection = (pred & gt).sum()
    return float(2.0 * intersection / (pred.sum() + gt.sum() + EPSILON))


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Intersection over Union (Jaccard index).
        |pred ∩ gt|
       ─────────────
        |pred ∪ gt|
    """
    _validate(pred, gt)
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(intersection / (union + EPSILON))


def precision_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Precision = TP / (TP + FP)
    """
    _validate(pred, gt)
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    return float(tp / (tp + fp + EPSILON))


def recall_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Recall = TP / (TP + FN)
    """
    _validate(pred, gt)
    tp = (pred & gt).sum()
    fn = (~pred & gt).sum()
    return float(tp / (tp + fn + EPSILON))


def compute_all_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Compute all four metrics and return as a dict.

    pred and gt must be boolean or uint8 arrays with values in {0, 1}.
    """
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    return {
        "dice":      dice_score(pred, gt),
        "iou":       iou_score(pred, gt),
        "precision": precision_score(pred, gt),
        "recall":    recall_score(pred, gt),
    }
