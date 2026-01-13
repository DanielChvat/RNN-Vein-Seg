import numpy as np
import torch


# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def dice_per_class(pred: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6):
    """
    pred: (N,H,W) int64
    target: (N,H,W) int64
    returns: list of dice for each class [C]
    """
    dices = []
    for c in range(num_classes):
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum().float()
        denom = p.sum().float() + t.sum().float()
        d = (2 * inter + eps) / (denom + eps)
        dices.append(d.item())
    return dices


# -------------------------
# Visualization helpers
# -------------------------
def _to_01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.min()
    return x / (x.max() + eps)


def mask_to_rgb(mask: np.ndarray, palette=None) -> np.ndarray:
    """
    mask: (H,W) int
    returns: (H,W,3) uint8
    """
    if palette is None:
        # background, class1, class2 ... (edit if you want)
        palette = [
            (0, 0, 0),        # 0
            (0, 255, 0),      # 1
            (255, 0, 0),      # 2
            (0, 0, 255),      # 3
            (255, 255, 0),    # 4
            (255, 0, 255),    # 5
            (0, 255, 255),    # 6
        ]

    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(int(mask.max()) + 1):
        if c < len(palette):
            rgb[mask == c] = palette[c]
        else:
            # deterministic fallback for extra classes
            rgb[mask == c] = ((37 * c) % 255, (17 * c) % 255, (97 * c) % 255)
    return rgb


def overlay_mask_on_gray(gray01: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    gray01: (H,W) float in [0,1]
    mask_rgb: (H,W,3) uint8
    returns: (H,W,3) uint8
    """
    base = (np.stack([gray01, gray01, gray01], axis=-1) * 255.0).astype(np.float32)
    m = mask_rgb.astype(np.float32)
    out = (1 - alpha) * base + alpha * m
    return np.clip(out, 0, 255).astype(np.uint8)


def chw_uint8(img_hwc_uint8: np.ndarray) -> np.ndarray:
    """HWC uint8 -> CHW uint8"""
    return np.transpose(img_hwc_uint8, (2, 0, 1))


def gray_to_chw_uint8(gray01: np.ndarray) -> np.ndarray:
    """(H,W) float [0,1] -> (1,H,W) uint8"""
    return (gray01[None, :, :] * 255.0).astype(np.uint8)
