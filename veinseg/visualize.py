"""Render per-frame comparisons of ground truth against predictions."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from . import config
from .datasets import SequenceDataset
from .inference import load_segmenter

COLORS = np.array(config.CLASS_COLORS, dtype=np.uint8)


def mask_to_rgb(mask: torch.Tensor | np.ndarray) -> np.ndarray:
    """(H, W) class indices -> (H, W, 3) uint8."""
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    return COLORS[np.asarray(mask, dtype=np.uint8)]


def overlay(gray01: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Alpha-blend a class map over a grayscale frame in [0, 1]."""
    base = np.repeat(np.asarray(gray01, dtype=np.float32)[..., None], 3, axis=-1) * 255.0
    blended = (1 - alpha) * base + alpha * mask_rgb.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def run(
    data_dir: os.PathLike | str = config.FILTERED_DIR,
    out_dir: os.PathLike | str = config.VIS_OUTPUT_DIR,
    checkpoint: os.PathLike | str = config.SEG_CHECKPOINT,
    cfg: config.SegConfig | None = None,
) -> None:
    """Write ``<seq>_frame<NNN>.png``: input, ground truth overlay, prediction overlay.

    The prediction used to be drawn as a bare class map next to an overlaid
    ground truth, so the two panels were not directly comparable. Both are now
    overlaid on the same frame.

    Existing PNGs in ``out_dir`` with matching names are overwritten.
    """
    import matplotlib

    matplotlib.use("Agg")  # no display needed; also much faster
    import matplotlib.pyplot as plt

    device = config.device()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = SequenceDataset(data_dir)
    model = load_segmenter(checkpoint, cfg, device)

    # One figure reused for every frame. Building a fresh 150-dpi figure per
    # frame was the dominant cost of this script.
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), dpi=150)
    titles = ("Input", "Ground truth", "Prediction")
    handles = [ax.imshow(np.zeros((2, 2, 3), dtype=np.uint8)) for ax in axes]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.axis("off")

    written = 0
    try:
        for idx in range(len(dataset)):
            sample = dataset[idx]
            seq_name = sample["seq_name"]
            if "_AUG_" in seq_name:
                continue

            print(f"Sequence: {seq_name}")
            images = sample["images"].to(device)
            masks = sample["masks"]
            model.reset_state()

            with torch.no_grad():
                for t in range(images.shape[0]):
                    logits = model(images[t].unsqueeze(0), t_idx=t)

                    gray = images[t, 0].cpu().numpy()
                    panels = (
                        np.repeat((gray * 255).astype(np.uint8)[..., None], 3, axis=-1),
                        overlay(gray, mask_to_rgb(masks[t])),
                        overlay(gray, mask_to_rgb(logits.argmax(dim=1)[0])),
                    )
                    for handle, panel in zip(handles, panels):
                        handle.set_data(panel)
                        handle.set_extent((0, panel.shape[1], panel.shape[0], 0))

                    fig.savefig(out_dir / f"{seq_name}_frame{t:03d}.png", bbox_inches="tight")
                    written += 1

            model.reset_state()
    finally:
        plt.close(fig)

    print(f"\nWrote {written} figures to {out_dir}")
