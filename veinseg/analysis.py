"""Exploratory analyses of the empty-frame problem.

Both read ``processed_data/``, so run them before the pipeline's ``--clean``
step removes it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import config


def intensity_correlation(
    data_dir: os.PathLike | str = config.PROCESSED_DIR,
    tsne_components: int = 2,
    pca_components: int = 50,
    show: bool = True,
    figures_dir: os.PathLike | str | None = config.FIGURES_DIR,
) -> None:
    """Correlate mean pixel intensity with mask emptiness, then embed with t-SNE.

    Both plots are written to ``figures_dir`` (pass ``None`` to skip). They used
    to be shown and then lost, so a headless run produced nothing at all.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    data_dir = Path(data_dir)
    if figures_dir is not None:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
    # Sequences are discovered rather than hardcoded; this script used to carry
    # its own list of four, which had drifted behind the other three copies.
    sequences = config.discover_sequences(data_dir)
    print(f"Reading {len(sequences)} sequences: {', '.join(sequences)}")

    intensities, is_empty, flat = [], [], []
    for name in sequences:
        for path in sorted((data_dir / name).glob("*.npz")):
            with np.load(path) as data:
                image, mask = data["image"], data["label"]
            intensities.append(image.mean())
            is_empty.append(int(mask.sum() == 0))
            flat.append(image.ravel())

    intensities = np.asarray(intensities)
    is_empty = np.asarray(is_empty)
    flat = np.asarray(flat, dtype=np.float32)

    corr, pval = pearsonr(intensities, is_empty)
    print(f"Correlation: {corr:.4f}  (p = {pval:.3g})")
    print(f"Mean intensity, empty masks:     {intensities[is_empty == 1].mean():.4f}")
    print(f"Mean intensity, non-empty masks: {intensities[is_empty == 0].mean():.4f}")

    plt.figure(figsize=(8, 6))
    plt.hist(intensities[is_empty == 1], bins=40, alpha=0.6, label="empty mask")
    plt.hist(intensities[is_empty == 0], bins=40, alpha=0.6, label="non-empty mask")
    plt.legend()
    plt.xlabel("Avg pixel intensity")
    plt.ylabel("Count")
    if figures_dir is not None:
        path = figures_dir / "intensity_histogram.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"wrote {path}")
    if show:
        plt.show()

    # Per-image standardisation was computed and then thrown away -- t-SNE was
    # run on the raw intensities, so the embedding mostly recovered overall
    # brightness, which is the very variable being tested against. PCA was
    # imported but never used; it now does the standard pre-reduction, which
    # also makes t-SNE tractable on 32k-dimensional inputs.
    normed = (flat - flat.mean(axis=1, keepdims=True)) / (flat.std(axis=1, keepdims=True) + 1e-6)

    n_components = min(pca_components, *normed.shape)
    reduced = PCA(n_components=n_components, random_state=0).fit_transform(normed)

    perplexity = min(30, max(5, (len(reduced) - 1) // 3))
    embedding = TSNE(
        n_components=tsne_components, perplexity=perplexity, init="pca", random_state=0, verbose=1
    ).fit_transform(reduced)

    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=is_empty, cmap="coolwarm", alpha=0.6)
    plt.title("t-SNE on OCT frames (red = empty mask, blue = non-empty)")
    if figures_dir is not None:
        path = figures_dir / "tsne_empty_masks.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"wrote {path}")
    if show:
        plt.show()


def detector_viewer(
    data_dir: os.PathLike | str = config.PROCESSED_DIR,
    checkpoint: os.PathLike | str = config.DETECTOR_CHECKPOINT,
) -> None:
    """Interactive slider over every frame, showing predicted vs true emptiness."""
    import matplotlib.pyplot as plt
    import torch
    from matplotlib.widgets import Slider

    from .datasets import EmptyMaskDataset
    from .models import EmptyMaskCNN, load_checkpoint

    device = config.device()
    model = EmptyMaskCNN().to(device)
    load_checkpoint(model, checkpoint, device=device)
    model.eval()

    dataset = EmptyMaskDataset(data_dir)
    n_frames = len(dataset)
    print(f"Loaded {n_frames} frames.")

    @torch.no_grad()
    def p_empty(img: torch.Tensor) -> float:
        return torch.sigmoid(model(img.unsqueeze(0).to(device))).item()

    def describe(idx: int, prob: float, truth: float) -> str:
        return f"Frame {idx} / {n_frames - 1}\np(empty) = {prob:.4f}   ground truth = {truth:.0f}"

    img0, label0 = dataset[0]
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.25)

    display = ax.imshow(img0.squeeze(0).numpy(), cmap="gray")
    title = ax.set_title(describe(0, p_empty(img0), label0.item()))
    ax.axis("off")

    slider = Slider(
        ax=plt.axes([0.15, 0.1, 0.7, 0.05]),
        label="Frame index",
        valmin=0,
        valmax=n_frames - 1,
        valinit=0,
        valstep=1,
    )

    def update(_value) -> None:
        idx = int(slider.val)
        img, label = dataset[idx]
        display.set_data(img.squeeze(0).numpy())
        title.set_text(describe(idx, p_empty(img), label.item()))
        fig.canvas.draw_idle()

    slider.on_changed(update)

    def on_key(event) -> None:
        if event.key == "right":
            slider.set_val(min(slider.val + 1, n_frames - 1))
        elif event.key == "left":
            slider.set_val(max(slider.val - 1, 0))

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
