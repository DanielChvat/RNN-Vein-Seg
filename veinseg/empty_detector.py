"""Stages 2 and 3: train the empty-frame detector, then use it to filter frames."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from . import config
from .datasets import EmptyMaskDataset
from .models import EmptyMaskCNN, load_checkpoint


def train(
    data_dir: os.PathLike | str = config.PROCESSED_DIR,
    checkpoint: os.PathLike | str = config.DETECTOR_CHECKPOINT,
    cfg: config.DetectorConfig | None = None,
) -> Path:
    cfg = cfg or config.DetectorConfig()
    device = config.device()
    print(f"Using device: {device}")

    dataset = EmptyMaskDataset(data_dir)
    val_size = max(1, int(cfg.val_fraction * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0)
    )

    print(f"Train frames: {len(train_set)}  Val frames: {len(val_set)}")

    # batch_size was 1, which made this stage dominate the pipeline's runtime
    # for no reason -- these are independent frames, not sequences.
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, num_workers=4)

    model = EmptyMaskCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs * len(train_loader), eta_min=cfg.eta_min
    )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Detector train {epoch}/{cfg.epochs}", leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device).unsqueeze(1)

            loss = criterion(model(imgs), labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device).unsqueeze(1)
                preds = (torch.sigmoid(model(imgs)) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.numel()

        # `total` is guaranteed non-zero: val_size is clamped to >= 1.
        print(
            f"Epoch {epoch:02d} | train loss {running / len(train_loader):.4f} "
            f"| val acc {correct / total:.4f}"
        )

    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    print(f"Detector saved to {checkpoint}")
    return checkpoint


def filter_frames(
    data_dir: os.PathLike | str = config.PROCESSED_DIR,
    out_dir: os.PathLike | str = config.FILTERED_DIR,
    checkpoint: os.PathLike | str = config.DETECTOR_CHECKPOINT,
    cfg: config.DetectorConfig | None = None,
) -> None:
    """Copy every frame the detector does *not* consider empty into ``out_dir``."""
    cfg = cfg or config.DetectorConfig()
    device = config.device()

    data_dir, out_dir = Path(data_dir), Path(out_dir)
    dataset = EmptyMaskDataset(data_dir)
    print(f"Loaded {len(dataset)} frames from {data_dir}")

    model = EmptyMaskCNN().to(device)
    load_checkpoint(model, checkpoint, device=device)
    model.eval()

    # Scored in batches rather than one frame at a time.
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
    probs: list[float] = []
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="Scoring frames"):
            logits = model(imgs.to(device))
            probs.extend(torch.sigmoid(logits).squeeze(1).cpu().tolist())

    kept = empty = 0
    for i, p_empty in enumerate(probs):
        if p_empty >= cfg.empty_threshold:
            empty += 1
            continue
        seq, fname = dataset.sequence_names[i], dataset.frame_names[i]
        dest = out_dir / seq
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(data_dir / seq / fname, dest / fname)
        kept += 1

    print("\n===== FILTERING COMPLETE =====")
    print(f"Total frames:          {len(dataset)}")
    print(f"Frames kept:           {kept}")
    print(f"Frames detected empty: {empty}")
    print(f"Output saved to:       {out_dir}")
