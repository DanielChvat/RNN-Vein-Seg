"""Stage 5: train the recurrent segmenter on ``filtered_data_augmented/``."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from . import config
from .datasets import SequenceDataset
from .losses import dice_per_class, segmentation_loss
from .models import RNN


def _split_by_group(names: list[str], val_groups: tuple[str, ...]) -> tuple[list[int], list[int]]:
    """Split sequence indices by root name so a sequence and its augmentations
    never straddle the train/val boundary."""
    train_idx, val_idx = [], []
    for i, name in enumerate(names):
        (val_idx if config.base_name(name) in val_groups else train_idx).append(i)
    return train_idx, val_idx


def run(
    data_dir: os.PathLike | str = config.AUGMENTED_DIR,
    checkpoint_dir: os.PathLike | str = config.CHECKPOINT_DIR,
    cfg: config.SegConfig | None = None,
) -> Path:
    cfg = cfg or config.SegConfig()
    device = config.device()
    use_amp = device.type == "cuda"
    print(f"Using device: {device} (AMP: {use_amp})")

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = SequenceDataset(data_dir)
    names = dataset.sequence_names
    train_idx, val_idx = _split_by_group(names, cfg.val_groups)

    if not train_idx:
        raise RuntimeError(f"No training sequences left after holding out {cfg.val_groups}")
    if not val_idx:
        raise RuntimeError(
            f"Validation groups {cfg.val_groups} matched nothing in {data_dir}. "
            f"Available roots: {sorted({config.base_name(n) for n in names})}"
        )

    print(f"Train sequences ({len(train_idx)}): {[names[i] for i in train_idx]}")
    print(f"Val   sequences ({len(val_idx)}): {[names[i] for i in val_idx]}")

    # batch_size is pinned to 1: one sample is a whole sequence of length T,
    # and T varies between sequences so they cannot be collated.
    loader_kwargs = dict(batch_size=1, num_workers=4, pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(Subset(dataset, train_idx), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(Subset(dataset, val_idx), shuffle=False, **loader_kwargs)

    model = RNN(
        in_channels=cfg.in_channels,
        base_channels=cfg.base_channels,
        num_classes=cfg.num_classes,
        dropout_p=cfg.dropout_p,
        use_checkpoint=cfg.use_checkpoint,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    # T_max tracks cfg.epochs; it used to hardcode 30 independently of the
    # epoch count, so changing one silently desynchronised the schedule.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs * len(train_loader), eta_min=cfg.eta_min
    )
    scaler = GradScaler(device=device.type, enabled=use_amp)

    best_val = float("inf")
    best_path = checkpoint_dir / "model_best.pth"
    last_path = checkpoint_dir / "model_last.pth"

    for epoch in range(1, cfg.epochs + 1):
        # ---------------------------- train ----------------------------
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train {epoch}/{cfg.epochs}", ncols=110)

        for batch in pbar:
            images = batch["images"].to(device, non_blocking=True)  # (1, T, C, H, W)
            masks = batch["masks"].to(device, non_blocking=True)
            t_len = images.shape[1]

            model.reset_state()
            optimizer.zero_grad(set_to_none=True)

            seq_loss = 0.0
            with autocast(device_type=device.type, enabled=use_amp):
                for t in range(t_len):
                    seq_loss = seq_loss + segmentation_loss(
                        model(images[:, t], t_idx=t),
                        masks[:, t],
                        cfg.focal_tversky_weight,
                        cfg.dice_weight,
                    )
            seq_loss = seq_loss / t_len

            scaler.scale(seq_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Drop the graph before the next sequence so it cannot be retained
            # through the dataloader's prefetch.
            model.reset_state()

            train_loss += seq_loss.item()
            pbar.set_postfix({"loss": f"{seq_loss.item():.4f}"})

        avg_train = train_loss / len(train_loader)

        # --------------------------- validate --------------------------
        model.eval()
        val_loss = 0.0
        dice_sum = [0.0] * cfg.num_classes
        frames = 0

        with torch.no_grad(), autocast(device_type=device.type, enabled=use_amp):
            for batch in val_loader:
                images = batch["images"].to(device, non_blocking=True)
                masks = batch["masks"].to(device, non_blocking=True)
                t_len = images.shape[1]

                model.reset_state()
                seq_loss = 0.0

                for t in range(t_len):
                    logits = model(images[:, t], t_idx=t)
                    seq_loss = seq_loss + segmentation_loss(
                        logits, masks[:, t], cfg.focal_tversky_weight, cfg.dice_weight
                    )

                    per_class = dice_per_class(
                        logits.argmax(dim=1), masks[:, t], cfg.num_classes
                    )
                    dice_sum = [a + b for a, b in zip(dice_sum, per_class)]
                    frames += 1

                val_loss += (seq_loss / t_len).item()

        model.reset_state()
        avg_val = val_loss / len(val_loader)
        dice = [d / frames for d in dice_sum]
        dice_str = "  ".join(f"{n}={d:.3f}" for n, d in zip(config.CLASS_NAMES, dice))

        print(f"Epoch {epoch:02d} | train {avg_train:.4f} | val {avg_val:.4f} | dice {dice_str}")

        # Only best-and-last are kept. Writing every epoch produced 30
        # checkpoints per run, none of which were ever read.
        torch.save(model.state_dict(), last_path)
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), best_path)
            print(f">> new best ({best_val:.4f}) -> {best_path}")

    print(f"\nTraining complete. Best val loss {best_val:.4f} at {best_path}")
    return best_path
