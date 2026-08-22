"""Segmentation objectives.

``compute_N_i`` and ``ClassBalancedSoftmaxCE`` used to live here. They were
already dead -- the CE term was computed every frame and then discarded without
ever entering the backward pass -- and ``compute_N_i`` cost a full extra pass
over the dataloader before training could start. Both were removed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """(B, H, W) int64 -> (B, C, H, W) float."""
    return F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft multi-class Dice loss.

    logits: (B, C, H, W) raw scores
    target: (B, H, W) integer class labels
    """
    probs = F.softmax(logits, dim=1)
    target_1hot = _one_hot(target, probs.size(1))

    dims = (0, 2, 3)
    intersection = (probs * target_1hot).sum(dims)
    union = probs.sum(dims) + target_1hot.sum(dims)

    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Multi-class Tversky loss. ``beta > alpha`` penalises false negatives."""
    probs = F.softmax(logits, dim=1)
    target_1hot = _one_hot(target, probs.size(1))

    dims = (0, 2, 3)
    tp = (probs * target_1hot).sum(dims)
    fp = (probs * (1 - target_1hot)).sum(dims)
    fn = ((1 - probs) * target_1hot).sum(dims)

    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - tversky.mean()


def focal_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.2,
    beta: float = 0.8,
    gamma: float = 0.85,
) -> torch.Tensor:
    return tversky_loss(logits, target, alpha, beta) ** gamma


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    focal_tversky_weight: float = 0.2,
    dice_weight: float = 0.8,
) -> torch.Tensor:
    """The objective actually used for training: a weighted focal-Tversky/Dice mix.

    Both terms share a softmax over ``logits``; keeping them in one function
    means callers cannot accidentally reweight one and not the other.
    """
    return (
        focal_tversky_weight * focal_tversky_loss(logits, target)
        + dice_weight * dice_loss(logits, target)
    )


@torch.no_grad()
def dice_per_class(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6
) -> list[float]:
    """Hard Dice per class from argmaxed predictions. For reporting, not training.

    pred / target: (..., H, W) int64
    """
    dices = []
    for c in range(num_classes):
        p = pred == c
        t = target == c
        inter = (p & t).sum().float()
        denom = p.sum().float() + t.sum().float()
        dices.append(((2 * inter + eps) / (denom + eps)).item())
    return dices
