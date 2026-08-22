"""Model definitions: the recurrent segmenter and the empty-frame detector."""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from . import config

# ----------------------------------------------------------------------
# Positional encodings
# ----------------------------------------------------------------------


def sinusoidal_position_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """Standard 1D transformer sinusoidal encoding. Returns ``(max_len, d_model)``."""
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def sinusoidal_2d_positional_encoding(h: int, w: int, d_model: int) -> torch.Tensor:
    """2D sinusoidal encoding, half the channels per axis. Returns ``(1, d_model, H, W)``."""
    if d_model % 4 != 0:
        raise ValueError(f"d_model must be divisible by 4 for 2D PE, got {d_model}")

    d_half = d_model // 2
    div_term = torch.exp(torch.arange(0, d_half, 2) * -(math.log(10000.0) / d_half))

    def axis_pe(n: int) -> torch.Tensor:
        pos = torch.arange(n).unsqueeze(1)
        pe = torch.zeros(n, d_half)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe

    pe_y = axis_pe(h).unsqueeze(1).expand(h, w, d_half)
    pe_x = axis_pe(w).unsqueeze(0).expand(h, w, d_half)

    pe = torch.cat([pe_y, pe_x], dim=-1)  # (H, W, d_model)
    return pe.permute(2, 0, 1).unsqueeze(0).contiguous()


# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------


class ConvBlock(nn.Module):
    """Two 3x3 convs with GroupNorm. GroupNorm rather than BatchNorm because
    sequences are fed one at a time, so the effective batch size is 1.

    Submodule names are spelled out rather than wrapped in a ``Sequential`` so
    that parameter keys stay stable and existing checkpoints keep loading.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.1, num_groups: int = 8):
        super().__init__()
        groups = min(num_groups, out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout_p)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.drop1(self.relu1(self.norm1(self.conv1(x))))
        return self.drop2(self.relu2(self.norm2(self.conv2(x))))


class ConvGRU(nn.Module):
    """Convolutional GRU cell operating on ``(B, C, H, W)`` feature maps."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels

        self.input_proj = nn.Conv2d(in_channels, hidden_channels, 1)

        cat_ch = hidden_channels * 2
        self.reset_gate = nn.Conv2d(cat_ch, hidden_channels, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(cat_ch, hidden_channels, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(cat_ch, hidden_channels, kernel_size, padding=padding)

    def forward(self, x, h_prev):
        x_proj = self.input_proj(x)

        if h_prev is None:
            h_prev = torch.zeros_like(x_proj)

        concat = torch.cat([x_proj, h_prev], dim=1)
        r = torch.sigmoid(self.reset_gate(concat))
        z = torch.sigmoid(self.update_gate(concat))

        h_tilde = torch.tanh(self.out_gate(torch.cat([x_proj, r * h_prev], dim=1)))
        return (1 - z) * h_prev + z * h_tilde


class TransformBlock(nn.Module):
    """Pointwise 1x1 bottleneck MLP with a residual connection."""

    def __init__(self, channels: int, dropout_p: float = 0.1):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, 4 * channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout_p)
        self.fc2 = nn.Conv2d(4 * channels, channels, 1)

    def forward(self, x):
        return self.fc2(self.drop(self.relu(self.fc1(x)))) + x


# ----------------------------------------------------------------------
# Segmentation model
# ----------------------------------------------------------------------


class RNN(nn.Module):
    """3-level U-Net encoder -> ConvGRU bottleneck -> decoder with e1/e2 skips.

    The recurrent state is a mutable attribute (:attr:`h_prev`) rather than a
    forward argument, and ``forward`` overwrites it. Call :meth:`reset_state`
    before every sequence and :meth:`detach_state` between backward passes, or
    the autograd graph grows without bound.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_classes: int = config.NUM_CLASSES,
        dropout_p: float = 0.1,
        use_checkpoint: bool = True,
        max_t: int = 256,
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        c = base_channels
        self._time_channels = c * 4

        self.enc1 = ConvBlock(in_channels, c, dropout_p)
        self.enc2 = ConvBlock(c, c * 2, dropout_p)
        self.enc3 = ConvBlock(c * 2, c * 4, dropout_p)
        self.pool = nn.MaxPool2d(2)

        self.memory = ConvGRU(c * 4, c * 4)
        self.transform = TransformBlock(c * 4, dropout_p)

        self.dec1 = ConvBlock(c * 4 + c * 2, c * 4, dropout_p)
        self.dec2 = ConvBlock(c * 4 + c, c * 2, dropout_p)
        self.dec3 = ConvBlock(c * 2, c, dropout_p)
        self.dec4 = ConvBlock(c, c, dropout_p)

        self.seg_head = nn.Conv2d(c, num_classes, 1)

        # Positional encodings are derived, not learned, so they are excluded
        # from the state dict. That keeps checkpoints smaller and lets the
        # temporal table grow for sequences longer than `max_t` -- the old
        # fixed 200-frame buffer raised IndexError instead.
        self.register_buffer(
            "pos_time", sinusoidal_position_encoding(max_t, self._time_channels), persistent=False
        )
        self.register_buffer("pos_spatial_e1", torch.empty(0), persistent=False)
        self.register_buffer("pos_spatial_e3", torch.empty(0), persistent=False)

        self.h_prev = None

    # ------------------------------------------------------------------
    # Recurrent state
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Clear the recurrent state. Call before each new sequence."""
        self.h_prev = None

    def detach_state(self) -> None:
        """Detach the recurrent state from the autograd graph."""
        if self.h_prev is not None:
            self.h_prev = self.h_prev.detach()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run(self, module, *args):
        """Run ``module`` under gradient checkpointing when it would actually help.

        Checkpointing trades compute for activation memory, so it is pure
        overhead under ``no_grad``. The old code also applied it to the decoder
        unconditionally, which meant ``use_checkpoint=False`` did not disable it.
        """
        if self.use_checkpoint and torch.is_grad_enabled() and self.training:
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def _spatial_pe(self, name: str, feat: torch.Tensor) -> torch.Tensor:
        """Fetch the cached 2D encoding for ``feat``, rebuilding on shape/device change.

        Cached in float32 and cast at the point of use, so autocast switching
        between fp16 and fp32 does not thrash the cache.
        """
        cached = getattr(self, name)
        if (
            cached.numel() == 0
            or cached.shape[1] != feat.shape[1]
            or cached.shape[-2:] != feat.shape[-2:]
            or cached.device != feat.device
        ):
            cached = sinusoidal_2d_positional_encoding(
                feat.shape[-2], feat.shape[-1], feat.shape[1]
            ).to(feat.device)
            setattr(self, name, cached)
        return cached.to(feat.dtype)

    def _time_pe(self, t_idx: int, feat: torch.Tensor) -> torch.Tensor:
        """Temporal encoding for frame ``t_idx``, growing the table if needed."""
        if t_idx >= self.pos_time.shape[0]:
            self.pos_time = sinusoidal_position_encoding(
                max(t_idx + 1, self.pos_time.shape[0] * 2), self._time_channels
            ).to(feat.device)
        elif self.pos_time.device != feat.device:
            self.pos_time = self.pos_time.to(feat.device)
        return self.pos_time[t_idx].to(feat.dtype).view(1, -1, 1, 1)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, t_idx: int) -> torch.Tensor:
        """``x``: (B, 1, H, W). ``t_idx``: index of this frame within its sequence."""
        # --- Encoder ---
        e1 = self._run(self.enc1, x)
        e1 = e1 + self._spatial_pe("pos_spatial_e1", e1)

        e2 = self._run(self.enc2, self.pool(e1))

        e3 = self._run(self.enc3, self.pool(e2))
        e3 = e3 + self._spatial_pe("pos_spatial_e3", e3)
        e3 = e3 + self._time_pe(t_idx, e3)

        # --- Temporal memory ---
        h = self._run(self.memory, e3, self.h_prev)
        self.h_prev = h
        h = self._run(self.transform, h)

        # --- Decoder ---
        d1 = F.interpolate(h, scale_factor=2, mode="bicubic", align_corners=False)
        d1 = self._run(self.dec1, torch.cat([d1, e2], dim=1))

        d2 = F.interpolate(d1, scale_factor=2, mode="bicubic", align_corners=False)
        d2 = self._run(self.dec2, torch.cat([d2, e1], dim=1))

        d3 = self._run(self.dec3, d2)
        d4 = self._run(self.dec4, d3)

        return self.seg_head(d4)


# ----------------------------------------------------------------------
# Empty-frame detector
# ----------------------------------------------------------------------


class EmptyMaskCNN(nn.Module):
    """Binary classifier: does this frame have an entirely-background mask?

    ``forward`` used to return ``self.classifier(x)`` -- the raw input rather
    than the conv features -- so the entire conv stack was dead and the
    "detector" was a linear probe on raw pixels. It ran without error only
    because ``1 * 256 * 128`` happens to equal ``64 * 32 * 16``. Checkpoints
    trained against the old behaviour are meaningless and must be regenerated.
    """

    def __init__(self, in_channels: int = 1, input_size: tuple[int, int] = config.TARGET_SIZE):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        feat_h, feat_w = input_size[0] // 8, input_size[1] // 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * feat_h * feat_w, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ----------------------------------------------------------------------
# Checkpoint IO
# ----------------------------------------------------------------------


def load_checkpoint(model: nn.Module, path: os.PathLike | str, device=None) -> nn.Module:
    """Load weights, tolerating checkpoints that still carry derived buffers.

    Positional encodings used to be persistent buffers; older checkpoints
    contain ``pos_time`` and would fail a strict load.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No checkpoint at {path}. Train first (`python -m veinseg train`)."
        )

    state = torch.load(path, map_location=device or "cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    derived = {"pos_time", "pos_spatial_e1", "pos_spatial_e3"}
    state = {k: v for k, v in state.items() if k not in derived}

    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected if k not in derived]
    missing = [k for k in missing if k not in derived]
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint {path} does not match the model.\n"
            f"  missing: {missing}\n  unexpected: {unexpected}\n"
            "Do `base_channels` and `num_classes` match the values used at training time?"
        )
    return model
