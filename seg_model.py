import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint
import math


# ----------------------------------------------------------------------
# SINUSOIDAL POSITIONAL ENCODINGS
# ----------------------------------------------------------------------

def sinusoidal_position_encoding(max_len, d_model):
    """
    Standard Transformer sinusoidal encoding (Vaswani et al., 2017)
    Returns shape: (max_len, d_model)
    """
    position = torch.arange(max_len).unsqueeze(1)                     # (T, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))

    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)                      # even dims
    pe[:, 1::2] = torch.cos(position * div_term)                      # odd dims
    return pe


def sinusoidal_2d_positional_encoding(H, W, d_model):
    """
    2D extension of sinusoidal positional encodings.
    Splits channels into two halves: Y and X encodings.
    Output shape: (1, d_model, H, W)
    """
    if d_model % 2 != 0:
        raise ValueError("d_model must be divisible by 2 for 2D PE.")

    d_half = d_model // 2

    # Encoding each axis with 1D sinusoidal encoding
    y_pos = torch.arange(H).unsqueeze(1)                                # (H, 1)
    x_pos = torch.arange(W).unsqueeze(1)                                # (W, 1)

    div_term_y = torch.exp(torch.arange(0, d_half, 2) *
                           -(math.log(10000.0) / d_half))
    div_term_x = torch.exp(torch.arange(0, d_half, 2) *
                           -(math.log(10000.0) / d_half))

    pe_y = torch.zeros(H, d_half)
    pe_x = torch.zeros(W, d_half)

    pe_y[:, 0::2] = torch.sin(y_pos * div_term_y)
    pe_y[:, 1::2] = torch.cos(y_pos * div_term_y)

    pe_x[:, 0::2] = torch.sin(x_pos * div_term_x)
    pe_x[:, 1::2] = torch.cos(x_pos * div_term_x)

    # Broadcast to grid
    pe_y = pe_y.unsqueeze(1).repeat(1, W, 1)      # (H, W, d_half)
    pe_x = pe_x.unsqueeze(0).repeat(H, 1, 1)      # (H, W, d_half)

    pe = torch.cat([pe_y, pe_x], dim=-1)          # (H, W, d_model)
    pe = pe.permute(2, 0, 1).unsqueeze(0)         # (1, d_model, H, W)

    return pe


# ----------------------------------------------------------------------
# MODEL COMPONENTS
# ----------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.1, num_groups=8):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout_p)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.drop1(self.relu1(self.norm1(self.conv1(x))))
        x = self.drop2(self.relu2(self.norm2(self.conv2(x))))
        return x


class ConvGRU(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()

        padding = kernel_size // 2
        self.hidden_channels = hidden_channels

        self.input_proj = nn.Conv2d(in_channels, hidden_channels, 1)

        cat_ch = hidden_channels * 2
        self.reset_gate  = nn.Conv2d(cat_ch, hidden_channels, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(cat_ch, hidden_channels, kernel_size, padding=padding)
        self.out_gate    = nn.Conv2d(cat_ch, hidden_channels, kernel_size, padding=padding)

    def forward(self, x, h_prev):
        x_proj = self.input_proj(x)

        if h_prev is None:
            h_prev = torch.zeros(
                x_proj.size(0), self.hidden_channels,
                x_proj.size(2), x_proj.size(3),
                device=x.device, dtype=x.dtype
            )

        concat = torch.cat([x_proj, h_prev], dim=1)

        r = torch.sigmoid(self.reset_gate(concat))
        z = torch.sigmoid(self.update_gate(concat))

        out_input = torch.cat([x_proj, r * h_prev], dim=1)
        h_tilde = torch.tanh(self.out_gate(out_input))

        h_new = (1 - z) * h_prev + z * h_tilde
        return h_new


class TransformBlock(nn.Module):
    def __init__(self, channels, dropout_p=0.1):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, 4 * channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout_p)
        self.fc2 = nn.Conv2d(4 * channels, channels, 1)

    def forward(self, x):
        out = self.fc2(self.drop(self.relu(self.fc1(x))))
        return out + x


# ----------------------------------------------------------------------
# FULL MODEL
# ----------------------------------------------------------------------

class RNN(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, num_classes=3,
                 max_T=200, dropout_p=0.1, use_checkpoint=True):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        C = base_channels

        # --- Encoder ---
        self.enc1 = ConvBlock(in_channels, C, dropout_p)
        self.enc2 = ConvBlock(C, C * 2, dropout_p)
        self.enc3 = ConvBlock(C * 2, C * 4, dropout_p)

        self.pool = nn.MaxPool2d(2)

        # --- Temporal memory ---
        self.memory = ConvGRU(C * 4, C * 4)
        self.transform = TransformBlock(C * 4, dropout_p)

        # --- Decoder ---
        self.dec1 = ConvBlock(C * 4 + C * 2, C * 4, dropout_p)
        self.dec2 = ConvBlock(C * 4 + C, C * 2, dropout_p)
        self.dec3 = ConvBlock(C * 2, C, dropout_p)
        self.dec4 = ConvBlock(C, C, dropout_p)

        self.seg_head = nn.Conv2d(C, num_classes, 1)

        # --- Positional encodings ---
        pe_time = sinusoidal_position_encoding(max_T, C * 4)
        self.register_buffer("pos_time", pe_time)

        self.pos_spatial_e1 = None   # created lazily
        self.pos_spatial_e3 = None

        self.h_prev = None


    # ------------------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------------------
    def forward(self, x, t_idx):
        # x: (B, 1, H, W)

        # ---------------------------------------------------------------
        # Encoder
        # ---------------------------------------------------------------
        e1 = checkpoint(self.enc1, x, use_reentrant=False) if self.use_checkpoint else self.enc1(x)

        # Spatial PE at e1 resolution
        if self.pos_spatial_e1 is None or \
           self.pos_spatial_e1.shape[-2:] != e1.shape[-2:]:
            self.pos_spatial_e1 = sinusoidal_2d_positional_encoding(
                e1.shape[-2], e1.shape[-1], e1.shape[1]
            ).to(e1.device)

        e1 = e1 + self.pos_spatial_e1

        p1 = self.pool(e1)
        e2 = checkpoint(self.enc2, p1, use_reentrant=False) if self.use_checkpoint else self.enc2(p1)
        p2 = self.pool(e2)

        e3 = checkpoint(self.enc3, p2, use_reentrant=False) if self.use_checkpoint else self.enc3(p2)

        # Spatial PE at e3 resolution
        if self.pos_spatial_e3 is None or \
           self.pos_spatial_e3.shape[-2:] != e3.shape[-2:]:
            self.pos_spatial_e3 = sinusoidal_2d_positional_encoding(
                e3.shape[-2], e3.shape[-1], e3.shape[1]
            ).to(e3.device)

        e3 = e3 + self.pos_spatial_e3

        # Temporal PE
        time_emb = self.pos_time[t_idx].view(1, -1, 1, 1)
        e3 = e3 + time_emb

        # ---------------------------------------------------------------
        # ConvGRU Memory
        # ---------------------------------------------------------------
        if self.h_prev is None:
            h = self.memory(e3, None)
        else:
            if self.use_checkpoint:
                h = checkpoint(lambda a, b: self.memory(a, b),
                               e3, self.h_prev,
                               use_reentrant=False)
            else:
                h = self.memory(e3, self.h_prev)

        self.h_prev = h

        h = checkpoint(self.transform, h, use_reentrant=False) if self.use_checkpoint else self.transform(h)

        # ---------------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------------
        d1 = F.interpolate(h, scale_factor=2, mode='bicubic', align_corners=False)
        d1 = checkpoint(self.dec1, torch.cat([d1, e2], dim=1), use_reentrant=False)

        d2 = F.interpolate(d1, scale_factor=2, mode='bicubic', align_corners=False)
        d2 = checkpoint(self.dec2, torch.cat([d2, e1], dim=1), use_reentrant=False)

        d3 = checkpoint(self.dec3, d2, use_reentrant=False)
        d4 = checkpoint(self.dec4, d3, use_reentrant=False)

        out = self.seg_head(d4)
        return out
