import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True),
            nn.GroupNorm(num_groups=min(num_groups, out_channels),
                         num_channels=out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True),
            nn.GroupNorm(num_groups=min(num_groups, out_channels),
                         num_channels=out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class ConvGRU(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels

        self.input_proj = nn.Conv2d(in_channels, hidden_channels, 1, bias=True)

        gate_channels = hidden_channels * 2
        self.reset  = nn.Conv2d(gate_channels, hidden_channels, kernel_size, padding=padding, bias=True)
        self.update = nn.Conv2d(gate_channels, hidden_channels, kernel_size, padding=padding, bias=True)
        self.output = nn.Conv2d(gate_channels, hidden_channels, kernel_size, padding=padding, bias=True)

    def forward(self, x, h_old):
        x_proj = self.input_proj(x)

        if h_old is None:
            h_old = torch.zeros(
                x.size(0), self.hidden_channels, x.size(2), x.size(3),
                device=x.device, dtype=x.dtype
            )

        mixed = torch.cat([x_proj, h_old], dim=1)

        reset  = torch.sigmoid(self.reset(mixed))
        update = torch.sigmoid(self.update(mixed))

        out_state = torch.tanh(
            self.output(torch.cat([x_proj, reset * h_old], dim=1))
        )

        h_new = (1 - update) * h_old + update * out_state
        return h_new


class TransformBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 4, channels, 1)
        )

    def forward(self, x):
        return self.block(x) + x


class RNN(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, num_classes=3,
                 max_T=200, use_checkpoint=True):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        C = base_channels

        self.enc1 = ConvBlock(in_channels, C)
        self.enc2 = ConvBlock(C, C * 2)
        self.enc3 = ConvBlock(C * 2, C * 4)

        self.pool = nn.MaxPool2d(2)

        self.memory = ConvGRU(C * 4, C * 4)
        self.transform = TransformBlock(C * 4)

        self.dec1 = ConvBlock(C * 4 + C * 2, C * 4)
        self.dec2 = ConvBlock(C * 4 + C, C * 2)
        self.dec3 = ConvBlock(C * 2, C)
        self.dec4 = ConvBlock(C, C)

        self.seg_head = nn.Conv2d(C, num_classes, 1)

        self.h_prev = None

        self.channel_weights = nn.Parameter(torch.randn(1, C, 1, 1))

        self.pos_time = nn.Parameter(torch.randn(max_T, C * 4))

    def forward(self, x, t_idx=None):
        
        if self.h_prev is not None:
            self.h_prev = self.h_prev.to(x.device)

        B, _, H, W = x.shape

        e1 = checkpoint(self.enc1, x, use_reentrant=False) if self.use_checkpoint else self.enc1(x)

        e1 = e1 + self.channel_weights

        p1 = self.pool(e1)

        e2 = checkpoint(self.enc2, p1, use_reentrant=False) if self.use_checkpoint else self.enc2(p1)
        p2 = self.pool(e2)

        e3 = checkpoint(self.enc3, p2, use_reentrant=False) if self.use_checkpoint else self.enc3(p2)

        time_emb = self.pos_time[t_idx].view(1, -1, 1, 1)  # (1, C, 1, 1)
        e3 = e3 + time_emb

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

        d1 = F.interpolate(h, scale_factor=2, mode='bicubic', align_corners=False)
        d1 = checkpoint(self.dec1, torch.cat([d1, e2], dim=1), use_reentrant=False)

        d2 = F.interpolate(d1, scale_factor=2, mode='bicubic', align_corners=False)
        d2 = checkpoint(self.dec2, torch.cat([d2, e1], dim=1), use_reentrant=False)

        d3 = checkpoint(self.dec3, d2, use_reentrant=False)
        d4 = checkpoint(self.dec4, d3, use_reentrant=False)

        out = self.seg_head(d4)
        return out