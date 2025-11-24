import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
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

        self.reset = nn.Conv2d(gate_channels, hidden_channels, kernel_size, padding=padding, bias=True)
        self.update = nn.Conv2d(gate_channels, hidden_channels, kernel_size, padding=padding, bias=True)
        self.output = nn.Conv2d(gate_channels, hidden_channels, kernel_size, padding=padding, bias=True)

    def forward(self, x, h_old):
        x_proj = self.input_proj(x)

        if h_old is None:
            h_old = torch.zeros(x.size(0), self.hidden_channels,
                                x.size(2), x.size(3), device=x.device, dtype=x.dtype)

        mixed = torch.cat([x_proj, h_old], dim=1)

        reset = torch.sigmoid(self.reset(mixed))
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
    def __init__(self, in_channels=1, base_channels=16, num_classes=3):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)

        self.pool = nn.MaxPool2d(2)

        self.memory = ConvGRU(base_channels * 4, base_channels * 4)
        self.transform = TransformBlock(base_channels * 4)

        self.dec1 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 4)
        self.dec2 = ConvBlock(base_channels * 4 + base_channels, base_channels * 2)
        self.dec3 = ConvBlock(base_channels * 2, base_channels)
        self.dec4 = ConvBlock(base_channels, base_channels)

        self.seg_head = nn.Conv2d(base_channels, num_classes, 1)

        self.h_prev = None

    def forward(self, x):
        if self.h_prev is not None:
            self.h_prev = self.h_prev.to(x.device)

        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)

        h = self.memory(e3, self.h_prev)
        # self.h_prev = h.detach()
        self.h_prev = h
        h = self.transform(h)

        d1 = F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e2], dim=1))

        d2 = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        d3 = self.dec3(d2)
        
        
        d4 = self.dec4(d3)

        out = self.seg_head(d4)
        return out

    

