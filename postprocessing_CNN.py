import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# 1. Simple Refinement CNN
# ----------------------------
class PostProcessCNN(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, kernel_size=1),
            nn.Sigmoid()  # output between 0 and 1
        )

    def forward(self, x):
        x_enc = self.encoder(x)
        out = self.decoder(x_enc)
        return out
