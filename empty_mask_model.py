import torch
import torch.nn as nn
ORIGINAL_SIZE = (1024, 512)
DOWNSCALE_FACTOR = 4 # changed this from 4 to 1 for original size images
PREPROCESSED_SIZE = (
    ORIGINAL_SIZE[0] // DOWNSCALE_FACTOR,   
    ORIGINAL_SIZE[1] // DOWNSCALE_FACTOR   
)

FEATURE_SIZE = (torch.tensor(PREPROCESSED_SIZE) // 8)
FLATTEN_DIM = 64 * FEATURE_SIZE[0] * FEATURE_SIZE[1]


class EmpyMaskCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FLATTEN_DIM, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.downsample(x)
        return self.classifier(x)