import torch
import torch.nn.functional as F
import torch.nn as nn

def compute_N_i(dataloader, num_classes):
    counts = torch.zeros(num_classes, dtype=torch.long)
    for sampled_batch in dataloader:
        labels = sampled_batch['masks'].view(-1)  # flatten (B*H*W)

        for c in range(num_classes):
            counts[c] += torch.sum(labels == c)

    return counts

class ClassBalancedSoftmaxCE(nn.Module):
    def __init__(self, class_counts):
        super().__init__()
        
        N = class_counts  # number of pixels per class
        betas = (N - 1) / N  # class-specific beta

        weights = (1 - betas) / (1 - torch.pow(betas, N) + 1e-8)
        weights = torch.softmax(weights, dim=0)
        self.register_buffer("weights", weights)

    def forward(self, logits, targets):
        B, C, H, W = logits.shape
        weights = self.weights.to(logits.device)

        logits = logits.permute(0, 2, 3, 1).reshape(-1, C)  # flatten spatial dims
        targets = targets.view(-1)                           # flatten targets

        loss = F.cross_entropy(
            input=logits,
            target=targets,
            weight=weights,
            reduction="mean"
        )

        return loss
    
def dice_loss(pred, target, eps=1e-6):
    """
    pred: (B, C, H, W) probabilities after softmax
    target: (B, H, W) integer class labels
    """
    num_classes = pred.size(1)
    pred = F.softmax(pred, dim=1)  # Ensure probabilities
    target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    intersection = (pred * target_onehot).sum(dim=(0,2,3))
    union = pred.sum(dim=(0,2,3)) + target_onehot.sum(dim=(0,2,3))
    
    dice = (2. * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()  # Dice loss