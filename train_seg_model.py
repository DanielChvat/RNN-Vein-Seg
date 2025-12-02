import torch
import torch.nn as nn
import torch.optim as optim
from dataset import SequenceDataset
from seg_model import RNN
from tqdm import tqdm
import os

from loss import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
train_dataset = SequenceDataset('./filtered_data')

batch_size = 1
num_epochs = 30
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

model = RNN(in_channels=1, base_channels=16, num_classes=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Compute class counts from dataset
num_classes = 3
class_counts = compute_N_i(train_loader, num_classes=num_classes)
print(f"Class counts: {class_counts}")

# Initialize ClassBalancedSoftmaxCE loss
criterion = ClassBalancedSoftmaxCE(class_counts)

# Cosine annealing scheduler per batch (T_max in steps)
steps_per_epoch = len(train_loader)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=(num_epochs * steps_per_epoch) // 20,  # step per image
    eta_min=0
)

best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)

    for batch in pbar:
        images = batch["images"].to(device)  # (B, T, C, H, W)
        masks = batch["masks"].to(device)    # (B, T, H, W)

        T = images.size(1)
        model.h_prev = None

        seq_loss_val = 0.0

        for t in range(T):
            # optimizer.zero_grad()

            out = model(images[:, t])           # (B, C, H, W)
            # ce_loss = criterion(out, masks[:, t])
            # d_loss = dice_loss(out, masks[:, t])

            ft_loss = focal_tversky_loss(out, masks[:, t])
            d_loss = dice_loss(out, masks[:, t])
            

            # loss = 0.3 * ce_loss + 0.7 * d_loss
            loss = 0.7 * ft_loss + 0.3 * d_loss
            
            loss.backward(retain_graph=True)
            if (t + 1) % 20 == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                model.h_prev = model.h_prev.detach()

            seq_loss_val += loss.item()

        seq_loss_val /= T
        epoch_loss += seq_loss_val
        pbar.set_postfix({"loss": seq_loss_val})

    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")

    # Save per-epoch checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch{epoch+1}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

    # Save best model
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        best_model_path = os.path.join(checkpoint_dir, "model_best.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model: {best_model_path}")
