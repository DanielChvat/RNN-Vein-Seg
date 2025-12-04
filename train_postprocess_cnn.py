import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from seg_model import RNN
from postprocess_cnn import PostProcessCNN
from dataset import SequenceDataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load pretrained RNN
# -----------------------------
seg_model = RNN(in_channels=1, base_channels=8, num_classes=3).to(device)
seg_model.load_state_dict(torch.load("checkpoints/model_best.pth"))
seg_model.eval()  # freeze

# -----------------------------
# PostProcess CNN
# -----------------------------
postprocess_cnn = PostProcessCNN(in_channels=6, out_channels=1).to(device)
optimizer = torch.optim.Adam(postprocess_cnn.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()  # predicting 0/1 refinement mask

# -----------------------------
# Data loader
# -----------------------------
train_dataset = SequenceDataset("./filtered_data_augmented")
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# -----------------------------
# Difference-guided mask
# -----------------------------
def class_specific_diff_mask(preds, targets, num_classes=3):
    pred_labels = torch.argmax(preds, dim=1)  # (B, H, W)
    diff_mask = torch.zeros_like(preds)
    for c in range(num_classes):
        mask = ((pred_labels == c) | (targets == c)) & (pred_labels != targets)
        diff_mask[:, c] = mask.float()
    return diff_mask

# -----------------------------
# Training loop
# -----------------------------
num_epochs = 10
for epoch in range(num_epochs):
    postprocess_cnn.train()
    epoch_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch in pbar:
        images = batch["images"].to(device)  # (B, T, C, H, W)
        masks = batch["masks"].to(device)

        optimizer.zero_grad()
        seq_loss = 0.0

        T = images.size(1)
        for t in range(T):
            with torch.no_grad():
                logits = seg_model(images[:, t])
                probs = F.softmax(logits, dim=1)

            diff_mask = class_specific_diff_mask(logits, masks[:, t])
            post_input = torch.cat([probs, diff_mask], dim=1)  # (B, 6, H, W)

            target_mask = (masks[:, t] != torch.argmax(logits, dim=1)).float().unsqueeze(1)  # refine where wrong
            refined_out = postprocess_cnn(post_input)

            loss = criterion(refined_out, target_mask)
            loss.backward()
            seq_loss += loss.item()

        optimizer.step()
        epoch_loss += seq_loss / T
        pbar.set_postfix({"loss": seq_loss / T})

    print(f"Epoch {epoch+1} average loss: {epoch_loss/len(train_loader):.4f}")

torch.save(postprocess_cnn.state_dict(), "checkpoints/postprocess_cnn.pth")
print("Saved PostProcessCNN weights")
