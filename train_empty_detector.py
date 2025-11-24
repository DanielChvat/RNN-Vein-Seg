import torch
import torch.nn as nn
from dataset import EmptyMaskDataset
from empty_mask_model import EmpyMaskCNN
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

root_dir = './processed_data'
epochs = 10
batch_size = 1
base_lr = 1e-3

dataset = EmptyMaskDataset(root_dir)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

model = EmpyMaskCNN().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

print(f"Using Train Dataset of length {len(train_dataset)}")
print(f"Using Validation Dataset of length {len(val_dataset)}")

for epoch in range(1, epochs + 1):
    print(f"\n===== Epoch {epoch}/{epochs} =====")
    model.train()
    running_loss = 0.0

    # ---------------- Training ----------------
    train_bar = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False)
    for imgs, labels in train_bar:
        imgs = imgs.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        logits = model(imgs)

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        train_bar.set_postfix({"loss": loss.item()})

    # ---------------- Validation ----------------
    model.eval()
    correct = 0
    total = 0

    val_bar = tqdm(val_loader, desc=f"Val Epoch {epoch}", leave=False)
    with torch.no_grad():
        for imgs, labels in val_bar:
            imgs = imgs.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(imgs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            acc = correct / total
            val_bar.set_postfix({"acc": acc})

    print(f"Epoch {epoch:02d} | Train Loss: {running_loss:.4f} | Val Acc: {acc:.4f}")

# Save model
torch.save(model.state_dict(), "empty_detector.pth")
print("\nModel saved as empty_detector.pth")
