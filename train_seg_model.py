import os
import re
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from tensorboardX import SummaryWriter

from dataset import SequenceDataset
from seg_model import RNN
from loss import focal_tversky_loss, dice_loss, ClassBalancedSoftmaxCE, compute_N_i
from tb_utils import (
    dice_per_class,
    mask_to_rgb,
    overlay_mask_on_gray,
    _to_01,
    chw_uint8,
    gray_to_chw_uint8,
)

# ============================================================
#                  CONFIG
# ============================================================
ROOT = "./filtered_data"
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

VAL_GROUPS = ["Cube15"]          # hold out whole group
NUM_CLASSES = 3                  # adjust if needed
NUM_EPOCHS = 30
BATCH_SIZE = 1
NUM_WORKERS = 4

LOG_HISTOGRAMS = False           # set True if you want weight/grad histograms
VIS_FRAMES = "auto"              # "auto" or list like [0, 5, 10]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = (device.type == "cuda")
print(f"Using device: {device}")

# ============================================================
#                 TENSORBOARDX SETUP
# ============================================================
RUN_NAME = datetime.now().strftime("%Y%m%d_%H%M%S")
TB_DIR = os.path.join("runs", f"rnn_seg_{RUN_NAME}")
writer = SummaryWriter(TB_DIR)
print(f"TensorBoard logs: {TB_DIR}")
print("Start UI with: tensorboard --logdir runs")

# ============================================================
#     1. GROUP SEQUENCES (Cube15, Cube16, OA, OA_AUGNNN, etc)
# ============================================================
def base_name(name):
    return re.sub(r"_AUG_\d+$", "", name)

all_folders = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
groups = {}
for folder in all_folders:
    bn = base_name(folder)
    groups.setdefault(bn, []).append(folder)

print("\nDiscovered groups:")
for g, members in groups.items():
    print(f"  {g}: {members}")

TRAIN_GROUPS = [g for g in groups if g not in VAL_GROUPS]

train_folders, val_folders = [], []
for g in TRAIN_GROUPS:
    train_folders.extend(groups[g])
for g in VAL_GROUPS:
    val_folders.extend(groups[g])

print("\nTrain groups:", TRAIN_GROUPS)
print("Val groups:  ", VAL_GROUPS)
print("Train folders:", train_folders)
print("Val folders:  ", val_folders)

# ============================================================
#     2. LOAD DATASET AND CREATE SUBSETS
# ============================================================
train_full = SequenceDataset(ROOT)
val_full   = SequenceDataset(ROOT)

# IMPORTANT:
# Your earlier dataset code uses `sequence_dirs`.
# If you switched to the improved dataset earlier (tuples of (seq_name, frame_files)),
# replace the next line appropriately.
train_seq_names = train_full.sequence_dirs
val_seq_names   = val_full.sequence_dirs
assert train_seq_names == val_seq_names, "Train/Val dataset sequence ordering mismatch!"

train_indices = [i for i, name in enumerate(train_seq_names) if name in train_folders]
val_indices   = [i for i, name in enumerate(train_seq_names) if name in val_folders]

train_dataset = Subset(train_full, train_indices)
val_dataset   = Subset(val_full, val_indices)

print(f"\nFinal Train sequences: {len(train_dataset)}")
print(f"Final Val sequences:   {len(val_dataset)}\n")

# ============================================================
#     3. DATA LOADERS
# ============================================================
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# ============================================================
#     4. MODEL + LOSSES + OPTIMIZER
# ============================================================
model = RNN(in_channels=1, base_channels=32, num_classes=NUM_CLASSES).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Class-balancing (computed from TRAIN ONLY)
class_counts = compute_N_i(train_loader, num_classes=NUM_CLASSES)
print("Class counts:", class_counts)
criterion_ce = ClassBalancedSoftmaxCE(class_counts)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS * len(train_loader),
    eta_min=1e-6
)

scaler = GradScaler(device="cuda") if use_cuda else None
best_val_loss = float("inf")

# (Optional) log model parameter count
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
writer.add_text("meta/params", f"Trainable params: {n_params}")
writer.add_text("meta/splits", f"VAL_GROUPS={VAL_GROUPS} TRAIN_GROUPS={TRAIN_GROUPS}")

# ============================================================
#     5. HELPERS
# ============================================================
def _autocast_ctx():
    if use_cuda:
        return autocast(device_type="cuda")
    return autocast(device_type="cpu")


@torch.no_grad()
def run_validation(epoch: int):
    model.eval()
    val_loss = 0.0

    # Dice accumulators
    dice_sum = np.zeros(NUM_CLASSES, dtype=np.float64)
    dice_count = 0

    # We'll also capture one example sequence for visualization
    example_logged = False

    for batch in val_loader:
        images = batch["images"].to(device)  # B, T, C, H, W
        masks  = batch["masks"].to(device)   # B, T, H, W
        seq_name = batch.get("seq_name", ["seq"])[0] if isinstance(batch.get("seq_name", "seq"), list) else batch.get("seq_name", "seq")

        T = images.shape[1]
        B = images.shape[0]

        model.h_prev = None
        seq_loss = 0.0

        preds_all = []
        gts_all = []

        with _autocast_ctx():
            for t in range(T):
                out = model(images[:, t], t_idx=t)  # logits B,C,H,W
                ft = focal_tversky_loss(out, masks[:, t])
                di = dice_loss(out, masks[:, t])
                seq_loss = seq_loss + (0.2 * ft + 0.8 * di)

                pred = out.argmax(dim=1)  # B,H,W
                preds_all.append(pred)
                gts_all.append(masks[:, t])

        seq_loss = seq_loss / T
        val_loss += seq_loss.item()

        # Dice over the whole sequence
        preds_seq = torch.cat(preds_all, dim=0)   # (B*T,H,W)
        gts_seq   = torch.cat(gts_all, dim=0)     # (B*T,H,W)
        dpc = dice_per_class(preds_seq, gts_seq, num_classes=NUM_CLASSES)
        dice_sum += np.array(dpc, dtype=np.float64)
        dice_count += 1

        # Visualization: log first sequence of epoch
        if not example_logged and B == 1:
            example_logged = True

            # choose a few frames
            if VIS_FRAMES == "auto":
                frame_ids = sorted(set([0, T // 2, T - 1]))
            else:
                frame_ids = [t for t in VIS_FRAMES if 0 <= t < T]
                if not frame_ids:
                    frame_ids = [0]

            for t in frame_ids:
                # input image: (1,C,H,W) -> (H,W)
                img = images[0, t].detach().float().cpu().numpy()
                if img.shape[0] == 1:
                    img2d = img[0]
                else:
                    img2d = img.mean(axis=0)

                img01 = _to_01(img2d)

                gt = masks[0, t].detach().cpu().numpy().astype(np.int32)
                # recompute pred for this frame from cached preds_all
                pred = preds_all[t][0].detach().cpu().numpy().astype(np.int32)

                gt_rgb = mask_to_rgb(gt)
                pr_rgb = mask_to_rgb(pred)

                gt_ov = overlay_mask_on_gray(img01, gt_rgb, alpha=0.45)
                pr_ov = overlay_mask_on_gray(img01, pr_rgb, alpha=0.45)

                # write to tensorboardX as CHW
                tag_base = f"val_vis/{seq_name}/frame_{t:03d}"
                writer.add_image(f"{tag_base}/input", gray_to_chw_uint8(img01), epoch)
                writer.add_image(f"{tag_base}/gt_mask", chw_uint8(gt_rgb), epoch)
                writer.add_image(f"{tag_base}/pred_mask", chw_uint8(pr_rgb), epoch)
                writer.add_image(f"{tag_base}/overlay_gt", chw_uint8(gt_ov), epoch)
                writer.add_image(f"{tag_base}/overlay_pred", chw_uint8(pr_ov), epoch)

    avg_val = val_loss / max(1, len(val_loader))
    dice_avg = dice_sum / max(1, dice_count)

    return avg_val, dice_avg


# ============================================================
#     6. TRAIN LOOP
# ============================================================
global_step = 0

for epoch in range(1, NUM_EPOCHS + 1):
    # ---------------- TRAIN ----------------
    model.train()
    train_loss = 0.0

    # optional: track dice on a small sample to avoid overhead
    train_dice_sum = np.zeros(NUM_CLASSES, dtype=np.float64)
    train_dice_count = 0

    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}", ncols=120)

    for batch in pbar:
        images = batch["images"].to(device)  # B, T,C,H,W
        masks  = batch["masks"].to(device)
        T = images.shape[1]

        model.h_prev = None
        optimizer.zero_grad(set_to_none=True)

        seq_loss = 0.0

        # For (cheap) train dice: compute on first frame only
        pred_first = None
        gt_first = None

        with _autocast_ctx():
            for t in range(T):
                out = model(images[:, t], t_idx=t)

                # your original losses (CE computed but not used; keep if you want to log it)
                ce = criterion_ce(out, masks[:, t])
                ft = focal_tversky_loss(out, masks[:, t])
                di = dice_loss(out, masks[:, t])

                loss = 0.2 * ft + 0.8 * di
                seq_loss += loss

                if t == 0:
                    pred_first = out.argmax(dim=1).detach()
                    gt_first = masks[:, t].detach()

        seq_loss = seq_loss / T

        if use_cuda:
            scaler.scale(seq_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            seq_loss.backward()
            optimizer.step()

        scheduler.step()

        if model.h_prev is not None:
            model.h_prev = model.h_prev.detach()

        train_loss += seq_loss.item()

        # ---- TensorBoardX per-step scalars ----
        writer.add_scalar("train/seq_loss_step", seq_loss.item(), global_step)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("train/focal_tversky_step", float(ft.item()), global_step)
        writer.add_scalar("train/dice_loss_step", float(di.item()), global_step)
        writer.add_scalar("train/ce_step", float(ce.item()), global_step)

        # ---- (Light) train dice on first frame only ----
        if pred_first is not None and gt_first is not None:
            dpc = dice_per_class(pred_first.cpu(), gt_first.cpu(), num_classes=NUM_CLASSES)
            train_dice_sum += np.array(dpc, dtype=np.float64)
            train_dice_count += 1

        # ---- Optional histograms ----
        if LOG_HISTOGRAMS and (global_step % 200 == 0):
            for name, p in model.named_parameters():
                writer.add_histogram(f"weights/{name}", p.detach().float().cpu().numpy(), global_step)
                if p.grad is not None:
                    writer.add_histogram(f"grads/{name}", p.grad.detach().float().cpu().numpy(), global_step)

        pbar.set_postfix({"loss": seq_loss.item()})
        global_step += 1

    avg_train = train_loss / max(1, len(train_loader))
    writer.add_scalar("train/seq_loss_epoch", avg_train, epoch)

    if train_dice_count > 0:
        train_dice_avg = train_dice_sum / train_dice_count
        for c in range(NUM_CLASSES):
            writer.add_scalar(f"train/dice_class_{c}", train_dice_avg[c], epoch)
        writer.add_scalar("train/dice_mean", float(train_dice_avg.mean()), epoch)

    print(f"Epoch {epoch} Train Loss: {avg_train:.4f}")

    # ---------------- VAL ----------------
    avg_val, val_dice_avg = run_validation(epoch)
    writer.add_scalar("val/seq_loss_epoch", avg_val, epoch)

    for c in range(NUM_CLASSES):
        writer.add_scalar(f"val/dice_class_{c}", val_dice_avg[c], epoch)
    writer.add_scalar("val/dice_mean", float(val_dice_avg.mean()), epoch)

    print(f"Epoch {epoch} VAL Loss: {avg_val:.4f}")
    print(f"Epoch {epoch} VAL Dice per class: {val_dice_avg} | mean={val_dice_avg.mean():.4f}")

    # ---------------- CHECKPOINTS ----------------
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_epoch{epoch}.pth")
    torch.save(model.state_dict(), ckpt_path)

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        print(">> Saving BEST model!")
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "model_best.pth"))

# done
writer.close()
print("Training finished. TensorBoard logs written to:", TB_DIR)
