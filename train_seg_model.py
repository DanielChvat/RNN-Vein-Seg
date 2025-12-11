import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from dataset import SequenceDataset
from seg_model import RNN
from loss import focal_tversky_loss, dice_loss, ClassBalancedSoftmaxCE, compute_N_i

def base_name(name):
    """Remove _AUG_xxx suffix to find the root sequence name."""
    return re.sub(r"_AUG_\d+$", "", name)

if __name__=="__main__":
    # ============================================================
    #                  DEVICE + PATHS
    # ============================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ROOT = "./filtered_data_augmented"
    CHECKPOINT_DIR = "./checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ============================================================
    #     1. GROUP SEQUENCES (Cube15, Cube16, OA, OA_AUGNNN, etc)
    # ============================================================


    all_folders = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
    groups = {}

    for folder in all_folders:
        bn = base_name(folder)
        groups.setdefault(bn, []).append(folder)

    print("\nDiscovered groups:")
    for g, members in groups.items():
        print(f"  {g}: {members}")


    # ============================================================
    #     2. PICK VALIDATION GROUPS (holds out entire sequences)
    # ============================================================

    VAL_GROUPS = ["Cube15"]      # <<--- HOLD OUT THIS WHOLE GROUP FOR VALIDATION
    TRAIN_GROUPS = [g for g in groups if g not in VAL_GROUPS]

    train_folders = []
    val_folders = []

    for g in TRAIN_GROUPS:
        train_folders.extend(groups[g])

    for g in VAL_GROUPS:
        val_folders.extend(groups[g])

    print("\nTrain groups:", TRAIN_GROUPS)
    print("Val groups:  ", VAL_GROUPS)
    print("Train folders:", train_folders)
    print("Val folders:  ", val_folders)


    # ============================================================
    #     3. LOAD DATASET AND CREATE SUBSETS
    # ============================================================

    full_dataset = SequenceDataset(ROOT)
    seq_names = full_dataset.sequence_dirs

    train_indices = [i for i, name in enumerate(seq_names) if name in train_folders]
    val_indices   = [i for i, name in enumerate(seq_names) if name in val_folders]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset   = Subset(full_dataset, val_indices)

    print(f"\nFinal Train sequences: {len(train_dataset)}")
    print(f"Final Val sequences:   {len(val_dataset)}\n")

    # ============================================================
    #     4. DATA LOADERS
    # ============================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


    # ============================================================
    #     5. MODEL + LOSSES + OPTIMIZER
    # ============================================================

    model = RNN(in_channels=1, base_channels=32, num_classes=3).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Class-balancing (computed from TRAIN ONLY)
    class_counts = compute_N_i(train_loader, num_classes=3)
    print("Class counts:", class_counts)
    criterion_ce = ClassBalancedSoftmaxCE(class_counts)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=30 * len(train_loader),
        eta_min=1e-6
    )

    scaler = GradScaler(device="cuda")

    best_val_loss = float("inf")


    # ============================================================
    #     6. TRAINING + VALIDATION LOOP
    # ============================================================

    num_epochs = 30
    TBPTT = 20 # backprop every 20 images

    for epoch in range(1, num_epochs + 1):

        # ------------------------------------------------------------
        #                     TRAINING
        # ------------------------------------------------------------
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}", ncols=120)

        for batch in pbar:
            images = batch["images"].to(device)  # B=1, T,C,H,W
            masks  = batch["masks"].to(device)

            T = images.shape[1]

            model.h_prev = None
            optimizer.zero_grad(set_to_none=True)
            
            running_seq_loss = 0.0  # for logging only

            # seq_loss = 0.0

            # with autocast(device_type="cuda"):
            #     for t in range(T):
            #         out = model(images[:, t], t_idx=t)

            #         ce = criterion_ce(out, masks[:, t])
            #         ft = focal_tversky_loss(out, masks[:, t])
            #         di = dice_loss(out, masks[:, t])

            #         loss = 0.2*ft + 0.8*di
            #         seq_loss += loss

            # seq_loss = seq_loss / T

            # scaler.scale(seq_loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            with autocast(device_type="cuda"):
                chunk_loss = 0.0
                counter = 0

                for t in range(T):
                    out = model(images[:, t], t_idx=t)

                    ce = criterion_ce(out, masks[:, t])
                    ft = focal_tversky_loss(out, masks[:, t])
                    di = dice_loss(out, masks[:, t])

                    loss = 0.2*ft + 0.8*di

                    # Just for logging
                    running_seq_loss += loss.item()

                    # For actual training
                    chunk_loss += loss
                    counter += 1

                    # ----- backprop every N frames -----
                    if counter == TBPTT or t == T-1:
                        chunk_loss = chunk_loss / counter
                        scaler.scale(chunk_loss).backward()

                        if model.h_prev is not None:
                            model.h_prev = model.h_prev.detach()

                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)

                        chunk_loss = 0.0
                        counter = 0

            scheduler.step()

            avg_seq_loss = running_seq_loss / T
            train_loss += avg_seq_loss
            pbar.set_postfix({"loss": avg_seq_loss})

            # scheduler.step()

            # if model.h_prev is not None:
            #     model.h_prev = model.h_prev.detach()

            # train_loss += seq_loss.item()
            # pbar.set_postfix({"loss": seq_loss.item()})

        avg_train = train_loss / len(train_loader)
        print(f"Epoch {epoch} Train Loss: {avg_train:.4f}")


        # ------------------------------------------------------------
        #                     VALIDATION
        # ------------------------------------------------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad(), autocast(device_type="cuda"):
            for batch in val_loader:
                images = batch["images"].to(device)
                masks  = batch["masks"].to(device)
                T = images.shape[1]

                model.h_prev = None
                seq_loss = 0.0

                for t in range(T):
                    out = model(images[:, t], t_idx=t)
                    ft = focal_tversky_loss(out, masks[:, t])
                    di = dice_loss(out, masks[:, t])
                    seq_loss += 0.2*ft + 0.8*di

                seq_loss = seq_loss / T
                val_loss += seq_loss.item()

        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch} VAL Loss: {avg_val:.4f}")

        # ------------------------------------------------------------
        #           SAVE CHECKPOINTS (BEST VAL LOSS)
        # ------------------------------------------------------------
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/model_epoch{epoch}.pth")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            print(">> Saving BEST model!")
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/model_best.pth")
