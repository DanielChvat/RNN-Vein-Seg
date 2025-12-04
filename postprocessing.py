import torch
import torch.nn.functional as F
from postprocess_cnn import PostProcessCNN  # your refinement CNN
from dataset import SequenceDataset
from seg_model import RNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load models
# ----------------------------
seg_model = RNN(in_channels=1, base_channels=8, num_classes=3).to(device)
seg_model.load_state_dict(torch.load("checkpoints/model_best.pth"))
seg_model.eval()

postprocess_cnn = PostProcessCNN(in_channels=6, out_channels=1).to(device)  # 3 pred + 3 diff channels
postprocess_cnn.load_state_dict(torch.load("checkpoints/postprocess_cnn.pth"))
postprocess_cnn.eval()

# ----------------------------
# Load data
# ----------------------------
val_dataset = SequenceDataset("./filtered_data_val")
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

# ----------------------------
# Difference-guided mask function
# ----------------------------
def class_specific_diff_mask(preds, targets, num_classes=3):
    pred_labels = torch.argmax(preds, dim=1)  # (B, H, W)
    diff_mask = torch.zeros_like(preds)
    for c in range(num_classes):
        mask = ((pred_labels == c) | (targets == c)) & (pred_labels != targets)
        diff_mask[:, c] = mask.float()
    return diff_mask

# ----------------------------
# Run post-processing
# ----------------------------
with torch.no_grad():
    for batch in val_loader:
        images = batch["images"].to(device)  # (B, T, C, H, W)
        masks = batch["masks"].to(device)    # (B, T, H, W)

        B, T, C, H, W = images.shape

        refined_preds = []

        for t in range(T):
            # Original segmentation prediction
            logits = seg_model(images[:, t])  # (B, num_classes, H, W)
            probs = F.softmax(logits, dim=1)

            # Generate difference-guided mask
            diff_mask = class_specific_diff_mask(logits, masks[:, t])  # (B, 3, H, W)

            # Concatenate original probs + difference mask as input to postprocess CNN
            post_input = torch.cat([probs, diff_mask], dim=1)  # (B, 6, H, W)
            refined = postprocess_cnn(post_input)               # (B, 1, H, W)

            refined_preds.append(refined)

        # Stack time dimension back
        refined_preds = torch.stack(refined_preds, dim=1)  # (B, T, 1, H, W)

        # Optionally: threshold
        final_mask = (refined_preds > 0.5).long()
        print("Refined mask shape:", final_mask.shape)