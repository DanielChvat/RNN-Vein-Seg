import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

from seg_model import RNN
from dataset import SequenceDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./checkpoints/model_best.pth"
DATA_PATH = "./filtered_data"
SAVE_OUTPUT = True  
OUTPUT_DIR = "./vis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = np.array([
    [0, 0, 0],        # class 0 -> black
    [0, 255, 0],      # class 1 -> green
    [255, 0, 0],      # class 2 -> red
], dtype=np.uint8)


def mask_to_rgb(mask):
    mask_np = mask.cpu().numpy().astype(np.uint8)
    return COLORS[mask_np]

def overlay_mask_on_image(img, mask_rgb, alpha=0.5):
    img_rgb = np.stack([img,img,img], axis=-1)
    img_rgb = (img_rgb * 255).astype(np.uint8)

    blended = (alpha * mask_rgb + (1 - alpha) * img_rgb).astype(np.uint8)
    return blended


def visualize_sequence(images, preds, gts, seq_name):
    T = images.size(0)

    for t in range(T):
        img = images[t, 0].cpu().numpy()
        pred = preds[t]
        gt_rgb = gts[t]
        gt_overlay = overlay_mask_on_image(img, gt_rgb)

        plt.figure(figsize=(10, 5), dpi=150)

        plt.subplot(1, 2, 1)
        plt.title("GT Overlay")
        plt.imshow(gt_overlay)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Predicted Mask")
        plt.imshow(pred)
        plt.axis("off")

        plt.tight_layout()

        if SAVE_OUTPUT:
            plt.savefig(f"{OUTPUT_DIR}/{seq_name}_frame{t:03d}.png")
            plt.close()
        else:
            plt.show()

def main():
    dataset = SequenceDataset(DATA_PATH)

    model = RNN(in_channels=1, base_channels=8, num_classes=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        for seq_idx in range(len(dataset)):
            sample = dataset[seq_idx]
            images = sample["images"].to(DEVICE)
            masks = sample["masks"].to(DEVICE)
            seq_name = sample["seq_name"]

            if "AUG" in seq_name:
                continue

            T = images.size(0)
            model.h_prev = None

            preds_rgb = []
            gts_rgb = []

            for t in range(T):
                print(f"Sequence: {seq_name}, Frame: {t}")
                out = model(images[t].unsqueeze(0), t_idx=t)
                pred_class = out.argmax(dim=1)[0]
                print(f"Unique predicted classes: {torch.unique(pred_class)}")

                pred_rgb = mask_to_rgb(pred_class)
                preds_rgb.append(pred_rgb)

                gt_rgb = mask_to_rgb(masks[t])
                gts_rgb.append(gt_rgb)
                

            visualize_sequence(images, preds_rgb, gts_rgb, seq_name)

            print(f"Sequence {seq_name} done.\n")

    print("Visualization complete.")


if __name__ == "__main__":
    main()
