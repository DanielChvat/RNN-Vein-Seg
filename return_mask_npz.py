import torch
import torch.nn.functional as F
import numpy as np
import os

from seg_model import RNN
from dataset import SequenceDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./checkpoints/model_best.pth"
DATA_PATH = "./filtered_data"
SAVE_OUTPUT = True  
OUTPUT_DIR = "./npz_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

UPSAMPLE_FACTOR = 4  # upscale 4×

def main():
    dataset = SequenceDataset(DATA_PATH)

    model = RNN(in_channels=1, base_channels=32, num_classes=3)
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

            for t in range(T):
                print(f"Sequence: {seq_name}, Frame: {t}")
                out = model(images[t].unsqueeze(0), t_idx=t)

                out_up = F.interpolate(out, scale_factor=UPSAMPLE_FACTOR,
                                       mode="bilinear", align_corners=False)

                pred_class = out_up.argmax(dim=1)[0]
                print("Upsampled pred shape:", tuple(pred_class.shape))

                unique, counts = torch.unique(pred_class, return_counts=True)
                print("Pred classes:", unique.tolist())
                print("Counts:", counts.tolist())

                if SAVE_OUTPUT:
                    np.savez_compressed(
                        os.path.join(OUTPUT_DIR, f"{seq_name}_frame{t}.npz"),
                        pred=pred_class.cpu().numpy().astype(np.uint8)
                    )


            print(f"Sequence {seq_name} done.\n")


if __name__ == "__main__":
    main()