import torch
import torch.nn.functional as F
import numpy as np
import os
import re

from seg_model import RNN
from dataset import SequenceDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./checkpoints/model_best.pth"
DATA_PATH = "./filtered_data"
SAVE_OUTPUT = True  
OUTPUT_DIR = "./npz_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

UPSAMPLE_FACTOR = 4  # upscale 4×

slice_re = re.compile(r"_slice_(\d+)\.npz$")

dict_slice_nums = {
    "OA": None, 
    "ICA": None, 
    "ICA2": None, 
    "Cube96": None, 
    "Cube15": None, 
    "Cube95": None, 
    "Cube16": None
}

for dataset_name in dict_slice_nums:
    npz_dir = os.path.join(DATA_PATH, dataset_name)

    slice_nums = []

    for fname in os.listdir(npz_dir):
        match = slice_re.search(fname)
        if match:
            slice_nums.append(int(match.group(1)))

    if slice_nums:
        dict_slice_nums[dataset_name] = min(slice_nums)


def main():
    dataset = SequenceDataset(DATA_PATH)

    model = RNN(in_channels=1, base_channels=32, num_classes=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        for seq_idx in range(len(dataset)):
            sample = dataset[seq_idx]
            # print("sample looks like: ", sample)
            images = sample["images"].to(DEVICE)
            masks = sample["masks"].to(DEVICE)
            seq_name = sample["seq_name"]
            slice_num = dict_slice_nums[seq_name]

            if "AUG" in seq_name:
                continue

            T = images.size(0)
            model.h_prev = None

            for t in range(T):
                print(f"Sequence: {seq_name}, Frame: {t}, Slice: {slice_num}")
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
                        os.path.join(OUTPUT_DIR, f"{seq_name}_slice_{slice_num}.npz"),
                        pred=pred_class.cpu().numpy().astype(np.uint8)
                    )

                slice_num += 1


            print(f"Sequence {seq_name} done.\n")


if __name__ == "__main__":
    main()