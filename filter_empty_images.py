import os
import numpy as np
import torch
import torch.nn as nn
import shutil

from empty_mask_model import EmpyMaskCNN
from dataset import EmptyMaskDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

RAW_DATA_DIR = "./processed_data"
FILTERED_DATA_DIR = "./filtered_data"
MODEL_PATH = "empty_detector.pth"
EMPTY_THRESHOLD = 0.5

print("Loading Empty Mask Model")
model = EmpyMaskCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def predict_empty(img):
    with torch.no_grad():
        x = img.unsqueeze(0).to(device)
        empty_logit = model(x)
        return torch.sigmoid(empty_logit).item()
    
dataset = EmptyMaskDataset(RAW_DATA_DIR)
print(f"Loaded {len(dataset)} Images.")

if not os.path.exists(FILTERED_DATA_DIR):
    os.makedirs(FILTERED_DATA_DIR)

for seq in sorted(os.listdir(RAW_DATA_DIR)):
    seq_path = os.path.join(RAW_DATA_DIR, seq)
    if os.path.isdir(seq_path):
        os.makedirs(os.path.join(FILTERED_DATA_DIR, seq), exist_ok=True)

print("Begin data filtering")
num_empty = 0
num_kept = 0

for i in range(len(dataset)):
    img_tensor, label = dataset[i]

    # Get file paths from dataset
    seq = dataset.sequence_names[i]
    filename = dataset.frame_names[i]

    input_path = os.path.join(RAW_DATA_DIR, seq, filename)
    output_path = os.path.join(FILTERED_DATA_DIR, seq, filename)

    p_empty = predict_empty(img_tensor)

    if p_empty >= EMPTY_THRESHOLD:
        num_empty += 1
    else:
        shutil.copy2(input_path, output_path)
        num_kept += 1


print("\n===== FILTERING COMPLETE =====")
print(f"Total frames:          {len(dataset)}")
print(f"Frames kept:           {num_kept}")
print(f"Frames detected empty: {num_empty}")
print(f"Output saved to:       {FILTERED_DATA_DIR}")
