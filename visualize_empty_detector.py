import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from empty_mask_model import EmpyMaskCNN
from dataset import EmptyMaskDataset

device = "cuda" if torch.cuda.is_available() else "cpu"


model = EmpyMaskCNN().to(device)
model.load_state_dict(torch.load("empty_detector.pth", map_location=device))
model.eval()
print("Loaded empty mask detector model")

root_dir = './processed_data'
dataset = EmptyMaskDataset(root_dir)
num_frames = len(dataset)

print(f"Loaded {num_frames} frames.")

def predict_empty(img_tensor):
    with torch.no_grad():
        x = img_tensor.unsqueeze(0).to(device)
        logit = model(x)
        return torch.sigmoid(logit).item()


img0, gt_label0 = dataset[0]
img0_np = img0.squeeze(0).numpy()
prob0 = predict_empty(img0)

fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(bottom=0.25)

image_display = ax.imshow(img0_np, cmap="gray")
title = ax.set_title(f"Frame 0 / {num_frames-1}\np(empty) = {prob0:.4f}\n gt p(empty) = {gt_label0}")
ax.axis("off")

ax_slider = plt.axes([0.15, 0.1, 0.7, 0.05])  # x, y, width, height
slider = Slider(
    ax=ax_slider,
    label="Frame Index",
    valmin=0,
    valmax=num_frames - 1,
    valinit=0,
    valstep=1
)

def update(idx):
    idx = int(slider.val)
    img, gt_label = dataset[idx]

    img_np = img.squeeze(0).numpy()
    prob = predict_empty(img)

    image_display.set_data(img_np)
    title.set_text(f"Frame {idx} / {num_frames-1}\np(empty) = {prob:.4f}\n gt p(empty) = {gt_label}")

    fig.canvas.draw_idle()

slider.on_changed(update)

def on_key(event):
    if event.key == "right":
        new_val = min(slider.val + 1, num_frames - 1)
        slider.set_val(new_val)
    elif event.key == "left":
        new_val = max(slider.val - 1, 0)
        slider.set_val(new_val)

fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()
