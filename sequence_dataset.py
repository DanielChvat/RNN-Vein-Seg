import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.sequences = []
        self.sequence_dirs = sorted(os.listdir(root_dir))

        for seq_name in sorted(os.listdir(root_dir)):
            seq_path = os.path.join(root_dir, seq_name)

            if not os.path.isdir(seq_path): continue

            frame_files = [
                os.path.join(seq_path, f)
                for f in sorted(os.listdir(seq_path))
                if f.endswith('.npz')
            ]

            if len(frame_files) > 0:
                self.sequences.append(frame_files)

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        frame_files = self.sequences[idx]
        seq_folder = self.sequence_dirs[idx]

        imgs = []
        masks = []

        for path in frame_files:
            npz = np.load(path)
            img = npz["image"]
            mask = npz["label"]          

            # Ensure image is CHW
            if img.ndim == 2:
                img = img[None, :, :]

            img = torch.tensor(img, dtype=torch.float32)
            mask = torch.tensor(mask, dtype=torch.long)

            if self.transform:
                img = self.transform(img)

            imgs.append(img)
            masks.append(mask)

        imgs = torch.stack(imgs, dim=0)
        masks = torch.stack(masks, dim=0)

        return {
            "images": imgs,
            "masks": masks,
            "seq_name": seq_folder
        }

            