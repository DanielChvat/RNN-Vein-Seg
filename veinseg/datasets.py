"""Dataset objects over the ``.npz`` frame archives.

Each ``.npz`` holds ``image`` (float32, HxW, min-max normalised) and ``label``
(uint8, HxW, class indices).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _npz_files(seq_path: Path) -> list[Path]:
    return sorted(p for p in seq_path.iterdir() if p.suffix == ".npz")


class SequenceDataset(Dataset):
    """One sample == one whole temporal sequence, shaped ``(T, C, H, W)``.

    ``T`` varies per sequence, so this is only usable with ``batch_size=1``.

    The previous implementation kept two parallel lists -- ``sequence_dirs``
    (every ``os.listdir`` entry) and ``sequences`` (only directories that
    actually contained ``.npz`` files) -- and indexed both with the same
    ``idx``. A stray file or an empty directory anywhere in the data root
    shifted the names against the data, which silently mislabelled samples and
    corrupted the train/val split built from those names. Names and frames are
    now stored together so they cannot drift apart.
    """

    def __init__(self, root_dir: os.PathLike | str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"No such data root: {self.root_dir}")

        self._samples: list[tuple[str, list[Path]]] = []
        for seq_path in sorted(p for p in self.root_dir.iterdir() if p.is_dir()):
            frames = _npz_files(seq_path)
            if frames:
                self._samples.append((seq_path.name, frames))

        if not self._samples:
            raise RuntimeError(f"No sequences containing .npz files under {self.root_dir}")

    @property
    def sequence_names(self) -> list[str]:
        """Names of the sequences actually loaded, aligned with ``__getitem__``."""
        return [name for name, _ in self._samples]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        seq_name, frame_files = self._samples[idx]

        imgs, masks = [], []
        for path in frame_files:
            with np.load(path) as npz:
                img = npz["image"]
                mask = npz["label"]

            if img.ndim == 2:  # HW -> CHW
                img = img[None]

            img = torch.from_numpy(np.ascontiguousarray(img)).float()
            mask = torch.from_numpy(np.ascontiguousarray(mask)).long()

            if self.transform:
                img = self.transform(img)

            imgs.append(img)
            masks.append(mask)

        return {
            "images": torch.stack(imgs),
            "masks": torch.stack(masks),
            "seq_name": seq_name,
            "frame_names": [p.name for p in frame_files],
        }


class EmptyMaskDataset(Dataset):
    """Flat per-frame view used to train and apply the empty-mask detector.

    Label is ``1.0`` when the ground-truth mask is entirely background.

    Frames are read lazily. The previous version decoded every ``.npz`` in
    ``__init__`` and held the whole corpus in RAM before the first batch.
    """

    def __init__(self, root_dir: os.PathLike | str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"No such data root: {self.root_dir}")

        self.paths: list[Path] = []
        self.sequence_names: list[str] = []
        self.frame_names: list[str] = []

        for seq_path in sorted(p for p in self.root_dir.iterdir() if p.is_dir()):
            for frame in _npz_files(seq_path):
                self.paths.append(frame)
                self.sequence_names.append(seq_path.name)
                self.frame_names.append(frame.name)

        if not self.paths:
            raise RuntimeError(f"No .npz frames found under {self.root_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        with np.load(self.paths[idx]) as npz:
            img = npz["image"]
            mask = npz["label"]

        if img.ndim == 2:
            img = img[None]

        label = float(mask.sum() == 0)

        img = torch.from_numpy(np.ascontiguousarray(img)).float()
        img = (img - img.mean()) / (img.std() + 1e-6)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)
