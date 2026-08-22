"""Stage 1: raw PNG slices -> normalised ``.npz`` frames under ``processed_data/``."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from . import config


def resize_to_target(arr: np.ndarray, target_size: tuple[int, int], is_label: bool) -> np.ndarray:
    """Resize a 2D array to ``(H, W)``; nearest for labels, bilinear for images."""
    resample = Image.NEAREST if is_label else Image.BILINEAR
    img = Image.fromarray(arr)
    img = img.resize((target_size[1], target_size[0]), resample)
    return np.array(img, dtype=arr.dtype)


def _pair_slices(img_folder: Path, lbl_folder: Path) -> list[tuple[Path, Path]]:
    """Pair image and label files.

    Prefers matching on filename stem. The previous implementation sorted both
    folders independently and zipped them, which silently mispaired images with
    labels whenever the two naming schemes sorted differently.
    """
    imgs = sorted(p for p in img_folder.iterdir() if p.is_file())
    lbls = sorted(p for p in lbl_folder.iterdir() if p.is_file())

    if not imgs:
        raise FileNotFoundError(f"No image slices in {img_folder}")

    by_stem = {p.stem: p for p in lbls}
    if len(by_stem) == len(lbls) and all(p.stem in by_stem for p in imgs):
        return [(p, by_stem[p.stem]) for p in imgs]

    if len(imgs) != len(lbls):
        raise ValueError(
            f"{img_folder} has {len(imgs)} images but {lbl_folder} has {len(lbls)} labels, "
            "and their stems do not match. Cannot pair them safely."
        )

    print(
        f"  ! stems differ between {img_folder.name}/ and {lbl_folder.name}/; "
        "falling back to positional pairing -- verify the result"
    )
    return list(zip(imgs, lbls))


def create_npz_slices(img_folder: Path, lbl_folder: Path, output_folder: Path, prefix: str) -> int:
    """Blank the watermark, downscale, window and min-max normalise each slice."""
    output_folder.mkdir(parents=True, exist_ok=True)
    pairs = _pair_slices(img_folder, lbl_folder)
    print(f"  Found {len(pairs)} paired slices.")

    lo, hi = config.INTENSITY_CLIP

    for idx, (img_p, lbl_p) in enumerate(pairs):
        img_raw = np.asarray(Image.open(img_p), dtype=np.float32)
        lbl_raw = np.asarray(Image.open(lbl_p))

        if lbl_raw.ndim != 2:
            raise ValueError(
                f"{lbl_p} has shape {lbl_raw.shape}; labels must be single-channel class indices."
            )

        img_raw[: config.WATERMARK_H, : config.WATERMARK_W] = 0

        img = resize_to_target(img_raw, config.TARGET_SIZE, is_label=False)
        lbl = resize_to_target(lbl_raw.astype(np.uint8), config.TARGET_SIZE, is_label=True)

        img = np.clip(img, lo, hi)
        span = img.max() - img.min()
        img = (img - img.min()) / (span + 1e-8)

        np.savez_compressed(
            output_folder / f"{prefix}_slice_{idx:04d}.npz",
            image=img.astype(np.float32),
            label=lbl,
        )

    print(f"  Saved {len(pairs)} NPZ slices to {output_folder}")
    return len(pairs)


def run(raw_dir: os.PathLike | str = config.RAW_DIR, out_dir: os.PathLike | str = config.PROCESSED_DIR) -> None:
    """Preprocess every sequence directory found under ``raw_dir``."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    sequences = config.discover_sequences(raw_dir)
    if not sequences:
        raise RuntimeError(f"No sequence directories under {raw_dir}")

    print(f"Preprocessing {len(sequences)} sequences: {', '.join(sequences)}")
    total = 0
    for name in sequences:
        print(f"\nProcessing dataset: {name}")
        total += create_npz_slices(
            img_folder=raw_dir / name / "imgs",
            lbl_folder=raw_dir / name / "masks",
            output_folder=out_dir / name,
            prefix=f"CASE_{name}",
        )
    print(f"\nPreprocessing complete: {total} frames -> {out_dir}")
