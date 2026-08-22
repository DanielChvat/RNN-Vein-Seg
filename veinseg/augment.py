"""Stage 4: 2D augmentation, one seed per augmented *sequence*.

KNOWN ISSUE -- the warp is NOT temporally consistent, despite what this module
used to claim. ``A.Compose(seed=n)`` seeds its RNG once at construction and then
*advances* it on every ``__call__``, so each frame of a sequence receives a
different random warp. Measured on a synthetic sequence: the vessel centroid
moves several pixels frame to frame, and HorizontalFlip fires on some frames but
not others.

This is inherited, not introduced -- ``augment2d_data_per_sequence.py`` built
one ``Compose`` per sequence and called it per frame in exactly the same way,
and reproduces the same drift. Every existing augmented sequence in
``filtered_data_augmented/`` was generated this way.

It matters because the model is recurrent: decorrelating consecutive frames
attacks the temporal structure the ConvGRU exists to exploit.

The fix is to rebuild the pipeline per frame from the same sequence seed
(verified to produce identical warps). It is not applied here because it would
change what training sees and invalidate comparison with existing results, and
because freezing the whole pipeline also freezes GaussNoise and brightness --
photometric jitter probably *should* vary per frame. Splitting geometric
(per-sequence) from photometric (per-frame) is the real fix.
"""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import albumentations as A
import numpy as np

from . import config


def build_pipeline(seed: int) -> A.Compose:
    """Albumentations pipeline, deterministic for a given ``seed``.

    Deterministic across *runs*, not across calls: the first call with a given
    seed always yields the same warp, but the second call yields a different
    one. See the module docstring.

    ``ColorJitter`` used to sit alongside ``RandomBrightnessContrast`` with
    ``saturation=0, hue=0``. On single-channel data that made it a second,
    redundant brightness/contrast jitter, so it was dropped.
    """
    return A.Compose(
        [
            A.Affine(scale=(0.9, 1.0), translate_percent=(0.0, 0.05), rotate=0, fit_output=False, p=1),
            A.ElasticTransform(alpha=5, sigma=3, approximate=True, p=1),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1),
            A.GaussNoise(std_range=(0.005, 0.02), mean_range=(0, 0), per_channel=False,
                         noise_scale_factor=1.0, p=1),
        ],
        seed=seed,
    )


def augment_sequence(src_dir: Path, dst_dir: Path, seed: int) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    pipeline = build_pipeline(seed)

    frames = sorted(p for p in src_dir.iterdir() if p.suffix == ".npz")
    for frame in frames:
        with np.load(frame) as data:
            img, mask = data["image"], data["label"]

        out = pipeline(image=img, mask=mask)
        np.savez_compressed(dst_dir / frame.name, image=out["image"], label=out["mask"])

    return len(frames)


def run(
    src_root: os.PathLike | str = config.FILTERED_DIR,
    dst_root: os.PathLike | str = config.AUGMENTED_DIR,
    cfg: config.AugmentConfig | None = None,
) -> None:
    """Write ``num_augments`` augmented copies of every sequence into ``dst_root``.

    With ``copy_originals`` set, the un-augmented sequences are copied across
    too. The old script only ever emitted the ``_AUG_N`` directories, yet the
    checked-in ``filtered_data_augmented/`` contains the originals as well --
    they were put there by some means outside this script. So a from-scratch
    re-run produced a *different* training root than the one the existing
    results came from: all-augmented, with the validation group augmented too.
    Copying the originals makes the script reproduce the layout that was
    actually trained on. Pass ``--no-copy-originals`` for the old behaviour.
    """
    cfg = cfg or config.AugmentConfig()
    src_root, dst_root = Path(src_root), Path(dst_root)

    sequences = config.discover_sequences(src_root)
    if not sequences:
        raise RuntimeError(f"No sequence directories under {src_root}")

    rng = random.Random(cfg.seed)

    for name in sequences:
        print(f"=== Augmenting: {name} ===")
        src_dir = src_root / name

        if cfg.copy_originals:
            dst = dst_root / name
            # Overlaid rather than replaced: this root holds checked-in data,
            # so no stage of the pipeline deletes a directory under it.
            shutil.copytree(src_dir, dst, dirs_exist_ok=True)
            print(f" -> copied originals to {dst}")

        for i in range(1, cfg.num_augments + 1):
            dst = dst_root / f"{name}_AUG_{i}"
            n = augment_sequence(src_dir, dst, seed=rng.randrange(2**32))
            print(f" -> augmented copy #{i}: {dst} ({n} frames)")

    print(f"\nAugmentation complete -> {dst_root}")
