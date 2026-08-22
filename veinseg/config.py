"""Single source of truth for paths, geometry and class definitions.

Sequence names used to be hardcoded in four separate scripts that drifted out of
sync. Nothing hardcodes them any more -- use :func:`discover_sequences`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# torch is deliberately not imported at module scope: this module is pure
# configuration, and the geometry-only tooling should not need a deep-learning
# stack just to read a path.

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "raw_data"
PROCESSED_DIR = ROOT / "processed_data"
FILTERED_DIR = ROOT / "filtered_data"
AUGMENTED_DIR = ROOT / "filtered_data_augmented"

CHECKPOINT_DIR = ROOT / "checkpoints"
SEG_CHECKPOINT = CHECKPOINT_DIR / "model_best.pth"
DETECTOR_CHECKPOINT = ROOT / "empty_detector.pth"

NPZ_OUTPUT_DIR = ROOT / "npz_outputs"
VIS_OUTPUT_DIR = ROOT / "vis_outputs"
RADII_REPORT = ROOT / "radii_report.txt"

# ----------------------------------------------------------------------
# Image geometry
# ----------------------------------------------------------------------

ORIGINAL_SIZE = (1024, 512)  # (H, W) of the raw acquisition
DOWNSCALE_FACTOR = 4
TARGET_SIZE = (
    ORIGINAL_SIZE[0] // DOWNSCALE_FACTOR,  # H = 256
    ORIGINAL_SIZE[1] // DOWNSCALE_FACTOR,  # W = 128
)

# The vendor watermark occupies the top-left corner of every raw frame.
WATERMARK_H = 150
WATERMARK_W = 200

# Intensity window applied before per-slice min-max normalisation.
INTENSITY_CLIP = (-125.0, 275.0)

# Physical extent of one frame. Used to convert pixel coordinates to
# millimetres in vessel_fitting; wrong values silently scale every radius.
FRAME_WIDTH_MM = 10.0
FRAME_HEIGHT_MM = 2.8

# ----------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------

NUM_CLASSES = 3
CLASS_NAMES = ("background", "wall", "vessel")
CLASS_COLORS = (
    (0, 0, 0),  # 0 background -> black
    (0, 255, 0),  # 1 wall       -> green
    (255, 0, 0),  # 2 vessel     -> red
)
VESSEL_CLASS = 2

# ----------------------------------------------------------------------
# Model / training defaults
# ----------------------------------------------------------------------


@dataclass
class SegConfig:
    """Hyper-parameters for the recurrent segmentation model.

    ``base_channels`` must match between training and inference or the
    checkpoint will not load.
    """

    in_channels: int = 1
    base_channels: int = 32
    num_classes: int = NUM_CLASSES
    dropout_p: float = 0.1
    use_checkpoint: bool = True

    epochs: int = 30
    lr: float = 1e-3
    eta_min: float = 1e-6

    val_groups: tuple[str, ...] = ("Cube15",)
    # Weights for `focal_tversky` / `dice` in the combined objective.
    focal_tversky_weight: float = 0.2
    dice_weight: float = 0.8


@dataclass
class DetectorConfig:
    """Hyper-parameters for the empty-mask detector."""

    epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-3
    eta_min: float = 1e-6
    val_fraction: float = 0.2
    #: Frames scoring >= this are considered empty and dropped.
    empty_threshold: float = 0.5


@dataclass
class AugmentConfig:
    """Hyper-parameters for sequence-consistent 2D augmentation."""

    num_augments: int = 5
    #: Copy the un-augmented sequences across as well, so that training and
    #: validation are not measured purely on augmented data.
    copy_originals: bool = True
    seed: int | None = None


@dataclass
class PipelineConfig:
    seg: SegConfig = field(default_factory=SegConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def device():
    """Return the best available torch device."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def discover_sequences(root: os.PathLike | str) -> list[str]:
    """Return the sorted names of every sequence directory under ``root``.

    Replaces the hardcoded sequence lists. A directory counts as a sequence if
    it is a directory; emptiness is the caller's problem.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"No such data root: {root}")
    return sorted(p.name for p in root.iterdir() if p.is_dir())


_AUG_SUFFIX = re.compile(r"_AUG_\d+$")


def base_name(name: str) -> str:
    """Strip the ``_AUG_<n>`` suffix to recover the root sequence name."""
    return _AUG_SUFFIX.sub("", name)
