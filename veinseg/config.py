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

CONFIG_DIR = ROOT / "configs"
DEFAULT_CONFIG_FILE = CONFIG_DIR / "config.yaml"

#: Everything the pipeline *generates* from raw acquisitions lives under here.
#: All three are gitignored.
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FILTERED_DIR = DATA_DIR / "filtered"

#: Not under ``data/``: this one is checked in (2760 tracked files), so it stays
#: where the existing results were produced.
AUGMENTED_DIR = ROOT / "filtered_data_augmented"

CHECKPOINT_DIR = ROOT / "checkpoints"
SEG_CHECKPOINT = CHECKPOINT_DIR / "model_best.pth"
DETECTOR_CHECKPOINT = CHECKPOINT_DIR / "empty_detector.pth"

NPZ_OUTPUT_DIR = ROOT / "npz_outputs"
VIS_OUTPUT_DIR = ROOT / "vis_outputs"
FIGURES_DIR = ROOT / "figures"
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


# ----------------------------------------------------------------------
# YAML configuration
# ----------------------------------------------------------------------
#
# The file is *optional*. Every value above is a working default, so a missing
# ``configs/config.yaml`` -- or a missing PyYAML -- degrades to those defaults
# rather than failing. Precedence is: CLI flag > YAML > default here.
#
# Only keys that something actually reads are listed. A config key that is
# parsed and then ignored is worse than no key at all, so geometry constants
# that are bound at import time (``TARGET_SIZE``, ``CLASS_COLORS``) are
# deliberately *not* exposed -- overriding them from YAML would appear to work
# and silently do nothing.

#: YAML ``(section, key)`` -> module-level path constant.
PATH_KEYS: dict[tuple[str, str], str] = {
    ("data", "raw"): "RAW_DIR",
    ("data", "processed"): "PROCESSED_DIR",
    ("data", "filtered"): "FILTERED_DIR",
    ("data", "augmented"): "AUGMENTED_DIR",
    ("data", "predictions"): "NPZ_OUTPUT_DIR",
    ("training", "checkpoint_dir"): "CHECKPOINT_DIR",
    ("inference", "checkpoint"): "SEG_CHECKPOINT",
    ("detector", "checkpoint"): "DETECTOR_CHECKPOINT",
    ("logging", "figures_dir"): "FIGURES_DIR",
    ("logging", "vis_dir"): "VIS_OUTPUT_DIR",
    ("fit", "report"): "RADII_REPORT",
}


def load_yaml(path: os.PathLike | str, *, required: bool = False) -> dict:
    """Parse a YAML config into a dict; ``{}`` when it cannot be read.

    ``required`` (set when the user names a file explicitly) turns every
    silent fallback into an error, so ``--config typo.yaml`` cannot quietly
    run with defaults.
    """
    path = Path(path)

    if not path.is_file():
        if required:
            raise FileNotFoundError(f"No such config file: {path}")
        return {}

    try:
        import yaml
    except ModuleNotFoundError:
        message = f"PyYAML is not installed, so {path} cannot be read"
        if required:
            raise RuntimeError(f"{message}. Install it with: pip install pyyaml") from None
        print(f"warning: {message}; using built-in defaults")
        return {}

    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def get_key(data: dict, section: str, key: str, default=None):
    """Read ``data[section][key]``, tolerating either level being absent."""
    value = data.get(section)
    if not isinstance(value, dict):
        return default
    got = value.get(key, default)
    return default if got is None else got


def resolve_paths(data: dict) -> dict[str, Path]:
    """Return ``{constant_name: Path}`` for every path key present in ``data``.

    Relative paths resolve against the repository root, not the working
    directory, so the pipeline behaves the same wherever it is invoked from.
    """
    out: dict[str, Path] = {}
    for (section, key), attr in PATH_KEYS.items():
        value = get_key(data, section, key)
        if value is not None:
            path = Path(value)
            out[attr] = path if path.is_absolute() else ROOT / path
    return out


def apply_geometry(data: dict) -> None:
    """Overlay the geometry constants that callers read at call time.

    ``FRAME_WIDTH_MM`` / ``FRAME_HEIGHT_MM`` are the documented landmine: they
    are assumed rather than measured, and every reported radius scales with
    them, so they need to be overridable for a different acquisition.
    """
    global FRAME_WIDTH_MM, FRAME_HEIGHT_MM, INTENSITY_CLIP

    frame_mm = get_key(data, "data", "frame_mm")
    if frame_mm is not None:
        FRAME_WIDTH_MM, FRAME_HEIGHT_MM = (float(v) for v in frame_mm)

    clip = get_key(data, "data", "intensity_clip")
    if clip is not None:
        INTENSITY_CLIP = tuple(float(v) for v in clip)


def pipeline_from_yaml(data: dict) -> PipelineConfig:
    """Build a :class:`PipelineConfig`, falling back to the dataclass defaults."""
    seg, detector, augment = SegConfig(), DetectorConfig(), AugmentConfig()

    seg.in_channels = get_key(data, "model", "in_channels", seg.in_channels)
    seg.base_channels = get_key(data, "model", "base_channels", seg.base_channels)
    seg.num_classes = get_key(data, "model", "num_classes", seg.num_classes)
    seg.dropout_p = get_key(data, "model", "dropout_p", seg.dropout_p)
    seg.use_checkpoint = get_key(data, "model", "use_checkpoint", seg.use_checkpoint)
    seg.epochs = get_key(data, "training", "epochs", seg.epochs)
    seg.lr = float(get_key(data, "training", "lr", seg.lr))
    seg.eta_min = float(get_key(data, "training", "eta_min", seg.eta_min))
    seg.val_groups = tuple(get_key(data, "training", "val_groups", seg.val_groups))
    seg.focal_tversky_weight = get_key(data, "loss", "focal_tversky_weight", seg.focal_tversky_weight)
    seg.dice_weight = get_key(data, "loss", "dice_weight", seg.dice_weight)

    detector.epochs = get_key(data, "detector", "epochs", detector.epochs)
    detector.batch_size = get_key(data, "detector", "batch_size", detector.batch_size)
    detector.lr = float(get_key(data, "detector", "lr", detector.lr))
    detector.eta_min = float(get_key(data, "detector", "eta_min", detector.eta_min))
    detector.val_fraction = get_key(data, "detector", "val_fraction", detector.val_fraction)
    detector.empty_threshold = get_key(data, "detector", "empty_threshold", detector.empty_threshold)

    augment.num_augments = get_key(data, "augment", "num_augments", augment.num_augments)
    augment.copy_originals = get_key(data, "augment", "copy_originals", augment.copy_originals)
    augment.seed = get_key(data, "augment", "seed", augment.seed)

    return PipelineConfig(seg=seg, detector=detector, augment=augment)
