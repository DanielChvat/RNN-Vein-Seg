# RNN Vein Segmentation

A recurrent (ConvGRU) U-Net that segments 3 classes from *sequences* of OCT
frames, plus geometry code that fits a polynomial through a predicted mask and
estimates the vessel's radius of curvature.

Everything runs through one entry point:

```bash
pip install -r requirements.txt   # or: pip install -e .
python -m veinseg <stage> [options]
```

## Layout

```
configs/config.yaml   pipeline settings (optional -- see Configuration)
data/                 raw/, processed/, filtered/   -- gitignored, regenerated
figures/              analyze/ plot output
checkpoints/          model weights -- gitignored
veinseg/              the package
```

Three data roots sit outside `data/` because they are checked into git:
`filtered_data_augmented/` (what `train` reads), `npz_outputs/` (what `predict`
writes) and `vis_outputs/` (what `visualize` writes).

## Stages

| Stage | Reads | Writes |
| --- | --- | --- |
| `preprocess` | `data/raw/<SEQ>/{imgs,masks}/` | `data/processed/<SEQ>/` |
| `train-detector` | `data/processed/` | `checkpoints/empty_detector.pth` |
| `filter` | `data/processed/` | `data/filtered/<SEQ>/` |
| `augment` | `data/filtered/` | `filtered_data_augmented/` |
| `train` | `filtered_data_augmented/` | `checkpoints/model_{best,last}.pth` |
| `predict` | `data/filtered/` | `npz_outputs/<seq>_slice_<N>.npz` |
| `visualize` | `data/filtered/` | `vis_outputs/<seq>_frame<NNN>.png` |
| `benchmark` | `data/filtered/` | per-frame latency to stdout |
| `fit` | `npz_outputs/` | `radii_report.txt` |
| `analyze` | `data/processed/` | `figures/{intensity_histogram,tsne_empty_masks}.png` |
| `viewer` | `data/processed/` | interactive empty-detector browser |

Two shorthands compose them:

```bash
python -m veinseg prepare   # preprocess -> train-detector -> filter -> augment
python -m veinseg all       # prepare -> train -> predict -> fit
```

Stages also compose explicitly, in the order given:

```bash
python -m veinseg train predict fit --epochs 50 --val-groups Cube15 Cube16
python -m veinseg predict --upsample 1 --npz-dir /tmp/preds
```

`--clean` deletes `data/processed/` at the end. It is opt-in: `analyze` and
`viewer` both read that directory, and the old shell script always removed it.

## Configuration

Settings resolve **CLI flag > `configs/config.yaml` > built-in default**.

The YAML file is optional. Every key in it has a working fallback in
`veinseg/config.py`, so a missing file — or a missing PyYAML — falls back with a
warning rather than failing. The shipped `configs/config.yaml` reproduces the
built-in defaults exactly, so it changes nothing until you edit it.

```bash
python -m veinseg train --config configs/experiment.yaml
python -m veinseg train --epochs 50        # flag still wins over the file
```

Naming a config that does not exist is an error; the default one being absent is
not.

`veinseg/config.py` remains the source of truth for anything not in the YAML.
Sequence names are **discovered from disk**, not listed — they used to be
hardcoded in four scripts that drifted out of sync.

Frame size and class colours are bound at import time and are deliberately *not*
exposed in the YAML: overriding them there would look like it worked and do
nothing.

The physical extent (`data.frame_mm`, default `[10.0, 2.8]`) is what converts
pixels to millimetres in `fit`. It is assumed, not read from the data, so it
silently rescales every reported radius if the acquisition geometry differs.
That one *is* settable from the YAML.

## Model

3-level encoder → `ConvGRU` bottleneck → decoder with skips from `e1`/`e2`. Uses
`GroupNorm` rather than BatchNorm because a batch is one sequence. Sinusoidal
spatial encodings are added at `e1`/`e3` resolution and a sinusoidal temporal
encoding is indexed by frame.

Two things callers must know:

- **`forward(x, t_idx)` requires the frame index**, not just the image.
- **The recurrent state is a mutable attribute**, not a forward argument.
  `RNN.h_prev` persists across calls and `forward` overwrites it. Call
  `model.reset_state()` before every sequence.

`base_channels` must match between training and inference or the checkpoint will
not load; `load_checkpoint` raises with the key mismatch rather than loading a
partially-initialised model.

Classes: `0` background, `1` wall (green), `2` vessel (red). `fit` depends on
class `2` marking the vessel.

Loss is `0.2 * focal_tversky + 0.8 * dice` (`veinseg/losses.py`). Training holds
out whole groups (`VAL_GROUPS = ["Cube15"]`), grouping by root name so a
sequence and its augmentations cannot straddle the split.

## Data layout

```
data/raw/<SEQ>/{imgs,masks}/     hand-provided, gitignored
data/processed/<SEQ>/CASE_<SEQ>_slice_NNNN.npz     keys: image float32, label uint8
data/filtered/<SEQ>/             frames the empty-detector kept
filtered_data_augmented/         <SEQ>/ and <SEQ>_AUG_N/  -- what training reads
```

Frames are resized to 256x128 (`ORIGINAL_SIZE` 1024x512 downscaled by 4),
windowed to [-125, 275] and min-max normalised per slice.

Augmentation draws **one seed per sequence**, but the warp is *not* consistent
across the frames of that sequence — `A.Compose` advances its RNG on every call.
This is inherited from the pre-refactor script and affects all existing
augmented data. See the `veinseg/augment.py` docstring.

`augment` copies the un-augmented sequences across as well as writing the
`_AUG_N` copies. The old script wrote only the `_AUG_N` copies even though the
checked-in `filtered_data_augmented/` contains the originals, so a from-scratch
re-run did not reproduce the layout that was actually trained on. Pass
`--no-copy-originals` for the old behaviour.

## Outputs already in the repo

`npz_outputs/` holds two generations of predictions with no overlap:
`<seq>_slice_<N>.npz` (current) and `<seq>_frame<N>.npz` (older, from the
`vessel_centre_inference` branch). `radii_report.txt` was generated from the
frame-named set.
