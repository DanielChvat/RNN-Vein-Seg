# RNN Vein Segmentation Model

A recurrent (ConvGRU) U-Net that segments 3 classes from *sequences* of medical
image frames, plus geometry code that fits a polynomial through a predicted mask
and estimates vessel radius of curvature.

## File structure

```
.
├── checkpoints              # model_epochN.pth / model_best.pth      (gitignored)
├── raw_data/<SEQ>/          # imgs/ + masks/, hand-provided          (gitignored)
├── processed_data/          # preprocess.py output                   (gitignored)
├── filtered_data/           # filter_empty_images.py output          (gitignored)
├── filtered_data_augmented/ # augment2d output; what training reads
├── npz_outputs/             # predicted masks as .npz
└── vis_outputs/             # rendered PNG visualizations            (gitignored)
```

## Commands

```bash
./prepare_data.sh              # preprocess → empty-detector → filter → augment
                               # (prepare_data.ps1 is the PowerShell equivalent)
python train_seg_model.py      # trains on ./filtered_data_augmented
python return_mask_npz.py      # writes predicted masks to ./npz_outputs
python vessel_fitting.py       # radii report from npz_outputs
python benchmark.py            # inference-time benchmark
```

`prepare_data.sh` deletes `processed_data/` when it finishes. `visualize_empty_detector.py`
and `pixel_intesity_mask_correlation.py` both read `./processed_data`, so they only
work if you re-run `preprocess.py` or stop the script before its cleanup.

Data flow:

```
raw_data/<SEQ>/{imgs,masks}/
  → preprocess.py                resize 224x224, clip [-125,275], min-max normalize
  → processed_data/<SEQ>/CASE_<SEQ>_slice_NNNN.npz   keys: image (float), label (int)
  → filter_empty_images.py       drops frames the empty-detector calls empty
  → filtered_data/<SEQ>/
  → augment2d_data_per_sequence.py  → filtered_data_augmented/<SEQ>_AUG_N/
  → train_seg_model.py
```

Note that `augment2d_data_per_sequence.py` reads `filtered_data/` but writes to a
*separate* `filtered_data_augmented/`, and training reads only the latter.

Sequence names are hardcoded in three places that must be kept in sync when adding
data: `datasets` in `preprocess.py`, `DATASETS` in `augment2d_data_per_sequence.py`,
and `preprocessed_image_folders` in `pixel_intesity_mask_correlation.py`. They are
currently **out of sync** — the first two list eight sequences including `Cube24`,
while `pixel_intesity_mask_correlation.py` still lists only the original four.

## Model

3-level encoder → `ConvGRU` bottleneck → decoder with skips. Uses `GroupNorm`
rather than BatchNorm because batch size is 1.

`RNN.forward(x, t_idx)` takes the frame index as a required argument and adds a
learned positional time embedding (`pos_time[t_idx]`), so callers must pass the
timestep, not just the image.

The recurrent state is a mutable attribute rather than a forward argument:
`RNN.h_prev` persists across `forward` calls. Every caller must set
`model.h_prev = None` before each sequence and detach it between sequences, or the
graph grows without bound.

`SequenceDataset` treats each subdirectory as one sample — a whole temporal
sequence of shape `(T, C, H, W)` — hence `batch_size=1` everywhere, with `T`
varying per sequence. Augmented copies are ordinary sibling directories, so they
are training samples too; `base_name()` strips the `_AUG_N` suffix so that a
sequence and all its augmentations land in the same train/val group and cannot
leak across the split.

Training holds out whole groups for validation (`VAL_GROUPS = ["Cube15"]`), runs
under AMP (`autocast` + `GradScaler`), accumulates loss across the full sequence
and steps once per sequence. Loss is `0.2 * focal_tversky + 0.8 * dice`.

`base_channels=32` here; checkpoints will not load if train and inference
disagree on this value.

Classes: `0` background, `1` green, `2` red.

## Vessel centre inference

`vessel_fitting.py` operates on predicted masks saved as `.npz` under
`npz_outputs/`, not on rendered PNGs.

- `fit_mask_polynomial(npz_path, mask_key=None, degree=4, plot=False)` — loads a
  mask, scales pixel coordinates to millimetres (the frame is treated as
  10.0 mm wide by 2.8 mm high), fits `y = f(x)` through all nonzero pixels, then
  for each connected component of class `2` finds the nearest point on the
  polynomial and computes the radius of curvature there. Returns
  `(coeffs, radii)`; a radius is `None` where curvature is zero. Pass `plot=True`
  to draw the mask, fit, centroids and osculating circle centres.
- `report_radii_for_slices(...)` — runs the above over a directory of `.npz`
  files and writes one line per slice.

The mm scaling is hardcoded to the 10.0 x 2.8 mm frame and must be changed if the
acquisition geometry differs.
