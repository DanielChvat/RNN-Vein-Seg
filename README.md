````markdown
# RNN Vein Segmentation Model

# File Structure
```
.
├── checkpoints
├── processed_data
│   ├── Cube96
│   ├── ICA
│   ├── ICA2
│   └── OA
├── raw_data
│   ├── Cube96
│   │   ├── imgs
│   │   └── masks
│   ├── ICA
│   │   ├── imgs
│   │   └── masks
│   ├── ICA2
│   │   ├── imgs
│   │   └── masks
│   └── OA
│       ├── imgs
│       └── masks
└── vis_outputs
```
````

## New utilities

This repository now includes a helper module `find_vessel_center.py` that
extracts a multiclass mask from RGB prediction/visualization images,
fits a polynomial through the non-background pixels, locates red-marked
local minima (vessel centers on the curve), computes the osculating
circle radius-of-curvature at each minima, and selects the minima whose
radius best matches an a-priori radius. It also contains a simple
visualization helper to draw the chosen circle and centroid on the
image and save an annotated copy.

Key functions
- `extract_mask_from_rgb(img)` — returns an HxW mask with values {0=black, 1=green, 2=red}.
- `fit_polynomial_through_nonzero(mask, degree=3)` — fits y = f(x) through non-background pixels.
- `find_red_centroids(red_mask)` — returns centroids of red connected components.
- `radius_of_curvature_at(poly_coeffs, x0)` — returns (R, kappa) at x0.
- `center_of_curvature(poly_coeffs, x0)` — returns ((cx, cy), radius) for osculating circle.
- `find_best_minima_center(image, apriori_radius, degree=None)` — full pipeline; returns chosen centroid, chosen center, chosen radius, polynomial coefficients, and data for all minima.
- `visualize_result(image, result, out_path, ...)` — draws chosen circle and centroid and saves annotated image.

Usage example
```
from find_vessel_center import find_best_minima_center, visualize_result
from PIL import Image
import numpy as np

img = np.array(Image.open('vis_outputs/ICA_frame010.png').convert('RGB'))
res = find_best_minima_center(img, apriori_radius=40.0)
visualize_result(img, res, 'vis_outputs/ICA_frame010_annotated.png')
```

Notes about thresholds and inputs
- The color extraction uses simple RGB thresholds; if your visualizations
  use different overlays or anti-aliased edges you may need to tweak
  thresholds in `extract_mask_from_rgb`.
- Input images must be RGB numpy arrays (HxWx3) or PIL images.

Bayesian approach for refining circle-center estimates across frames
---------------------------------------------------------------

Yes — you can (and usually should) use a Bayesian filtering approach to
refine the estimated circle center and radius across a temporal sequence
of frames. Below are recommended approaches and how they map to this
problem.

1) Kalman Filter (online)
- State vector suggestion: `[cx, cy, vx, vy, r, vr]` where `(cx,cy)` is
  circle centre, `vx,vy` are its velocities, `r` is radius and `vr` its
  rate of change.
- Process model: constant-velocity for centre and radius (linear Gaussian).
- Measurement model: direct observation of `(cx_meas, cy_meas, r_meas)` from
  `find_best_minima_center`. Use a measurement covariance that reflects
  uncertainty (e.g., large for near-flat curvature where radius is ill-conditioned).
- Advantages: fast, online, robust to noisy individual frame fits.

2) Extended/Unscented Kalman Filter
- If you prefer to treat measurements or dynamics nonlinearly (e.g. if
  you want to fit circle parameters from raw pixel data in the filter
  step) use EKF/UKF.

3) Rauch–Tung–Striebel (RTS) smoother (offline)
- If you have the full sequence and want to improve past estimates using
  future frames, run a backward RTS smoother after Kalman filtering. It
  produces smoothed state estimates that combine past and future data.

4) Full Bayesian smoothing (e.g., particle filter / MCMC)
- For multimodal uncertainty or strongly nonlinear geometry, consider a
  particle filter or batch MCMC. These are heavier but give richer
  uncertainty quantification.

Practical implementation notes
- You can implement the Kalman filter yourself using `numpy` or use
  `filterpy` or `pykalman` for convenience. For smoothing use the RTS
  implementation from literature or the same libraries.
- Key detail: measurement covariance for radius must reflect curvature
  uncertainty. When curvature is near zero your `find_best_minima_center`
  returns very large or infinite R — treat these as high-uncertainty
  measurements (very large variance) or drop the radius measurement and
  update only `(cx, cy)`.
- If multiple minima exist per frame, you can incorporate assignment by
  (a) selecting the closest centroid to the predicted centre before the
  Kalman update, or (b) performing a multi-hypothesis filter.

Would you like me to:
- implement an online Kalman filter and integrate it into `find_vessel_center.py` (fast, online), or
- implement an offline RTS smoother + example script to run across a sequence (improves past fits), or
- prototype a particle-filter approach (more complex, slower, but robust to multimodal cases)?

If you approve one option I can implement it and run a small demo over
some frames in `vis_outputs/` to show smoothed centers and annotated
images.
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
