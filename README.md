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