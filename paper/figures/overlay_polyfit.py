"""Overlay the fitted polynomial (and predicted mask) on the original bmp.

Recomputes the fit from the ``.npz`` mask via ``veinseg.vessel_fitting.fit_mask``
-- the same code path as ``python -m veinseg fit`` -- rather than compositing
the rendered polyfit PNG, so the overlay is exact and fully deterministic.

The bmp and the mask must share shape and orientation (both 1024x512 here);
mm coordinates follow ``config.FRAME_WIDTH_MM`` x ``config.FRAME_HEIGHT_MM``,
identical to the standalone polyfit figure.

Usage (from anywhere):
    python figures/paper/overlay_polyfit.py
    python figures/paper/overlay_polyfit.py --bmp X.bmp --npz Y.npz --out Z.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))

from veinseg import config
from veinseg.vessel_fitting import fit_mask


def overlay(bmp_path: Path, npz_path: Path, out_path: Path,
            degree: int = 4, mask_alpha: float = 0.35) -> None:
    image = np.array(Image.open(bmp_path).convert("L"))
    with np.load(npz_path) as data:
        mask = data[list(data.keys())[0]]

    if image.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: bmp {image.shape} vs mask {mask.shape}. "
            "The overlay assumes both are the same frame at the same resolution."
        )

    result = fit_mask(npz_path, degree=degree)
    mm_w, mm_h = result.extent_mm
    extent = [0, mm_w, mm_h, 0]
    poly = np.poly1d(result.coeffs)

    # Mask classes as a translucent RGBA layer: class 1 red, vessel class green.
    rgba = np.zeros((*mask.shape, 4), dtype=float)
    rgba[mask == 1] = (1.0, 0.0, 0.0, mask_alpha)
    rgba[mask == config.VESSEL_CLASS] = (0.0, 1.0, 0.0, mask_alpha)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image, cmap="gray", extent=extent, aspect="auto")
    ax.imshow(rgba, extent=extent, aspect="auto")

    y_idx, x_idx = np.nonzero(mask)
    x_scale = mm_w / mask.shape[1]
    xs = np.linspace(x_idx.min() * x_scale, x_idx.max() * x_scale, 1000)
    ys = poly(xs)
    inside = (ys >= 0) & (ys <= mm_h)
    ax.plot(xs[inside], ys[inside], color="blue", linewidth=2, label="Fitted polynomial")

    for i, cluster in enumerate(result.clusters):
        first = i == 0
        ax.scatter(*cluster.centroid_mm, color="yellow", s=30, marker="x",
                   label="Centroid" if first else None)
        ax.scatter(*cluster.contact_mm, color="cyan", s=30, marker="o",
                   label="Closest poly pt" if first else None)
        if cluster.curvature_centre_mm is not None:
            ax.scatter(*cluster.curvature_centre_mm, color="magenta", s=30, marker="*",
                       label="Curvature centre" if first else None)

    ax.set_xlabel("Width (mm)")
    ax.set_ylabel("Height (mm)")
    ax.legend(loc="upper right")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bmp", type=Path, default=_HERE / "oa_060.bmp",
                        help="original frame (default: oa_060.bmp)")
    parser.add_argument("--npz", type=Path,
                        default=_HERE.parents[1] / "npz_outputs" / "OA_slice_60.npz",
                        help="predicted mask (default: npz_outputs/OA_slice_60.npz)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output PNG (default: <bmp stem>_polyfit_overlay.png)")
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--mask-alpha", type=float, default=0.35,
                        help="mask overlay opacity; 0 hides the mask")
    args = parser.parse_args()

    out = args.out if args.out is not None else _HERE / f"{args.bmp.stem}_polyfit_overlay.png"
    overlay(args.bmp, args.npz, out, degree=args.degree, mask_alpha=args.mask_alpha)


if __name__ == "__main__":
    main()
