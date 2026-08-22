"""Stage 7: vessel-centre geometry from predicted masks.

Fits ``y = f(x)`` through the predicted mask, then for each connected component
of the vessel class finds the nearest point on that curve and reports the radius
of curvature there.

Operates on the ``.npz`` masks written by :mod:`veinseg.inference`, not on
rendered PNGs.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import center_of_mass, label

from . import config


@dataclass
class ClusterFit:
    """Geometry for one connected component of the vessel class, in millimetres."""

    centroid_mm: tuple[float, float]
    #: Closest point on the fitted polynomial to the centroid.
    contact_mm: tuple[float, float]
    radius_mm: float | None
    #: Centre of the osculating circle; ``None`` when curvature is zero.
    curvature_centre_mm: tuple[float, float] | None


@dataclass
class MaskFit:
    coeffs: np.ndarray
    clusters: list[ClusterFit]
    #: Physical extent used for the pixel -> mm conversion.
    extent_mm: tuple[float, float]

    @property
    def radii(self) -> list[float | None]:
        return [c.radius_mm for c in self.clusters]


def _nearest_point_on_poly(
    poly: np.poly1d, x_lo: float, x_hi: float, cx: float, cy: float, samples: int = 1000
) -> float:
    """Return the x of the closest point on ``poly`` to ``(cx, cy)``.

    Coarse sweep followed by a refinement pass over the winning interval. The
    old single-pass 1000-point grid was also rebuilt inside the per-cluster
    loop, so its cost scaled with the number of clusters for no extra accuracy.
    """
    xs = np.linspace(x_lo, x_hi, samples)
    best = int(np.argmin((xs - cx) ** 2 + (poly(xs) - cy) ** 2))

    step = (x_hi - x_lo) / (samples - 1) if samples > 1 else 0.0
    lo, hi = xs[best] - step, xs[best] + step
    xs = np.linspace(lo, hi, samples)
    return float(xs[int(np.argmin((xs - cx) ** 2 + (poly(xs) - cy) ** 2))])


def _curvature(poly: np.poly1d, x: float) -> tuple[float | None, tuple[float, float] | None]:
    """Radius of curvature and osculating-circle centre at ``x``."""
    dy = float(np.polyder(poly, 1)(x))
    ddy = float(np.polyder(poly, 2)(x))

    # (1 + dy**2) is always >= 1, so the old guard against it being zero was
    # unreachable. Curvature vanishes only when the second derivative does.
    denom = (1 + dy**2) ** 1.5
    kappa = abs(ddy) / denom
    if kappa == 0:
        return None, None

    radius = 1.0 / kappa
    norm = np.sqrt(1 + dy**2)
    centre = (x + (-dy / norm) * radius, float(poly(x)) + (1 / norm) * radius)
    return radius, centre


def fit_mask(
    npz_path: os.PathLike | str,
    mask_key: str | None = None,
    degree: int = 4,
    vessel_class: int = config.VESSEL_CLASS,
) -> MaskFit:
    """Fit the centreline and measure curvature at each vessel cluster.

    The pixel -> mm conversion assumes the frame spans
    ``config.FRAME_WIDTH_MM`` x ``config.FRAME_HEIGHT_MM``, independent of the
    array shape. It is a ratio, so it is invariant to the upsampling applied at
    prediction time -- but it is wrong for any acquisition with a different
    physical field of view, and it fails silently when it is.
    """
    with np.load(npz_path) as data:
        key = mask_key if mask_key is not None else list(data.keys())[0]
        mask = data[key]

    y_idx, x_idx = np.nonzero(mask)
    if x_idx.size == 0:
        raise ValueError(f"No nonzero pixels in {npz_path}")

    height, width = mask.shape
    x_scale = config.FRAME_WIDTH_MM / width
    y_scale = config.FRAME_HEIGHT_MM / height

    x_mm = x_idx * x_scale
    y_mm = y_idx * y_scale

    coeffs = np.polyfit(x_mm, y_mm, degree)
    poly = np.poly1d(coeffs)
    x_lo, x_hi = float(x_mm.min()), float(x_mm.max())

    labelled, n_clusters = label(mask == vessel_class)
    centroids = center_of_mass(mask == vessel_class, labelled, range(1, n_clusters + 1))

    clusters = []
    for cy_px, cx_px in centroids:
        cx, cy = cx_px * x_scale, cy_px * y_scale
        x_contact = _nearest_point_on_poly(poly, x_lo, x_hi, cx, cy)
        radius, centre = _curvature(poly, x_contact)
        clusters.append(
            ClusterFit(
                centroid_mm=(cx, cy),
                contact_mm=(x_contact, float(poly(x_contact))),
                radius_mm=radius,
                curvature_centre_mm=centre,
            )
        )

    return MaskFit(
        coeffs=coeffs,
        clusters=clusters,
        extent_mm=(config.FRAME_WIDTH_MM, config.FRAME_HEIGHT_MM),
    )


def fit_mask_polynomial(
    npz_path: os.PathLike | str,
    mask_key: str | None = None,
    degree: int = 4,
    plot: bool = False,
) -> tuple[np.ndarray, list[float | None]]:
    """Backwards-compatible wrapper returning ``(coeffs, radii)``.

    The radius computation used to be written out twice -- once inside the
    ``plot`` branch and once inside the ``not plot`` branch -- so a fix to one
    silently left the other wrong. Both paths now share :func:`fit_mask`.
    """
    result = fit_mask(npz_path, mask_key=mask_key, degree=degree)
    if plot:
        plot_fit(npz_path, result, mask_key=mask_key)
    return result.coeffs, result.radii


def plot_fit(npz_path: os.PathLike | str, result: MaskFit, mask_key: str | None = None) -> None:
    """Draw the mask, the fitted curve, cluster centroids and curvature centres."""
    import matplotlib.pyplot as plt

    with np.load(npz_path) as data:
        key = mask_key if mask_key is not None else list(data.keys())[0]
        mask = data[key]

    mm_w, mm_h = result.extent_mm
    x_scale, y_scale = mm_w / mask.shape[1], mm_h / mask.shape[0]
    poly = np.poly1d(result.coeffs)

    plt.imshow(mask, cmap="gray", extent=[0, mm_w, mm_h, 0], aspect="auto")
    for value, colour in ((1, "red"), (config.VESSEL_CLASS, "green")):
        ys, xs = np.where(mask == value)
        if xs.size:
            plt.scatter(xs * x_scale, ys * y_scale, s=1, color=colour, label=f"Mask={value}")

    y_idx, x_idx = np.nonzero(mask)
    xs = np.linspace(x_idx.min() * x_scale, x_idx.max() * x_scale, 1000)
    ys = poly(xs)
    inside = (ys >= 0) & (ys <= mm_h)
    plt.plot(xs[inside], ys[inside], color="blue", linewidth=2, label="Fitted polynomial")

    for i, cluster in enumerate(result.clusters):
        first = i == 0
        plt.scatter(*cluster.centroid_mm, color="yellow", s=30, marker="x",
                    label="Centroid" if first else None)
        plt.scatter(*cluster.contact_mm, color="cyan", s=30, marker="o",
                    label="Closest poly pt" if first else None)
        if cluster.curvature_centre_mm is not None:
            plt.scatter(*cluster.curvature_centre_mm, color="magenta", s=30, marker="*",
                        label="Curvature centre" if first else None)

    plt.xlabel("Width (mm)")
    plt.ylabel("Height (mm)")
    plt.legend()
    plt.show()


def run(
    input_dir: os.PathLike | str = config.NPZ_OUTPUT_DIR,
    output_txt: os.PathLike | str = config.RADII_REPORT,
    degree: int = 4,
) -> None:
    """Fit every ``.npz`` under ``input_dir`` and write a per-cluster radius report."""
    input_dir, output_txt = Path(input_dir), Path(output_txt)

    # Sorted and extension-filtered. `os.listdir` order is arbitrary, and the
    # old loop fed every entry -- including non-npz files -- straight to np.load.
    npz_files = sorted(p for p in input_dir.iterdir() if p.suffix == ".npz")
    if not npz_files:
        raise RuntimeError(f"No .npz masks under {input_dir}")

    by_sequence: dict[str, list[float]] = defaultdict(list)
    skipped: list[str] = []

    with open(output_txt, "w") as out:
        for path in npz_files:
            seq_type = path.name.split("_")[0]
            try:
                result = fit_mask(path, degree=degree)
            except ValueError as exc:  # empty mask
                skipped.append(f"{path.name}: {exc}")
                continue

            out.write(f"{path.name}\n")
            for i, cluster in enumerate(result.clusters, start=1):
                if cluster.radius_mm is None:
                    out.write(f"  Cluster {i}: None\n")
                else:
                    by_sequence[seq_type].append(cluster.radius_mm)
                    out.write(f"  Cluster {i}: {cluster.radius_mm:.4f} mm\n")
            if not result.clusters:
                out.write("  (no vessel clusters)\n")

    print(f"Radii written to {output_txt}")
    if skipped:
        print(f"Skipped {len(skipped)} empty masks.")

    print("\n=== Statistics per sequence ===")
    for seq in sorted(by_sequence):
        values = by_sequence[seq]
        print(f"{seq}: mean {np.mean(values):.2f} mm  std {np.std(values):.2f} mm  n={len(values)}")
    if not by_sequence:
        print("No valid radii found.")
