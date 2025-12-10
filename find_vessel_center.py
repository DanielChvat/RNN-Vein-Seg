"""Utilities to extract a multiclass mask from an RGB prediction image,
fit a polynomial through non-background pixels, identify local minima
(red spots), compute the radius of curvature at those minima, select the
minima whose radius best matches a supplied a-priori radius, and return
the centre of curvature (circle centre) coordinates.

Class mapping: class 0 = black/background, class 1 = green, class 2 = red

Functions:
- extract_mask_from_rgb(image)
- fit_polynomial_through_nonzero(mask, degree)
- find_red_centroids(red_mask)
- radius_of_curvature_at(poly_coeffs, x0)
- center_of_curvature(poly_coeffs, x0)
- find_best_minima_center(image, apriori_radius, degree=None)

The module works with numpy arrays (H,W,3) in RGB order. It contains
small fallbacks to label connected components without external libs.
"""

from typing import Optional, Tuple, List, Dict
import numpy as np
from PIL import Image, ImageDraw


def crop_rightmost_subfigure(img: np.ndarray, white_thresh: int = 245, min_region_width: int = 10) -> np.ndarray:
    """Crop and return the rightmost subfigure inside an RGB image.

    The images contain three subfigures side-by-side on a white canvas.
    We detect non-white columns, find contiguous non-white column regions,
    and select the rightmost region as the subfigure bounding box. A
    small margin is applied. Returns the cropped RGB image array.

    Args:
        img: HxWx3 RGB numpy array (uint8 or float 0..1).
        white_thresh: grayscale threshold above which pixels are considered white.
        min_region_width: ignore tiny regions narrower than this.
    """
    arr = img.copy()
    arr = arr[38:385, 1061:1408, :]  # Crop to expected area first
    return arr


def extract_mask_from_rgb(img: np.ndarray) -> np.ndarray:
    """Extract a multiclass mask from an RGB image.

    Args:
        img: HxWx3 RGB uint8 (or float 0..1) image containing black,
             green and red colored regions.

    Returns:
        mask: HxW integer array with values {0,1,2} mapping to
              background(black), green, red respectively.
    """
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("Expected an HxWx3 RGB image")

    arr = img.copy()
    if arr.dtype != np.uint8:
        # assume 0..1 floats
        arr = (arr * 255).astype(np.uint8)

    R = arr[..., 0].astype(int)
    G = arr[..., 1].astype(int)
    B = arr[..., 2].astype(int)

    mask = np.zeros(R.shape, dtype=np.uint8)

    # black: all channels small
    black = (R < 50) & (G < 50) & (B < 50)
    # green: strong G, small R and B
    green = (G > 150) & (R < 120) & (B < 120)
    # red: strong R, small G and B
    red = (R > 150) & (G < 120) & (B < 120)

    mask[black] = 0
    mask[green] = 1
    mask[red] = 2

    return mask


def fit_polynomial_through_nonzero(mask: np.ndarray, degree: int = 3) -> np.ndarray:
    """Fit polynomial y = f(x) through all non-zero pixels in `mask`.

    Coordinates use image convention: x = column indices, y = row indices.

    Returns:
        poly_coeffs: numpy poly coefficients (highest degree first)
    """
    ys, xs = np.nonzero(mask != 0)
    if len(xs) < degree + 1:
        # Not enough points: reduce degree
        degree = max(1, min(len(xs) - 1, degree)) if len(xs) > 1 else 1

    if len(xs) == 0:
        raise ValueError("No non-zero pixels to fit polynomial")

    # Fit y as a function of x
    coeffs = np.polyfit(xs, ys, deg=degree)
    return coeffs


def _manual_label_components(binarr: np.ndarray) -> Tuple[np.ndarray, int]:
    """A small pure-numpy/python 8-connected component labeler fallback.

    Returns labelled array and number of labels.
    """
    h, w = binarr.shape
    visited = np.zeros_like(binarr, dtype=bool)
    labels = np.zeros_like(binarr, dtype=int)
    cur_label = 0
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for r in range(h):
        for c in range(w):
            if not visited[r, c] and binarr[r, c]:
                cur_label += 1
                # flood fill
                stack = [(r, c)]
                while stack:
                    rr, cc = stack.pop()
                    if rr < 0 or rr >= h or cc < 0 or cc >= w:
                        continue
                    if visited[rr, cc] or not binarr[rr, cc]:
                        continue
                    visited[rr, cc] = True
                    labels[rr, cc] = cur_label
                    for dr, dc in neighbors:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and binarr[nr, nc]:
                            stack.append((nr, nc))

    return labels, cur_label


def find_red_centroids(red_mask: np.ndarray) -> List[Tuple[float, float]]:
    """Find centroids (x, y) of connected components in binary `red_mask`.

    Returns list of (x_centroid, y_centroid) in pixel coordinates.
    """
    binarr = (red_mask != 0).astype(bool)
    try:
        # prefer scipy if available
        import scipy.ndimage as ndi

        labeled, n = ndi.label(binarr)
        if n == 0:
            return []
        centers = ndi.center_of_mass(binarr, labeled, range(1, n + 1))
        # centers are (row, col) -> convert to (x, y)
        centroids = [(float(c), float(r)) for (r, c) in centers]
        return centroids
    except Exception:
        # fallback to cv2 if present
        try:
            import cv2

            # cv2 expects uint8
            lab = cv2.connectedComponentsWithStats((binarr).astype('uint8'), connectivity=8)
            n = lab[0]
            stats = lab[2]
            centroids = []
            for i in range(1, n):
                cx = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] / 2.0
                cy = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2.0
                centroids.append((float(cx), float(cy)))
            return centroids
        except Exception:
            # final fallback: manual label
            labels, n = _manual_label_components(binarr)
            if n == 0:
                return []
            centroids = []
            for labid in range(1, n + 1):
                ys, xs = np.nonzero(labels == labid)
                if xs.size == 0:
                    continue
                cx = xs.mean()
                cy = ys.mean()
                centroids.append((float(cx), float(cy)))
            return centroids


def radius_of_curvature_at(poly_coeffs: np.ndarray, x0: float) -> Tuple[Optional[float], Optional[float]]:
    """Compute radius of curvature R and signed curvature kappa at x0 for y=f(x).

    Returns (R, kappa). R is positive (np.inf if near-flat) and kappa may be signed.
    """
    # derivative coeffs
    p = np.poly1d(poly_coeffs)
    dp = p.deriv(1)
    ddp = p.deriv(2)

    yp = float(dp(x0))
    ypp = float(ddp(x0))
    denom = (1.0 + yp ** 2) ** 1.5
    if abs(ypp) < 1e-12:
        return (np.inf, 0.0)
    kappa = ypp / denom
    R = abs(1.0 / kappa)
    return (R, kappa)


def center_of_curvature(poly_coeffs: np.ndarray, x0: float) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
    """Return (center_x, center_y) and radius for osculating circle at x0.

    If curvature is near zero, returns (None, None).
    """
    p = np.poly1d(poly_coeffs)
    dp = p.deriv(1)
    ddp = p.deriv(2)

    y0 = float(p(x0))
    yp = float(dp(x0))
    ypp = float(ddp(x0))
    denom = (1.0 + yp ** 2) ** 1.5
    if abs(ypp) < 1e-12:
        return (None, None)
    kappa = ypp / denom
    R_signed = 1.0 / kappa

    # unit normal pointing towards centre of curvature
    norm = np.sqrt(1.0 + yp ** 2)
    nx = -yp / norm
    ny = 1.0 / norm

    cx = x0 + nx * R_signed
    cy = y0 + ny * R_signed

    return ((float(cx), float(cy)), abs(R_signed))


def find_best_minima_center(image: np.ndarray, apriori_radius: float,
                            degree: Optional[int] = None) -> Optional[Dict]:
    """Process `image` and return info for the best-matching local minima.

    Args:
        image: HxWx3 RGB image (numpy array)
        apriori_radius: a-priori radius of curvature to match (pixels)
        degree: optional polynomial degree to use. If None, degree is
                chosen from number of red spots (min 1, capped at 5).

    Returns:
        dict with keys: 'chosen_centroid', 'chosen_center', 'chosen_radius',
                          'poly_coeffs', 'all_minima' or None if ignored.
        'all_minima' is a list with entry per red spot: (centroid_x, centroid_y, R, center_x, center_y)

    Notes:
        If there are no red spots the function returns None (image ignored).
    """
    # Crop to rightmost subfigure first (many inputs contain three subplots).
    try:
        image = crop_rightmost_subfigure(image)
    except Exception:
        # if cropping fails, proceed with original image
        pass
    mask = extract_mask_from_rgb(image)
    red_mask = (mask == 2)
    centroids = find_red_centroids(red_mask)
    if len(centroids) == 0:
        # ignore image
        return None

    # decide polynomial degree if not provided
    if degree is None:
        degree = min(max(1, len(centroids)), 5)

    poly = fit_polynomial_through_nonzero(mask, degree=degree)

    minima_info = []
    for (cx, cy) in centroids:
        x0 = float(cx)
        R, kappa = radius_of_curvature_at(poly, x0)
        center, radius = center_of_curvature(poly, x0)
        minima_info.append({'centroid': (cx, cy), 'R': R, 'kappa': kappa, 'center': center})

    # choose minima with R closest to apriori_radius
    diffs = [abs(mi['R'] - apriori_radius) if mi['R'] is not None and np.isfinite(mi['R']) else np.inf for mi in minima_info]
    best_idx = int(np.argmin(diffs))
    best = minima_info[best_idx]

    return {
        'chosen_centroid': best['centroid'],
        'chosen_center': best['center'],
        'chosen_radius': best['R'],
        'poly_coeffs': poly,
        'all_minima': minima_info,
        'best_index': best_idx,
    }


def visualize_result(image: np.ndarray,
                     result: Optional[Dict],
                     out_path: str,
                     draw_circle: bool = True,
                     draw_centroid: bool = True,
                     draw_curve: bool = True,
                     circle_color=(255, 0, 0),
                     centroid_color=(0, 255, 0),
                     curve_color=(255, 255, 0),
                     thickness: int = 2) -> None:
    """Draw selected osculating circle, centroid, and fitted curve on the image and save.

    Args:
        image: HxWx3 numpy RGB image or PIL Image.
        result: dict returned by `find_best_minima_center` or None.
        out_path: path to save annotated image.
        draw_circle: whether to draw the osculating circle.
        draw_centroid: whether to draw the chosen centroid.
        draw_curve: whether to draw the fitted polynomial curve.
        circle_color: RGB tuple for circle outline.
        centroid_color: RGB tuple for centroid marker.
        curve_color: RGB tuple for curve.
        thickness: line thickness in pixels.

    Notes:
        If `result` is None, the original image is saved unchanged.
    """
    # Convert numpy array to PIL.Image if needed
    if isinstance(image, np.ndarray):
        pil = Image.fromarray(image.astype('uint8'), 'RGB')
    elif isinstance(image, Image.Image):
        pil = image.convert('RGB')
    else:
        raise ValueError('image must be a numpy array or PIL.Image')

    draw = ImageDraw.Draw(pil)

    if result is None:
        pil.save(out_path)
        return

    chosen_centroid = result.get('chosen_centroid')
    chosen_center = result.get('chosen_center')
    chosen_radius = result.get('chosen_radius')

    W, H = pil.size

    if draw_circle and chosen_center is not None and chosen_radius is not None and np.isfinite(chosen_radius):
        cx, cy = chosen_center
        R = chosen_radius
        # bounding box for ellipse
        left = cx - R
        top = cy - R
        right = cx + R
        bottom = cy + R
        # Clip to image
        bbox = [left, top, right, bottom]
        # Draw multiple concentric ellipses to emulate thickness
        for t in range(thickness):
            b = [bbox[0]-t, bbox[1]-t, bbox[2]+t, bbox[3]+t]
            draw.ellipse(b, outline=tuple(circle_color))

    if draw_centroid and chosen_centroid is not None:
        gx, gy = chosen_centroid
        # draw small cross at centroid
        l = max(2, thickness * 2)
        draw.line([(gx - l, gy), (gx + l, gy)], fill=tuple(centroid_color), width=thickness)
        draw.line([(gx, gy - l), (gx, gy + l)], fill=tuple(centroid_color), width=thickness)

    if draw_curve and result is not None and 'poly_coeffs' in result:
        poly_coeffs = result.get('poly_coeffs')
        if poly_coeffs is not None:
            # Draw polynomial curve
            p = np.poly1d(poly_coeffs)
            x_vals = np.arange(W)
            y_vals = p(x_vals)
            y_vals = np.clip(y_vals, 0, H - 1)
            points = [(int(x), int(y)) for x, y in zip(x_vals, y_vals)]
            for i in range(len(points) - 1):
                draw.line([points[i], points[i + 1]], fill=tuple(curve_color), width=thickness)

    pil.save(out_path)


if __name__ == "__main__":
    # tiny usage example (requires numpy and an RGB image array `img`):
    print("Module `find_vessel_center` provides `find_best_minima_center`.")
