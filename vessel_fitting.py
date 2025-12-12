import numpy as np
import matplotlib.pyplot as plt

def fit_mask_polynomial(npz_path, mask_key=None, degree=2, plot=False):
    """
    Load a .npz file as a mask, extract nonzero pixel coordinates, and fit a polynomial.
    Args:
        npz_path (str): Path to the .npz file.
        mask_key (str or None): Key for the mask array in the .npz file. If None, use the first key.
        degree (int): Degree of the polynomial to fit.
        plot (bool): Whether to plot the mask and fitted polynomial.
    Returns:
        coeffs (np.ndarray): Polynomial coefficients.
    """
    data = np.load(npz_path)
    if mask_key is None:
        mask_key = list(data.keys())[0]
    mask = data[mask_key]

    # Find all nonzero pixels
    y_indices, x_indices = np.nonzero(mask)
    if len(x_indices) == 0:
        raise ValueError("No nonzero pixels found in mask.")

    # Get mask shape and scaling factors
    height, width = mask.shape
    mm_width = 10.0
    mm_height = 2.8
    x_scale = mm_width / width
    y_scale = mm_height / height

    # Scale pixel coordinates to mm
    x_indices_mm = x_indices * x_scale
    y_indices_mm = y_indices * y_scale

    # Fit polynomial to all nonzero pixels (in mm)
    coeffs = np.polyfit(x_indices_mm, y_indices_mm, degree)
    poly = np.poly1d(coeffs)

    if plot:
        plt.imshow(mask, cmap='gray', extent=[0, mm_width, mm_height, 0], aspect='auto')
        # Plot mask pixels with value 1 and 2 separately (in mm)
        for val, color, label in zip([1, 2], ['red', 'green'], ['Mask=1', 'Mask=2']):
            y_val, x_val = np.where(mask == val)
            if len(x_val) > 0:
                x_val_mm = x_val * x_scale
                y_val_mm = y_val * y_scale
                plt.scatter(x_val_mm, y_val_mm, s=1, color=color, label=label)

        # Generate x_fit so that the polynomial stays within the y=[0, mm_height] range
        x_min = x_indices_mm.min()
        x_max = x_indices_mm.max()
        x_fit_full = np.linspace(x_min, x_max, 1000)
        y_fit_full = poly(x_fit_full)
        # Only keep points where y_fit is within [0, mm_height]
        valid = (y_fit_full >= 0) & (y_fit_full <= mm_height)
        x_fit = x_fit_full[valid]
        y_fit = y_fit_full[valid]

        # Ensure the polynomial ends at the lower edge (y=mm_height)
        # If not, extend x_fit to the x where y=mm_height (if within x_min, x_max)
        from numpy.polynomial import Polynomial
        p = Polynomial(coeffs[::-1])  # poly1d uses descending order, Polynomial uses ascending
        roots = (p - mm_height).roots()
        # Only consider real roots within the x range
        real_roots = roots[np.isreal(roots)].real
        real_roots = real_roots[(real_roots >= x_min) & (real_roots <= x_max)]
        if len(real_roots) > 0:
            x_end = real_roots.max()
            if x_end > x_fit.max():
                x_fit = np.append(x_fit, x_end)
                y_fit = np.append(y_fit, mm_height)

        plt.plot(x_fit, y_fit, color='blue', linewidth=2, label='Fitted Polynomial')

        # --- Find clusters of value 2, compute centroids, and plot osculating circle centers ---
        from scipy.ndimage import label, center_of_mass
        mask2 = (mask == 2)
        labeled, num_features = label(mask2)
        centroids = center_of_mass(mask2, labeled, range(1, num_features+1))
        for centroid in centroids:
            cy, cx = centroid
            # Convert centroid to mm
            cx_mm = cx * x_scale
            cy_mm = cy * y_scale
            # Find closest x on the polynomial to the centroid
            x_search = np.linspace(x_indices_mm.min(), x_indices_mm.max(), 1000)
            y_search = poly(x_search)
            dists = np.sqrt((x_search - cx_mm)**2 + (y_search - cy_mm)**2)
            min_idx = np.argmin(dists)
            x_poly = x_search[min_idx]
            y_poly = y_search[min_idx]

            # Compute curvature at x_poly
            dy = np.polyder(poly, 1)(x_poly)
            ddy = np.polyder(poly, 2)(x_poly)
            kappa = np.abs(ddy) / (1 + dy**2)**1.5 if (1 + dy**2) != 0 else 0
            R = None
            if kappa != 0:
                R = 1 / kappa
                # Normal vector (pointing to center of curvature)
                nx = -dy / np.sqrt(1 + dy**2)
                ny = 1 / np.sqrt(1 + dy**2)
                # Center of osculating circle
                x_center = x_poly + nx * R
                y_center = y_poly + ny * R
                # Plot centroid, closest point, and osculating circle center
                plt.scatter([cx_mm], [cy_mm], color='yellow', s=30, marker='x', label='Centroid' if 'Centroid' not in plt.gca().get_legend_handles_labels()[1] else None)
                plt.scatter([x_poly], [y_poly], color='cyan', s=30, marker='o', label='Closest Poly Pt' if 'Closest Poly Pt' not in plt.gca().get_legend_handles_labels()[1] else None)
                plt.scatter([x_center], [y_center], color='magenta', s=30, marker='*', label='Curvature Center' if 'Curvature Center' not in plt.gca().get_legend_handles_labels()[1] else None)
        plt.xlabel('Width (mm)')
        plt.ylabel('Height (mm)')
        plt.legend()
        plt.show()

    return coeffs, R

# Example usage
if __name__ == "__main__":
    # Use the provided example file and always plot
    npz_file = "npz_outputs/OA_frame30.npz"
    degree = 2
    print(f"Fitting polynomial of degree {degree} to mask in {npz_file}...")
    coeffs, R = fit_mask_polynomial(npz_file, degree=degree, plot=True)
    # print("Polynomial coefficients:", coeffs)
    if R is not None:
        print(f"Radius of curvature at last evaluated point: {R:.2f} mm")
    else:
        print("Cannot compute radius of curvature (possibly due to zero curvature).")
