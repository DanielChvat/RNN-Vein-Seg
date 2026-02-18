import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import label
from collections import defaultdict


def geometric_median(points, eps=1e-5, max_iter=200):
    y = points.mean(axis=0)
    for _ in range(max_iter):
        d = np.linalg.norm(points - y, axis=1)
        d[d < eps] = eps
        y_new = np.sum(points / d[:, None], axis=0) / np.sum(1 / d)
        if np.linalg.norm(y - y_new) < eps:
            break
        y = y_new
    return y


def reconstruct_collapsed_vessel(npz_path, mask_key=None,
                                  alpha=0.25,
                                  plot=False):

    data = np.load(npz_path)
    if mask_key is None:
        mask_key = list(data.keys())[0]
    mask = data[mask_key]

    height, width = mask.shape
    mm_width = 10.0
    mm_height = 2.8
    x_scale = mm_width / width
    y_scale = mm_height / height

    mask2 = (mask == 2)
    labeled, num_features = label(mask2)

    if num_features < 2:
        return None, None
        # raise ValueError("Need at least two clusters.")

    clusters = []
    for i in range(1, num_features + 1):
        ys, xs = np.where(labeled == i)
        if len(xs) > 10:
            pts = np.column_stack([xs * x_scale, ys * y_scale])
            clusters.append(pts)

    clusters = sorted(clusters, key=lambda c: len(c), reverse=True)[:2]
    print("clusters: ", clusters)

    if not clusters:
        return None, None

    P1 = geometric_median(clusters[0])

    if len(clusters) <= 1:
        return None, None

    P2 = geometric_median(clusters[1])

    chord = P2 - P1
    d = np.linalg.norm(chord)
    t = chord / d
    n = np.array([-t[1], t[0]])

    midpoint = (P1 + P2) / 2

    # Reconstruction depth
    h = alpha * d

    # Reconstructed center (vertex)
    C = midpoint + h * n

    # Parabola parameter
    a = -4 * h / d**2

    # Radius of curvature at vertex
    R = d**2 / (8 * h)

    # Build arc for visualization
    x_local = np.linspace(-d/2, d/2, 200)
    y_local = a * x_local**2 + h

    arc_pts = []
    for x_l, y_l in zip(x_local, y_local):
        pt = midpoint + x_l * t + y_l * n
        arc_pts.append(pt)
    arc_pts = np.array(arc_pts)

    if plot:
        plt.imshow(mask, cmap='gray',
                   extent=[0, mm_width, mm_height, 0],
                   aspect='auto')

        for cluster in clusters:
            plt.scatter(cluster[:, 0], cluster[:, 1], s=3, color='red')

        plt.scatter(P1[0], P1[1], color='yellow', s=60, marker='x')
        plt.scatter(P2[0], P2[1], color='yellow', s=60, marker='x')

        plt.plot(arc_pts[:, 0], arc_pts[:, 1],
                 color='blue', linewidth=2,
                 label='Reconstructed Parabola')

        plt.scatter(C[0], C[1],
                    color='magenta', s=80,
                    label='Reconstructed Center')

        # Draw osculating circle
        circle = plt.Circle((C[0], C[1] - R),
                            R,
                            color='green',
                            fill=False,
                            linestyle='--',
                            label='Osculating Circle')
        plt.gca().add_patch(circle)

        plt.legend()
        plt.xlabel("Width (mm)")
        plt.ylabel("Height (mm)")

        out_dir = "./collapsed_reconstruction_vis"
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(npz_path))[0]
        plt.savefig(os.path.join(out_dir, f"{base}_reconstruction.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    return C, R


# Example usage
if __name__ == "__main__":
    # Use the provided example file and always plot
    npz_outputs = "npz_outputs"
    for npz_file in os.listdir(npz_outputs):
        npz_file = os.path.join(npz_outputs, npz_file)
        degree = 4
        print(f"Fitting polynomial of degree {degree} to mask in {npz_file}...")
        coeffs, radii = reconstruct_collapsed_vessel(npz_file, plot=True)
        print("radii: ", radii)
        # for i, r in enumerate(radii, start=1):
        #     if r is not None:
        #         print(f"Cluster {i}: Radius = {r:.2f} mm")
        #     else:
        #         print(f"Cluster {i}: Radius = None")