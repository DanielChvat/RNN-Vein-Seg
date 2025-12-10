"""Hidden Markov Model (HMM) for selecting vessel-centre candidates
across a temporal sequence.

The module builds per-frame candidate lists using `find_vessel_center`'s
helpers, scores each candidate by how well its radius matches an
a-priori radius (emission score), and uses a transition model that
penalizes large jumps in centre location. The Viterbi algorithm finds
the most likely sequence of candidates across frames.

Functions:
- build_candidates_for_frame(image): returns list of candidates per frame
- viterbi_select(candidates_seq, apriori_radius, sigma_r, sigma_trans): runs Viterbi
- annotate_hmm_sequence(image_paths, out_dir, apriori_radius, ...): demo runner

This file depends on `find_vessel_center.py` functions.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import os
from PIL import Image

from find_vessel_center import extract_mask_from_rgb, fit_polynomial_through_nonzero, find_red_centroids, radius_of_curvature_at, center_of_curvature, visualize_result, crop_rightmost_subfigure


def _has_strict_red(image: np.ndarray, min_pixels: int = 5, r_min: int = 180, g_max: int = 120, b_max: int = 120) -> bool:
    """Return True if image contains at least `min_pixels` pixels that
    satisfy a strict red threshold. This is a fast pre-filter to avoid
    false positives from faint artifacts.
    """
    if image.ndim != 3 or image.shape[2] < 3:
        return False
    arr = image.copy()
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8)
    R = arr[..., 0]
    G = arr[..., 1]
    B = arr[..., 2]
    red_pixels = (R >= r_min) & (G <= g_max) & (B <= b_max)
    return int(red_pixels.sum()) >= int(min_pixels)


def build_candidates_for_frame(image: np.ndarray) -> List[Dict]:
    """Return a list of candidate dicts for the frame.

    Each candidate has keys:
      - 'centroid': (x,y)
      - 'R': radius (float or np.inf)
      - 'center': (cx,cy) or None
      - 'score': placeholder for emission log-likelihood
    """
    mask = extract_mask_from_rgb(image)
    # Fit polynomial through non-background pixels once per frame
    try:
        poly = fit_polynomial_through_nonzero(mask, degree=3)
    except Exception:
        poly = None

    red_mask = (mask == 2)
    centroids = find_red_centroids(red_mask)

    candidates = []
    for (cx, cy) in centroids:
        x0 = float(cx)
        if poly is None:
            R, kappa = (np.inf, 0.0)
            center = None
        else:
            R, kappa = radius_of_curvature_at(poly, x0)
            center, rabs = center_of_curvature(poly, x0)
        candidates.append({'centroid': (cx, cy), 'R': R, 'center': center})

    return candidates


def _emission_logprob(candidate: Dict, apriori_radius: float, sigma_r: float) -> float:
    """Return log-probability of observing candidate radius given apriori.

    We use a Gaussian likelihood on radius with std `sigma_r`. If R is
    infinite or non-finite, return a very low log-prob (i.e., large negative).
    """
    R = candidate.get('R', np.inf)
    if R is None or not np.isfinite(R):
        # very uncertain; give a low emission probability but not -inf to
        # allow path continuity. Use a small constant.
        return -1e3
    # Gaussian log-likelihood up to additive constant
    return -0.5 * ((R - apriori_radius) ** 2) / (sigma_r ** 2)


def _transition_logprob(cand_from: Dict, cand_to: Dict, sigma_trans: float) -> float:
    """Return log-probability of transitioning from cand_from to cand_to.

    Penalize squared Euclidean distance between centres. If a centre is
    missing, use centroid distance as fallback. If both missing, return
    a neutral log-probability (0).
    """
    if cand_from is None or cand_to is None:
        return -1e3

    c1 = cand_from.get('center') or cand_from.get('centroid')
    c2 = cand_to.get('center') or cand_to.get('centroid')
    if c1 is None or c2 is None:
        return -0.5 * (1e4) / (sigma_trans ** 2)
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    dist2 = dx * dx + dy * dy
    return -0.5 * dist2 / (sigma_trans ** 2)


def viterbi_select(candidates_seq: List[List[Dict]], apriori_radius: float,
                   sigma_r: float = 50.0, sigma_trans: float = 100.0) -> List[Optional[int]]:
    """Run Viterbi on a sequence of candidate-lists.

    Args:
        candidates_seq: list of frames; each frame is a list of candidate dicts.
        apriori_radius: prior expected radius (px)
        sigma_r: std for radius measurement
        sigma_trans: spatial std for allowed motion between frames

    Returns:
        List of chosen candidate indices per frame (or None if frame had no candidates).
    """
    T = len(candidates_seq)
    if T == 0:
        return []

    # Precompute emission logprobs
    emis = []
    for frame_cands in candidates_seq:
        emis.append([_emission_logprob(c, apriori_radius, sigma_r) for c in frame_cands])

    # DP arrays
    V = []  # list of arrays of best log-prob up to state
    backptr = []

    # Initialization
    if len(candidates_seq[0]) == 0:
        V.append(np.array([]))
        backptr.append([])
    else:
        V.append(np.array(emis[0]))
        backptr.append([-1] * len(emis[0]))

    # Recurrence
    for t in range(1, T):
        prev_cands = candidates_seq[t - 1]
        cur_cands = candidates_seq[t]
        if len(cur_cands) == 0:
            V.append(np.array([]))
            backptr.append([])
            continue

        cur_V = np.full((len(cur_cands),), -np.inf)
        cur_bp = [-1] * len(cur_cands)

        if len(prev_cands) == 0:
            # No previous candidates: just take emission for current
            for j in range(len(cur_cands)):
                cur_V[j] = emis[t][j]
                cur_bp[j] = -1
        else:
            for j, cj in enumerate(cur_cands):
                best_val = -np.inf
                best_i = -1
                for i, pi in enumerate(prev_cands):
                    trans = _transition_logprob(pi, cj, sigma_trans)
                    val = V[t - 1][i] + trans + emis[t][j]
                    if val > best_val:
                        best_val = val
                        best_i = i
                cur_V[j] = best_val
                cur_bp[j] = best_i

        V.append(cur_V)
        backptr.append(cur_bp)

    # Termination: pick best in final frame
    chosen = [None] * T
    # find last non-empty frame from end to handle trailing empty frames
    last = None
    for t in reversed(range(T)):
        if len(V[t]) != 0:
            last = t
            break
    if last is None:
        return chosen

    best_last_idx = int(np.argmax(V[last]))
    # backtrack
    idx = best_last_idx
    for t in range(last, -1, -1):
        if len(candidates_seq[t]) == 0:
            chosen[t] = None
            continue
        chosen[t] = int(idx)
        idx = backptr[t][idx]
        if idx == -1:
            # previous frames before this may have candidates but no predecessor
            for tt in range(t - 1, -1, -1):
                if len(candidates_seq[tt]) == 0:
                    chosen[tt] = None
                else:
                    # choose emission-only best for this earlier frame
                    chosen[tt] = int(np.argmax(emis[tt]))
            break

    return chosen


def annotate_hmm_sequence(image_paths: List[str], out_dir: str, apriori_radius: float,
                          sigma_r: float = 50.0, sigma_trans: float = 100.0,
                          prefilter: bool = True, pre_min_pixels: int = 5,
                          pre_r_min: int = 180, pre_g_max: int = 120, pre_b_max: int = 120) -> Tuple[List[Optional[int]], List[List[Dict]]]:
    """Run candidate extraction on each image path, run Viterbi, and save annotated images.

    This version supports a strict pre-filter that checks for a minimum
    number of red pixels before running candidate extraction / HMM. Frames
    failing the pre-filter are skipped entirely (no output saved for
    skipped frames) and returned as `None` in the chosen list.

    Returns chosen indices per frame and the candidates per frame.
    """
    os.makedirs(out_dir, exist_ok=True)
    candidates_seq: List[List[Dict]] = []
    poly_coeffs_seq: List[Optional[np.ndarray]] = []
    images_orig: List[np.ndarray] = []
    images_cropped: List[Optional[np.ndarray]] = []
    prefilter_mask: List[bool] = []
    for p in image_paths:
        pil = Image.open(p).convert('RGB')
        arr = np.array(pil)
        images_orig.append(arr)
        if prefilter:
            ok = _has_strict_red(arr, min_pixels=pre_min_pixels, r_min=pre_r_min, g_max=pre_g_max, b_max=pre_b_max)
        else:
            ok = True
        prefilter_mask.append(ok)
        if ok:
            # crop to the rightmost subfigure before extracting mask/candidates
            try:
                arr_c = crop_rightmost_subfigure(arr)
            except Exception:
                arr_c = arr
            mask = extract_mask_from_rgb(arr_c)
            try:
                poly = fit_polynomial_through_nonzero(mask, degree=3)
            except Exception:
                poly = None
            poly_coeffs_seq.append(poly)
            # build candidates using the cropped image
            cands = build_candidates_for_frame(arr_c)
            images_cropped.append(arr_c)
        else:
            cands = []
            poly_coeffs_seq.append(None)
            images_cropped.append(None)
        candidates_seq.append(cands)

    # Build reduced sequence indices containing frames that passed prefilter and have candidates
    indices_with_cands = [i for i, c in enumerate(candidates_seq) if len(c) > 0]
    if len(indices_with_cands) == 0:
        # nothing to run HMM on; return list of Nones and empty candidates
        return [None] * len(images), candidates_seq

    reduced_candidates = [candidates_seq[i] for i in indices_with_cands]

    # Run Viterbi on reduced sequence
    chosen_reduced = viterbi_select(reduced_candidates, apriori_radius, sigma_r=sigma_r, sigma_trans=sigma_trans)

    # Map reduced choices back to full sequence indices. Frames without
    # candidates remain ignored (None).
    chosen_full: List[Optional[int]] = [None] * len(images)
    for idx_full, choice in zip(indices_with_cands, chosen_reduced):
        chosen_full[idx_full] = choice

    # Annotate and save only frames that passed prefilter and have a chosen candidate
    for i in range(len(images_orig)):
        # only save visualization for the cropped/mask subfigure
        cropped = images_cropped[i]
        if cropped is None:
            # skipped by prefilter or no cropped image available
            continue
        if chosen_full[i] is None:
            # passed prefilter but no candidate chosen
            continue
        idx = chosen_full[i]
        c = candidates_seq[i][idx]
        res = {'chosen_centroid': c['centroid'], 'chosen_center': c['center'], 'chosen_radius': c['R'], 'poly_coeffs': poly_coeffs_seq[i]}
        out_path = os.path.join(out_dir, os.path.basename(image_paths[i])[:-4] + '_hmm.png')
        visualize_result(cropped, res, out_path)

    return chosen_full, candidates_seq


if __name__ == '__main__':
    # quick demo on a short ICA sequence if module executed directly
    import glob
    seq = sorted(glob.glob('vis_outputs/ICA_frame00*.png'))[:8]
    if len(seq) == 0:
        print('No ICA frames found in vis_outputs to demo HMM')
    else:
        chosen, cands = annotate_hmm_sequence(seq, 'vis_outputs/hmm_demo', apriori_radius=400.0, sigma_r=200.0, sigma_trans=200.0)
        print('HMM chosen indices:', chosen)
