import numpy as np

# 1. extract the rightmost image from the visualization output
def extract_mask_from_vis_output(vis_output:np.ndarray) -> np.ndarray:
    mask = vis_output[1061:]


