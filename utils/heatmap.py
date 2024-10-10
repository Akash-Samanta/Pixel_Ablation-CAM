# utils/heatmap.py

import numpy as np

def postprocess_cam(cam):
    """
    Normalize and convert CAM to uint8 format.

    Args:
        cam (np.ndarray): CAM array.

    Returns:
        np.ndarray: Postprocessed CAM.
    """
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)  # Avoid division by zero
    cam = np.uint8(255 * cam)
    return cam
