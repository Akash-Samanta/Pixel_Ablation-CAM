import cv2
import numpy as np

def overlay_heatmap_on_image(original_image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlay a heatmap on the original image.

    Args:
        original_image (np.array): The original image.
        cam (np.array): The computed CAM.
        alpha (float): The transparency of the heatmap overlay.
        colormap: The colormap to use for the heatmap.

    Returns:
        overlay_image (np.array): The image with the heatmap overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
    overlay_image = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
    return overlay_image

import cv2

def draw_bounding_box(image, bbox_coords):
    """
    Draws a bounding box on the image.

    Args:
        image (numpy.ndarray): The image on which to draw the bounding box.
        bbox_coords (tuple): Bounding box coordinates (x1, y1, x2, y2).

    Returns:
        numpy.ndarray: The image with the bounding box drawn on it.
    """
    # Ensure image is in the expected format (height, width, channels)
    if image.ndim == 2:  # If it's a grayscale image, convert to BGR
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    x1, y1, x2, y2 = bbox_coords
    # Draw rectangle on the image
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green color
    return image


def threshold_cam(cam, percentile=80):
    """
    Threshold the CAM to create a binary mask.

    Args:
        cam (np.array): The computed CAM.
        percentile (float): The percentile threshold to apply.

    Returns:
        binary_mask (np.array): The binary mask indicating important regions.
    """
    threshold_value = np.percentile(cam, percentile)
    binary_mask = (cam >= threshold_value).astype(np.uint8) * 255
    return binary_mask


def generate_explanation_map(original_image, cam, original_shape):
    """
    Generate an explanation map where only the top regions are highlighted.

    Args:
        original_image (np.array): The original image.
        cam (np.array): The computed CAM.
        original_shape (tuple): The original shape of the image.

    Returns:
        explanation_image (np.array): The explanation image with highlighted top regions.
    """
    # Resize the CAM to the original image shape
    cam_resized = cv2.resize(cam, original_shape[::-1], interpolation=cv2.INTER_LINEAR)
    
    # Normalize the CAM
    cam_resized = cam_resized.astype(np.float32) / cam_resized.max()
    
    # Threshold the CAM to only keep the top 20% or 30% regions
    threshold_value = np.percentile(cam_resized, 80)  # 80th percentile threshold by default
    cam_thresholded = np.where(cam_resized >= threshold_value, cam_resized, 0)
    
    # Normalize the thresholded CAM
    cam_thresholded = cam_thresholded.astype(np.float32) / cam_thresholded.max()
    
    # Multiply the original image by the thresholded CAM
    explanation_image = original_image.astype(np.float32) / 255.0
    explanation_image *= cam_thresholded[..., np.newaxis]
    
    # Convert back to uint8 format
    explanation_image = (explanation_image * 255).astype(np.uint8)
    
    return explanation_image

