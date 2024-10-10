import argparse
import os
import cv2 
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from models.resnet50_model import load_resnet50
from models.vgg16_model import load_vgg16
from cam.pixel_ablation_cam import PixelAblationCAM
from utils.preprocessing import preprocess_image
from utils.explanation_map import overlay_heatmap_on_image, draw_bounding_box, threshold_cam, generate_explanation_map
import warnings

# Suppress specific torchvision warnings
warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated")
warnings.filterwarnings("ignore", message="Arguments other than a weight enum or `None` for 'weights' are deprecated")

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(image_path, model_name, bbox_coords, threshold_percentile=80):
    """
    Main function to generate and display Pixel Ablation-CAM.

    Args:
        image_path (str): Path to the input image.
        model_name (str): Model architecture to use ('resnet50' or 'vgg16').
        bbox_coords (tuple): Bounding box coordinates (x1, y1, x2, y2).
        threshold_percentile (float): Percentile for CAM thresholding (default: 80).
    """
    try:
        logging.info("Starting Pixel Ablation-CAM process.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        
        # Load the specified model
        if model_name.lower() == "resnet50":
            logging.info("Loading ResNet50 model.")
            model = load_resnet50(device)
            target_layer = model.layer4[-1]
        elif model_name.lower() == "vgg16":
            logging.info("Loading VGG16 model.")
            model = load_vgg16(device)
            target_layer = model.features[-1]
        else:
            logging.error("Model not supported. Choose 'resnet50' or 'vgg16'.")
            raise ValueError("Model not supported. Choose 'resnet50' or 'vgg16'.")
    
        # Initialize Pixel Ablation-CAM
        logging.info("Initializing Pixel Ablation-CAM.")
        pixel_ablation_cam = PixelAblationCAM(model, target_layer)
    
        # Preprocess the image
        if not os.path.exists(image_path):
            logging.error(f"Image path does not exist: {image_path}")
            raise FileNotFoundError(f"Image path does not exist: {image_path}")
    
        logging.info(f"Preprocessing image: {image_path}")
        image = preprocess_image(image_path, device)
        original_image = Image.open(image_path).resize((224, 224))
        original_image_cv = cv2.imread(image_path)  # For generating explanation map
    
        # Compute CAM
        logging.info("Computing Pixel Ablation-CAM.")
        cam = pixel_ablation_cam.compute_pixel_ablation_cam(image)
    
        # Normalize CAM for heatmap visualization
        cam_normalized = cam / cam.max() if cam.max() != 0 else cam
    
        # Generate Binary Thresholded Explanation Map
        logging.info("Generating Binary Thresholded Explanation Map.")
        binary_mask = threshold_cam(cam_normalized, percentile=threshold_percentile)
    
        # Explanation map where only the top regions are highlighted
        logging.info("Generating explanation map with top regions highlighted.")
        explanation_map = generate_explanation_map(original_image_cv, binary_mask, original_image_cv.shape[:2])
    
        # Overlay Binary Mask on Original Image
        logging.info("Overlaying Binary Mask on Original Image.")
        binary_mask_rgb = cv2.applyColorMap(binary_mask, cv2.COLORMAP_JET)
        binary_mask_rgb = cv2.cvtColor(binary_mask_rgb, cv2.COLOR_BGR2RGB)
        binary_mask_pil = Image.fromarray(binary_mask_rgb).resize(original_image.size, Image.Resampling.LANCZOS)
        binary_mask_pil = binary_mask_pil.convert("RGBA")
        original_rgba = original_image.convert("RGBA")
        overlayed_binary = Image.alpha_composite(original_rgba, binary_mask_pil)

        # Convert overlayed binary to NumPy array before drawing bounding box
        overlayed_binary_np = np.array(overlayed_binary)  # Ensure this is a NumPy array
    
        # Draw Bounding Box on Overlayed Binary Mask
        logging.info("Drawing Bounding Box on Binary Mask Overlay.")
        overlayed_binary_with_bbox = draw_bounding_box(overlayed_binary_np, bbox_coords)

        # Draw the Ground Truth Bounding Box on Explanation Map
        logging.info("Drawing Ground Truth Bounding Box on Explanation Map.")
        cv2.rectangle(explanation_map, 
                      (bbox_coords[0], bbox_coords[1]), 
                      (bbox_coords[2], bbox_coords[3]), 
                      color=(255, 0, 0),  # Red color
                      thickness=2)  # Thickness of the bounding box
    
        # Create the Heatmap in 'jet' Colormap
        logging.info("Creating Heatmap in 'jet' Colormap.")
        heatmap_jet = cam_normalized * 255  # Scale to [0,255]
        heatmap_jet = heatmap_jet.astype(np.uint8)
    
        # Plotting the Results
        logging.info("Plotting the Results.")
        plt.figure(figsize=(18, 6))
    
        # Plot 1: Original Image
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(original_image)
        plt.axis('off')
    
        # Plot 2: Heatmap in 'jet' Colormap
        plt.subplot(1, 3, 2)
        plt.title("Heatmap (Jet)")
        plt.imshow(original_image)
        plt.imshow(heatmap_jet, cmap='jet', alpha=0.5)
        plt.axis('off')
    
        # Plot 3: Explanation Map with Bounding Box
        plt.subplot(1, 3, 3)
        plt.title("Explanation Map with Ground Truth Box")
        plt.imshow(explanation_map)
        plt.axis('off')
    
        plt.tight_layout()
        plt.show()
        logging.info("Pixel Ablation-CAM process completed successfully.")

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except ValueError as e:
        logging.error(f"Value error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pixel Ablation-CAM Implementation")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model', type=str, required=True, choices=['resnet50', 'vgg16'], help='Model architecture to use')
    parser.add_argument('--bbox', type=int, nargs=4, default=[152, 18, 392, 464], help='Bounding box coordinates (x1 y1 x2 y2)')
    parser.add_argument('--threshold', type=float, default=80, help='Threshold percentile for binary mask (default: 80)')
    
    args = parser.parse_args()
    
    main(args.image_path, args.model, tuple(args.bbox), args.threshold)
