# examples/example_usage.py

import sys
import os

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Append the parent directory to sys.path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from main import main
try:
    if __name__ == "__main__":
        # Define the image path and model
        image_path = '/Users/akashsamanta/Desktop/panda_00012.jpg'
        model_name = 'resnet50'
        
        # Define bounding box coordinates (example)
        bbox_coords = (152, 18, 392, 464) # Update based on your image
        
        # Define threshold percentile
        threshold_percentile = 80  # Top 20%
        
        # Run Pixel Ablation-CAM
        main(image_path, model_name, bbox_coords, threshold_percentile)
except Exception as e:
    print(f"An error occurred: {e}")
