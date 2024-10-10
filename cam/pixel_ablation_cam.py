# cam/pixel_ablation_cam.py

import torch
import numpy as np
from PIL import Image
import cv2
from utils.heatmap import postprocess_cam

class PixelAblationCAM:
    def __init__(self, model, target_layer):
        """
        Initialize PixelAblationCAM with the model and target layer.

        Args:
            model (torch.nn.Module): Pretrained model.
            target_layer (torch.nn.Module): Target convolutional layer for CAM.
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = []

    def hook_function(self, module, input, output):
        """
        Hook function to capture activations from the target layer.

        Args:
            module (torch.nn.Module): Module.
            input (torch.Tensor): Input tensor.
            output (torch.Tensor): Output tensor.
        """
        self.activations.append(output)

    def register_hook(self):
        """
        Register the forward hook to capture activations.
        """
        self.hook = self.target_layer.register_forward_hook(self.hook_function)

    def remove_hook(self):
        """
        Remove the registered forward hook.
        """
        self.hook.remove()

    def compute_pixel_ablation_cam(self, image, target_class=None):
        """
        Compute the Pixel Ablation-CAM for the given image and target class.

        Args:
            image (torch.Tensor): Preprocessed image tensor.
            target_class (int, optional): Target class index. Defaults to prediction.

        Returns:
            np.ndarray: Generated CAM as a NumPy array.
        """
        self.activations = []
        self.register_hook()
        output = self.model(image)
        self.remove_hook()

        if len(self.activations) == 0:
            raise RuntimeError("Hook did not capture any activations.")

        prediction = output.argmax(dim=1).item()
        if target_class is None:
            target_class = prediction

        activations = self.activations[0].squeeze(0).detach().cpu().numpy()
        grid_size = activations.shape[1]

        cam = np.zeros((grid_size, grid_size))

        for i in range(grid_size):
            for j in range(grid_size):
                # Ablate the pixel
                modified_activations = activations.copy()
                modified_activations[:, i, j] = 0

                # Convert modified activations back to tensor
                modified_activations_tensor = torch.from_numpy(modified_activations).unsqueeze(0).to(image.device)

                # Forward pass with modified activations
                with torch.no_grad():
                    # Replace the activations in the target layer
                    def replace_activations(module, input, output):
                        return modified_activations_tensor

                    handle = self.target_layer.register_forward_hook(replace_activations)
                    modified_output = self.model(image)
                    handle.remove()

                modified_score = modified_output[0, target_class].item()
                original_score = output[0, target_class].item()

                # Compute importance
                cam[i, j] = (original_score - modified_score) / (original_score + 1e-9)

        # Normalize CAM
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Apply weights to the original activations
        weighted_activations = activations * cam

        # Sum across the channel dimension to get the final CAM
        cam = weighted_activations.sum(axis=0)

        cam = postprocess_cam(cam)

        # Resize using OpenCV and return as NumPy array
        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LANCZOS4)

        return cam
