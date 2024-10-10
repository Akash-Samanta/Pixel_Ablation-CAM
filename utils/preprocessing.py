# utils/preprocessing.py

import torch
import torchvision.transforms as transforms
from PIL import Image

def preprocess_image(image_path, device):
    """
    Preprocess the input image for the model.

    Args:
        image_path (str): Path to the input image.
        device (torch.device): Device to load the image tensor.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0).to(device)
    return image
