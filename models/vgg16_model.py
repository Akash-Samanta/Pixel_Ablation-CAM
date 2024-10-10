# models/vgg16_model.py

import torch
import torchvision.models as models
from torchvision.models import VGG16_Weights

def load_vgg16(device):
    """
    Load the pretrained VGG16 model.

    Args:
        device (torch.device): Device to load the model.

    Returns:
        torch.nn.Module: Pretrained VGG16 model.
    """
    # Update to use 'weights' instead of 'pretrained'
    weights = VGG16_Weights.IMAGENET1K_V1
    model = models.vgg16(weights=weights).to(device)
    model.eval()
    return model
