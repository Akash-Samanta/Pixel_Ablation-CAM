# models/resnet50_model.py

import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights

def load_resnet50(device):
    """
    Load the pretrained ResNet50 model.

    Args:
        device (torch.device): Device to load the model.

    Returns:
        torch.nn.Module: Pretrained ResNet50 model.
    """
   
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights).to(device)
    model.eval()
    return model

