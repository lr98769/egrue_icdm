import torch.nn as nn
from lightning.pytorch import seed_everything
from torchvision.models import resnet18, densenet201
from torchvision.models import ResNet18_Weights

from src.configs.default_configs import device

class DEC:
    name = "dec"

def instantiate_dec_model(num_outputs, seed, feat_extractor="densenet"):
    seed_everything(seed, workers=True)
    if feat_extractor == "densenet":
        densenet = densenet201()
        feature_extractor = nn.Sequential(*list(densenet.children())[:-1])
        width = 1920
    elif feat_extractor == "resnet":
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        width = 512
    model = nn.Sequential(
        feature_extractor, 
        nn.Flatten(),
        nn.Linear(width, num_outputs)
    )
    return model
    

