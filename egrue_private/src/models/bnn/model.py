from torch import nn
from torchhk import transform_model
import torchbnn as bnn
from torchvision.models import resnet18, densenet201
from torchvision.models import ResNet18_Weights

from src.configs.default_configs import device
from src.misc import set_seed_pytorch

class BNN(nn.Module):
    # This is a placeholder
    name = "bnn"

class GlobalMaxPool2d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool2d, self).__init__()
    
    def forward(self, x):
        return nn.MaxPool2d(kernel_size=x.size()[2:])(x)

def instantiate_bnn_model(
        num_outputs, seed, feat_extractor="densenet", width=None, num_layers=None, prior_sigma=0.1):
    set_seed_pytorch(seed)

    if feat_extractor == "densenet":
        densenet = densenet201()
        feature_extractor = nn.Sequential(*list(densenet.children())[:-1]).to(device)
        width = 1920
    elif feat_extractor == "resnet":
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        feature_extractor = nn.Sequential(*list(resnet.children())[:-1]).to(device)
        width = 512
    else:
        layers = []
        input_dim = 3
        for layer in range(num_layers):
            layers.append(nn.Conv2d(input_dim,width,3, padding=1))
            layers.append(nn.BatchNorm2d(width))
            layers.append(nn.ReLU())
            input_dim = width
        layers.append(GlobalMaxPool2d().to(device))
        feature_extractor = nn.Sequential(*layers)
    
    # Change all conv2 layers to bayesconv2
    feature_extractor = transform_model(feature_extractor, nn.Conv2d, bnn.BayesConv2d, 
                args={"prior_mu":0, "prior_sigma":prior_sigma, "in_channels" : ".in_channels",
                      "out_channels" : ".out_channels", "kernel_size" : ".kernel_size",
                      "stride" : ".stride", "padding" : ".padding", "bias":".bias"
                     }, 
                attrs={"weight_mu" : ".weight"},
        inplace=False
    ).to(device)
    for i in feature_extractor.named_parameters():
        i[1].to(device)
    layers = [
        feature_extractor.to(device),
        nn.Flatten().to(device),
        bnn.BayesLinear(prior_mu=0, prior_sigma=prior_sigma, in_features=width, out_features=num_outputs).to(device)
    ]
    model = nn.Sequential(*layers).to(device)
    model.name = "bnn"
    return model


