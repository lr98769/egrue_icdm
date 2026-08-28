# Code With Reference to: https://pyimagesearch.com/2023/10/02/a-deep-dive-into-variational-autoencoders-with-pytorch/
from torch.nn import Module, Sequential, Flatten
import torch

from src.models.resnet.model import ResNet18
from src.models.resnet_rue.decoder import ResNet18Decoder, BasicBlockDec

class RueResNet18(Module):
    name = "resnet18_rue"
    def __init__(
        self, resnet_model: ResNet18, 
        last_feat_extractor_layer_index, num_classes):
        super().__init__()
        # Assume ResNet has dropout in pentultimate layer
        # Derived Params
        self.num_classes = resnet_model.num_classes
        # New Params
        self.last_feat_extractor_layer_index = last_feat_extractor_layer_index
        # ResNet
        resnet_layers = list(resnet_model.children())
        # Load Encoder
        self.encoder = Sequential(
            *(resnet_layers[:self.last_feat_extractor_layer_index+1]))
        # Generate VaeDecoder
        # self.decoder = Decoder(
        #     input_channels=512, num_layers=5, width=128
        # )
        self.decoder = ResNet18Decoder(BasicBlockDec, [2, 2, 2, 2]) # [2, 2, 2, 2]
        # Load the Predictor
        self.classifier = Sequential(
            *(resnet_layers[self.last_feat_extractor_layer_index+1:-1]),
            Flatten(), resnet_layers[-1] # fc_layer
        )
        
    def forward(self, x, activate_dropout=False):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        self.classifier.training = activate_dropout
        prediction = self.classifier(z)
        return prediction, reconstruction
    
    def recon_predictions(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction
    
    def recon_error(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        image_dim = tuple([i for i in range(1, len(x.shape))])
        return torch.abs(reconstruction-x).mean(dim = image_dim)
        
        
