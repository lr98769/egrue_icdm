import tensorflow as tf
import numpy as np
from scipy.stats import pearsonr
from keras import regularizers
from keras.layers import Dense, Dropout
from torch.nn import Module, Softplus, Linear, Sequential, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from src.display.display_history import display_history

class AE_Predictor(Module):
    def __init__(
        self, num_features, 
        num_encoder_layers, encoder_width, 
        num_decoder_layers, decoder_width):
        super().__init__()
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.encoder_width = encoder_width
        self.decoder_width = decoder_width

        self.num_features = num_features

        self.define_encoder()
        self.define_decoder()
        self.define_classifier()

    def define_encoder(self):
        self.encoder = define_mlp(
            num_layers=self.num_encoder_layers, # Tunable
            intermediate_dim=self.encoder_width, # Tunable
            input_dim=self.num_features, 
            output_dim=self.encoder_width
        )

    def define_decoder(self):
        self.decoder = define_mlp(
            num_layers=self.num_decoder_layers, # Tunable
            intermediate_dim=self.decoder_width, # Tunable
            input_dim=self.encoder_width, 
            output_dim=self.num_features
        )

    def define_classifier(self):
        modules = []
        modules.append(Dropout(p=0.5))
        modules.append(Linear(self.encoder_width, 1)) # Binary classification
        # No activation since we are using BCEWithLogitsLoss
        self.classifier = Sequential(*modules)
    
    def forward(self, x):
        encoder_output = self.encoder(x)
        classifier_output = self.classifier(encoder_output)
        decoder_output = self.decoder(encoder_output)
        return classifier_output.flatten(), decoder_output
    
def define_mlp(num_layers, input_dim, intermediate_dim, output_dim):
        modules = []
        in_size, out_size = input_dim, intermediate_dim
        for i in range(num_layers-1):
            modules.append(Linear(in_size, out_size))
            modules.append(Softplus())
            in_size = intermediate_dim
        modules.append(Linear(out_size, output_dim))
        modules.append(Softplus())
        return Sequential(*modules)