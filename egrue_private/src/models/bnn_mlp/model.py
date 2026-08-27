import torchbnn as bnn
from torch.nn import ReLU, Sequential

from src.misc import set_seed_pytorch

def instantiate_bnn_model(num_layers, width, num_inputs, num_outputs, seed):
    set_seed_pytorch(seed)
    layers = []
    in_features, out_features = num_inputs, width
    for i in range(num_layers):
        if (i == (num_layers-1)):
            out_features = num_outputs
        layers.append(bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=in_features, out_features=out_features))
        if (i != num_layers-1):
            layers.append(ReLU())
        in_features = width
    return Sequential(*layers)

