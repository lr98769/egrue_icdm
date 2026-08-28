from torch.nn import Module, Linear, Sequential, Dropout
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights
from torch import flatten, stack

class ResNet18(Module):
    name = "resnet18"
    def __init__(self, num_classes, dropout=0.5):
        super().__init__()
        self.num_classes = num_classes
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = Sequential()
        for name, child in list(resnet.named_children())[:-2]:
            self.encoder.add_module(name, child)
        self.pooling = list(resnet.children())[-2]
        self.dropout = Dropout(p=dropout)
        # Replace layer to suit number of classes
        self.fc = Linear(512, self.num_classes)

    def forward(self, x, activate_dropout=False):
        encoded = self.encoder(x)
        pooled = self.pooling(encoded)
        pooled = flatten(pooled, start_dim=1)
        self.dropout.training = activate_dropout
        pooled = self.dropout(pooled)
        output = self.fc(pooled)
        return output
    
    def sample_predictions_dropout(self, x, T):
        self.dropout.training = True
        encoded = self.encoder(x)
        pooled = self.pooling(encoded)
        pooled = flatten(pooled, start_dim=1)
        preds = []
        for _ in range(T):
            dropped = self.dropout(pooled) # batch, num_pred
            output = self.fc(dropped)
            preds.append(output)
        return stack(preds, dim=0) # T, batch, num_pred
