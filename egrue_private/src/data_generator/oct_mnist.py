from medmnist import OCTMNIST
from torchvision.models import ResNet18_Weights 
from torchvision import transforms
import torch

from src.configs.octmnist_config import data_name, IMG_SIZE, target_col, num_classes

def target_transform(target):
    return torch.tensor(target, dtype=torch.long)[0]

def load_octmnist_data_dict(fp_preprocessed, greyscale_transform=True):
    if greyscale_transform:
        transform = transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(size=IMG_SIZE),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
    
    # Create datasets for training & validation, download if necessary
    train_set = OCTMNIST(
        root=fp_preprocessed, split="train", transform=transform, download=True,
        target_transform=target_transform, size=224
    )
    train_set.labels
    val_set = OCTMNIST(
        root=fp_preprocessed, split="val", transform=transform, download=True,
        target_transform=target_transform, size=224
    )
    test_set = OCTMNIST(
        root=fp_preprocessed, split="test", transform=transform, download=True,
        target_transform=target_transform, size=224
    )
    
    return {
        "target_col": target_col,
        "data_name": data_name,
        "num_classes": num_classes,
        "train_df": train_set,
        "val_df": val_set,
        "test_df": test_set,
        "classes": ('CNV', 'DME', 'Drusen', 'Normal')
    }