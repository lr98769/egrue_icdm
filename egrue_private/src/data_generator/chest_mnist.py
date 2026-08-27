from medmnist import ChestMNIST
from torchvision.models import ResNet18_Weights 
from torchvision import transforms
import torch

from src.configs.chestmnist_config import data_name, IMG_SIZE

def target_transform(target):
    return torch.tensor(target, dtype=torch.long)[0]

def load_chestmnist_data_dict(fp_preprocessed, greyscale_transform=True, only_test=False, num_classes=2):
    if greyscale_transform:
        transform = transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(size=IMG_SIZE),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
    
    # Create datasets for training & validation, download if necessary
    if only_test:
        test_set = ChestMNIST(root=fp_preprocessed, split="test", transform=transform, download=True, size=224, target_transform=target_transform)
        return {
            "target_col": "class",
            "data_name": data_name,
            "num_classes": 2,
            "test_df": test_set,
            "classes": ("no atelectasis", "atelectasis")
        }
        
    train_set = ChestMNIST(root=fp_preprocessed, split="train", transform=transform, download=True, size=224, target_transform=target_transform)
    val_set = ChestMNIST(root=fp_preprocessed, split="val", transform=transform, download=True, size=224, target_transform=target_transform)
    test_set = ChestMNIST(root=fp_preprocessed, split="test", transform=transform, download=True, size=224, target_transform=target_transform)
    
    return {
        "target_col": "class",
        "data_name": data_name,
        "num_classes": num_classes,
        "train_df": train_set,
        "val_df": val_set,
        "test_df": test_set,
        "classes": ("no atelectasis", "atelectasis")
    }