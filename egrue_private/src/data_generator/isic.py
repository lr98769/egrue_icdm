from os.path import join

from src.configs.isic_config import data_name, IMG_SIZE, num_classes, target_col
from torchvision.datasets import ImageFolder
from torchvision import transforms

def load_isic_data_dict(fp_processed_data_folder):
    transform = transforms.Compose([
        transforms.Resize(size=IMG_SIZE),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    train_ds = ImageFolder(
        root=join(fp_processed_data_folder, "train"), transform=transform)
    valid_ds = ImageFolder(
        root=join(fp_processed_data_folder, "valid"), transform=transform)
    test_ds = ImageFolder(
        root=join(fp_processed_data_folder, "test"), transform=transform)
    assert train_ds.classes == valid_ds.classes
    assert train_ds.classes == test_ds.classes

    return {
        "target_col": target_col,
        "data_name": data_name,
        "num_classes": num_classes,
        "classes": train_ds.classes,
        "train_df": train_ds,
        "val_df": valid_ds,
        "test_df": test_ds,
    }