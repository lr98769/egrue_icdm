from torchvision import transforms
import torch
from torchvision.datasets import ImageFolder
from os.path import join, exists

from src.configs.raabin_config import data_name, IMG_SIZE, target_col, num_classes

# 1. Define your strict folder string to class index mapping
STRING_MAPPING = {
    "basophil": 0,
    "eosinophil": 1,
    "lymphocyte": 4,
    "monocyte": 5,
    "neutrophil": 6,
    # Added fallback keys to prevent KeyError if these folders exist in Test-A
    "erythroblast": 2,          
    "immature granulocytes": 3, 
    "platelet": 7              
}

def load_raabin_data_dict(fp_preprocessed):
    transform = transforms.Compose([
        transforms.Resize(size=IMG_SIZE),
        transforms.ToTensor(),
    ])
    
    # Create datasets for training & validation, download if necessary
    fp = join(fp_preprocessed, "Test-A")
    test_set = ImageFolder(root=fp, transform=transform)
    
    # Change idx
    idx_mapping = {
        class_idx: torch.tensor(STRING_MAPPING[class_name])
        for class_idx, class_name in enumerate(test_set.classes)
    }
    test_set.targets = [idx_mapping[old_idx] for old_idx in test_set.targets]
    test_set.samples = [(path, idx_mapping[old_idx]) for path, old_idx in test_set.samples]
    
    return {
        "target_col": target_col,
        "data_name": data_name,
        "num_classes": num_classes,
        "test_df": test_set,
        "classes": (
            'basophil',  'eosinophil', 'erythroblast', 
            'immature granulocytes', 'lymphocyte', 'monocyte',
            'neutrophil', 'platelet')
            # immature granulocytes (myelocytes, metamyelocytes and promyelocytes)
    }