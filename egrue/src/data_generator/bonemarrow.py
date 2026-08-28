from torchvision import transforms
import torch
from torchvision.datasets import ImageFolder
from os.path import join, exists
from datasets import load_dataset
import pandas as pd

from src.configs.bonemarrow_config import data_name, IMG_SIZE, target_col, num_classes
from src.data_processing.dataset import HugggingFaceDataset

# 1. Define your strict folder string to class index mapping
existing_classes = ('basophil',  'eosinophil', 'erythroblast', 
 'immature granulocytes', 'lymphocyte', 'monocyte',
  'neutrophil', 'platelet')


def get_ood_classes(fp_preprocessed, meaning_col="Meaning"):
    df_abbrev = pd.read_csv(
        join(fp_preprocessed, "abbreviations.csv"), sep=";").reset_index()
    df_abbrev[meaning_col] = df_abbrev[meaning_col].str.lower()
    pattern = '|'.join(existing_classes)
    ood_df = df_abbrev[~df_abbrev[meaning_col].str.contains(pattern, case=False, na=False)]
    ood_idx = ood_df["index"].to_list()
    full_classnames = df_abbrev[meaning_col].to_list()
    return ood_idx, full_classnames

def load_bonemarrow_data_dict(fp_preprocessed, dataset_name="ekim15/bone_marrow_cell_dataset"):
    transform = transforms.Compose([
        transforms.Resize(size=IMG_SIZE),
        transforms.ToTensor(),
    ])
    
    # Load Data
    full_ds = load_dataset(dataset_name, cache_dir=fp_preprocessed)
    test_set = full_ds["test"]
    
    # Remove IN classes
    ood_class_indices, ood_classnames = get_ood_classes(fp_preprocessed)
    test_set = test_set.filter(lambda example: example["label"] in ood_class_indices)
    # test_set.set_format("torch")
    test_set = HugggingFaceDataset(test_set, transform=transform)

    return {
        "target_col": target_col,
        "data_name": data_name,
        "num_classes": num_classes,
        "test_df": test_set,
        "classes": ood_classnames
    }