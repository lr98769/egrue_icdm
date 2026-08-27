from medmnist import ChestMNIST
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from os.path import join
import pandas as pd
from src.configs.octdl_config import data_name, IMG_SIZE, fn_in, fn_out, fn_images, fn_col, target_col
import cv2
from PIL import Image


def load_octdl_data_dict(fp_preprocessed, resize=True):
    if resize:
        transform = transforms.Compose([
            transforms.Resize(size=IMG_SIZE),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
        
    fp_in = join(fp_preprocessed, fn_in)
    fp_out = join(fp_preprocessed, fn_out)
    fp_img = join(fp_preprocessed, fn_images)
    in_test_df = OCTDLDataset(
        fp_csv=fp_in, fp_image_dir=fp_img, fn_col=fn_col, target_col=target_col, 
        transform=transform)

    out_test_df = OCTDLDataset(
        fp_csv=fp_out, fp_image_dir=fp_img, fn_col=fn_col, target_col=target_col, 
        transform=transform)

    in_dict = {
        "target_col": "class",
        "data_name": data_name,
        "num_classes": 3,
        "test_df": in_test_df,
        "classes": ["AMD","DME", "Drusen", "Normal", "ERM", "RVO", "VID", "RAO"]
    }
    
    out_dict = {
        "target_col": "class",
        "data_name": data_name,
        "num_classes": 5,
        "test_df": out_test_df,
        "classes": ["AMD","DME", "Drusen", "Normal", "ERM", "RVO", "VID", "RAO"]
    }
    
    return in_dict, out_dict

from os.path import exists
    
class OCTDLDataset(Dataset):
    def __init__(
        self, fp_csv, fp_image_dir, 
        fn_col, target_col, transform=None):
        self.df = pd.read_csv(fp_csv)
        self.root_dir = fp_image_dir
        self.transform = transform
        self.fn_col = fn_col
        self.target_col = target_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fn_image  = self.df[self.fn_col].iloc[idx]
        fp_image = join(self.root_dir, fn_image)
        
        if exists(fp_image):
            image = Image.open(fp_image).convert('RGB')
        else:
            raise Exception(f"fp_image not found {fp_image}")
    
        if self.transform is not None:
            image = self.transform(image)
            
        target = self.df[self.target_col].iloc[idx]
        target = torch.tensor(target, dtype=torch.long)
        return image, target