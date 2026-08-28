
from torch.utils.data import Dataset
import pandas as pd
from os.path import join
from PIL import Image
import torch


class ImageDataset(Dataset):
    def __init__(self, fp_csv_file, fp_img_dir, img_fn_col, target_col, transform=None):
        self.df = pd.read_csv(fp_csv_file)
        self.img_dir = fp_img_dir
        self.transform = transform
        self.img_fn_col = img_fn_col
        self.target_col = target_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fp_img = join(self.img_dir, row[self.img_fn_col])
        image = Image.open(fp_img)
        label = row[self.target_col]
        if self.transform:
            image = self.transform(image)
        return image, label

class TabularDataset(Dataset):
    def __init__(self, df, feat_cols, target_col):
        self.data_df = df
        self.feat_cols = feat_cols
        self.target_col = target_col

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        features = torch.from_numpy(row[self.feat_cols].values).float()
        label = row[self.target_col]
        return features, label
    

class HugggingFaceDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, label_key="label", image_key="image"):
        self.dataset = hf_dataset
        self.transform = transform
        self.label_key = label_key
        self.image_key = image_key
        self.targets = torch.tensor(self.dataset[self.label_key]).numpy()
        self.labels = self.targets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item[self.image_key]
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(item[self.label_key])