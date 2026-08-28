import numpy as np
import torch
import matplotlib.pyplot as plt
from math import ceil
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import pandas as pd
from tqdm.auto import tqdm
from IPython.display import display
from collections import Counter
from torch.utils.data import Subset

def transform_for_visualisation(img: torch.tensor, data_type=np.float32, permute=True, reverse_transform=True):
    if permute:
        img = img.permute((1, 2, 0)).numpy()
    img_copy = img.copy()
    img_copy = img_copy.astype(data_type) 
    if reverse_transform:
        img_copy[:, :, 0] = img_copy[:, :, 0]* 0.229 + 0.485
        img_copy[:, :, 1] = img_copy[:, :, 1]* 0.224 + 0.456
        img_copy[:, :, 2] = img_copy[:, :, 2]*0.225 + 0.406
    return (img_copy.clip(0, 1) * 255).astype(np.uint8)

def show_img_examples_ds(ds: Dataset, num_classes, classes, ncols=1, img_size=2, dpi=300, transform=True):
    displayed_class_labels = []
    nrows = ceil(num_classes/ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(img_size*ncols, img_size*nrows), dpi=dpi)
    axes = axes.flatten()
    for img, label in tqdm(ds):
        if label not in displayed_class_labels:
            img = transform_for_visualisation(img, reverse_transform=transform)
            axes[label].imshow(img)
            axes[label].set_title(f"Class: {classes[label]}")
            displayed_class_labels.append(label)
        if len(displayed_class_labels) >= num_classes:
            break
    for j in range(num_classes, nrows*ncols):
        axes[j].set_axis_off()
    plt.tight_layout()
    plt.show()
    
def get_image_class_distribution_labels(data_dict):
    splits = ["train_df", "val_df", "test_df"]
    index = []
    all_class_counts = []
    for split_name in tqdm(splits):
        if split_name not in data_dict:
            continue
        ds = data_dict[split_name]
        cur_dist_dict = {"# Instances": len(ds)}
        if isinstance(ds, Subset):
            targets = np.array(ds.dataset.labels)
            indices = ds.indices
            class_count = dict(Counter(targets[indices]))
        elif isinstance(ds, ImageFolder):
            class_count = dict(Counter(ds.targets))
        else:
            class_count = dict(Counter(ds.labels.flatten()))
        cur_dist_dict.update(class_count)
        all_class_counts.append(cur_dist_dict)
        index.append(split_name)
    class_dist_df = pd.DataFrame(all_class_counts) 
    if "classes" in data_dict:
        class_dist_df.columns = [
            f"# Instances {col} ({data_dict['classes'][col]})" if col != "# Instances" else col
            for col in class_dist_df.columns]
    else:
        class_dist_df.columns = [
            f"# Instances {col}" if col != "# Instances" else col
            for col in class_dist_df.columns]
    class_dist_df = class_dist_df[class_dist_df.columns.sort_values()]
    class_dist_df.index = index
    return class_dist_df