from torch.utils.data import Subset
import numpy as np
from collections import Counter
import torch
from torchvision.datasets import ImageFolder

from src.configs.default_configs import device

# https://naadispeaks.blog/2021/07/31/handling-imbalanced-classes-with-weighted-loss-in-pytorch/
def get_class_weights(data_dict):
    split_name = "train_df"
    ds = data_dict[split_name]
    num_classes = data_dict["num_classes"]
    num_instances = len(ds)
    if isinstance(ds, Subset):
        targets = np.array(ds.dataset.labels)
        indices = ds.indices
        class_count = dict(Counter(targets[indices]))
    elif isinstance(ds, ImageFolder):
            class_count = dict(Counter(ds.targets))
    else:
        class_count = dict(Counter(ds.labels.flatten()))

    class_count_array = np.array([class_count[i] for i in range(num_classes)])
    assert num_instances == class_count_array.sum()
    return torch.tensor(1-(class_count_array/num_instances), dtype=torch.float32).to(device)

def get_pos_weight(split_dict, target_col):
    num_pos = (split_dict["train_df"][target_col]==1).sum()
    num_neg = (split_dict["train_df"][target_col]==0).sum()
    # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    return torch.Tensor([num_neg/num_pos]).to(device)