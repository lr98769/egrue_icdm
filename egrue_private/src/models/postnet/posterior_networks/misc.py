import numpy as np
import torch
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder



from collections import Counter


def get_class_count(ds, output_dim):
    if isinstance(ds, Subset):
        targets = np.array(ds.dataset.labels)
        indices = ds.indices
        # print(targets[indices])
        class_count = dict(Counter(targets[indices].flatten()))
    elif isinstance(ds, ImageFolder):
        class_count = dict(Counter(ds.targets))
    else:
        class_count = dict(Counter(ds.labels.flatten()))
    ordered_counts = [class_count[i] for i in range(output_dim)]
    N = torch.tensor(ordered_counts)
    return N

def get_class_count_from_df(df, target_col, output_dim):
    val_counts = df[target_col].value_counts()
    ordered_counts = [val_counts.loc[i] for i in range(output_dim)]
    N = torch.tensor(ordered_counts)
    return N