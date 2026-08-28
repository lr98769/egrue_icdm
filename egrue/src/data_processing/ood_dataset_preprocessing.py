from copy import deepcopy
from torch import Generator
from torch.utils.data.dataset import random_split
from torch.utils.data import ConcatDataset

def process_dataset_for_ood(data_dict, data_dict_ood, seed):
    data_dict_ood = deepcopy(data_dict_ood)
    test_set_size = len(data_dict["test_df"])
    cur_ood_set_size = len(data_dict_ood["test_df"])
    if cur_ood_set_size > test_set_size:
        data_dict_ood["test_df"], _ = random_split(
            data_dict_ood["test_df"], [test_set_size, cur_ood_set_size-test_set_size],
            generator=Generator().manual_seed(seed))
    data_dict_ood["num_classes"] = data_dict["num_classes"]
    return data_dict_ood

def left_join_datasets(data_dict1, data_dict2, split="test_df"):
    data_dict1 = deepcopy(data_dict1)
    data_dict1[split] = ConcatDataset([data_dict1[split], data_dict2[split]])
    return data_dict1