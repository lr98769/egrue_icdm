# Make OOD Dicts
from copy import deepcopy
import numpy as np
def create_ood_dicts(split_dict, noise_mu_list, seed, sigma=0.1):
    np.random.seed(seed)
    feat_cols = split_dict["feat_cols"]
    ood_dicts = {}
    for mu in noise_mu_list:
        cur_split_dict = deepcopy(split_dict)
        cur_split_dict["test_df"][feat_cols] += np.random.normal(
            loc=mu, scale=sigma, size=cur_split_dict["test_df"][feat_cols].shape)
        ood_dicts[f"mu={mu}"] = cur_split_dict
    return ood_dicts

def create_ood_dicts_mimic(split_dict, noise_mu_list, seed, predictors, sigma=0.1):
    np.random.seed(seed)
    ood_dicts = {}
    for mu in noise_mu_list:
        cur_split_dict = deepcopy(split_dict)
        for time_label in cur_split_dict.keys():
            cur_split_dict[time_label]["test_df"][predictors] += np.random.normal(
                loc=mu, scale=sigma, size=cur_split_dict[time_label]["test_df"][predictors].shape)
        ood_dicts[f"mu={mu}"] = cur_split_dict
    return ood_dicts