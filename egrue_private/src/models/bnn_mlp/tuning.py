from src.models.bnn_mlp.model import instantiate_bnn_model
from src.models.bnn_mlp.training import train_bnn_model


import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm


import time


def tune_bnn_model(
    param_grid, train_df, valid_df, feat_cols, target_cols, epochs, patience, seed, batch_size, fp_model):
    parameter_list = list(ParameterGrid(param_grid))
    loss_list, time_list, epoch_list = [], [], []
    with tqdm(parameter_list) as pbar:
        for param_dict in pbar:
            bnn_model = instantiate_bnn_model(
                seed=seed, num_inputs=len(feat_cols), num_outputs=len(target_cols), **param_dict)
            start = time.time()
            loss, best_epoch = train_bnn_model(
                bnn_model, train_df, valid_df,
                feat_cols, target_cols, epochs, patience, seed, batch_size, fp_model)
            time_list.append(time.time()-start)
            epoch_list.append(best_epoch)
            loss_list.append(loss)
    tuning_df = pd.DataFrame(parameter_list)
    tuning_df["loss"] = loss_list
    tuning_df["epoch"] = epoch_list
    tuning_df["time/s"] = time_list
    best_index = np.argmin(tuning_df["loss"])
    tuning_df["best_hyperparameter"] = False
    tuning_df.iloc[best_index, -1] = True
    return tuning_df, parameter_list[best_index]