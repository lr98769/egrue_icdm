from src.models.der.training import train_der_w_param


import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm


import time


def tune_der_model(
    param_grid, train_df, valid_df, feat_cols, target_cols, epochs, patience, batch_size, seed,):
    parameter_list = list(ParameterGrid(param_grid))
    loss_list, time_list = [], []
    with tqdm(parameter_list) as pbar:
        for param_dict in pbar:
            start = time.time()
            der_model, result = train_der_w_param(
                train_df=train_df, valid_df=valid_df,
                inputs=feat_cols, outputs=target_cols, batch_size=batch_size,
                seed=seed, max_epochs=epochs, patience=patience, **param_dict
            )
            # print(result)
            time_list.append(time.time()-start)
            loss_list.append(result['test/MSE'])
    tuning_df = pd.DataFrame(parameter_list)
    tuning_df["loss"] = loss_list
    tuning_df["time/s"] = time_list
    best_index = np.argmin(tuning_df["loss"])
    tuning_df["best_hyperparameter"] = False
    tuning_df.iloc[best_index, -1] = True
    return tuning_df, parameter_list[best_index]