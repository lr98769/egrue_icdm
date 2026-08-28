import numpy as np
from src.misc import set_seed_tf
from src.models.rue.model import AE_Regressor, get_model_error_corr


import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm
import time

def model_tuning_regressor(
    param_grid, predictors, pred_cols, train_df, valid_df, seed,
    batch_size=128, max_epochs=5000, verbose=1, patience=10):
    train_X, train_y = (
        train_df[predictors].values.astype('float32'), train_df[pred_cols].values.astype('float32'))
    valid_X, valid_y = (
        valid_df[predictors].values.astype('float32'), valid_df[pred_cols].values.astype('float32'))
    loss_list = []
    epoch_list = []
    time_spent_list = []
    tuning_df_list = []
    parameter_list = list(ParameterGrid(param_grid))
    pbar = tqdm(parameter_list)
    for param_dict in pbar:
        pbar.set_description(f"{param_dict}")
        set_seed_tf(seed)
        regressor = AE_Regressor(**param_dict, predictors=predictors, output_features=pred_cols)
        start = time.time()
        val_loss, best_epoch = regressor.train_regressor(
            train_X, train_y, valid_X, valid_y, batch_size, max_epochs, verbose, patience)
        cur_param_dict = param_dict.copy()
        cur_param_dict["loss"] = val_loss
        cur_param_dict["epochs"] = best_epoch
        cur_param_dict["time/s"] = time.time()-start
        tuning_df_list.append(cur_param_dict)
    tuning_df = pd.DataFrame(tuning_df_list)
    best_param_idx = tuning_df['loss'].idxmin()
    tuning_df["best_hyperparameter"] = [True if i==best_param_idx else False for i in range(len(tuning_df))]
    best_param = parameter_list[best_param_idx]
    return tuning_df, best_param


def model_tuning_decoder(
    param_grid, predictors, pred_cols, train_df, valid_df, seed, prev_model,
    batch_size=128, max_epochs=5000, verbose=1, patience=10):
    train_X, train_y = (
        train_df[predictors].values.astype('float32'), train_df[pred_cols].values.astype('float32'))
    valid_X, valid_y = (
        valid_df[predictors].values.astype('float32'), valid_df[pred_cols].values.astype('float32'))
    tuning_df_list = []
    parameter_list = list(ParameterGrid(param_grid))
    best_corr = -np.inf
    pbar = tqdm(parameter_list)
    for param_dict in pbar:
        pbar.set_description(f"{param_dict}, best corr: {best_corr:.5f}")
        set_seed(seed)
        predictor = AE_Regressor(**param_dict, predictors=predictors, output_features=pred_cols)
        predictor.replace_encoder_predictor(prev_model)
        start = time.time()
        val_loss, best_epoch = predictor.train_decoder(
            train_X, valid_X, batch_size, max_epochs, verbose, patience)
        timing = time.time()-start
        corr = get_model_error_corr(predictor, valid_X, valid_y)
        cur_param_dict = param_dict.copy()
        cur_param_dict["loss"] = val_loss
        cur_param_dict["corr"] = corr
        cur_param_dict["epochs"] = best_epoch
        cur_param_dict["time/s"] = timing
        tuning_df_list.append(cur_param_dict)
        best_corr = max(corr, best_corr)
    tuning_df = pd.DataFrame(tuning_df_list)
    best_param_idx = tuning_df["corr"].idxmax()
    tuning_df["best_hyperparameter"] = [True if i==best_param_idx else False for i in range(len(tuning_df))]
    best_param = parameter_list[best_param_idx]
    return tuning_df, best_param