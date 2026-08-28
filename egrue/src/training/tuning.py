from scipy.stats import pearsonr
from src.training.train import transfer_encoder_n_classifier
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.model_selection import ParameterGrid
import time
from shutil import copyfile

from src.data_processing.dataloader import get_pytorch_split_dict_image
from src.file_manager.filepath import FilePath
from src.misc import set_seed_pytorch
from src.evaluation.ue_metrics import calculate_aurc, thresholded_loss


def tune_model(
    ModelClass, param_grid, 
    feature_cols, target_col, 
    train_param_dict, train_model_func, 
    split_dict, pytorch_split_dict_func, # Split dict is a dictionary of dataframes, pytorch_split_dict_func converts these dfs to dl
    fp_model, seed, 
    batch_size, eval_batch_size, 
    fp_history=None, 
    metric_to_monitor = "auc", maximise=True, # Metric used for Tuning
    prev_model=None, weight_cols=None, 
    class_weight=None
):
    split_to_monitor = "valid"
    
    metric_label = f"{split_to_monitor.capitalize()} {metric_to_monitor.capitalize()}"
    tuning_df_list = []
    parameter_list = list(ParameterGrid(param_grid))
    pbar = tqdm(parameter_list) # , position=0
    best_score = -np.inf if maximise else np.inf
    val_score = None
    for param_dict in pbar:
        print(param_dict)
        if val_score is not None:
            pbar.set_description(f"Current Param: {param_dict}, Best {metric_label}: {best_score:.5f}, Lastest {metric_label}: {val_score:.5f}")
        else:
            pbar.set_description(f"Current Param: {param_dict}")
        set_seed_pytorch(seed)
        split_dict_pytorch = pytorch_split_dict_func(
            **split_dict, feat_cols=feature_cols, target_col=target_col, weight_cols=weight_cols,
            batch_size=batch_size, eval_batch_size=eval_batch_size
        )
        model = ModelClass(
            **param_dict, num_features=len(feature_cols)
        )
        if prev_model:
            model = transfer_encoder_n_classifier(model, prev_model) 
        start = time.time()
        history = train_model_func(
            model=model, **split_dict_pytorch, 
            fp_model=fp_model, **train_param_dict, fp_history=fp_history,
            verbose=False, metric_to_monitor=metric_to_monitor, maximise=maximise, class_weight=class_weight
        )
        best_metrics = get_best_val(history, split_to_monitor, metric_to_monitor, maximise=maximise)
        cur_param_dict = param_dict.copy()
        cur_param_dict.update(best_metrics)
        cur_param_dict["Time/s"] = time.time()-start
        tuning_df_list.append(cur_param_dict)

        val_score = cur_param_dict[metric_label]
        if maximise:
            if val_score > best_score:
                best_score = val_score
        else:
            if val_score < best_score:
                best_score = val_score
        
    tuning_df = pd.DataFrame(tuning_df_list)
    if maximise:
        best_param_idx = tuning_df[metric_label].idxmax()
    else:
        best_param_idx = tuning_df[metric_label].idxmin()
    tuning_df["best_hyperparameter"] = [
        True if i==best_param_idx else False for i in range(len(tuning_df))]
    best_param = parameter_list[best_param_idx]
    return tuning_df, best_param

def get_best_val(history, split_to_monitor, metric_to_monitor, maximise=True):
    if maximise:
        best_epoch = np.argmax(history[split_to_monitor][metric_to_monitor])
    else:
        best_epoch = np.argmin(history[split_to_monitor][metric_to_monitor])
    best_metrics = {"Epochs": best_epoch}
    for split, split_dict in history.items():
        for metric, metric_list in split_dict.items():
            best_metrics[f"{split.capitalize()} {metric.capitalize()}"] = metric_list[best_epoch]
    return best_metrics
