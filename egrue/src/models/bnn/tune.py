from src.models.bnn.model import instantiate_bnn_model
from src.models.bnn.train import train_bnn_model
from src.file_manager.filepath import FilePath
from src.misc import set_seed_pytorch
from src.data_processing.dataloader import get_pytorch_split_dict_image

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

import time

def tune_bnn_model(
        param_grid, data_dict, 
        epochs, patience, seed, fp: FilePath, 
        batch_size, eval_batch_size
    ):
    num_outputs = data_dict["num_classes"]
    parameter_list = list(ParameterGrid(param_grid))
    loss_list, time_list, epoch_list = [], [], []
    with tqdm(parameter_list) as pbar:
        for param_dict in pbar:
            bnn_model = instantiate_bnn_model(seed=seed, num_outputs=num_outputs, **param_dict)
            fp_model = fp.get_fp_model(bnn_model, cur_model_name="tuning")
            start = time.time()
            set_seed_pytorch(seed)
            split_dict_pytorch = get_pytorch_split_dict_image(
                data_dict=data_dict, batch_size=batch_size, eval_batch_size=eval_batch_size
            )
            train_dl, valid_dl = split_dict_pytorch["train_dl"], split_dict_pytorch["val_dl"]
            ce_loss, best_epoch = train_bnn_model(
                bnn_model, train_dl, valid_dl, epochs, patience, seed, fp_model, 
                class_weight=None)
            time_list.append(time.time()-start)
            epoch_list.append(best_epoch)
            loss_list.append(ce_loss)
    tuning_df = pd.DataFrame(parameter_list)
    tuning_df["loss"] = loss_list
    tuning_df["epoch"] = epoch_list
    tuning_df["time/s"] = time_list
    best_index = np.argmin(tuning_df["loss"])
    tuning_df["best_hyperparameter"] = False
    tuning_df.iloc[best_index, -1] = True
    return tuning_df, parameter_list[best_index]