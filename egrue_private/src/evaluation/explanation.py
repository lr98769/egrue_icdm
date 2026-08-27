from IPython.display import display
import torch
from tqdm.auto import tqdm
import pandas as pd

from src.configs.default_configs import device
from src.data_processing.dataloader import *



def explanation(
    model, dl, expl_func, y_pred_labels, verbose=True, active_dropout=False, additional_param=dict()):
    model.eval()
    if active_dropout:
        model.train()
    else:
        model.eval()
    model.to(device)
    num_batches = len(dl)
    all_attr = []
    batch_size = dl.batch_size
    start_index = 0
    with torch.no_grad():
        with tqdm(dl, leave=False, position=0, total=num_batches, disable=not verbose) as pbar:
            for inputs in pbar:
                x_batch = inputs[0].to(device)
                cur_y_pred_labels = y_pred_labels[start_index:start_index+batch_size]
                attr = expl_func(x_batch, cur_y_pred_labels, additional_param)
                all_attr.append(attr)
                start_index+=batch_size
    return torch.concatenate(all_attr)

def explanation_reg(
    model, dl, expl_func, verbose=True, active_dropout=False, additional_param=dict()):
    model.eval()
    if active_dropout:
        model.train()
    else:
        model.eval()
    model.to(device)
    num_batches = len(dl)
    all_attr = []
    batch_size = dl.batch_size
    start_index = 0
    with torch.no_grad():
        with tqdm(dl, leave=False, position=0, total=num_batches, disable=not verbose) as pbar:
            for inputs in pbar:
                x_batch = inputs[0].to(device)
                attr = expl_func(x_batch, additional_param)
                all_attr.append(attr)
                start_index+=batch_size
    return torch.concatenate(all_attr)

def get_all_explanations(
        model, data_dict, pred_df, expl_func, 
        batch_size, eval_batch_size, seed, 
        pytorch_split_dict_func=get_pytorch_split_dict, additional_pred_args=dict()
    ):
    feat_cols, target_col = data_dict["feat_cols"], data_dict["target_col"]
    num_classes = data_dict["num_classes"] if "num_classes" in data_dict else data_dict["num_outputs"]
    all_expl_dfs = []
    # Generate DLs
    dls = pytorch_split_dict_func(
        data_dict=data_dict, batch_size=batch_size, eval_batch_size=eval_batch_size, shuffle_train=False)
    new_dls = {}
    name_map = {"train_dl":"Train", "val_dl": "Valid", "test_dl": "Test", "grid_dl": "Grid"}
    for label, dl in dls.items():
        if label in name_map:
            new_dls[name_map[label]] = dl
    dls = new_dls
    # Get expl dfs
    with tqdm(dls.items(), total=len(dls)) as pbar:
        for split_name, dl in pbar:
            pbar.set_description(split_name)
            cur_pred_df = pred_df[pred_df["split"]==split_name]
            expl_df = expl_func(
                model, dl, feat_cols, target_col, num_classes, cur_pred_df=cur_pred_df, seed=seed, **additional_pred_args)
            expl_df["split"] = split_name
            all_expl_dfs.append(expl_df)
    return pd.concat(all_expl_dfs)


