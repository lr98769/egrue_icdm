import numpy as np
from torch.utils.data import DataLoader
from src.misc import set_seed_pytorch
from src.configs.default_configs import device


import torch
from tqdm.auto import tqdm

from src.data_processing.dataset import TabularDataset


def predict_bnn_model(bnn_model, dl, feat_cols, target_cols, seed, silent):
    set_seed_pytorch(seed)
    # Send model to gpu
    bnn_model.to(device)
    # Predictions
    all_pred = []
    # Predict with model
    with tqdm(dl, total=len(dl), disable=silent) as pbar:
        for x_batch, y_batch in pbar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = bnn_model(x_batch)
            all_pred.append(pred.detach().cpu())
    return torch.cat(all_pred)

def mae_fn(y_true, y_pred):
    return np.mean(np.abs(y_true-y_pred), axis=-1)

def bnn_model_prediction(
    bnn_model, test_df, feat_cols, target_cols, T, seed,  regressor_label, batch_size):
    test_df = test_df.copy()
    set_seed_pytorch(seed)
    # Prepare dataset
    test_ds = TabularDataset(df=test_df, feat_cols=feat_cols, target_cols=target_cols)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    seed_list = list(range(seed, seed+T))
    all_logits = []
    for cur_seed in tqdm(seed_list):
        logits = predict_bnn_model(bnn_model, test_dl, feat_cols, target_cols, seed=cur_seed, silent=True)
        all_logits.append(logits)

    all_logits = torch.stack(all_logits) # T, N, O
    test_y_pred = all_logits.mean(axis=0) # N, O
    test_y_std = all_logits.std(axis=0) # N, O
    test_y = torch.from_numpy(test_df[target_cols].values).float()

    test_df["bnn_uncertainty"] = test_y_std.mean(axis=-1) # N
    test_df["bnn_mae"] = mae_fn(y_true=test_y.numpy(), y_pred=test_y_pred.numpy())
    predicted_colnames = [col + "_bnn_"+ regressor_label for col in target_cols]
    test_df[predicted_colnames] = test_y_pred
    return test_df