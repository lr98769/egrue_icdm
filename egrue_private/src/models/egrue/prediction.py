from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from torch.nn import Sequential
import torch
import shap

def buffered_shap(explainer, matrix, chunksize, seed):
    from math import ceil
    all_exp = []
    matrix_len = len(matrix)
    num_chunks = ceil(matrix_len/chunksize)
    for chunk_i in tqdm(range(num_chunks), total=num_chunks):
        start = chunk_i*chunksize
        end = (chunk_i+1)*chunksize
        if end < matrix_len:
            matrix_chunk = matrix[chunk_i*chunksize:(chunk_i+1)*chunksize]
        else:
            matrix_chunk = matrix[chunk_i*chunksize:]
        shap_chunk = explainer.shap_values(matrix_chunk, rseed=seed)
        all_exp.append(shap_chunk)
    return np.vstack(all_exp)

def get_eg_values(model, split_dict, feat_cols, seed):
    split_col = "split"
    # Get classifier
    classifier = Sequential(model.encoder, model.classifier).cpu()
    # Get tensors
    train_tensor = torch.tensor(split_dict["train_df"][feat_cols].values).to(torch.float)
    all_df = []
    for split_name in ["train_df", "val_df", "test_df"]:
        cur_df = split_dict[split_name].copy()
        cur_df[split_col] = split_name[:-3]
        all_df.append(cur_df)
    all_df = pd.concat(all_df)
    all_tensor = torch.tensor(all_df[feat_cols].values).to(torch.float)
    # Get explanation
    explainer = shap.GradientExplainer(classifier, train_tensor)
    eg_values = buffered_shap(explainer, all_tensor, chunksize=100, seed=seed)
    eg_df = pd.DataFrame(np.squeeze(eg_values), columns=[feat+"_eg" for feat in feat_cols])
    eg_df[split_col] = all_df[split_col]
    return eg_df

def get_egRUE(pred_df, eg_df, feat_cols):
    pred_df = pred_df.copy()
    eg_cols = [feat+"_eg" for feat in feat_cols]
    eg_weight_cols = [feat+"_eg_weight" for feat in feat_cols]
    recon_cols = [feat+"_recon" for feat in feat_cols]
    # RUE_featwise_cols = [feat+"_rue" for feat in feat_cols]
    egRUE_featwise_cols = [feat+"_egRUE" for feat in feat_cols]
    egRUE_col = "egRUE"
    pred_df[eg_cols] = eg_df[eg_cols].values
    
    # Calculate egRUE
    abs_eg = np.abs(pred_df[eg_cols].values) # Get the magnitude of egs
    eg_weights = abs_eg/np.expand_dims(np.sum(abs_eg, axis=-1), axis=-1) # normalise the absolute eg by total eg of each input
    pred_df[eg_weight_cols] = eg_weights
    recon_error = np.abs(pred_df[feat_cols].values-pred_df[recon_cols].values) # element wise recon error
    # pred_df[RUE_featwise_cols] = recon_error
    pred_df[egRUE_featwise_cols] = recon_error*eg_weights
    egRUE = np.sum(recon_error*eg_weights, axis=-1)
    pred_df[egRUE_col] = egRUE

    return pred_df