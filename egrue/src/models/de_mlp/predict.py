import pandas as pd
from tqdm.auto import tqdm
from os.path import join, exists
import torch
import numpy as np

from src.configs.default_configs import fn_model, device, fn_pred
from src.models.rue.predicting import get_all_predictions, get_nn_predictions
from src.models.de_mlp.train import get_ensemble_seeds
from src.file_manager.filepath import FilePath
from src.evaluation.perf_metrics import get_bce_from_pred_prob

def get_all_ensemble_predictions(
    train_dl, val_dl, test_dl, 
    feature_cols, target_col, 
    seed, seed_interval_size, ensemble_size, data_name,
    pred_function=get_nn_predictions, T=None):

    all_ensemble_seeds = get_ensemble_seeds(seed, seed_interval_size, ensemble_size)
    all_pred_dfs = []
    with tqdm(all_ensemble_seeds) as pbar:
        for cur_seed in pbar:
            pbar.set_description(f"Seed: {cur_seed}")
            cur_fp = FilePath(data_name, seed=cur_seed)
            fp_model = join(cur_fp.get_parent_folder(fn_model), "classifier_tuned.pt")
            fp_classifier_predictions_file = join(cur_fp.get_parent_folder(fn_pred), "tuning_classifier.csv")
            if exists(fp_classifier_predictions_file):
                pred_df = pd.read_csv(fp_classifier_predictions_file, index_col=0)
            else:
                model = torch.load(fp_model, mmap=device)
                pred_df = get_all_predictions(
                    model, train_dl, val_dl, test_dl, 
                    feature_cols, target_col, seed, 
                    pred_function=pred_function, T=T)
                pred_df.to_csv(fp_classifier_predictions_file)
            all_pred_dfs.append(pred_df)

    all_pred_dfs = process_ensemble_pred_dfs(all_pred_dfs, target_col)
    return all_pred_dfs

def process_ensemble_pred_dfs(all_pred_dfs, target_col, label="de"):
    # Columns
    pred_prob_col = f"{target_col}_pred_prob"
    pred_label_col = f"{target_col}_pred_label"
    bce_col = "bce"
    std_col = "de_std"

    all_pred_probs = []
    for pred_df in all_pred_dfs:
        all_pred_probs.append(pred_df[pred_prob_col].values)
    all_pred_probs = np.array(all_pred_probs) # (num_models, num_samples)

    mean_pred_prob = all_pred_probs.mean(axis=0) # (num_samples)
    std_pred_prob = all_pred_probs.std(axis=0) # (num_samples)
    pred_label = mean_pred_prob.round()
    y_true = pred_df[target_col].values
    index = pred_df.index
    split = pred_df["split"]

    pred_df = pd.DataFrame(np.expand_dims(y_true, axis=-1), columns=[target_col])
    pred_df[pred_prob_col+"_de"] = mean_pred_prob
    pred_df[pred_label_col+"_de"] = pred_label
    pred_df[bce_col+"_de"] = get_bce_from_pred_prob(y_true, mean_pred_prob)
    pred_df[std_col] = std_pred_prob
    pred_df.index = index
    pred_df["split"] = split

    return pred_df
    