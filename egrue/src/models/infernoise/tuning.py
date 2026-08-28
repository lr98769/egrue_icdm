import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import pearsonr
import numpy as np
from src.models.infernoise.predicting import infernoise_test_predictions

def tune_infernoise(ae_predictor, stddev_list, valid_df, inputs, outputs, seed, T, regressor_label):
    corr_list = []
    loss_list = []
    for stddev in tqdm(stddev_list):
        valid_pred_df = infernoise_test_predictions(ae_predictor, valid_df, inputs, outputs, seed, T, stddev, regressor_label, False)
        loss = valid_pred_df["infernoise_mae"]
        ue = valid_pred_df["infernoise_uncertainty"]
        corr, _ = pearsonr(ue, loss)
        loss_list.append(valid_pred_df["infernoise_mae"].mean())
        corr_list.append(corr)
    tuning_df = pd.DataFrame({"std": stddev_list, "correlation": corr_list, "loss":loss_list})
    tuning_df["best_hyperparameter"] = False
    best_index = tuning_df["loss"].idxmin()
    tuning_df.iloc[best_index, -1] = True
    return tuning_df