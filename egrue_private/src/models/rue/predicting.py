import pandas as pd
import warnings
import numpy as np
from tqdm.auto import tqdm

from src.evaluation.inference import predict
from src.evaluation.perf_metrics import get_pred_prob, get_pred_labels, get_bce_from_pred_prob, \
    get_entropy_from_pred_prob, get_rue, get_rue_featwise
from src.misc import set_seed_pytorch


def get_nn_predictions(model, dl, feature_cols, target_col, seed, T=None):
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    
    # Columns
    reconstruction_cols = [f"{col}_recon"   for col in feature_cols]
    rue_featwise_cols = [f"{col}_rue"   for col in feature_cols]
    logit_col = f"{target_col}_logit"
    pred_prob_col = f"{target_col}_pred_prob"
    pred_label_col = f"{target_col}_pred_label"
    entropy_col = "entropy"
    bce_col = "bce"
    rue_col = "rue"
    
    # Predictions
    y_true, y_logits, x_true, x_logits = predict(model, dl)
    pred_prob = get_pred_prob(y_logits)

    # Output pred_df
    pred_df = pd.DataFrame(x_true, columns=feature_cols)
    pred_df[reconstruction_cols] = x_logits
    pred_df[target_col] = y_true
    pred_df[logit_col] = y_logits
    pred_df[pred_prob_col] = pred_prob
    pred_df[pred_label_col] = get_pred_labels(y_logits)

    # Uncertainty and loss
    pred_df[bce_col] = get_bce_from_pred_prob(y_true, pred_prob)
    pred_df[entropy_col] = get_entropy_from_pred_prob(pred_prob)
    pred_df[rue_col] = get_rue(x_true, x_logits)
    pred_df[rue_featwise_cols] = get_rue_featwise(x_true, x_logits)

    set_seed_pytorch(seed)
    # Get MC Dropout predictions
    if T is not None:
        mc_pred_prob_col = f"{target_col}_pred_prob_mc"
        mc_pred_label_col = f"{target_col}_pred_label_mc"
        mc_ue_col = "ue_mc"
        mc_bce_col = "bce_mc"
        all_y_probs = [] 
        for _ in tqdm(range(T), total=T):
            y_logits = predict(model, dl, verbose=False, active_dropout=True)[1]
            y_pred_prob = get_pred_prob(y_logits)
            all_y_probs.append(y_pred_prob)
        mc_pred_prob = np.mean(all_y_probs, axis=0)
        pred_df[mc_pred_prob_col] = mc_pred_prob
        pred_df[mc_pred_label_col] = pred_df[mc_pred_prob_col].round()
        pred_df[mc_ue_col] = np.std(all_y_probs, axis=0)
        pred_df[mc_bce_col] = get_bce_from_pred_prob(y_true, mc_pred_prob)

    return pred_df

def get_all_predictions(
    model, train_dl, val_dl, test_dl, 
    feature_cols, target_col, seed, 
    pred_function=get_nn_predictions, T=None):
    all_pred_dfs = []
    dls = {"Train": train_dl, "Valid": val_dl, "Test": test_dl}
    for split_name, dl in dls.items():
        pred_df = pred_function(model, dl, feature_cols, target_col, T= T, seed=seed)
        pred_df["split"] = split_name
        all_pred_dfs.append(pred_df)
    return pd.concat(all_pred_dfs)