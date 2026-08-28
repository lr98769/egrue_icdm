import pandas as pd
from scipy.stats import entropy
from scipy.stats import pearsonr
import numpy as np
from torch.nn.functional import cross_entropy, l1_loss 
from sklearn.metrics import auc, roc_auc_score
from IPython.display import display
from torch.nn.functional import binary_cross_entropy_with_logits

from src.display.display_df import highlight_first_n_second_highest, highlight_first_n_second_lowest

# Calculate AURC
def calculate_aurc(ue, loss):
    num_samples = len(ue)
    ue_loss_df = pd.DataFrame({"ue":ue, "loss":loss})
    ue_loss_df = ue_loss_df.sort_values(by="ue", ascending=True)
    ue_loss_df["cumulative_loss"] = ue_loss_df["loss"].expanding().mean()
    ue_loss_df["coverage"] = (np.arange(num_samples)+1)/num_samples
    return auc(ue_loss_df["coverage"], ue_loss_df["cumulative_loss"].values)

# Calculate Sigma Risk Loss
def scale_ue_minmax(ue_array, split_array):
    train_ue = ue_array[split_array=="Train"]
    min_ue, max_ue = min(train_ue), max(train_ue)
    return (ue_array - min_ue)/(max_ue - min_ue)

def scale_ue_znorm(ue_array, split_array):
    train_ue = ue_array[split_array=="Train"]
    mean, std = np.mean(train_ue), np.std(train_ue)
    return (ue_array - mean)/std

def remove_outliers(vec):
    # vector within 3 std away from mean
    # data_mean, data_std = np.mean(vec), np.std(vec)
    # num_std = 3
    # return vec[(vec <= data_mean + num_std*data_std) & (vec >= data_mean - num_std*data_std)]
    Q1 = np.percentile(vec, 25, method= 'midpoint') 
    Q3 = np.percentile(vec, 75, method= 'midpoint') 
    IQR = Q3 - Q1 
    low_lim = Q1 - 1.5 * IQR
    up_lim = Q3 + 1.5 * IQR
    return vec[(vec <= up_lim )]

def min_max_norm_wo_outliers(vec):
    vec_wo_outliers = remove_outliers(vec)
    vec_min, vec_max = min(vec_wo_outliers), max(vec_wo_outliers)
    return (vec - vec_min) / (vec_max - vec_min)

def thresholded_loss(pred_df, ue_col, loss_col, cutoff):
    ue = pred_df[ue_col]
    # ue = (ue - ue.min()) / (ue.max() - ue.min())
    ue = min_max_norm_wo_outliers(ue)
    loss = pred_df[loss_col]
    return loss[ue<=cutoff].mean()

def get_entropy_from_pred_prob(pred_prob):
    base = 2  # work in units of bits
    if len(pred_prob.shape)<2:
        pred_prob = [1-pred_prob, pred_prob]
        return entropy(pred_prob, base=base, axis=0)
    return entropy(pred_prob, base=base, axis=1)

def get_correlation_w_bce(y_batch, y_logits, x_batch, x_logits):
    # Get mae
    maes = l1_loss(x_logits, x_batch, reduction="none")
    maes = maes.mean(axis=tuple([i for i in range(1, len(maes.shape))]))
    if len(y_logits.shape) == 1:
        bces =  binary_cross_entropy_with_logits(y_logits, y_batch, reduction="none")
    else:
        bces = cross_entropy(y_logits, y_batch.long(), reduction="none")
    corr, _ = pearsonr(bces, maes)
    return corr

def get_ood_auroc(x_true, x_logits, x_true_ood, x_logits_ood):
    maes = l1_loss(x_logits, x_true, reduction="none")
    maes = maes.mean(axis=tuple([i for i in range(1, len(maes.shape))]))
    maes_ood = l1_loss(x_logits_ood, x_true_ood, reduction="none")
    maes_ood = maes_ood.mean(axis=tuple([i for i in range(1, len(maes_ood.shape))]))
    y_true = np.concatenate([np.repeat(0, len(maes)), np.repeat(1, len(maes_ood))])
    y_score = np.concatenate([maes, maes_ood])
    return roc_auc_score(y_true=y_true, y_score=y_score)

def restructure_ue_df(ue_perf_df):
    ue_perf_df = ue_perf_df.copy()
    # Split df into time label
    num_time, num_metrics = 3, 7
    all_dfs = []
    for i in range(num_time):
        column_indices = list(range(i*num_metrics, (i+1)*num_metrics))
        cur_df = ue_perf_df.iloc[:,column_indices]
        cur_df.columns = cur_df.columns.str.split(" ").str[-1] # remove time label from column names
        cur_df.loc[:,"Time Horizon"] = f"t+{i+1}"
        all_dfs.append(cur_df)
    all_dfs = pd.concat(all_dfs)
    all_dfs = all_dfs.reset_index().set_index(["Time Horizon", "Model"])
    return all_dfs

def remove_absent_cols(cols, df):
    return [col for col in cols if col in df]

def get_subset_of_multiindex_col(df, col_list, level=1):
    return [col for col in df.columns if col[level] in col_list]

def display_ue_perf(ue_perf_df, consolidated=False):
    ue_perf_df = ue_perf_df.copy()
    high_cols = ["Pearson's Correlation","AUROC","AUPR"]
    low_cols = [
        "AURC (CE Loss)","AURC (0/1 Loss)","AURC",
        "Sigma Risk (0.1)","Sigma Risk (0.2)","Sigma Risk (0.3)","Sigma Risk (0.4)",
        "Sigma-Risk Score (0.1)", "Sigma-Risk Score (0.2)", "Sigma-Risk Score (0.3)", "Sigma-Risk Score (0.4)"
    ]
    if isinstance(ue_perf_df.columns, pd.MultiIndex):
        high_cols = get_subset_of_multiindex_col(ue_perf_df, high_cols)
        low_cols = get_subset_of_multiindex_col(ue_perf_df, low_cols)
    high_cols = remove_absent_cols(high_cols, ue_perf_df)
    low_cols = remove_absent_cols(low_cols, ue_perf_df)
    display(
        ue_perf_df.style.apply(
            highlight_first_n_second_highest, subset=high_cols, split_value=consolidated).apply(
                highlight_first_n_second_lowest, subset=low_cols, split_value=consolidated
            )
    )

def display_ood_perf(ood_perf_df, consolidated=False):
    ood_perf_df = ood_perf_df.copy()
    display(
        ood_perf_df.style.apply(
            highlight_first_n_second_highest, split_value=consolidated)
    )
