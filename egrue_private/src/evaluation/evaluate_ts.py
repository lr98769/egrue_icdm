import numpy as np
import pandas as pd
from IPython.display import display
from scipy.stats import pearsonr

from src.evaluation.ue_metrics import calculate_aurc, thresholded_loss
from src.display.display_df import highlight_first_n_second_highest, highlight_first_n_second_lowest

def evaluate_ue_ts(test_df_dict, ue_dict):
    perf_df_dict = []
    for ue_name, ue_info in ue_dict.items():
        ue_row_dict = {"Model": ue_name}
        pred_label = ue_info["pred_label"]
        ue_col = ue_info["ue"]
        for regressor_label, test_df_info in test_df_dict.items():
            test_df = test_df_info["test_df"].copy()
            pred_cols = test_df_info["pred_cols"]
            y_pred_cols = [col+pred_label+"_"+regressor_label for col in pred_cols]
            loss_col = "loss"

            y_true = test_df[pred_cols].values
            y_pred = test_df[y_pred_cols].values
            ues = test_df[ue_col].values
            
            mean_abs_errors = np.mean(np.abs(y_true-y_pred), axis=1)
            test_df[loss_col] = mean_abs_errors
            
            corr, p_value = pearsonr(ues, mean_abs_errors)
            aurc = calculate_aurc(ues, mean_abs_errors)

            ue_row_dict[regressor_label+" Corr"] = corr
            ue_row_dict[regressor_label+" AURC"] = aurc

            for thres in [0.1, 0.2, 0.3, 0.4]:
                ue_row_dict[regressor_label+f" Sigma={thres}"] = thresholded_loss(
                    test_df, ue_col=ue_col, loss_col=loss_col, cutoff=thres)
            
        perf_df_dict.append(ue_row_dict)
    perf_df = pd.DataFrame(perf_df_dict)
    perf_df = perf_df.set_index("Model")
    return perf_df

def restructure_ue_df(ue_perf_df):
    ue_perf_df = ue_perf_df.copy()
    # Split df into time label
    num_time = 3 
    num_metrics = round(ue_perf_df.shape[-1]/num_time)
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


def display_ue_perf(ue_perf_df, consolidated=False):
    ue_perf_df = ue_perf_df.copy()
    # Split df into time label
    num_time, num_metrics = 3, 7
    for time_horizon, cur_df in ue_perf_df.groupby(level="Time Horizon"):
        print(time_horizon)
        display(
            cur_df.style.apply(
                highlight_first_n_second_highest, subset=cur_df.columns[0], split_value=consolidated).apply(
                    highlight_first_n_second_lowest, subset=cur_df.columns[1:], split_value=consolidated
                )
        )


def display_ood_perf(ood_perf_df, consolidated=False):
    ood_perf_df = ood_perf_df.copy()
    # Split df into time label
    num_time, num_metrics = 3, 7
    for time_horizon, cur_df in ood_perf_df.groupby(level="Time Horizon"):
        print(time_horizon)
        display(
            cur_df.style.apply(
                highlight_first_n_second_highest, split_value=consolidated)
        )
