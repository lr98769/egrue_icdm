import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, mean_squared_error, mean_absolute_error
from os.path import join
from IPython.display import display

from src.evaluation.misc import get_ue_loss_correct_col
from src.evaluation.perf_metrics import *
from src.display_df import highlight_first_n_second_highest
from src.plotting.ue_evaluation import \
    get_risk_coverage_curve, get_risk_coverage_curve_zero_one, get_ue_loss_scatterplot
from src.display_df import highlight_first_n_second_lowest
from src.evaluation.ue_metrics import calculate_aurc, thresholded_loss
from src.configs.default_configs import split_col, split_names
from src.file_manager.filepath import FilePath

# Evalueate Prediction Performance
def get_model_performance(all_pred_df, data_dict, label):
    target_col, num_outputs = data_dict["target_col"], data_dict["num_outputs"]
    logit_cols = [f"{target_col}_logits_{i}_{label}" for i in range(num_outputs)]
    model_perf_list = []
    split_list = []
    for split_name, split_df in all_pred_df.groupby(split_col):
        if split_name not in split_names:
            continue
        y_true = split_df[target_col].values
        y_logits = split_df[logit_cols].values
        model_perf_list.append({
            "MAE": mean_absolute_error(y_true, y_logits),
            "MSE": mean_squared_error(y_true, y_logits),
        })
        split_list.append(split_name)
    return pd.DataFrame(model_perf_list, index=split_list).loc[split_names,:]

def get_cur_ue_perf_df(pred_df, ue_col, loss_col):
    pred_df = pred_df.copy()
    ue_perf_list = []
    index_list = []
    for split_name, split_df in pred_df.groupby("split"):
        loss = split_df[loss_col]
        ue = split_df[ue_col]
        pearson_corr, pearson_pval = pearsonr(ue, loss)
        # spearman_corr, spearman_pval = spearmanr(ue, loss)
        aurc = calculate_aurc(ue, loss)
        aurc_zero_one = calculate_aurc(ue, loss)
        ue_perf_list.append({
            "Pearson's Correlation": pearson_corr,
            "Pearson's P-Value": pearson_pval,
            "AURC (0/1 Loss)": aurc_zero_one,
            f"Sigma Risk (0.1)": thresholded_loss(split_df, ue_col, loss_col, 0.1), 
            f"Sigma Risk (0.2)": thresholded_loss(split_df, ue_col, loss_col, 0.2), 
            f"Sigma Risk (0.3)": thresholded_loss(split_df, ue_col, loss_col, 0.3), 
            f"Sigma Risk (0.4)": thresholded_loss(split_df, ue_col, loss_col, 0.4), 
            # "AURC": aurc,
            # "Spearman's Correlation": spearman_corr,
            # "Spearman's P-Value": spearman_pval,
            # "Mean Loss":np.mean(loss),
            # "Mean UE":np.mean(ue),
            # "Mean UE Correct": np.mean(ue[correct]),
            # "Mean UE Incorrect": np.mean(ue[np.logical_not(correct)])
        })
        index_list.append(split_name)
    # print(ue_col)
    perf_df = pd.DataFrame(ue_perf_list)
    perf_df.index = index_list
    perf_df = perf_df.loc[split_names]
    perf_df.index.name = "Split"
    return perf_df

def evaluate_ue(
    pred_df, data_dict, ue_dict, fp: FilePath=None,
    only_test=True, 
    loss_col="mae_loss"
):
    if fp is not None:
        fp_evaluation = fp.get_fp_ue_folder()
    else:
        fp_evaluation = None
    pred_df  = pred_df.copy()
    test_df = pred_df[pred_df[split_col]=="Test"]
    target_col = data_dict["target_col"]

    # Show Performance Tables
    if only_test:
        show_test_performance_tables(pred_df, ue_dict, fp_evaluation, loss_col)
    else:
        show_all_performance_tables(pred_df, ue_dict, fp_evaluation, loss_col)

    if only_test:
        test_df = pred_df[pred_df[split_col]=="Test"]
        # Show ue-loss scatterplot
        get_ue_loss_scatterplot(test_df, ue_dict, loss_col, title="Test")
        if fp_evaluation:
            plt.savefig(join(fp_evaluation, "ue_loss_scatterplot.jpg"), bbox_inches="tight")

        # Show risk coverage curve
        get_risk_coverage_curve(test_df, ue_dict, loss_col, title="Test")
        if fp_evaluation:
            plt.savefig(join(fp_evaluation, "risk_coverage_curve.jpg"), bbox_inches="tight")
    else:
        for key, val in pred_df.groupby(split_col):
            print(key+":")
            # Show ue-loss scatterplot
            get_ue_loss_scatterplot(val, ue_dict, loss_col, title=key)

            # Show risk coverage curve
            get_risk_coverage_curve(val, ue_dict, loss_col, title=key)
    
def show_test_performance_tables(
    pred_df, ue_dict, fp_evaluation, 
    loss_col
):
    all_test_perf = []
    index = []
    for key, values in ue_dict.items():
        ue_col, cur_loss_col, _ = get_ue_loss_correct_col(cur_ue_dict=values, loss_col=loss_col)
        perf_df = get_cur_ue_perf_df(pred_df, ue_col=ue_col, loss_col=cur_loss_col)
        if fp_evaluation:
            perf_df.to_csv(join(fp_evaluation, key+".csv"))
        test_perf_row = perf_df.loc["Test", :]
        all_test_perf.append(test_perf_row)
        index.append(key)
    all_test_perf = pd.DataFrame(all_test_perf)
    all_test_perf.index = index
    lowest_cols = ["Pearson's P-Value", "AURC (0/1 Loss)", "Sigma Risk (0.1)", "Sigma Risk (0.2)", "Sigma Risk (0.3)", "Sigma Risk (0.4)"]
    highest_cols = ["Pearson's Correlation"]
    display(
        all_test_perf.style.apply(
            highlight_first_n_second_highest, subset=highest_cols, split_value=False).apply(
                highlight_first_n_second_lowest, subset=lowest_cols, split_value=False
            )
    )
    
def show_all_performance_tables(
    pred_df, ue_dict, fp_evaluation,
    loss_col\
):
    perf_lists = {}
    index = []
    for key, values in ue_dict.items():
        ue_col, cur_loss_col, _ = get_ue_loss_correct_col(cur_ue_dict=values, loss_col=loss_col)
        perf_df = get_cur_ue_perf_df(pred_df, ue_col=ue_col, loss_col=cur_loss_col)
        if fp_evaluation:
            perf_df.to_csv(join(fp_evaluation, key+".csv"))
        for split, row in perf_df.iterrows():
            if split not in perf_lists:
                perf_lists[split] = []
            perf_lists[split].append(row)
        index.append(key)
    for split, df_list in perf_lists.items():
        print(split+":")
        all_perf = pd.DataFrame(df_list)
        all_perf.index = index
        lowest_cols = ["Pearson's P-Value", "AURC (0/1 Loss)", "Sigma Risk (0.1)", "Sigma Risk (0.2)", "Sigma Risk (0.3)", "Sigma Risk (0.4)"]
        highest_cols = ["Pearson's Correlation"]
        display(
            all_perf.style.apply(
                highlight_first_n_second_highest, subset=highest_cols, split_value=False).apply(
                    highlight_first_n_second_lowest, subset=lowest_cols, split_value=False
                )
        )

# def show_all_performance_tables(
#     pred_df, ue_dict, fp_evaluation,
#     loss_col\
# ):
#     for key, values in ue_dict.items():
#         print(f"{key}:")
#         ue_col, cur_loss_col, _ = get_ue_loss_correct_col(cur_ue_dict=values, loss_col=loss_col)
#         perf_df = get_cur_ue_perf_df(pred_df, ue_col=ue_col, loss_col=cur_loss_col)
#         display(perf_df)
#         if fp_evaluation is not None:
#             fp_perf_df = join(fp_evaluation, key+".csv")
#             perf_df.to_csv(fp_perf_df)

