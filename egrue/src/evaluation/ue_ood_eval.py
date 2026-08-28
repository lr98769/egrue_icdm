import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from IPython.display import display
from os.path import join

from src.display.display_df import highlight_first_n_second_highest

def evaluate_ue_ood(
    pred_df, pred_df_ood, ue_dict, fp_fig=None, split_col="split", test_label="Test"
):
    pred_df, pred_df_ood = pred_df.copy(), pred_df_ood.copy()
    ood_col = "ood"
    # Get combined df
    pred_df_test = pred_df[pred_df[split_col]==test_label]
    pred_df_test.loc[:,[ood_col]] = 0
    pred_df_ood.loc[:,[ood_col]] = 1
    combined_df = pd.concat([pred_df_test, pred_df_ood])
    # Output DF
    eval_df = create_ue_ood_evaluation_df(ue_dict, combined_df, ood_col)
    # Display OOD Eval Fig
    create_ue_ood_evaluation_fig(ue_dict, combined_df, ood_col, fp_fig=fp_fig)
    return eval_df

def group_columns(columns, sep="-"):
    col_groups = {}
    for col in columns:
        col_group = col.split(sep=sep)[0]
        if col_group not in col_groups:
            col_groups[col_group] = [col]
        else:
            col_groups[col_group].append(col)
    # print(col_groups)
    return [col for key, val in col_groups.items() for col in val]

def evaluate_all_ue_ood(
    pred_df, pred_df_ood_dict, ue_dict, fp_fig=None, 
    split_col="split", test_label="Test", fp=None, show=True
):
    pred_df = pred_df.copy()
    all_ood_dfs = []
    for ood_name, pred_df_ood in tqdm(pred_df_ood_dict.items()):
        ood_df = evaluate_ue_ood(
            pred_df, pred_df_ood, ue_dict, fp_fig=fp_fig, 
            split_col=split_col, test_label=test_label
        )
        ood_df.columns = [f"{col}-{ood_name}" for col in ood_df.columns]
        all_ood_dfs.append(ood_df)
    perf_df = pd.concat(all_ood_dfs, axis=1)
    perf_df = perf_df[group_columns(perf_df.columns)]
    if show:
        display(
            perf_df.style.apply(
                highlight_first_n_second_highest, split_value=False)
        )
    if fp is not None:
        fp_evaluation = fp.get_fp_ue_folder()
        perf_df.to_csv(join(fp_evaluation, "ood_detection.csv"))
    return perf_df

def display_ood_perf(ood_perf_df, consolidated=False):
    ood_perf_df = ood_perf_df.copy()
    # Split df into time label
    num_time, num_metrics = 3, 7
    for time_horizon, cur_df in ood_perf_df.groupby(level="Time Horizon"):
        display(
            cur_df.style.apply(
                highlight_first_n_second_highest, split_value=consolidated)
        )

def create_ue_ood_evaluation_df(ue_dict, combined_df, ood_col):
    output_df, index = [], []
    for ue_name, ue_info_dict in ue_dict.items():
        ue_col = ue_info_dict["ue"]
        # Evaluate AUROC
        output_df.append({
            "OOD AUROC": roc_auc_score(y_true=combined_df[ood_col], y_score=combined_df[ue_col]),
            "OOD AUPR": average_precision_score(y_true=combined_df[ood_col], y_score=combined_df[ue_col]),
        })
        index.append(ue_name)
    return pd.DataFrame(output_df, index=index)

def create_ue_ood_evaluation_fig(ue_dict, combined_df, ood_col, dpi=300, fp_fig=None):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=dpi)
    for ue_name, ue_info_dict in ue_dict.items():
        ue_col = ue_info_dict["ue"]
        fpr, tpr, threshold =  roc_curve(y_true=combined_df[ood_col], y_score=combined_df[ue_col])
        ax.plot(fpr, tpr, label = ue_name)
    ax.set_ylabel("True Postive Rate")
    ax.set_xlabel("False Postive Rate")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(len(ue_dict), 2))
    fig.tight_layout()
    if fp_fig is not None:
        fig.savefig(fp_fig, bbox_inches="tight")
    else:
        fig.show()
        