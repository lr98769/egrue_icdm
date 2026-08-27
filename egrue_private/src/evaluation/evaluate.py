import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, \
    mean_squared_error, mean_absolute_error, average_precision_score
from os.path import join
from IPython.display import display
from dcor import distance_correlation
from scipy.stats import spearmanr

from src.evaluation.misc import get_ue_loss_correct_col
from src.evaluation.perf_metrics import *
from src.display.display_df import highlight_first_n_second_highest, highlight_first_n_second_lowest
from src.display.ue_evaluation import \
    get_risk_coverage_curve, get_risk_coverage_curve_zero_one, get_ue_loss_scatterplot
from src.evaluation.ue_metrics import calculate_aurc, thresholded_loss
from src.configs.default_configs import split_col, split_names
from src.file_manager.filepath import FilePath

# Evalueate Prediction Performance
def get_model_performance(all_pred_df, data_dict, label, perf_split_col=split_col):
    target_col, num_classes = data_dict["target_col"], data_dict["num_classes"]
    pred_prob_cols = [f"{target_col}_class_prob_{i}_{label}" for i in range(num_classes)]
    model_perf_list = []
    split_list = []
    labels = [i for i in range(num_classes)]
    for split_name, split_df in all_pred_df.groupby(perf_split_col):
        # if split_name not in split_names:
        #     continue
        y_true = split_df[target_col].values
        y_score = split_df[pred_prob_cols].values
        y_pred = split_df[f"{target_col}_pred_label_{label}"].values
        if num_classes > 2:
            model_perf_list.append({
                "AUC": roc_auc_score(y_true=y_true, y_score=y_score, multi_class="ovo", labels=labels),
                "Accuracy": accuracy_score(y_true=y_true, y_pred=y_pred), 
                "Crossentropy Loss":get_crossentropyloss_from_pred_prob(y_true, pred_prob=y_score), 
                "Recall": recall_score(y_true=y_true, y_pred=y_pred, average="macro"), 
                "Precision": precision_score(y_true=y_true, y_pred=y_pred, average="macro"),
                "F1-Score": f1_score(y_true=y_true, y_pred=y_pred, average="macro"),
            })
        else:
            if len(y_score.shape)>1:
                y_score = y_score[:,-1]
            model_perf_list.append({
                "AUC": roc_auc_score(y_true=y_true, y_score=y_score),
                "Accuracy": accuracy_score(y_true=y_true, y_pred=y_pred), 
                "Crossentropy Loss":get_crossentropyloss_from_pred_prob(y_true, pred_prob=y_score), 
                "Recall": recall_score(y_true=y_true, y_pred=y_pred, average="binary"), 
                "Precision": precision_score(y_true=y_true, y_pred=y_pred, average="binary"),
                "F1-Score": f1_score(y_true=y_true, y_pred=y_pred, average="binary")
            })
        split_list.append(split_name)
    perf_df = pd.DataFrame(model_perf_list, index=split_list)
    if len(perf_df) == len(split_names):
        perf_df = perf_df.loc[split_names,:]
    elif ("Train" in perf_df.index) and ("Valid" in perf_df.index):
        train_val_idxs = ["Train", "Valid"]
        perf_df = perf_df.sort_values(by="Accuracy", ascending=False)
        # print(train_val_idxs+[idx for idx in perf_df.index if idx not in train_val_idxs])
        perf_df = perf_df.loc[train_val_idxs+[idx for idx in perf_df.index if idx not in train_val_idxs]]
    return perf_df

def get_cur_ue_perf_df(pred_df, ue_col, loss_col, correct_col, split_col="split"):
    pred_df = pred_df.copy()
    ue_perf_list = []
    index_list = []
    for split_name, split_df in pred_df.groupby(split_col):
        if np.max(split_df[loss_col])==np.inf:
            split_df = split_df.copy()
            split_df = split_df[split_df[loss_col]!=np.inf]
        loss = split_df[loss_col]
        ue = split_df[ue_col]
        zero_one_loss_col = "zero_one"
        correct = split_df[correct_col]
        zero_one_loss = np.invert(correct.values).astype(int)
        split_df[zero_one_loss_col] = zero_one_loss
        pearson_corr, pearson_pval = pearsonr(ue, loss)
        # dcor = distance_correlation(ue, loss)
        # spearman_corr, spearman_pval = spearmanr(ue, loss)
        aurc = calculate_aurc(ue, loss)
        aurc_zero_one = calculate_aurc(ue, zero_one_loss)
        # Misclassification
        auroc = roc_auc_score(y_true=zero_one_loss, y_score=ue)
        aupr = average_precision_score(y_true=zero_one_loss, y_score=ue)
        ue_perf_list.append({
            "Pearson's Correlation": pearson_corr,
            # "Pearson's P-Value": pearson_pval,
            f"AUROC": auroc,
            "AUPR": aupr,
            "AURC (CE Loss)": aurc,
            "AURC (0/1 Loss)": aurc_zero_one,
            f"Sigma Risk (0.1)": thresholded_loss(split_df, ue_col, zero_one_loss_col, 0.1), 
            f"Sigma Risk (0.2)": thresholded_loss(split_df, ue_col, zero_one_loss_col, 0.2), 
            f"Sigma Risk (0.3)": thresholded_loss(split_df, ue_col, zero_one_loss_col, 0.3), 
            f"Sigma Risk (0.4)": thresholded_loss(split_df, ue_col, zero_one_loss_col, 0.4), 
            # "dCor": dcor
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
    if len(split_names) == len(perf_df):
        perf_df = perf_df.loc[split_names]
    perf_df.index.name = "Split"
    return perf_df

def evaluate_ue(
    pred_df, ue_dict, fp: FilePath=None,
    only_test=True, 
    loss_col="crossentropy_loss", correct_col="correct", split_col=split_col, split="Test"
):
    if fp is not None:
        fp_evaluation = fp.get_fp_ue_folder()
    else:
        fp_evaluation = None
    pred_df  = pred_df.copy()
    test_df = pred_df[pred_df[split_col]==split]

    # Show Performance Tables
    if only_test:
        show_test_performance_tables(pred_df, ue_dict, fp_evaluation, loss_col, correct_col, split_col=split_col, split=split)
    else:
        show_all_performance_tables(pred_df, ue_dict, fp_evaluation, loss_col, correct_col, split_col=split_col)

    # Show ue-loss scatterplot
    get_ue_loss_scatterplot(test_df, ue_dict, loss_col, correct_col)
    if fp_evaluation:
        plt.savefig(join(fp_evaluation, "ue_loss_scatterplot.jpg"), bbox_inches="tight")

    # Show risk coverage curve
    get_risk_coverage_curve(test_df, ue_dict, loss_col, correct_col)
    if fp_evaluation:
        plt.savefig(join(fp_evaluation, "risk_coverage_curve.jpg"), bbox_inches="tight")

    # Show risk coverage curve (0/1 Loss)
    get_risk_coverage_curve_zero_one(test_df, ue_dict, loss_col, correct_col)
    if fp_evaluation:
        plt.savefig(join(fp_evaluation, "risk_coverage_curve_zero_one.jpg"), bbox_inches="tight")
    
def show_test_performance_tables(
    pred_df, ue_dict, fp_evaluation, 
    loss_col, correct_col, split="Test", split_col=split_col
):
    all_test_perf = []
    index = []
    for key, values in ue_dict.items():
        ue_col, cur_loss_col, cur_correct_col = get_ue_loss_correct_col(
            values, loss_col, correct_col)
        perf_df = get_cur_ue_perf_df(
            pred_df, ue_col=ue_col, loss_col=cur_loss_col, correct_col=cur_correct_col, split_col=split_col)
        if fp_evaluation:
            perf_df.to_csv(join(fp_evaluation, key+".csv"))
        test_perf_row = perf_df.loc[split, :]
        all_test_perf.append(test_perf_row)
        index.append(key)
    all_test_perf = pd.DataFrame(all_test_perf)
    all_test_perf.index = index
    if fp_evaluation:
        all_test_perf.to_csv(join(fp_evaluation, "ue_perf.csv"))
    lowest_cols = [
        "AURC (0/1 Loss)", "AURC (CE Loss)",
        "Sigma Risk (0.1)", "Sigma Risk (0.2)", "Sigma Risk (0.3)", "Sigma Risk (0.4)"]
    highest_cols = ["Pearson's Correlation", "AUROC", "AUPR"] # "Spearman's Correlation"
    display(
        all_test_perf.style.apply(
            highlight_first_n_second_highest, subset=highest_cols, split_value=False).apply(
                highlight_first_n_second_lowest, subset=lowest_cols, split_value=False
            )
    )

def show_all_performance_tables(
    pred_df, ue_dict, fp_evaluation,
    loss_col, correct_col, split_col
):
    for key, values in ue_dict.items():
        print(f"{key}:")
        ue_col, cur_loss_col, cur_correct_col = get_ue_loss_correct_col(
            values, loss_col, correct_col)
        perf_df = get_cur_ue_perf_df(
            pred_df, ue_col=ue_col, loss_col=cur_loss_col, correct_col=cur_correct_col, split_col=split_col)
        display(perf_df)
        fp_perf_df = join(fp_evaluation, key+".csv")
        perf_df.to_csv(fp_perf_df)


def get_model_performance_tabular(all_pred_df, target_col, label=None):
    model_perf_list = []
    split_list = []
    for split_name, split_df in all_pred_df.groupby("split"):
        y_true = split_df[target_col].values
        pred_prob_col, pred_label_col = f"{target_col}_pred_prob", f"{target_col}_pred_label"
        if label is not None:
            pred_prob_col+=f"_{label}"
            pred_label_col+=f"_{label}"
        y_score = split_df[pred_prob_col].values
        y_pred = split_df[pred_label_col].values
        model_perf_list.append({
            "AUC": roc_auc_score(y_true=y_true, y_score=y_score),
            "Accuracy": accuracy_score(y_true=y_true, y_pred=y_pred), 
            "Crossentropy Loss": get_crossentropyloss_from_pred_prob(torch.from_numpy(y_true), torch.from_numpy(y_score)),
            "Recall": recall_score(y_true=y_true, y_pred=y_pred), 
            "Precision": precision_score(y_true=y_true, y_pred=y_pred),
            "F1-Score": f1_score(y_true=y_true, y_pred=y_pred)
        })
        split_list.append(split_name)
    return pd.DataFrame(model_perf_list, index=split_list).loc[["Train", "Valid", "Test"],:]

