
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os.path import join
from IPython.display import display
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import auc, roc_auc_score, average_precision_score


def calculate_aurc(ue, loss):
    num_samples = len(ue)
    ue_loss_df = pd.DataFrame({"ue":ue, "loss":loss})
    ue_loss_df = ue_loss_df.sort_values(by="ue", ascending=True)
    ue_loss_df["cumulative_loss"] = ue_loss_df["loss"].expanding().mean()
    ue_loss_df["coverage"] = (np.arange(num_samples)+1)/num_samples
    return auc(ue_loss_df["coverage"], ue_loss_df["cumulative_loss"].values)

def calculate_thresholded_loss(ue, loss, threshold):
    ue = (ue - ue.min()) / (ue.max() - ue.min())
    return loss[ue<threshold].mean()

def calculate_prop_thresh_loss(ue, loss, threshold):
    ue_threshold = np.percentile(ue, threshold*100)
    return loss[ue<ue_threshold].mean()

def evaluate_cur_ue(pred_df, ue_col, loss_col, pred_label_col, target_col):
    from scipy.stats import pearsonr
    ue_perf_list = []
    index_list = []
    for split_name, split_df in pred_df.groupby("split"):
        loss = split_df[loss_col]
        ue = split_df[ue_col]
        correct = split_df[target_col] == split_df[pred_label_col]
        zero_one_loss = np.invert(correct.values).astype(int)
        # print(loss.isna().sum())
        pearson_corr, pearson_pval = pearsonr(ue, loss)
        spearman_corr, spearman_pval = spearmanr(ue, loss)
        aurc = calculate_aurc(ue, loss)
        aurc_zero_one = calculate_aurc(ue, zero_one_loss)
        auroc = roc_auc_score(zero_one_loss, ue)
        aupr = average_precision_score(zero_one_loss, ue)
        
        ue_perf_list.append({
            "Pearson's Correlation": pearson_corr,
            "Pearson's P-Value": pearson_pval,
            "AURC (0/1 Loss)": aurc_zero_one,
            "AUROC": auroc,
            "AUPR": aupr,
            "Sigma-Risk Score (0.1)": calculate_thresholded_loss(ue, zero_one_loss, threshold=0.1), 
            "Sigma-Risk Score (0.2)": calculate_thresholded_loss(ue, zero_one_loss, threshold=0.2), 
            "Sigma-Risk Score (0.3)": calculate_thresholded_loss(ue, zero_one_loss, threshold=0.3), 
            "Sigma-Risk Score (0.4)": calculate_thresholded_loss(ue, zero_one_loss, threshold=0.4), 
            "AURC": aurc,
            # "Spearman's Correlation": spearman_corr,
            # "Spearman's P-Value": spearman_pval,
            # "Mean Loss":np.mean(loss),
            # "Mean UE":np.mean(ue),
            # "Mean UE Correct": np.mean(ue[correct]),
            # "Mean UE Incorrect": np.mean(ue[np.logical_not(correct)])
        })
        index_list.append(split_name)
    print(ue_col)
    perf_df = pd.DataFrame(ue_perf_list)
    perf_df.index = index_list
    split_list = [split for split in ["Train", "Valid", "Test"] if split in perf_df.index]
    perf_df = perf_df.loc[split_list]
    perf_df.index.name = "Split"
    display(perf_df)
    return perf_df

def scale_ue_minmax(ue_array, split_array):
    train_ue = ue_array[split_array=="Train"]
    min_ue, max_ue = min(train_ue), max(train_ue)
    return (ue_array - min_ue)/(max_ue - min_ue)

def scale_ue_znorm(ue_array, split_array):
    train_ue = ue_array[split_array=="Train"]
    mean, std = np.mean(train_ue), np.std(train_ue)
    return (ue_array - mean)/std

def get_ue_loss_scatterplot(pred_df, stat_dict, cutoff=None, nbins=10):
    eqn_label_fs = 7
    line_col = 'black'
    point_color = "#0090C1"
    point_size = 75
    point_alpha = 0.5
    marker = "."
    formatting_dict = dict(color=point_color, s=point_size, alpha=point_alpha, marker=marker, edgecolors='none')
    num_cols = len(stat_dict)
    bin_width = int(100/nbins)
    bin_edges = np.array([0]+list(np.linspace(bin_width, 100, num=10)/100))
    
    fig, axes = plt.subplots(2, num_cols, figsize=(num_cols*2, 1.5*2), dpi=300, sharey="row", sharex="col")
    for i, (ue_type, ue_info_dict) in enumerate(stat_dict.items()):
        ue_col, loss_col = ue_info_dict["ue"], ue_info_dict["loss"]
        ue_df = pred_df.copy()
        ue, loss = ue_df[ue_col], ue_df[loss_col]
        ue = (ue - ue.min()) / (ue.max() - ue.min())
        ue_df[ue_col] = ue
        
        # Plot scatter
        # axes[0, i].axvspan(0, bin_width/100, color='grey', alpha=0.2, linewidth=0)
        axes[0,i].scatter(ue, loss, **formatting_dict)
        # Plot line
        m, c = np.polyfit(ue, loss, 1) 
        ue = np.sort(ue)
        axes[0, i].plot(ue, m*ue+c, color=line_col, linestyle='-', label=f'y = {m:.3f}x + {c:.3f}', linewidth=1.5)
        axes[0, i].legend(fontsize=eqn_label_fs)
        if i == 0:
            axes[0, i].set_ylabel("Error")
            
        # Plot Barplot
        ue_df['bin'] = pd.cut(
            ue_df[ue_col], bins=bin_edges, labels=bin_edges[1:], include_lowest=True, right=True)
        grouped = ue_df.groupby("bin", observed=False)
        grouped_loss = grouped[loss_col].mean()
        # Comparison of losses
        axes[1,i].bar(
            bin_edges[1:]-bin_width/100/2, grouped_loss, width=bin_width/100*0.9, color="#0090C1")
        ticks = list(axes[1,i].get_yticks(minor=True)) + [grouped_loss[0.1]]
        axes[1,0].set_yticks(ticks, minor=True)
        if i == 0:
            axes[1,i].set_ylabel("Mean Error")
        axes[1,i].set_xlabel(ue_type)

    plt.tight_layout()

def get_risk_coverage_curve(pred_df, stat_dict, cutoff=None):
    fig, ax = plt.subplots(1, 1, figsize=(5, 1.5), dpi=300)
    if cutoff:
        plt.axvspan(0, cutoff, color='grey', alpha=0.2, linewidth=0)
    for ue_type, ue_info_dict in stat_dict.items():
        loss_list, thres_list, prop_list = [], [], []
        ue_col, loss_col = ue_info_dict["ue"], ue_info_dict["loss"]
        ue = pred_df[ue_col]
        loss = pred_df[loss_col]
        n = len(ue)
        thres_list = np.unique(ue)
        thres_list = np.sort(ue) # sorted in increasing order
        for threshold in thres_list:
            confident_mask = ue<=threshold
            confident_proportion = np.sum(confident_mask)/n
            cur_loss = np.mean(loss[confident_mask])
            prop_list.append(confident_proportion)
            loss_list.append(cur_loss)
        ax.plot(np.array(prop_list)*100, loss_list, label=ue_type, alpha=0.8)
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Prediction Error")
    if cutoff:
        ax.set_xticks(list(ax.get_xticks()) + [cutoff])
        ax.set_xlim(xmin=-5, xmax=105)
    ax.legend(
        loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=3)

def get_risk_coverage_curve_zero_one(pred_df, stat_dict, target_col, cutoff=None):
    fig, ax = plt.subplots(1, 1, figsize=(5, 1.5), dpi=300)
    if cutoff:
        plt.axvspan(0, cutoff, color='grey', alpha=0.2, linewidth=0)
    for ue_type, ue_info_dict in stat_dict.items():
        loss_list, thres_list, prop_list = [], [], []
        ue_col, pred_label_col = ue_info_dict["ue"], ue_info_dict["pred_label"]
        correct = pred_df[target_col] == pred_df[pred_label_col]
        loss = np.invert(correct.values).astype(int)
        ue = pred_df[ue_col]
        n = len(ue)
        thres_list = np.unique(ue)
        thres_list = np.sort(ue) # sorted in increasing order
        for threshold in thres_list:
            confident_mask = ue<=threshold
            confident_proportion = np.sum(confident_mask)/n
            cur_loss = np.mean(loss[confident_mask])
            prop_list.append(confident_proportion)
            loss_list.append(cur_loss)
        ax.plot(np.array(prop_list)*100, loss_list, label=ue_type, alpha=0.8)
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Prediction Error")
    if cutoff:
        ax.set_xticks(list(ax.get_xticks()) + [cutoff])
        ax.set_xlim(xmin=-5, xmax=105)
    ax.legend(
        loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=3)
    
def get_incorrect_class_prob(true, pred_prob):
    pred_prob = np.array([1-pred_prob, pred_prob]).transpose()
    return pred_prob[range(len(true)), 1-true]

def evaluate_ue(pred_df, stat_dict, target_col, fp_evaluation,):
    pred_df  = pred_df.copy()
    split_array = pred_df["split"]
    test_df = pred_df[pred_df["split"]=="Test"]

    # Show Performance Tables
    for key, values in stat_dict.items():
        perf_df = evaluate_cur_ue(
            pred_df, 
            ue_col=values["ue"], loss_col=values["loss"], 
            pred_label_col=values["pred_label"], target_col=target_col
        )
        fp_perf_df = join(fp_evaluation, key+".csv")
        perf_df.to_csv(fp_perf_df)

    # Show ue-loss scatterplot
    get_ue_loss_scatterplot(test_df, stat_dict)
    fp_ue_loss_scaterplot = join(fp_evaluation, "ue_loss_scatterplot.jpg")
    plt.savefig(fp_ue_loss_scaterplot, bbox_inches="tight")

    # Show risk coverage curve
    get_risk_coverage_curve(test_df, stat_dict)
    fp_risk_coverage_curve = join(fp_evaluation, "risk_coverage_curve.jpg")
    plt.savefig(fp_risk_coverage_curve, bbox_inches="tight")

    # Show risk coverage curve (0/1 Loss)
    get_risk_coverage_curve_zero_one(test_df, stat_dict, target_col)
    fp_risk_coverage_curve_zero_one = join(fp_evaluation, "risk_coverage_curve_zero_one.jpg")
    plt.savefig(fp_risk_coverage_curve_zero_one, bbox_inches="tight")
