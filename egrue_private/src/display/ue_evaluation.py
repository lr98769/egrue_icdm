import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

from src.display.display_image import transform_for_visualisation
from src.evaluation.misc import get_ue_loss_correct_col
from src.evaluation.ue_metrics import min_max_norm_wo_outliers

def get_ue_loss_scatterplot(
    pred_df, ue_dict, loss_col, correct_col=None, cutoff=None, nbins=10, title=None):
    eqn_label_fs = 7
    line_col = 'black'
    point_color = "#0090C1"
    point_size = 75
    point_alpha = 0.5
    marker = "."
    formatting_dict = dict(color=point_color, s=point_size, alpha=point_alpha, marker=marker, edgecolors='none')
    num_cols = len(ue_dict)
    bin_width = int(100/nbins)
    bin_edges = np.array([0]+list(np.linspace(bin_width, 100, num=10)/100))
    
    fig, axes = plt.subplots(2, num_cols, figsize=(num_cols*2, 1.5*2), dpi=300, sharey="row", sharex="col")
    if len(axes.shape) == 1:
        axes = np.expand_dims(axes, -1)
    for i, (ue_type, ue_info_dict) in enumerate(ue_dict.items()):
        cur_formatting_dict = formatting_dict.copy()
        if "scatter_color" in ue_info_dict:
            cur_formatting_dict["color"] = pred_df[ue_info_dict["scatter_color"]].values
        ue_df = pred_df.copy()
        ue_col, cur_loss_col, _ = get_ue_loss_correct_col(
            ue_info_dict, loss_col, correct_col)
        ue, loss = ue_df[ue_col], ue_df[cur_loss_col]
        ue = min_max_norm_wo_outliers(ue)
        # ue = (ue - ue.min()) / (ue.max() - ue.min())
        ue_df[ue_col] = ue
        # Plot scatter
        # axes[0, i].axvspan(0, bin_width/100, color='grey', alpha=0.2, linewidth=0)
        axes[0,i].scatter(ue, loss, **cur_formatting_dict)
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
        grouped_loss = grouped[cur_loss_col].mean()
        # Comparison of losses
        axes[1,i].bar(
            bin_edges[1:]-bin_width/100/2, grouped_loss, width=bin_width/100*0.9, color="#0090C1")
        ticks = list(axes[1,i].get_yticks(minor=True)) + [grouped_loss[0.1]]
        axes[1,0].set_yticks(ticks, minor=True)
        if i == 0:
            axes[1,i].set_ylabel("Mean Error")
        axes[1,i].set_xlabel(ue_type)
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()

def get_risk_coverage_curve(pred_df, ue_dict, loss_col, correct_col=None, cutoff=None, title=None):
    fig, ax = plt.subplots(1, 1, figsize=(5, 1.5), dpi=300)
    if cutoff:
        plt.axvspan(0, cutoff, color='grey', alpha=0.2, linewidth=0)
    for ue_type, ue_info_dict in ue_dict.items():
        loss_list, thres_list, prop_list = [], [], []
        ue_col, cur_loss_col, _ = get_ue_loss_correct_col(
            ue_info_dict, loss_col, correct_col)
        ue, loss = pred_df[ue_col], pred_df[cur_loss_col]
        ue = pred_df[ue_col]
        loss = pred_df[cur_loss_col]
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
    if title is not None:
        fig.suptitle(title)

def get_risk_coverage_curve_zero_one(pred_df, ue_dict, loss_col, correct_col, cutoff=None):
    fig, ax = plt.subplots(1, 1, figsize=(5, 1.5), dpi=300)
    if cutoff:
        plt.axvspan(0, cutoff, color='grey', alpha=0.2, linewidth=0)
    for ue_type, ue_info_dict in ue_dict.items():
        loss_list, thres_list, prop_list = [], [], []
        ue_col, _, cur_correct_col = get_ue_loss_correct_col(
            ue_info_dict, loss_col, correct_col)
        correct = pred_df[cur_correct_col]
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
    

def get_aggregated_values(pred_df, ue_dict):
    output_df = []
    splits = ["Train", "Valid", "Test"]
    for split in splits:
        split_df = pred_df[pred_df["split"]==split]
        cur_dict = {}
        for ue_name, cur_ue_dict in ue_dict.items():
            ue_col = cur_ue_dict["ue"]
            cur_dict[ue_name] = split_df[ue_col].mean()
        output_df.append(cur_dict)
    output_df = pd.DataFrame(output_df)
    output_df.index = splits
    return output_df

def show_examples_of_high_uncertainty(
    pred_df, pred_df_ood, data_dict, data_dict_ood, n_highest=0):
    pred_df, pred_df_ood = pred_df.copy(), pred_df_ood.copy()
    ood_col = "ood"
    test_pred_df = pred_df[pred_df["split"]=="Test"].reset_index()
    pred_df_ood = pred_df_ood.reset_index()
    test_pred_df[ood_col] = False
    pred_df_ood[ood_col] = True
    new_pred_df = pd.concat([test_pred_df, pred_df_ood])
    ue_dict = {
        "Aleatoric": "rue++m_alea_ue", 
        "Epistemic": "rue++m_epis_ue", 
        "Distributional": "rue++m_dist_ue"
    }
    target_col, pred_col = "class", "class_pred_label_rue++m"
    class_map = data_dict["classes"]
    class_map_ood = data_dict_ood["classes"]
    for ue_name, ue_col in ue_dict.items():
        incorrect_df = new_pred_df[
            (new_pred_df[target_col] != new_pred_df[pred_col]) | new_pred_df[ood_col]]
        incorrect_df = incorrect_df.sort_values(ue_col, ascending=False)
        ue = incorrect_df.iloc[n_highest][ue_col]
        max_idx = incorrect_df[incorrect_df[ue_col] == ue].index[0]
        row = new_pred_df[new_pred_df[ue_col]==ue].loc[max_idx]
        ood = row[ood_col]
        true = row[target_col]
        pred = row[pred_col]
        if ood:
            image, label = data_dict_ood["test_df"][max_idx]
            assert label == true
            plt.imshow(transform_for_visualisation(image))
            plt.show()
            print(ue_name, "Example: ")
            print("- True:", class_map_ood[int(true)])
            print("- Predicted:", class_map[int(pred)])
            for ue_name, ue_col in ue_dict.items():
                ue = row[ue_col]
                print(f"- {ue_name} UE:", ue)
        else:
            image, label = data_dict["test_df"][max_idx]
            assert label == true
            plt.imshow(transform_for_visualisation(image))
            plt.show()
            print(ue_name, "Example: ")
            print("True:", class_map[int(true)])
            print("Predicted:", class_map[int(pred)])
            for ue_name, ue_col in ue_dict.items():
                ue = row[ue_col]
                print(f"- {ue_name} UE:", ue)

from math import ceil
import seaborn as sns

def show_distribution_of_uncertainty_old(
    pred_df, pred_df_ood, ue_dict, num_cols=3, size=2, dpi=300):
    pd.set_option('mode.chained_assignment', None)
    pred_df, pred_df_ood = pred_df.copy(), pred_df_ood.copy()
    test_pred_df = pred_df[pred_df["split"]=="Test"]
    label_col = "Label"
    pred_df_ood.loc[:, label_col] = "OOD"
    num_ue = len(ue_dict)
    num_rows = ceil(num_ue/num_cols)
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(num_cols*size, num_rows*size), dpi=dpi)
    if num_rows > 1:
        axes = axes.flatten()
    for i, (ue, ue_info_dict) in tqdm(enumerate(ue_dict.items()), total=num_ue):
        ax = axes[i]
        ue_col = ue_info_dict["ue"]
        model_name = ue_info_dict["model"]
        correct_col = f"correct_{model_name}"
        new_labels = test_pred_df.loc[:,correct_col].replace(
            {True: "Correct", False:"Incorrect"})
        test_pred_df.loc[:,label_col] = new_labels
        new_pred_df = pd.concat([test_pred_df, pred_df_ood])
        sns.kdeplot(
            data=new_pred_df, x=ue_col, hue=label_col, ax=ax, fill=True, linewidth=0.2,
            legend=False #i==num_ue-1
        )
        
        ax.set_xlabel(ue)
        # if i == num_ue-1:
        #     legend = ax.legend()
        #     print(legend)
    for j in range(i+1,num_rows*num_cols):
        axes[j].set_axis_off()
    # sns.move_legend(axes[i], "center left", bbox_to_anchor=(0, .5), frameon=False)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def show_distribution_of_uncertainty(
    pred_df, pred_df_ood, ue_dict, num_cols=3, size=2, dpi=300):
    pd.set_option('mode.chained_assignment', None)
    pred_df, pred_df_ood = pred_df.copy(), pred_df_ood.copy()
    test_pred_df = pred_df[pred_df["split"]=="Test"]
    num_ue = len(ue_dict)
    combined_df = []
    ue_name_col, ue_value_col, label_col = "UE Name", "UE Value", "Label"
    pred_df_ood.loc[:, label_col] = "OOD"
    for ue_name, ue_info_dict in tqdm(ue_dict.items(), total=num_ue):
        ue_col = ue_info_dict["ue"]
        model_name = ue_info_dict["model"]
        correct_col = f"correct_{model_name}"
        test_pred_df.loc[:,label_col] = test_pred_df.loc[:,correct_col].replace(
            {True: "Correct", False:"Incorrect"})
        new_pred_df = pd.concat([test_pred_df, pred_df_ood])
        new_df = pd.DataFrame({
            ue_name_col: [ue_name for _ in range(len(new_pred_df))],
            ue_value_col: new_pred_df[ue_col],
            label_col: new_pred_df[label_col]
        })
        combined_df.append(new_df)
    combined_df = pd.concat(combined_df).reset_index()
    g = sns.displot(
        kind='kde', 
        data=combined_df, x=ue_value_col, hue=label_col, col=ue_name_col, col_wrap=num_cols,
        fill=True, linewidth=0.2, facet_kws={'sharex': False, 'sharey': False},
        height=size
        )
    g.set_titles("{col_name}")
    g.figure.set_dpi(dpi)
    sns.move_legend(g, "lower center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True)
    
# def show_distribution_of_uncertainty(
#     pred_df, pred_df_ood_dict, ue_dict, num_cols=3, size=2, dpi=300):
#     pd.set_option('mode.chained_assignment', None)
    
#     pred_df= pred_df.copy()
#     test_pred_df = pred_df[pred_df["split"]=="Test"]
#     num_ue = len(ue_dict)
#     combined_df = []
#     ue_name_col, ue_value_col, label_col = "UE Name", "UE Value", "Label"
#     pred_df_ood = []
#     for ood_name, df in pred_df_ood_dict.items():
#         df = df.copy()
#         df[label_col] = f"OOD {ood_name}"
#         pred_df_ood.append(df)
#     pred_df_ood = pd.concat(pred_df_ood)
#     for ue_name, ue_info_dict in tqdm(ue_dict.items(), total=num_ue):
#         ue_col = ue_info_dict["ue"]
#         model_name = ue_info_dict["model"]
#         correct_col = f"correct_{model_name}"
#         test_pred_df.loc[:,label_col] = test_pred_df.loc[:,correct_col].replace(
#             {True: "Correct", False:"Incorrect"})
#         new_pred_df = pd.concat([test_pred_df, pred_df_ood])
#         new_df = pd.DataFrame({
#             ue_name_col: [ue_name for _ in range(len(new_pred_df))],
#             ue_value_col: new_pred_df[ue_col],
#             label_col: new_pred_df[label_col]
#         })
#         combined_df.append(new_df)
#     combined_df = pd.concat(combined_df).reset_index()
#     g = sns.displot(
#         kind='kde', 
#         data=combined_df, x=ue_value_col, hue=label_col, col=ue_name_col, col_wrap=num_cols,
#         fill=True, linewidth=0.2, facet_kws={'sharex': False, 'sharey': False},
#         height=size, aspect=1
#         )
#     g.set_titles("{col_name}")
#     g.figure.set_dpi(dpi)
#     sns.move_legend(g, "lower center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True)
    