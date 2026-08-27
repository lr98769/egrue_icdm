from os.path import join
import pandas as pd
import numpy as np

from src.configs.default_configs import fp_checkpoint_folder, fp_checkpoint_folder, fn_consolidated

def process_tabular_pred_perf(df):
    df = df.copy()
    # Change Column Name
    df.columns = [col if col != "Accuracy" else "Acc" for col in df.columns]
    # Change Method Names
    name_map = {
        "egRUE":  "egRUE & RUE", "gpc": "GPC", "MC Dropout": "MCD", "postNet": "PN", "de": "DE"
    }
    df.index = [method if method not in name_map else name_map[method] for method in df.index]
    return df

def process_image_pred_perf(df):
    df = df.copy()
    df = df.drop(["ResNet18 RUE", "PostNet Aleatoric"])
    # Change Method Names
    name_map = {
        "ResNet18 Entropy": "egRUE & RUE", "Deep Ensemble": "DE", "PostNet Epistemic": "PN", "ResNet18 MC Dropout": "MCD"
    }
    df.index = [method if method not in name_map else name_map[method] for method in df.index]
    return df

def process_tabular_ue_perf(df):
    df = df.copy()
    # rename index
    name_map = {
        "MC Dropout": "MCD", "Deep Ensemble": "DE", 
        "PostNet Aleatoric": "PN Alea", "PostNet Epistemic": "PN Epis", 
        "GPC Entropy": "GPC"
    }
    df.index = [method if method not in name_map else name_map[method] for method in df.index]
    return df

def process_image_ue_perf(df):
    df = df.copy()
    # rename cols
    col_map = {"Sigma Risk (0.1)": "Sigma-Risk Score (0.1)"}
    df.columns = [col if col not in col_map else col_map[col] for col in df.columns]
    # rename index
    name_map = {
        "ResNet18 egRUE": "egRUE", "ResNet18 RUE": "RUE", "ResNet18 Entropy": "Entropy",
        "ResNet18 MC Dropout": "MCD", "Deep Ensemble": "DE", 
        "PostNet Aleatoric": "PN Alea", "PostNet Epistemic": "PN Epis", 
    }
    df.index = [method if method not in name_map else name_map[method] for method in df.index]
    return df

def process_image_ood_perf(df):
    df = df.copy()
    # rename index
    name_map = {
        "ResNet18 egRUE": "egRUE", "ResNet18 RUE": "RUE", "ResNet18 Entropy": "Entropy",
        "ResNet18 MC Dropout": "MCD", "Deep Ensemble": "DE", 
        "PostNet Aleatoric": "PN Aleatoric", "PostNet Epistemic": "PN Epistemic", 
    }
    df.index = [method if method not in name_map else name_map[method] for method in df.index]
    return df

def all_data_perf(data_names, process_df_funcs=None, columns=None, order=None, pref_file="pred_perf.csv"):
    all_pred_perf_df = None
    def empty_func(df):
        return df
    if process_df_funcs is None:
        process_df_funcs = [empty_func for _ in data_names]
    for data_name, process_df in zip(data_names, process_df_funcs):
        fp_consolidated = join(fp_checkpoint_folder, fn_consolidated, data_name)
        pred_perf_df = pd.read_csv(join(fp_consolidated, pref_file), index_col=0)
        pred_perf_df = process_df(pred_perf_df)
        if columns is not None:
            pred_perf_df = pred_perf_df[columns]
        # Make multilevel column names
        level_0 = [data_name for i in range(pred_perf_df.shape[-1])]
        level_1 = pred_perf_df.columns
        multi_index = pd.MultiIndex.from_arrays([level_0, level_1], names=['Data', 'Metric'])
        pred_perf_df.columns = multi_index
        if all_pred_perf_df is None:
            all_pred_perf_df = pred_perf_df
        else:
            all_pred_perf_df = all_pred_perf_df.join(pred_perf_df, how="outer")
    if order is not None:
        all_pred_perf_df = all_pred_perf_df.loc[order]
    return all_pred_perf_df

def get_other_methods(pred_perf_df, proposed_method):
    other_methods = [method for method in pred_perf_df.index if method != proposed_method]
    return other_methods

def calculate_avg_improvement(pred_perf_df, proposed_method, column_direction):
    other_methods = get_other_methods(pred_perf_df, proposed_method)
    improvement_dict = {}
    for col, direction in zip(pred_perf_df.columns, column_direction):
        proposed_method_perf = float(pred_perf_df.loc[proposed_method, [col]])
        other_method_perf = pred_perf_df.loc[other_methods, col].values.astype(float).mean()
        # print(proposed_method_perf, other_method_perf)
        if direction == "max":
            improvement_array = (proposed_method_perf-other_method_perf)/np.abs(other_method_perf)
        else:
            improvement_array = (other_method_perf-proposed_method_perf)/np.abs(other_method_perf)
        # print(improvement_array)
        improvement_dict[col] = improvement_array
    return improvement_dict

def calculate_improvement_over_sota(pred_perf_df, proposed_method, column_direction, sota="DE"):
    improvement_dict = {}
    for col, direction in zip(pred_perf_df.columns, column_direction):
        proposed_method_perf = float(pred_perf_df.loc[proposed_method, col])
        sota_method_perf = float(pred_perf_df.loc[sota, col])
        # print(proposed_method_perf, other_method_perf)
        if direction == "max":
            improvement_array = (proposed_method_perf-sota_method_perf)/np.abs(sota_method_perf)
        else:
            improvement_array = (sota_method_perf-proposed_method_perf)/np.abs(sota_method_perf)
        improvement_dict[col] = improvement_array*100
    return improvement_dict

def all_data_perf_improvement(
        proposed_method, data_names, 
        process_df_funcs=None, 
        calculate_improvement_func=calculate_improvement_over_sota, 
        column_direction=["max", "max", "min", "min"],
        columns=None, order=None, pref_file="pred_perf.csv", sf=3):
    improvement_df = []
    def empty_func(df):
        return df
    if process_df_funcs is None:
        process_df_funcs = [empty_func for _ in data_names]
    for data_name, process_df in zip(data_names, process_df_funcs):
        fp_consolidated = join(fp_checkpoint_folder, fn_consolidated, data_name)
        pred_perf_df = pd.read_csv(join(fp_consolidated, pref_file), index_col=0)
        pred_perf_df = process_df(pred_perf_df)
        if columns is not None:
            pred_perf_df = pred_perf_df[columns]
        if order is not None:
            cur_order = [method for method in order if method in pred_perf_df.index]
            pred_perf_df = pred_perf_df.loc[cur_order]
        for col in pred_perf_df.columns:
            pred_perf_df[col] = pred_perf_df[col].str.split(" ± ").str[0]
        # display(pred_perf_df)
        improvement_dict = calculate_improvement_func(
            pred_perf_df, proposed_method, column_direction)
        improvement_df.append(improvement_dict)
    return pd.DataFrame(improvement_df, index=data_names).round(sf)