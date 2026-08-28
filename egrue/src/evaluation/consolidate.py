import pandas as pd
import numpy as np
from os.path import join
from IPython.display import display

def combine_mean_n_std_matrices(mean, std, sp=3):
    assert mean.shape == std.shape
    shape = mean.shape
    returned_list = []
    for i in range(shape[0]):
        cur_list = []
        for j in range(shape[1]):
            cur_list.append(f"{round(mean[i][j],sp):.{sp}f} ± {std[i][j]:.{sp}f}")
        returned_list.append(cur_list)
    return returned_list

def get_mean_std_of_all_seed_csvs(seed_list, fp_folder, filename, sp=3, reindex=None):
    result_list = []
    for cur_seed in seed_list:
        fp_perf = join(fp_folder, str(cur_seed), filename)
        df = pd.read_csv(fp_perf, index_col=0)
        # display(df)
        if reindex is not None:
            df = df.reset_index()
            df = df.set_index(reindex)
        result_list.append(df.values)
    results = np.array(result_list)
    combined_mean = np.mean(results, axis=0)
    combined_std = np.std(results, axis=0)
    return pd.DataFrame(
        combine_mean_n_std_matrices(combined_mean, combined_std, sp=sp), 
        index=df.index, columns=df.columns)
    
def get_mean_std_of_all_seed_csvs_one_column(seed_list, fp_folder, filename, sp=3, reindex=None):
    result_list = []
    for cur_seed in seed_list:
        fp_perf = join(fp_folder, str(cur_seed), filename)
        df = pd.read_csv(fp_perf, index_col=0)
        if reindex is not None:
            df = df.reset_index()
            df = df.set_index(reindex)
        result_list.append(df.mean(axis=1).values[:, None])
    results = np.array(result_list)
    combined_mean = np.mean(results, axis=0)
    combined_std = np.std(results, axis=0)
    return pd.DataFrame(
        combine_mean_n_std_matrices(combined_mean, combined_std, sp=sp), 
        index=df.index, columns=["Aggregated"])

def get_mean_of_all_seed_csvs(seed_list, fp_folder, filename, reindex):
    result_list = []
    for cur_seed in seed_list:
        fp_perf = join(fp_folder, str(cur_seed), filename)
        df = pd.read_csv(fp_perf, index_col=0)
        if reindex is not None:
            df = df.reset_index()
            df = df.set_index(reindex)
        result_list.append(df.values)
    results = np.array(result_list)
    combined_mean = np.mean(results, axis=0)
    return pd.DataFrame(
        combined_mean, 
        index=df.index, columns=df.columns)
    
def get_std_of_all_seed_csvs(seed_list, fp_folder, filename, reindex):
    result_list = []
    for cur_seed in seed_list:
        fp_perf = join(fp_folder, str(cur_seed), filename)
        df = pd.read_csv(fp_perf, index_col=0)
        if reindex is not None:
            df = df.reset_index()
            df = df.set_index(reindex)
        result_list.append(df.values)
    results = np.array(result_list)
    combined_std = np.std(results, axis=0)
    return pd.DataFrame(
        combined_std, 
        index=df.index, columns=df.columns)
    
def consolidate_pred_perf(seed_list, fp_evaluation, one_col=False, filename="pred_perf.csv", sp=4):
    if one_col:
        return get_mean_std_of_all_seed_csvs_one_column(
            seed_list, fp_evaluation, filename=filename, sp=sp)
    else:
        return get_mean_std_of_all_seed_csvs(
            seed_list, fp_evaluation, filename=filename, sp=sp)
    
def consolidate_ue_perf(seed_list, fp_evaluation, exclude_columns=None, reindex=None):
    output_df = get_mean_std_of_all_seed_csvs(
        seed_list, fp_evaluation, filename="ue_perf.csv", sp=3, reindex=reindex)
    if exclude_columns is not None:
        output_df = output_df.drop(columns=exclude_columns)
    return output_df

def consolidate_ood_perf(seed_list, fp_evaluation, exclude_columns=None, reindex=None):
    output_df = get_mean_std_of_all_seed_csvs(
        seed_list, fp_evaluation, filename="ood_detection.csv", sp=3, reindex=reindex)
    if exclude_columns is not None:
        output_df = output_df.drop(columns=exclude_columns)
    return output_df

    

