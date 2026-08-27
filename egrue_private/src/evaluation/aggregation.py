import pandas as pd
import numpy as np
from os.path import join, exists
from IPython.display import display
from tqdm.auto import tqdm

from src.file_manager.load_save_df import load_pred_perf_df, load_ue_perf_df, load_all_pred_dfs
from src.file_manager.filepath import FilePath
from src.configs.default_configs import fp_checkpoint_folder, fn_consolidated, \
    fn_consolidated_pred_perf, fn_consolidated_ue_perf
from src.file_manager.filepath import create_folder
from src.evaluation.evaluate import evaluate_ue
from src.display_df import highlight_first_n_second_highest, highlight_first_n_second_lowest     
from src.display_df import underline_second_best, bold_best
from src.evaluation.ue_ood_eval import evaluate_ue_ood, evaluate_all_ue_ood

perf_lowest_cols = ["Crossentropy Loss"]
perf_highest_cols = ["AUC", "Accuracy", "Recall","Precision", "F1-Score"]
# "AURC (CE Loss)", 
ue_lowest_cols = ["AURC (0/1 Loss)", "Sigma Risk (0.1)", "Sigma Risk (0.2)", "Sigma Risk (0.3)", "Sigma Risk (0.4)"]
ue_highest_cols = ["Pearson's Correlation"] 
# "AUROC", "AUPR"
ood_lowest_cols = []
ood_highest_cols = ["OOD AUROC"]

# "Pearson's P-Value", 

def combine_mean_n_std_matrices(mean, std):
    assert mean.shape == std.shape
    shape = mean.shape
    returned_list = []
    for i in range(shape[0]):
        cur_list = []
        for j in range(shape[1]):
            cur_list.append(f"{mean[i][j]:.3f} ± {std[i][j]:.3f}")
        returned_list.append(cur_list)
    return returned_list

def get_perf_of_all_seed_csvs(seed_list, data_name, load_func):
    result_list = []
    for cur_seed in seed_list:
        fp = FilePath(data_name=data_name, seed=cur_seed)
        df = load_func(fp)
        # display(df)
        result_list.append(df.values)
    results = np.array(result_list)
    combined_mean = np.mean(results, axis=0)
    combined_std = np.std(results, axis=0)
    return pd.DataFrame(
        combine_mean_n_std_matrices(combined_mean, combined_std), 
        index=df.index, columns=df.columns)

def derive_consolidated_excel_pred_perf(data_name, ModelClass_dict, seed_list):
    fp_consolidated_folder = join(fp_checkpoint_folder, data_name, fn_consolidated)
    create_folder(fp_consolidated_folder)
    fp_consolidated_pred_perf = join(fp_consolidated_folder, fn_consolidated_pred_perf)
    df_dict = {}
    with pd.ExcelWriter(fp_consolidated_pred_perf) as writer:  
        for ModelClass, op_labels in ModelClass_dict.items():
            op_labels.insert(0, None)
            for optional_label in op_labels:
                def load_df(fp):
                    return load_pred_perf_df(fp, ModelClass, optional_label)
                avg_results_df = get_perf_of_all_seed_csvs(seed_list, data_name, load_df)
                name = f"{ModelClass.name}_{optional_label}" if optional_label is not None else ModelClass.name
                avg_results_df.to_excel(writer, sheet_name=name)
                df_dict[name]=avg_results_df
    return df_dict

def derive_consolidated_excel_ue_perf(
    data_name, data_dict, ue_dict, ModelClass_dict, seed_list, index="multi", override=False,):
    # Get all ue performance if it is not available
    last_fp = FilePath(data_name, seed=seed_list[-1])
    fp_last_ue = join(last_fp.get_fp_ue_folder(), "risk_coverage_curve_zero_one.jpg")
    if override or not exists(fp_last_ue):
        get_all_ue_performance_dfs(data_name, data_dict, ue_dict, ModelClass_dict, seed_list, index)
    fp_consolidated_folder = join(fp_checkpoint_folder, data_name, fn_consolidated)
    create_folder(fp_consolidated_folder)
    fp_consolidated_ue_perf = join(fp_consolidated_folder, fn_consolidated_ue_perf)
    df_dict = {}
    with pd.ExcelWriter(fp_consolidated_ue_perf) as writer:  
        for ue_label in ue_dict.keys(): 
            def load_df(fp):
                return load_ue_perf_df(fp, ue_label)
            avg_results_df = get_perf_of_all_seed_csvs(
                seed_list, data_name, load_df)
            avg_results_df = avg_results_df[ue_highest_cols+ue_lowest_cols]
            # print(f"{ue_label}:")
            # display(avg_results_df)
            avg_results_df.to_excel(writer, sheet_name=ue_label)
            df_dict[ue_label] = avg_results_df
    return df_dict

def derive_consolidated_ood_perf(data_name, ModelClass_dict, ue_dict, seed_list, override=False):
    fn = "ood_perf.csv"
    fn_fig = "ood_perf_roc.jpg"
    last_fp = FilePath(data_name, seed=seed_list[-1])
    fp_last_ood = join(last_fp.get_fp_ue_folder(), fn)
    if override or not exists(fp_last_ood):
        # Evaluate All OOD
        for cur_seed in tqdm(seed_list):
            fp = FilePath(data_name=data_name, seed=cur_seed)
            fp_ood = join(fp.get_fp_ue_folder(), fn)
            fp_ood_fig = join(fp.get_fp_ue_folder(), fn_fig)
            pred_df = load_all_pred_dfs(fp, ModelClass_dict=ModelClass_dict)
            pred_df_ood = load_all_pred_dfs(fp, ModelClass_dict=ModelClass_dict, pred_optional_label="ood")
            ood_df = evaluate_ue_ood(
                pred_df=pred_df, pred_df_ood=pred_df_ood, ue_dict=ue_dict, fp_fig=fp_ood_fig
            )
            ood_df.to_csv(fp_ood)
    def load_func(fp):
        fp_ood = join(fp.get_fp_ue_folder(), fn)
        return pd.read_csv(fp_ood, index_col=0)
    avg_results_df = get_perf_of_all_seed_csvs(seed_list, data_name, load_func)
    return avg_results_df

def derive_consolidated_all_ood_perf(
    data_name, ModelClass_dict, ue_dict, seed_list, ood_list, override=False, index="single"):
    fn = "ood_perf.csv"
    fn_fig = "ood_perf_roc.jpg"
    last_fp = FilePath(data_name, seed=seed_list[-1])
    fp_last_ood = join(last_fp.get_fp_ue_folder(), fn)
    if override or not exists(fp_last_ood):
        # Evaluate All OOD
        for cur_seed in tqdm(seed_list):
            fp = FilePath(data_name=data_name, seed=cur_seed)
            fp_ood = join(fp.get_fp_ue_folder(), fn)
            fp_ood_fig = join(fp.get_fp_ue_folder(), fn_fig)
            pred_df = load_all_pred_dfs(fp, ModelClass_dict=ModelClass_dict, index=index)
            pred_df_ood_dict = {}
            for key in ood_list:
                pred_df_ood_dict[key] = load_all_pred_dfs(
                    fp, ModelClass_dict=ModelClass_dict, pred_optional_label=f"ood_{key}", index=index)
            ood_df = evaluate_all_ue_ood(
                pred_df=pred_df, pred_df_ood_dict=pred_df_ood_dict, ue_dict=ue_dict
            )
            ood_df.to_csv(fp_ood)
    def load_func(fp):
        fp_ood = join(fp.get_fp_ue_folder(), fn)
        return pd.read_csv(fp_ood, index_col=0)
    avg_results_df = get_perf_of_all_seed_csvs(seed_list, data_name, load_func)
    return avg_results_df
 
def get_all_ue_performance_dfs(data_name, data_dict, ue_dict, ModelClass_dict, seed_list, index):
    for cur_seed in tqdm(seed_list):
        fp = FilePath(data_name, seed=cur_seed)
        pred_df = load_all_pred_dfs(fp, ModelClass_dict=ModelClass_dict, index=index)
        evaluate_ue(
            pred_df=pred_df, ue_dict=ue_dict, data_dict=data_dict, only_test=False, fp=fp
        )

def get_test_df(perf_dict):
    test_perf_df = []
    index_list = []
    for label, df in perf_dict.items():
        test_perf_df.append(df.loc[["Test"],:]) 
        index_list.append(label)
    test_perf_df = pd.concat(test_perf_df)
    test_perf_df.index = index_list
    return test_perf_df

def show_perf_dict(perf_dict, lowest_cols, highest_cols, only_test=True):
    if not only_test:
        for label, df in perf_dict.items():
            print(f"{label}:")
            display(df)
    else:
        perf_df = get_test_df(perf_dict)
        show_perf_df(perf_df, lowest_cols, highest_cols)
        
def show_perf_df(perf_df, lowest_cols, highest_cols):
    display(
        perf_df.style.apply(
            highlight_first_n_second_highest, subset=highest_cols, split_value=True).apply(
                highlight_first_n_second_lowest, subset=lowest_cols, split_value=True
            )
    )

def show_pred_perf_dict(pred_perf_dict, only_test=True):
    show_perf_dict(pred_perf_dict, perf_lowest_cols, perf_highest_cols, only_test=only_test)
    
def show_ue_perf_dict(ue_perf_dict, only_test=True):
    show_perf_dict(ue_perf_dict, ue_lowest_cols, ue_highest_cols, only_test=only_test)
    
def show_ood_perf_df(ood_perf_df):
    all_lowest_cols, all_highest_cols = get_lowest_highest_col(
        columns=ood_perf_df.columns,
        lowest_cols=ood_lowest_cols, highest_cols=ood_highest_cols)
    show_perf_df(ood_perf_df, lowest_cols=all_lowest_cols, highest_cols=all_highest_cols)
    
def show_ue_ood_perf(ue_perf_dict, ood_perf_df):
    ue_perf_df = get_test_df(ue_perf_dict)
    all_perf_df = ue_perf_df.join(ood_perf_df)
    all_lowest_cols, all_highest_cols = get_lowest_highest_col(
        columns=all_perf_df,
        lowest_cols=ue_lowest_cols+ood_lowest_cols, 
        highest_cols=ue_highest_cols+ood_highest_cols)
    show_perf_df(all_perf_df, lowest_cols=all_lowest_cols, highest_cols=all_highest_cols)
    
def get_superset_w_keywords(query_list, keyword_list):
    return [
        query for query in query_list 
        if any(keyword in query for keyword in keyword_list)]

def get_lowest_highest_col(columns, lowest_cols, highest_cols):
    all_lowest_cols = get_superset_w_keywords(columns, lowest_cols)
    all_highest_cols = get_superset_w_keywords(columns, highest_cols)
    return all_lowest_cols, all_highest_cols
    
def get_latex_output_perf_dict(perf_dict, lowest_cols, highest_cols):
    perf_df = get_test_df(perf_dict)
    perf_df = perf_df.copy()
    return get_latex_output_perf_df(perf_df, lowest_cols, highest_cols)
    
def get_latex_output_perf_df(perf_df, lowest_cols, highest_cols):
    column_format_dict = {col: "min" for col in lowest_cols}
    column_format_dict.update({col: "max" for col in highest_cols})
    for col, direction in column_format_dict.items():
        perf_df[col] = underline_second_best(perf_df[col], direction)
    styler = perf_df.style
    # Bold column names
    styler.map_index(lambda v: "textbf:--rwrap;", axis="columns")
    # Bold best
    for col, direction in column_format_dict.items():
        styler.apply(bold_best, subset=[col], direction=direction)
    return styler.to_latex(column_format='c'*(perf_df.shape[1]+perf_df.index.nlevels))

def get_pred_perf_latex(pred_perf_dict):
    return get_latex_output_perf_dict(pred_perf_dict, perf_lowest_cols, perf_highest_cols)
    
def get_ue_perf_latex(ue_perf_dict):
    return get_latex_output_perf_dict(ue_perf_dict, ue_lowest_cols, ue_highest_cols)

def get_ood_perf_latex(ood_perf_df):
    all_lowest_cols, all_highest_cols = get_lowest_highest_col(
        columns=ood_perf_df.columns,
        lowest_cols=ood_lowest_cols, highest_cols=ood_highest_cols)
    return get_latex_output_perf_df(ood_perf_df, all_lowest_cols, all_highest_cols)