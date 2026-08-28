import pandas as pd
from os.path import join, exists
from IPython.display import display

from src.file_manager.filepath import FilePath
from src.configs.default_configs import fn_tuning, fn_pred, fn_pred_perf, fn_ue_perf

def save_df(df, fp: FilePath, ModelClass, df_type, optional_label=None, limit=45000000):
    fp_df = fp.get_fp_df(ModelClass=ModelClass, df_type=df_type, optional_label=optional_label)
    if df.memory_usage(index=True, deep=True).sum()>limit:
        # print("Compressing...")
        fp_df += ".gz"
    df.to_csv(fp_df)
    return fp_df

def load_df(fp: FilePath, ModelClass, df_type, optional_label=None):
    fp_df = fp.get_fp_df(ModelClass=ModelClass, df_type=df_type, optional_label=optional_label)
    if exists(fp_df+".gz"):
        fp_df += ".gz"
    return pd.read_csv(fp_df, index_col=0)

# Specific df saving and loading functions
def save_tuning_df(tuning_df, fp, ModelClass, optional_label=None):
    save_df(tuning_df, fp, ModelClass, df_type=fn_tuning, optional_label=optional_label)
    
def load_tuning_df(fp, ModelClass, optional_label=None):
    return load_df(fp=fp, ModelClass=ModelClass, df_type=fn_tuning, optional_label=optional_label)

def save_pred_df(pred_df, fp, ModelClass, optional_label=None):
    save_df(pred_df, fp, ModelClass, df_type=fn_pred, optional_label=optional_label)

def load_pred_df(fp, ModelClass, optional_label=None):
    return load_df(fp, ModelClass, df_type=fn_pred, optional_label=optional_label)

def save_pred_perf_df(pred_perf_df, fp, ModelClass, optional_label=None):
    return save_df(pred_perf_df, fp, ModelClass, df_type=fn_pred_perf,  optional_label=optional_label)

def load_pred_perf_df(fp, ModelClass, optional_label=None):
    return load_df(fp, ModelClass, df_type=fn_pred_perf, optional_label=optional_label)

def save_expl_df(expl_df, fp, ModelClass, optional_label=None):
    save_df(expl_df, fp, ModelClass, df_type=fn_expl, optional_label=optional_label)

def load_expl_df(fp, ModelClass, optional_label=None):
    return load_df(fp, ModelClass, df_type=fn_expl, optional_label=optional_label)

# def save_grid_pred_df(pred_df, fp, ModelClass, optional_label=None):
#     return save_df(pred_df, fp, ModelClass, df_type=fn_grid_pred, optional_label=optional_label)

# def load_grid_pred_df(fp, ModelClass, optional_label=None):
#     return load_df(fp, ModelClass, df_type=fn_grid_pred, optional_label=optional_label)

def load_ue_perf_df(fp: FilePath, ue_label):
    fp_perf_df = join(fp.get_fp_ue_folder(), ue_label+".csv")
    return pd.read_csv(fp_perf_df, index_col=0)

def load_all_pred_dfs(fp, ModelClass_dict: dict, index="multi", pred_optional_label=None, remove_grid=False):
    cols = set()
    all_dfs = []
    def add_df(df, cols):
        if index == "multi":
            df = df.reset_index()
            df = df.set_index(["split", "index"])
        else:
            df = df.reset_index(drop=True)
        cur_cols = set(df.columns)
        added_cols = cur_cols.difference(cols)
        all_dfs.append(df[list(added_cols)])
        cols = cols.union(added_cols)
        return cols
    # ModelClass_dict: is a dictionary of list
    for ModelClass, optional_labels in ModelClass_dict.items():
        df = load_pred_df(fp=fp, ModelClass=ModelClass, optional_label=pred_optional_label)
        if remove_grid:
            df = df[df["split"]!="Grid"]
        # print("split" in df.columns)
        cols = add_df(df, cols)
        # if len(optional_labels) > 0:
        #     for optional_label in optional_labels:
        #         df = load_pred_df(fp=fp, ModelClass=ModelClass, optional_label=pred_optional_label)
        #         cols = add_df(df, cols)
    all_dfs = pd.concat(all_dfs, axis=1)
    # display(all_dfs.shape)
    if "split" not in all_dfs.columns:
        all_dfs=all_dfs.reset_index()
    return all_dfs