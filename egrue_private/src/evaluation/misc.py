import pandas as pd

def get_ue_loss_correct_col(cur_ue_dict, loss_col, correct_col=None):
    ue_col, model_label = cur_ue_dict["ue"], cur_ue_dict["model"]
    cur_loss_col = loss_col + "_" + model_label
    if correct_col is not None:
        cur_correct_col = correct_col + "_" + model_label
    else:
        cur_correct_col = None
    return ue_col, cur_loss_col, cur_correct_col

def combine_pred_df(pred_df_list):
    column_list = []
    df_list = []
    for pred_df in pred_df_list:
        new_cols = [col for col in pred_df.columns if col not in column_list]
        df_list.append(pred_df[new_cols].reset_index(drop=True))
        column_list.extend(new_cols)
    return pd.concat(df_list, axis=1)