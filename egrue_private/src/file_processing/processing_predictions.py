from os.path import join, exists
import pandas as pd
import joblib
from tqdm.auto import tqdm

from src.file_manager.filepath import create_folder

def load_all_predictions(
    time_label, fp_cur_predictions_folder, split="test", 
    pred_file_names=["rue", "gpr", "infernoise", "bnn", "der"], additional_label=None):
    columns = []
    df_list = [] 
    for model in pred_file_names: # Update this when you have a new ue
        if additional_label is None:
            fp = join(fp_cur_predictions_folder, f"{model}_{split}_{time_label[-1]}.csv")
        else:
            fp = join(fp_cur_predictions_folder, f"{model}_{split}_{time_label[-1]}_{additional_label}.csv")
        df = pd.read_csv(fp, index_col=0)
        # find new columns not in current column list
        new_cols = [col for col in df.columns if col not in columns]
        columns.extend(new_cols)
        df_list.append(df[new_cols])
    return pd.concat(df_list, axis=1)

def add_new_ue_to_df_dict(df_dict, fp_cur_pi_prediction_folder, model, time_labels=["t+1", "t+2", "t+3"]):
    for time_label in tqdm(time_labels):
        for split in ["valid_df", "test_df"]:
            fp = join(fp_cur_pi_prediction_folder, f"{model}_{split[:-3]}_{time_label[-1]}.csv")
            df = pd.read_csv(fp, index_col=0)
            df = df.reset_index(drop=True)
            # find new columns not in current column list
            new_cols = [col for col in df.columns if col not in df_dict[time_label][split].columns]
            if len(new_cols)==0:
                print(f"No new columns in {split}")
                continue
            df_dict[time_label][split] = pd.concat([df_dict[time_label][split], df[new_cols]], axis=1)
    return df_dict

def load_prediction_df_dict(
    split_dict, fp_cur_predictions_folder, 
    pred_file_names=["rue", "gpr", "infernoise", "bnn", "der"], additional_label=None, only_test=False):
    df_dict = {}
    for time_label, target_cols in tqdm(split_dict["target_cols"].items()):
        df_dict[time_label] = {"pred_cols": target_cols}
        if not only_test:
            df_dict[time_label]["valid_df"] = load_all_predictions(
                time_label, fp_cur_predictions_folder, split="valid", 
                pred_file_names=pred_file_names, additional_label=additional_label)
        df_dict[time_label]["test_df"] = load_all_predictions(
            time_label, fp_cur_predictions_folder, split="test", 
            pred_file_names=pred_file_names, additional_label=additional_label)
    return df_dict

def load_prediction_df_dict_mimic(
    split_dict, fp_cur_predictions_folder, 
    pred_file_names=["rue", "gpr", "infernoise", "bnn", "der"], additional_label=None, only_test=False):
    df_dict = {}
    for time_label, time_info_dict in tqdm(split_dict.items()):
        df_dict[time_label] = {"pred_cols": time_info_dict["outputs"]}
        if not only_test:
            df_dict[time_label]["valid_df"] = load_all_predictions(
                time_label, fp_cur_predictions_folder, split="valid", 
                pred_file_names=pred_file_names, additional_label=additional_label)
        df_dict[time_label]["test_df"] = load_all_predictions(
            time_label, fp_cur_predictions_folder, split="test", 
            pred_file_names=pred_file_names, additional_label=additional_label)
    return df_dict


# Save all predictions
def save_pi_df_dict(df_dict, fp_cur_pi_predictions_folder):
    for time_label, time_info in tqdm(df_dict.items()):
        create_folder(fp_cur_pi_predictions_folder)
        val_df, test_df, pred_cols = time_info["valid_df"], time_info["test_df"], time_info["pred_cols"]
        val_df.to_csv(join(fp_cur_pi_predictions_folder, "val_"+time_label+".csv"))
        test_df.to_csv(join(fp_cur_pi_predictions_folder, "test_"+time_label+".csv"))
        joblib.dump(pred_cols, join(fp_cur_pi_predictions_folder, "pred_cols_"+time_label+".joblib"))
    print("Saved df_dict!")

# Load all predictions
def load_pi_df_dict(fp_cur_pi_predictions_folder, time_labels=["t+1", "t+2", "t+3"]):
    df_dict = {}
    for time_label in tqdm(time_labels):
        val_df = pd.read_csv(join(fp_cur_pi_predictions_folder, "val_"+time_label+".csv"), index_col=0)
        val_df = val_df.loc[:, ~val_df.columns.str.contains('^Unnamed')]
        test_df = pd.read_csv(join(fp_cur_pi_predictions_folder, "test_"+time_label+".csv"), index_col=0)
        test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]
        pred_cols = joblib.load(join(fp_cur_pi_predictions_folder, "pred_cols_"+time_label+".joblib"))
        df_dict[time_label] = {"valid_df": val_df, "test_df": test_df, "pred_cols":pred_cols}
    print("Loaded df_dict!")
    return df_dict
