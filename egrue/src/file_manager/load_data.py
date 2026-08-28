from os.path import join, exists
import json
import pandas as pd

def load_split_dict(fp_output_data_folder):
    fp_json = join(fp_output_data_folder, "data_info.json")
    with open(fp_json, 'r') as f:
        split_dict = json.load(f)
    splits = ["train_df", "valid_df", "test_df"]
    for split in splits:
        fp_df = join(fp_output_data_folder, split+".csv")
        split_dict[split] = pd.read_csv(fp_df, index_col=0)
    return split_dict