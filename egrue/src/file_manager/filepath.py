import os
from os.path import join
from src.configs.default_configs import fp_checkpoint_folder, fp_data_folder 
from src.configs.default_configs import fn_model, fn_ue_perf, fn_history

def create_folder(fp):
    if not os.path.exists(fp):
        os.makedirs(fp)

class FilePath:
    def __init__(self, data_name, seed):
        self.data_name = data_name
        self.seed = str(seed)

    # Filepath functions
    def get_preprocessed_folder(self, modified_label=None):
        if modified_label:
            fp = join(fp_data_folder, self.data_name+"_"+modified_label)
        else:
            fp = join(fp_data_folder, self.data_name)
        create_folder(fp)
        return fp
    
    def get_parent_folder(self, folder_name):
        fp = join(fp_checkpoint_folder, folder_name, self.data_name, self.seed)
        create_folder(fp)
        return fp

    def get_fp_model(self, ModelClass, cur_model_name):
        fp_model_folder = self.get_parent_folder(folder_name=fn_model)
        fp_model = join(fp_model_folder, f"{ModelClass.name}_{cur_model_name}.pt")
        return fp_model

    def get_fp_df(self, ModelClass, df_type, optional_label=None):
        fp_folder = self.get_parent_folder(folder_name=df_type)
        fn = f"{ModelClass.name}_{optional_label}.csv" if optional_label else f"{ModelClass.name}.csv"
        fp = join(fp_folder, fn)
        return fp

    def get_fp_ue_folder(self):
        return self.get_parent_folder(folder_name=fn_ue_perf)
    
    def get_fp_history(self, ModelClass, cur_model_name):
        fp_history_folder = self.get_parent_folder(folder_name=fn_history)
        fp_history = join(fp_history_folder, f"{ModelClass.name}_{cur_model_name}.jpg")
        return fp_history