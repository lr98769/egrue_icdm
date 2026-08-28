# Load and save
import torch
from src.file_manager.filepath import FilePath
from src.configs.default_configs import device

def load_model(fp: FilePath, ModelClass, cur_model_name):
    fp_model = fp.get_fp_model(ModelClass, cur_model_name)
    return torch.load(fp_model, map_location=device, weights_only=False)
