from os.path import join
import torch

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
split_col = "split"
split_names = ["Train", "Valid", "Test"]

fp_project_folder = "../../../"
fp_code_folder = join(fp_project_folder, "code")
fp_data_folder = join(fp_project_folder, "Data")

fp_checkpoint_folder = join(fp_code_folder, "checkpoints")

# Filenames
fn_model="models"
fn_ue_perf="evaluation"
fn_pred_perf = "evaluation"
fn_pred = "predictions"
fn_grid_pred = "grid_pred"
fn_tuning = "hyperparameter_tuning"
fn_history = "history"
fn_consolidated  = "consolidated"
fn_expl = "explanations"