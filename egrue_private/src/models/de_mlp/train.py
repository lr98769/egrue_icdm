import torch
from tqdm.auto import tqdm
from os.path import join, exists

from src.configs.default_configs import device, fn_model
from src.misc import set_seed_pytorch
from src.training.train import transfer_encoder_n_classifier
from src.file_manager.filepath import FilePath

def get_ensemble_seeds(seed, seed_interval_size, ensemble_size):
    return [seed+i*seed_interval_size for i in range(ensemble_size)]

def train_ensemble_w_best_param_tabular(
    ModelClass, best_param, feature_cols, target_col, 
    split_dict, pytorch_split_dict_func,
    train_param_dict, train_model_func,
    seed, seed_interval_size, ensemble_size, data_name,
    batch_size, eval_batch_size,
    metric_to_monitor="auc", maximise=True,
    prev_model=None, class_weight=None, 
):
    all_ensemble_seeds = get_ensemble_seeds(seed, seed_interval_size, ensemble_size)
    with tqdm(all_ensemble_seeds) as pbar:
        for cur_seed in pbar:
            pbar.set_description(f"Seed: {cur_seed}")
            set_seed_pytorch(cur_seed)
            cur_fp = FilePath(data_name, seed=cur_seed)
            fp_model = join(cur_fp.get_parent_folder(fn_model), "classifier_tuned.pt")
            fp_history = join(cur_fp.get_parent_folder(fn_model), "classifier_tuned_history.jpg")
            if exists(fp_model):
                continue
            split_dict_pytorch = pytorch_split_dict_func(
                **split_dict, feat_cols=feature_cols, target_col=target_col, 
                batch_size=batch_size, eval_batch_size=eval_batch_size, 
            )
            model = ModelClass(
                **best_param, num_features=len(feature_cols)
            )
            if prev_model:
                model = transfer_encoder_n_classifier(model, prev_model) 
            history = train_model_func(
                model=model, **split_dict_pytorch, 
                fp_model=fp_model, **train_param_dict, fp_history=fp_history, 
                metric_to_monitor=metric_to_monitor, maximise=maximise, class_weight=class_weight
            )
