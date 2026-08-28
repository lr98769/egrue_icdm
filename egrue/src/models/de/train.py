import torch
from os.path import join, exists
from tqdm.auto import tqdm

from src.misc import set_seed_pytorch
from src.data_processing.dataloader import get_pytorch_split_dict_image
from src.file_manager.filepath import FilePath

def get_ensemble_seeds(seed, seed_interval_size, ensemble_size):
    return [seed+i*seed_interval_size for i in range(ensemble_size)]

def train_ensemble_w_best_param(
    ModelClass, best_param, cur_model_name,
    data_dict, batch_size, eval_batch_size, 
    train_param_dict, train_model_func, seed, data_name,
    metric_to_monitor="auc", maximise=True, 
    pytorch_split_dict_func=get_pytorch_split_dict_image, num_workers=4,
    seed_interval_size=100, ensemble_size=5
):
    if "num_classes" not in data_dict:
        num_classes = data_dict["num_outputs"]
    else:
        num_classes = data_dict["num_classes"]

    all_ensemble_seeds = get_ensemble_seeds(seed, seed_interval_size, ensemble_size)
    for cur_seed in tqdm(all_ensemble_seeds):
        fp = FilePath(data_name=data_name, seed=cur_seed)
        fp_model = fp.get_fp_model(ModelClass, cur_model_name)
        fp_history = fp.get_fp_history(ModelClass, cur_model_name)
        if exists(fp_model):
            continue
        
        set_seed_pytorch(cur_seed)
        split_dict_pytorch = pytorch_split_dict_func(
            data_dict=data_dict, 
            batch_size=batch_size, eval_batch_size=eval_batch_size, num_workers=num_workers
        )

        if "feat_cols" in data_dict:
            feat_cols = data_dict["feat_cols"]
            model = ModelClass(
                **best_param, 
                num_features=len(feat_cols),
                num_classes=num_classes
            )
        else:
            model = ModelClass(
                **best_param, 
                num_classes=num_classes
            )

        history = train_model_func(
            model=model, 
            train_dl=split_dict_pytorch["train_dl"], 
            val_dl=split_dict_pytorch["val_dl"],
            test_dl=split_dict_pytorch["test_dl"],
            fp_model=fp_model, **train_param_dict, fp_history=fp_history, 
            metric_to_monitor=metric_to_monitor, maximise=maximise,
            seed=seed, 
        )