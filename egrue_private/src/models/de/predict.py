from tqdm.auto import tqdm
import pandas as pd
from os.path import exists
import numpy as np


from src.configs.default_configs import fn_pred
from src.file_manager.load_save_df import load_pred_df, save_pred_df
from src.data_processing.dataloader import *
from src.data_processing.dataloader import get_pytorch_split_dict_image
from src.models.de.train import get_ensemble_seeds
from src.file_manager.load_save_model import load_model
from src.file_manager.filepath import FilePath
from src.evaluation.perf_metrics import get_incorrect_class_prob, get_elementwise_crossentropyloss_from_pred_prob
from src.evaluation.timer import Timer

def get_all_ensemble_predictions(
    data_dict, pred_func, 
    batch_size, eval_batch_size, 
    data_name, seed, 
    ModelClass, cur_model_name,
    model_label,
    seed_interval_size=100, ensemble_size=5,
    additional_pred_args={},
    pytorch_split_dict_func=get_pytorch_split_dict_image, 
    optional_label=None, override=False
):
    # Get DLs
    dls = pytorch_split_dict_func(
        data_dict=data_dict, batch_size=eval_batch_size, eval_batch_size=eval_batch_size, 
        shuffle_train=False
    )
    new_dls = {}
    name_map = {"train_dl":"Train", "val_dl": "Valid", "test_dl": "Test", "grid_dl": "Grid"}
    for label, dl in dls.items():
        if label in name_map:
            new_dls[name_map[label]] = dl
    dls = new_dls
    # Get data info
    if "feat_cols" in data_dict:
        feat_cols = data_dict["feat_cols"]
    else:
        feat_cols = None
    target_col = data_dict["target_col"]
    num_classes = data_dict["num_classes"] if "num_classes" in data_dict else data_dict["num_outputs"]

    all_ensemble_seeds = get_ensemble_seeds(seed, seed_interval_size, ensemble_size)

    ensemble_pred_dfs = []
    with tqdm(all_ensemble_seeds) as ensemble_pbar:
        for cur_seed in ensemble_pbar:
            ensemble_pbar.set_description(f"seed: {cur_seed}")
            fp = FilePath(data_name=data_name, seed=cur_seed)
            fp_pred_df = fp.get_fp_df(ModelClass=ModelClass, df_type=fn_pred, optional_label=optional_label)
            # If prediction exists, load it 
            if not override and exists(fp_pred_df):
                all_pred_dfs = load_pred_df(fp, ModelClass, optional_label=optional_label)
            # Else predict
            else:
                model = load_model(fp=fp, ModelClass=ModelClass, cur_model_name=cur_model_name)
                all_pred_dfs = []
                with tqdm(dls.items(), total=len(dls)) as pbar:      
                    for split_name, dl in pbar:
                        pbar.set_description(split_name)  
                        pred_df = pred_func(
                            model, dl, feat_cols, target_col, num_classes, seed=seed, **additional_pred_args)
                        pred_df["split"] = split_name
                        all_pred_dfs.append(pred_df)
                all_pred_dfs = pd.concat(all_pred_dfs)
                save_pred_df(all_pred_dfs, fp, ModelClass, optional_label)
            ensemble_pred_dfs.append(all_pred_dfs)

    ensemble_pred_df = process_ensemble_pred_dfs(ensemble_pred_dfs, ModelClass, target_col, num_classes, model_label=model_label)
    return ensemble_pred_df

def get_all_ensemble_predictions_time(
    data_dict, pred_func, 
    batch_size, eval_batch_size, 
    data_name, seed, 
    ModelClass, cur_model_name,
    model_label,
    seed_interval_size=100, ensemble_size=5,
    additional_pred_args={},
    pytorch_split_dict_func=get_pytorch_split_dict_image, 
    optional_label=None, override=False
):
    # Get DLs
    dls = pytorch_split_dict_func(
        data_dict=data_dict, batch_size=eval_batch_size, eval_batch_size=eval_batch_size, 
        shuffle_train=False
    )
    new_dls = {}
    name_map = {"train_dl":"Train", "val_dl": "Valid", "test_dl": "Test", "grid_dl": "Grid"}
    for label, dl in dls.items():
        if label in name_map:
            new_dls[name_map[label]] = dl
    dls = new_dls
    # Get data info
    if "feat_cols" in data_dict:
        feat_cols = data_dict["feat_cols"]
    else:
        feat_cols = None
    target_col = data_dict["target_col"]
    num_classes = data_dict["num_classes"] if "num_classes" in data_dict else data_dict["num_outputs"]

    all_ensemble_seeds = get_ensemble_seeds(seed, seed_interval_size, ensemble_size)

    ensemble_pred_dfs = []
    timer = Timer("Deep Ensemble")
    with tqdm(all_ensemble_seeds) as ensemble_pbar:
        for cur_seed in ensemble_pbar:
            ensemble_pbar.set_description(f"seed: {cur_seed}")
            fp = FilePath(data_name=data_name, seed=cur_seed)
            model = load_model(fp=fp, ModelClass=ModelClass, cur_model_name=cur_model_name)
            all_pred_dfs = []
            with tqdm(dls.items(), total=len(dls)) as pbar:      
                for split_name, dl in pbar:
                    pbar.set_description(split_name)  
                    pred_df = pred_func(
                        model, dl, feat_cols, target_col, num_classes, seed=seed, **additional_pred_args)
                    pred_df["split"] = split_name
                    all_pred_dfs.append(pred_df)
            all_pred_dfs = pd.concat(all_pred_dfs)
            ensemble_pred_dfs.append(all_pred_dfs)
    ensemble_pred_df = process_ensemble_pred_dfs(
        ensemble_pred_dfs, ModelClass, target_col, num_classes, model_label=model_label, timer=timer)
    return ensemble_pred_df

def process_ensemble_pred_dfs(
        ensemble_pred_dfs, ModelClass, target_col, num_classes, 
        label="de", model_label="resnet", timer=None):
    # Name Columns to Compute
    pred_prob_cols = [f"{target_col}_class_prob_{i}_{label}" for i in range(num_classes)]
    loss_col = f"crossentropy_loss_{label}"
    pred_label_col = f"{target_col}_pred_label_{label}"
    incorrect_class_prob_col = f"incorrect_class_prob_{label}"
    correct_col = f"correct_{label}"
    ue_col = f"pred_std_{label}"

    # Name Columns to Derive Them From
    model_pred_prob_cols = [f"{target_col}_class_prob_{i}_{model_label}" for i in range(num_classes)]

    # Compute Values
    pred_probs = []
    for pred_df in ensemble_pred_dfs:
        pred_probs.append(pred_df[model_pred_prob_cols].values)
    pred_probs = np.array(pred_probs) # (num_models, num_samples, num_outputs)
    mean_pred_prob = pred_probs.mean(axis=0 ) # (num_samples, num_outputs)
    std_pred_prob = pred_probs.std(axis=0).mean(axis=-1) # (num_samples, num_outputs)
    if timer is not None:
        timer.end()
    pred_labels = mean_pred_prob.argmax(axis=-1)
    
    # Make output pred_df
    y_true = pred_df[target_col].values
    split = pred_df["split"].values
    pred_df = pd.DataFrame(np.expand_dims(y_true, axis=-1), columns=[target_col])
    pred_df["split"] = split
    pred_df[pred_prob_cols] = mean_pred_prob
    pred_df[pred_label_col] = pred_labels
    pred_df[ue_col] = std_pred_prob
    if y_true.max() >= num_classes:
        pred_df[loss_col] = np.nan
        pred_df[incorrect_class_prob_col] = np.nan
        pred_df[correct_col] = np.nan
    else:
        ce = get_elementwise_crossentropyloss_from_pred_prob(y_true, mean_pred_prob)
        incorrect_class_prob = get_incorrect_class_prob(y_true, mean_pred_prob)
        correct = pred_labels == y_true
        pred_df[loss_col] = ce
        pred_df[incorrect_class_prob_col] = incorrect_class_prob
        pred_df[correct_col] = correct

    pred_df.index = ensemble_pred_dfs[0].index
    
    return pred_df

    






