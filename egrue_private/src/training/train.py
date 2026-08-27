import torch
from src.misc import set_seed_pytorch
from src.data_processing.dataloader import get_pytorch_split_dict_image
from src.file_manager.filepath import FilePath

def train_model_w_best_param(
    ModelClass, best_param, cur_model_name,
    data_dict, batch_size, eval_batch_size, 
    train_param_dict, train_model_func, seed, fp: FilePath, 
    metric_to_monitor="auc", maximise=True, 
    pytorch_split_dict_func=get_pytorch_split_dict_image, data_dict_ood=None, num_workers=4
):
    fp_model = fp.get_fp_model(ModelClass, cur_model_name)
    fp_history = fp.get_fp_history(ModelClass, cur_model_name)
    
   
    if "num_classes" not in data_dict:
        num_classes = data_dict["num_outputs"]
    else:
        num_classes = data_dict["num_classes"]

    set_seed_pytorch(seed)
    split_dict_pytorch = pytorch_split_dict_func(
        data_dict=data_dict, batch_size=batch_size, eval_batch_size=eval_batch_size
    )
    
    if data_dict_ood:
        split_dict_pytorch_ood = pytorch_split_dict_func(
            data_dict=data_dict_ood, batch_size=batch_size, eval_batch_size=eval_batch_size,
            num_workers=num_workers
        )
        train_param_dict["ood_dl"] = split_dict_pytorch_ood["test_dl"]
    
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
    model = torch.load(fp_model, only_weights=False)
    return model

def train_model_w_best_param_tabular(
    ModelClass, best_param, feature_cols, target_col, 
    split_dict, pytorch_split_dict_func,
    train_param_dict, train_model_func,
    seed,
    batch_size, eval_batch_size,
    fp_model, fp_history=None, 
    metric_to_monitor="auc", maximise=True,
    prev_model=None, class_weight=None, 
):
    set_seed_pytorch(seed)
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
    model = torch.load(fp_model, weights_only=False)
    return model


def transfer_encoder_n_classifier(model, prev_model):
    model.encoder = prev_model.encoder
    model.num_encoder_layers = prev_model.num_encoder_layers
    model.encoder_width = prev_model.encoder_width
    model.classifier = prev_model.classifier
    return model