from src.configs.default_configs import device
from src.data_processing.dataloader import *
from IPython.display import display
from src.data_processing.dataloader import get_pytorch_split_dict_image

import torch
from tqdm.auto import tqdm
import pandas as pd

def predict(model, dl, verbose=True, active_dropout=False):
    y_true, y_logits, x_true, x_logits, weights = [], [], [], [], []
    if active_dropout:
        model.train()
    else:
        model.eval()
    model.to(device)
    num_batches = len(dl)
    with torch.no_grad():
        with tqdm(dl, leave=False, position=0, total=num_batches, disable=not verbose) as pbar:
            for batch in pbar:
                x_batch, y_batch = batch[0].to(device), batch[1].to(device)
                y_logit, x_logit = model(x_batch)

                x_batch = x_batch.cpu()
                y_logit = y_logit.cpu()
                y_batch = y_batch.cpu()
                x_logit = x_logit.cpu()

                y_true.append(y_batch)
                y_logits.append(y_logit) 
                x_true.append(x_batch)
                x_logits.append(x_logit)

                if len(batch)>2:
                    weights.append(batch[2].cpu())
                    
    y_true = torch.cat(y_true)
    y_logits = torch.cat(y_logits)
    x_true = torch.cat(x_true)
    x_logits = torch.cat(x_logits)

    if len(batch)>2:
        weights = torch.cat(weights)
        return y_true, y_logits, x_true, x_logits, weights
    else:
        return y_true, y_logits, x_true, x_logits


def prediction(model, dl, verbose=True, active_dropout=False, additional_param=None):
    all_outputs = [] # Shape: (num_samples, )
    model.eval()
    if active_dropout:
        model.train()
    model.to(device)
    num_batches = len(dl)
    first_batch = True
    with torch.no_grad():
        with tqdm(dl, leave=False, position=0, total=num_batches, disable=not verbose) as pbar:
            for inputs in pbar:
                x_batch = inputs[0].to(device)
                if additional_param:
                    outputs =  model(x_batch, **additional_param)
                else:
                    outputs =  model(x_batch)
                x_batch = x_batch.cpu()
                # Convert to list of inputs
                if (type(inputs) is not list) and (type(inputs) is not tuple):
                    inputs = [inputs]
                elif (type(inputs) is tuple):
                    inputs = list(inputs)
                # Convert to list of outputs
                if (type(outputs) is not list) and (type(outputs) is not tuple):
                    outputs = [outputs]
                elif (type(outputs) is tuple):
                    outputs = list(outputs)
                # Add the list of inputs and outputs
                cur_batch_output = inputs + outputs
                for i, element in enumerate(cur_batch_output):
                    element = element.cpu()
                    # first batch
                    if first_batch:
                        # start list of vectors for each element that we want to save
                        all_outputs.append([element]) 
                    else:
                        all_outputs[i].append(element)
                first_batch = False
                del x_batch, outputs, inputs, cur_batch_output
    # Concat all elements
    for i, element_list in enumerate(all_outputs):
        all_outputs[i] = torch.cat(element_list , dim=0)
    return all_outputs
    
def get_all_inputs(dl, verbose=True):
    all_inputs = []
    num_batches = len(dl)
    with torch.no_grad():
        with tqdm(dl, leave=False, position=0, total=num_batches, disable=not verbose) as pbar:
            for inputs in pbar:
                for i, input in enumerate(inputs):
                    if i>=len(all_inputs):
                        all_inputs.append([input])
                    else:
                        all_inputs[i].append(input)
    new_all_inputs = []
    for input_list in all_inputs:
        new_all_inputs.append(torch.cat(input_list, dim=0))
    return new_all_inputs

def prediction_recon(model, dl, verbose=True):
    all_outputs = [] # Shape: (T, num_samples, )
    model.eval()
    model.to(device)
    num_batches = len(dl)
    with torch.no_grad():
        with tqdm(dl, leave=False, position=0, total=num_batches, disable=not verbose) as pbar:
            for inputs in pbar:
                x_batch = inputs[0].to(device)
                output =  model.recon_predictions(x_batch) # Shape: (batch, recon...)
                all_outputs.append(output.cpu())
    return torch.concatenate(all_outputs, axis=0)  # Shape: (n_samples, recon...)

def prediction_recon_error(model, dl, verbose=True):
    all_outputs = [] # Shape: (T, num_samples, )
    model.eval()
    model.to(device)
    num_batches = len(dl)
    with torch.no_grad():
        with tqdm(dl, leave=False, position=0, total=num_batches, disable=not verbose) as pbar:
            for inputs in pbar:
                x_batch = inputs[0].to(device)
                output =  model.recon_error(x_batch) # Shape: (batch, recon...)
                all_outputs.append(output.cpu())
    return torch.concatenate(all_outputs, axis=0)  # Shape: (n_samples, recon...)

def get_imp_weights(attr, image_dim, squared=False):
    abs_attr = torch.abs(attr)
    if squared:
        abs_attr = torch.square(abs_attr)
    return abs_attr/abs_attr.sum(dim=image_dim, keepdims=True)

def prediction_egrue(model, dl, grad_shap, baselines, squared=True, verbose=True):
    all_outputs = [] # Shape: (T, num_samples, )
    model.eval()
    model.to(device)
    num_batches = len(dl)
    with torch.no_grad():
        with tqdm(dl, leave=False, position=0, total=num_batches, disable=not verbose) as pbar:
            for inputs in pbar:
                x_batch = inputs[0].to(device)
                image_dim = [i for i in range(1, len(x_batch.shape))]
                pred, recon =  model(x_batch) # Shape: (batch, num_classes), (batch, recon...)
                target = pred.argmax(axis=-1) # Shape: (batch, )
                attr = grad_shap.attribute(inputs=x_batch, baselines=baselines, target=target) # (batch, recon...)
                imp_weights = get_imp_weights(attr, image_dim, squared)
                recon_errors = torch.abs(recon-x_batch) # (batch, recon...)
                egrue = (recon_errors*imp_weights).sum(axis=image_dim)
                all_outputs.append(egrue.cpu())
    return torch.concatenate(all_outputs, axis=0)  # Shape: (n_samples, recon...)

def prediction_resnet(model, dl, verbose=True):
    all_outputs = [] # Shape: (T, num_samples, )
    model.train()
    # Freeze fc_mean of encoder
    for param in model.encoder.fc_mean.parameters():
        param.requires_grad = False
    # Freeze classifier
    for param in model.predictor.parameters():
        param.requires_grad = False
    model.encoder.eval()
    model.to(device)
    num_batches = len(dl)
    with torch.no_grad():
        with tqdm(dl, leave=False, position=0, total=num_batches, disable=not verbose) as pbar:
            for inputs in pbar:
                x_batch = inputs[0].to(device)
                output =  model.resnet_predictions(x_batch) # Shape: (batch, pred)
                all_outputs.append(output)
    return torch.concatenate(all_outputs, axis=0)  # Shape: (n_samples, pred)
    
# def prediction_resnet(model, dl, verbose=True):
#     all_outputs = [] # Shape: (T, num_samples, )
#     model.eval()
#     model.to(device)
#     num_batches = len(dl)
#     with torch.no_grad():
#         with tqdm(dl, leave=False, position=0, total=num_batches, disable=not verbose) as pbar:
#             for inputs in pbar:
#                 x_batch = inputs[0].to(device)
#                 output =  model.resnet_predictions(x_batch) # Shape: (batch, pred)
#                 all_outputs.append(output)
#     return torch.concatenate(all_outputs, axis=0)  # Shape: (n_samples, pred)

def prediction_sampling_dropout(model, dl, T, verbose=True):
    all_outputs = [] # Shape: (T, num_samples, )
    model.eval()
    model.to(device)
    num_batches = len(dl)
    with torch.no_grad():
        with tqdm(dl, leave=False, position=0, total=num_batches, disable=not verbose) as pbar:
            for inputs in pbar:
                x_batch = inputs[0].to(device)
                output =  model.sample_predictions_dropout(x_batch, T) # Shape: (T, batch, )
                all_outputs.append(output)
    return torch.concatenate(all_outputs, axis=1) # join on batch

def prediction_sampling_noise(model, dl, T, verbose=True):
    all_outputs = [] # Shape: (T, num_samples, )
    model.eval()
    model.to(device)
    num_batches = len(dl)
    with torch.no_grad():
        with tqdm(dl, leave=False, position=0, total=num_batches, disable=not verbose) as pbar:
            for inputs in pbar:
                x_batch = inputs[0].to(device)
                output =  model.sample_predictions_noise(x_batch, T) # Shape: (T, batch, )
                all_outputs.append(output)
    return torch.concatenate(all_outputs, axis=1) # join on batch

def get_all_predictions(
    model, data_dict, pred_func, batch_size, eval_batch_size, seed, 
    pytorch_split_dict_func=get_pytorch_split_dict_image, additional_pred_args={}
    ):
    all_pred_dfs = []
    dls = pytorch_split_dict_func(
        data_dict=data_dict, batch_size=eval_batch_size, eval_batch_size=eval_batch_size, 
        shuffle_train=False)
    new_dls = {}
    name_map = {"train_dl":"Train", "val_dl": "Valid", "test_dl": "Test", "grid_dl": "Grid"}
    for label, dl in dls.items():
        if label in name_map:
            new_dls[name_map[label]] = dl
    dls = new_dls
    if "feat_cols" in data_dict:
        feat_cols = data_dict["feat_cols"]
    else:
        feat_cols = None
    target_col = data_dict["target_col"]
    num_classes = data_dict["num_classes"] if "num_classes" in data_dict else data_dict["num_outputs"]
    with tqdm(dls.items(), total=len(dls)) as pbar:
        for split_name, dl in pbar:
            pbar.set_description(split_name)
            pred_df = pred_func(
                model, dl, feat_cols, target_col, num_classes, seed=seed, **additional_pred_args)
            pred_df["split"] = split_name
            all_pred_dfs.append(pred_df)
    return pd.concat(all_pred_dfs)

def split_test_set(pred_df, split_col, new_split_col, num_ori_test, labels):
    pred_df = pred_df.copy()
    pred_df[new_split_col] = pred_df[split_col]
    pred_df[new_split_col][pred_df[new_split_col]=="Test"] = (
        [labels[0] for i in range(num_ori_test)] + \
        [labels[1] for i in range((pred_df["split"]=="Test").sum()-num_ori_test)])
    return pred_df



