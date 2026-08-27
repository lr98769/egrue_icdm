from src.configs.default_configs import device
from src.misc import set_seed_pytorch
from src.data_processing.dataloader import get_pytorch_split_dict_image


import numpy as np
import torch
import torch.optim as optim
import torchbnn as bnn
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models.bnn.model import instantiate_bnn_model
from src.models.bnn.inference import predict_bnn_model


def evaluate_bnn_perf(target, pred):
    ce_loss = nn.CrossEntropyLoss()
    ce = ce_loss(pred, target).item()
    acc = (pred.argmax(-1) == target).float().mean().item()
    return ce, acc


def train_bnn_model(
        bnn_model, train_dl, valid_dl, epochs, patience, seed, fp_model, class_weight, kl_weight=0.1, lr=0.001):
    set_seed_pytorch(seed)

    ce_loss = nn.CrossEntropyLoss(class_weight).to(device)
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False).to(device)
    optimizer = optim.Adam(bnn_model.parameters(), lr=lr)

    best_epoch, best_val_loss, patience_count = -1, np.inf, 0

    bnn_model = bnn_model.to(device)
    bnn_model.device = device
    for i in bnn_model.named_parameters():
        if i[1].device != device:
            print(f"{i[0]} -> {i[1].device}")

    with tqdm(range(epochs), total=epochs) as pbar:
        for epoch in pbar:
            for x_batch, y_batch in tqdm(train_dl, total=len(train_dl)):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = bnn_model(x_batch)
                ce = ce_loss(pred, y_batch)
                kl = kl_loss(bnn_model)
                cost = ce + kl_weight*kl

                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                x_batch = x_batch.detach().cpu()
                y_batch = y_batch.detach().cpu()
                ce = ce.detach().cpu()
                kl = kl.detach().cpu()
                cost = cost.detach().cpu()
                del x_batch, y_batch, ce, kl, cost

            # Evaluate performance on validation set
            valid_target, valid_pred = predict_bnn_model(bnn_model, valid_dl, seed=seed, silent=True)
            valid_ce, valid_acc = evaluate_bnn_perf(valid_target, valid_pred)
            pbar.set_description(f"valid_ce: {valid_ce:.3f}, valid_acc: {valid_acc:.3f}")

            # Early stopping
            if valid_ce < best_val_loss:
                best_epoch, best_val_loss = epoch, valid_ce
                patience_count = 0
                torch.save(bnn_model, fp_model)
            else:
                patience_count += 1
                if patience_count > patience:
                    print(f"Early stopping! Model achieved best performance at Epoch {best_epoch} with loss = {best_val_loss}.")
                    break

    return best_val_loss, best_epoch


def train_bnn_w_best_param(
    best_param, data_dict, epochs, patience, seed, fp, class_weight,
    batch_size, eval_batch_size, kl_weight=0.1, lr=0.001
):
    set_seed_pytorch(seed)
    split_dict_pytorch = get_pytorch_split_dict_image(
        data_dict=data_dict, batch_size=batch_size, eval_batch_size=eval_batch_size
    )
    train_dl, valid_dl = split_dict_pytorch["train_dl"], split_dict_pytorch["val_dl"]
    num_outputs = data_dict["num_classes"]
    bnn_model = instantiate_bnn_model(**best_param, seed=seed, num_outputs=num_outputs)
    fp_model = fp.get_fp_model(bnn_model, cur_model_name="tuned")
    train_bnn_model(
        bnn_model, train_dl, valid_dl, epochs, patience, seed, fp_model, class_weight, kl_weight=kl_weight, lr=lr)
    return torch.load(fp_model, weights_only=False)