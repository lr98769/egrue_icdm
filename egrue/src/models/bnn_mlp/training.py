from src.misc import set_seed_pytorch
from src.data_processing.dataset import TabularDataset
from src.models.bnn.model import instantiate_bnn_model
from src.configs.default_configs import device


import numpy as np
import torch
import torch.optim as optim
import torchbnn as bnn
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models.bnn_mlp.predicting import predict_bnn_model


def train_bnn_model(
    bnn_model, train_df, valid_df, feat_cols, target_cols, epochs, patience, seed, batch_size, fp_model):
    set_seed_pytorch(seed)
    # Prepare dataset
    train_ds = TabularDataset(df=train_df, feat_cols=feat_cols, target_cols=target_cols)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Prepare dataset
    set_seed_pytorch(seed)
    valid_ds = TabularDataset(df=valid_df, feat_cols=feat_cols, target_cols=target_cols)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    mse_loss = MSELoss()
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    kl_weight = 0.1
    optimizer = optim.Adam(bnn_model.parameters(), lr=0.001)

    best_epoch, best_val_loss, patience_count = -1, np.inf, 0

    bnn_model.to(device)

    with tqdm(range(epochs), total=epochs) as pbar:
        for epoch in pbar:
            for x_batch, y_batch in train_dl:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = bnn_model(x_batch)
                mse = mse_loss(pred, y_batch)
                kl = kl_loss(bnn_model)
                cost = mse + kl_weight*kl

                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                x_batch = x_batch.detach().cpu()
                y_batch = y_batch.detach().cpu()
                mse = mse.detach().cpu()
                kl = kl.detach().cpu()
                cost = cost.detach().cpu()

            # Evaluate performance on validation set
            valid_pred = predict_bnn_model(bnn_model, valid_dl, feat_cols, target_cols, seed=seed, silent=True)
            valid_loss = evaluate_bnn_perf(valid_df, feat_cols, target_cols, valid_pred)
            pbar.set_description(f"valid_loss: {valid_loss:.5f}")

            # Early stopping
            if valid_loss < best_val_loss:
                best_epoch, best_val_loss = epoch, valid_loss
                patience_count = 0
                torch.save(bnn_model, fp_model)
            else:
                patience_count += 1
                if patience_count > patience:
                    print(f"Early stopping! Model achieved best performance at Epoch {best_epoch} with loss = {best_val_loss}.")
                    break

    return best_val_loss, best_epoch


def train_model_w_best_param(
    best_param, train_df, valid_df, feat_cols, target_cols, epochs, patience, seed, batch_size, fp_model):
    bnn_model =  instantiate_bnn_model(seed=seed, num_inputs=len(feat_cols), num_outputs=len(target_cols), **best_param)
    train_bnn_model(
        bnn_model, train_df, valid_df, feat_cols, target_cols, epochs, patience, 
        seed, batch_size, fp_model)
    return torch.load(fp_model)

def evaluate_bnn_perf(df, feat_cols, target_cols, pred):
    y = df[target_cols].values
    mse = torch.mean(torch.square(pred-y)).item()
    return mse