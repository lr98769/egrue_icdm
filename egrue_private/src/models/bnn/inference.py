from src.configs.default_configs import device
from src.misc import set_seed_pytorch


import torch
from tqdm.auto import tqdm



def predict_bnn_model(bnn_model, dl, seed, silent):
    set_seed_pytorch(seed)
    # Send model to gpu
    bnn_model.to(device)
    # Predictions
    all_pred = []
    all_target = []
    # Predict with model
    with tqdm(dl, total=len(dl), disable=silent) as pbar:
        for x_batch, y_batch in pbar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = bnn_model(x_batch)
            all_pred.append(pred.detach().cpu())
            all_target.append(y_batch.detach().cpu())
    return torch.cat(all_target), torch.cat(all_pred)


