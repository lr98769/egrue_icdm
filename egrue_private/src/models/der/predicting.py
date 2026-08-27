

import numpy as np
import torch
from lightning.pytorch import seed_everything
from src.configs.default_configs import device


def mae_fn(y_true, y_pred):
    return np.mean(np.abs(y_true-y_pred), axis=-1)


def der_model_prediction(der_model, test_df, feat_cols, target_cols, seed, silent, regressor_label):
    seed_everything(seed, workers=True, verbose=False)
    test_df = test_df.copy()

    with torch.no_grad():
        x = torch.from_numpy(test_df[feat_cols].values).float().to(device)
        dists = der_model(x)
        means = dists.loc.cpu()
        std = torch.sqrt(dists.variance_loc).cpu()

    test_y = torch.from_numpy(test_df[target_cols].values).float().cpu()

    test_df["der_uncertainty"] = std.mean(axis=-1) # N
    test_df["der_mae"] = mae_fn(y_true=test_y.numpy(), y_pred=means.numpy())
    predicted_colnames = [col + "_der_"+regressor_label for col in target_cols]
    test_df[predicted_colnames] = means
    return test_df