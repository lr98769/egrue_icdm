import numpy as np
from src.training.callbacks import *
from src.training.display_progress import *
from src.evaluation.inference import prediction_egrue
import pandas as pd
import warnings
from captum.attr import GradientShap
from torch.utils.data import DataLoader
from src.misc import set_seed_pytorch
from torch.nn import Sequential
from src.configs.default_configs import device
from src.evaluation.timer import Timer

def get_egrue_predictions_resnet(
    model, dl, feat_cols, target_col, num_classes, seed, train_ds, num_baselines=100, squared=False, label="egrue"):
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    set_seed_pytorch(seed)

    # Get baseline instances
    dataloader = DataLoader(train_ds, batch_size=num_baselines, shuffle=True)
    batch = next(iter(dataloader))
    baselines, _ = batch # Get a single batch of 100
    baselines = baselines.to(device)

    # Get grad shap explainer
    pred_model = Sequential(model.encoder, model.classifier)
    grad_shap = GradientShap(pred_model)
    
    # Columns
    egrue_col = label
    
    # Predictions
    timer = Timer("egRUE")
    egrue = prediction_egrue(model, dl, grad_shap, baselines, squared=squared).numpy()
    timer.end()
    
    # Ouput pred_df
    pred_df = pd.DataFrame(np.expand_dims(egrue, -1), columns=[egrue_col])

    baselines = baselines.cpu()
    
    return pred_df
