import numpy as np
from src.training.callbacks import *
from src.training.display_progress import *
from src.evaluation.inference import prediction, prediction_recon_error, get_all_inputs
import pandas as pd
import warnings

def get_rue_predictions_resnet(
    model, dl, feat_cols, target_col, num_classes, seed, label="rue"):
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    
    # Columns
    reconstruction_error_col = label
    
    # Predictions
    rue = prediction_recon_error(model, dl).numpy()

    # Ouput pred_df
    pred_df = pd.DataFrame(np.expand_dims(rue, -1), columns=[reconstruction_error_col])
    
    return pred_df
