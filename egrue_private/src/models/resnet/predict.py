from tqdm.auto import tqdm
import torch
import warnings
import pandas as pd

from src.misc import set_seed_pytorch
from src.models.resnet.inference import predict_w_resnet, predict_w_resnet_dropout_speedup
from src.training.callbacks import *
from src.evaluation.timer import Timer

# Timing purposes
def get_resnet_predictions(model, dl, feat_cols, target_col, num_classes, seed, T=10, mc=True, label="resnet", ):
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    
    # Columns
    logit_cols = [f"{target_col}_class_logit_{i}_{label}" for i in range(num_classes)]
    pred_prob_cols = [f"{target_col}_class_prob_{i}_{label}" for i in range(num_classes)]
    loss_col = f"crossentropy_loss_{label}"
    pred_label_col = f"{target_col}_pred_label_{label}"
    incorrect_class_prob_col = f"incorrect_class_prob_{label}"
    entropy_col = f"entropy_{label}"
    correct_col = f"correct_{label}"
    
    # Predictions
    timer = Timer("Entropy")
    pred_list = predict_w_resnet(model, dl)
    y_true, y_logits = pred_list[0].type(torch.LongTensor), pred_list[1]
    y_pred_prob = torch.softmax(y_logits, dim=-1)
    y_true = y_true.type(torch.LongTensor)
    entropy = get_entropy_from_pred_prob(y_pred_prob.numpy())
    timer.end()

    # Ouput pred_df
    pred_df = pd.DataFrame(y_true.unsqueeze(-1), columns=[target_col])
    pred_df[logit_cols] = y_logits
    pred_df[pred_prob_cols] = y_pred_prob    
    pred_df[pred_label_col] = y_pred_prob.argmax(dim=-1)
    pred_df[loss_col] = cross_entropy(y_logits, y_true, reduction="none").numpy()
    pred_df[incorrect_class_prob_col] = get_incorrect_class_prob(y_true.numpy(), y_pred_prob.numpy())
    pred_df[entropy_col] = entropy
    pred_df[correct_col] = y_true.numpy().flatten() == y_pred_prob.argmax(dim=-1).numpy().flatten()

    # MC Dropout Predictions
    if mc:
        mc_logit_cols = [colname+"_mc" for colname in logit_cols]
        mc_pred_prob_cols = [colname+"_mc" for colname in pred_prob_cols]
        mc_loss_col = loss_col+"_mc"
        mc_pred_label_col = pred_label_col+"_mc"
        mc_incorrect_class_prob_col = incorrect_class_prob_col+"_mc"
        mc_ue_col = "pred_std_mc"
        mc_correct_col = correct_col+"_mc"
        
        set_seed_pytorch(seed)
        all_y_logits = [] 
        timer = Timer("MC Dropout")
        for i in tqdm(range(T), total=T):
            pred_list = predict_w_resnet(model, dl, active_dropout=True)
            y_true, y_logits = pred_list[0].type(torch.LongTensor), pred_list[1]
            all_y_logits.append(y_logits) 
        all_y_logits = torch.stack(all_y_logits) # (T, num_samples, num_output)
        all_y_logits = all_y_logits.permute((1, 2, 0)) # (num_samples, num_output, T)
        all_y_pred_probs = torch.softmax(all_y_logits, dim=1) # (num_samples, num_output, T)
        mc_ue = all_y_pred_probs.std(axis=-1).mean(axis=-1) # (num_samples)
        timer.end()
        mc_logits = all_y_logits.mean(axis=-1)  # (num_samples, num_output)
        mc_pred_prob = all_y_pred_probs.mean(axis=-1) # (num_samples, num_output)
        
        # MC Dropout
        pred_df[mc_logit_cols] = mc_logits
        pred_df[mc_pred_prob_cols] = mc_pred_prob    
        pred_df[mc_pred_label_col] = mc_pred_prob.argmax(dim=-1)
        pred_df[mc_loss_col] = cross_entropy(mc_logits, y_true, reduction="none").numpy()
        pred_df[mc_incorrect_class_prob_col] = get_incorrect_class_prob(y_true.numpy(), mc_pred_prob)
        pred_df[mc_ue_col] = mc_ue
        pred_df[mc_correct_col] = mc_pred_prob.argmax(dim=-1).numpy().flatten() == y_true.numpy().flatten()
    
    return pred_df


# Assumes that only the last layer is dropout
def get_resnet_predictions_speedup(
    model, dl, feat_cols, target_col, num_classes, seed, T=10, mc=True, label="resnet", ood=False):
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    
    # Columns
    logit_cols = [f"{target_col}_class_logit_{i}_{label}" for i in range(num_classes)]
    pred_prob_cols = [f"{target_col}_class_prob_{i}_{label}" for i in range(num_classes)]
    loss_col = f"crossentropy_loss_{label}"
    pred_label_col = f"{target_col}_pred_label_{label}"
    incorrect_class_prob_col = f"incorrect_class_prob_{label}"
    entropy_col = f"entropy_{label}"
    correct_col = f"correct_{label}"
    
    # Predictions
    pred_list = predict_w_resnet(model, dl)
    y_true, y_logits = pred_list[0].type(torch.LongTensor), pred_list[1]
    y_pred_prob = torch.softmax(y_logits, dim=-1)
    y_true = y_true.type(torch.LongTensor)

    # Ouput pred_df
    pred_df = pd.DataFrame(y_true.unsqueeze(-1), columns=[target_col])
    pred_df[logit_cols] = y_logits
    pred_df[pred_prob_cols] = y_pred_prob    
    pred_df[pred_label_col] = y_pred_prob.argmax(dim=-1)
    if (y_true.max() >= num_classes) or ood:
        pred_df[loss_col] = np.nan
        pred_df[incorrect_class_prob_col] = np.nan
        pred_df[correct_col] = np.nan
    else:
        pred_df[loss_col] = cross_entropy(y_logits, y_true, reduction="none").numpy()
        pred_df[incorrect_class_prob_col] = get_incorrect_class_prob(y_true.numpy(), y_pred_prob.numpy())
        pred_df[correct_col] = y_true.numpy().flatten() == y_pred_prob.argmax(dim=-1).numpy().flatten()
    pred_df[entropy_col] = get_entropy_from_pred_prob(y_pred_prob.numpy())
    # MC Dropout Predictions
    if mc:
        mc_logit_cols = [colname+"_mc" for colname in logit_cols]
        mc_pred_prob_cols = [colname+"_mc" for colname in pred_prob_cols]
        mc_loss_col = loss_col+"_mc"
        mc_pred_label_col = pred_label_col+"_mc"
        mc_incorrect_class_prob_col = incorrect_class_prob_col+"_mc"
        mc_ue_col = "pred_std_mc"
        mc_correct_col = correct_col+"_mc"
        
        set_seed_pytorch(seed)
        all_y_logits = predict_w_resnet_dropout_speedup(model, dl, T).cpu() # (T, num_samples, num_output)
        all_y_logits = all_y_logits.permute((1, 2, 0)) # (num_samples, num_output, T)
        all_y_pred_probs = torch.softmax(all_y_logits, dim=1) # (num_samples, num_output, T)
        mc_logits = all_y_logits.mean(axis=-1)  # (num_samples, num_output)
        mc_pred_prob = all_y_pred_probs.mean(axis=-1) # (num_samples, num_output)
        mc_ue = all_y_pred_probs.std(axis=-1).mean(axis=-1) # (num_samples)
        
        # MC Dropout
        pred_df[mc_logit_cols] = mc_logits
        pred_df[mc_pred_prob_cols] = mc_pred_prob    
        pred_df[mc_pred_label_col] = mc_pred_prob.argmax(dim=-1)
        if (y_true.max() >= num_classes) or ood:
            pred_df[mc_loss_col] = np.nan
            pred_df[mc_incorrect_class_prob_col] = np.nan
            pred_df[mc_correct_col] = np.nan
        else:
            pred_df[mc_loss_col] = cross_entropy(mc_logits, y_true, reduction="none").numpy()
            pred_df[mc_incorrect_class_prob_col] = get_incorrect_class_prob(y_true, mc_pred_prob)
            pred_df[mc_correct_col] = mc_pred_prob.argmax(dim=-1).numpy().flatten() == y_true.numpy().flatten()
        pred_df[mc_ue_col] = mc_ue
        
    
    return pred_df
