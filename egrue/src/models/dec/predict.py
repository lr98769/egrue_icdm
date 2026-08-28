from src.configs.default_configs import device
from src.evaluation.perf_metrics import get_elementwise_crossentropyloss_from_pred_prob, get_incorrect_class_prob
from src.evaluation.timer import Timer

import torch
from lightning.pytorch import seed_everything
from tqdm.auto import tqdm
import pandas as pd

# model, dl, feat_cols, target_col, num_classes, seed=seed, **additional_pred_args
def get_dec_model_prediction(
        dec_model, dl, feat_cols, target_col, num_outputs, seed, label="dec", ood=False):
    seed_everything(seed, workers=True)

    entropy_list, test_y_list, test_y_pred_prob = [], [], []
    dec_model.to(device)
    timer = Timer("EDL")
    with torch.no_grad():
        for x_batch, y_batch in tqdm(dl, total=len(dl)):
            evidence = dec_model(x_batch.to(device))
            alpha = torch.relu(evidence) + 1
            strength = torch.sum(alpha, dim=1, keepdim=True)
            probs = alpha / strength
            entropy = -1 * torch.sum(probs * torch.log(probs), dim=1, keepdim=True)
            entropy_list.append(entropy)
            test_y_pred_prob.append(probs)
            test_y_list.append(y_batch)
            x_batch.cpu()

    entropy = torch.cat(entropy_list).cpu().numpy()
    timer.end()
    test_y_pred_prob = torch.cat(test_y_pred_prob).cpu().numpy()
    test_y_pred = test_y_pred_prob.argmax(axis=1)
    test_y = torch.cat(test_y_list).cpu().numpy()

    pred_prob_cols = [f"{target_col}_class_prob_{i}_{label}" for i in range(num_outputs)]
    ue_col = f"{label}_uncertainty"
    pred_label_col = f"{target_col}_pred_label_{label}"
    incorrect_prob_col = f"incorrect_class_prob_{label}"
    ce_col = f"crossentropy_loss_{label}"
    correct_col = f"correct_{label}"
    
    df = pd.DataFrame(test_y_pred_prob, columns=pred_prob_cols)
    df[target_col] = test_y
    df[ue_col] = entropy # N
    df[pred_label_col] = test_y_pred
    if not ood:
        df[incorrect_prob_col] = get_incorrect_class_prob(
            true = df[target_col].values, pred_prob=df[pred_prob_cols].values
        )
        df[ce_col] = get_elementwise_crossentropyloss_from_pred_prob(
            true = df[target_col].values, pred_prob=df[pred_prob_cols].values
        )
        df[correct_col] = test_y == test_y_pred
        
    return df