from src.misc import set_seed_pytorch
from src.models.bnn.inference import predict_bnn_model
from src.models.bnn.train import evaluate_bnn_perf
from src.evaluation.perf_metrics import get_elementwise_crossentropyloss_from_pred_prob, get_incorrect_class_prob
from src.evaluation.timer import Timer

import pandas as pd
import torch
from tqdm.auto import tqdm

# model, dl, feat_cols, target_col, num_classes, seed=seed
def get_bnn_model_prediction(bnn_model, dl,  feat_cols, target_col, num_classes, seed, T, label="bnn", ood=False):
    set_seed_pytorch(seed)
    # Prepare dataset
    timer = Timer("BNN")
    seed_list = list(range(seed, seed+T))
    all_logits, all_targets = [], []
    for cur_seed in tqdm(seed_list):
        targets, logits = predict_bnn_model(bnn_model, dl, seed=cur_seed, silent=True)
        all_logits.append(logits)
        all_targets.append(targets)

    all_logits = torch.stack(all_logits) # T, N, C
    all_pred_probs = torch.softmax(all_logits, axis=-1)
    test_y_pred = all_pred_probs.mean(axis=0) # N, C
    test_y_std = all_pred_probs.std(axis=0)
    timer.end()
    test_y_pred_labels = torch.argmax(test_y_pred, axis=-1) # N
    test_y = targets

    pred_prob_cols = [f"{target_col}_class_prob_{i}_{label}" for i in range(num_classes)]
    ue_col = f"{label}_uncertainty"
    pred_label_col = f"{target_col}_pred_label_{label}"
    incorrect_prob_col = f"incorrect_class_prob_{label}"
    ce_col = f"crossentropy_loss_{label}"
    correct_col = f"correct_{label}"

    
    df = pd.DataFrame(test_y_pred, columns=pred_prob_cols)
    df[target_col] = test_y
    df[pred_label_col] = test_y_pred_labels
    df[ue_col] = test_y_std[range(len(test_y_pred_labels)), test_y_pred_labels]
    if not ood:
        df[incorrect_prob_col] = get_incorrect_class_prob(
            true = df[target_col].values, pred_prob=df[pred_prob_cols].values
        )
        df[ce_col] = get_elementwise_crossentropyloss_from_pred_prob(
            true = df[target_col].values, pred_prob=df[pred_prob_cols].values
        )
        df[correct_col] = test_y.numpy() == test_y_pred_labels.numpy()

    return df


def evaluate_bnn(bnn_model, dl, seed):
    target, pred = predict_bnn_model(bnn_model, dl, seed=seed, silent=True)
    return evaluate_bnn_perf(target, pred)