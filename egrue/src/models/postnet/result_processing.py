import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import pickle

from src.evaluation.perf_metrics import get_elementwise_crossentropyloss_from_pred_prob, get_incorrect_class_prob
from src.file_manager.load_save_df import save_pred_df
from src.models.postnet.posterior_networks.PosteriorNetwork import PosteriorNetwork

def process_pn_results(data_dict, ood_data_dicts, directory_results, fp, only_test=False):
    label_map = {"train": "Train", "val": "Valid", "test": "Test"}
    # Load normal predictions
    pred_df = []
    for split_name, split_df in tqdm(data_dict.items()):
        if "df" not in split_name:
            continue
        elif only_test and split_name != "test_df":
            continue
        label = split_name.split("_")[0]
        fp_result = f'{directory_results}/{label}_pn_pred.pickle'
        with open(fp_result, 'rb') as f:
            results = pickle.load(f)
        split_df = convert_pn_results_to_df(results, data_dict)
        split_df["split"] = label_map[label]
        pred_df.append(split_df)
    save_pred_df(pred_df=pd.concat(pred_df), fp=fp, ModelClass=PosteriorNetwork)

    # Load ood predictions
    for ood_dataset_name, data_dict_ood in tqdm(ood_data_dicts.items()):
        split_df = data_dict_ood["test_df"]
        fp_result = f'{directory_results}/{ood_dataset_name}_pred.pickle'
        with open(fp_result, 'rb') as f:
            results = pickle.load(f)
            pred_df_ood = convert_pn_results_to_df(results, data_dict, ood=True)
            save_pred_df(
                pred_df=pred_df_ood, fp=fp, ModelClass=PosteriorNetwork, optional_label=ood_dataset_name)


def convert_pn_results_to_df(results, data_dict, label="pn", ood=False):
    target_col = data_dict["target_col"]
    num_classes = data_dict["num_classes"]
    df = pd.DataFrame(np.expand_dims(results["Y"], -1), columns=[target_col])
    
    # Columns
    alpha_cols = [f"alpha_{i}" for i in range(num_classes)]
    pred_prob_cols = [f"{target_col}_class_prob_{i}_{label}" for i in range(num_classes)]
    pred_label_col = f"{target_col}_pred_label_{label}"
    alea_conf_col = f"aleatoric_conf_{label}"
    epis_conf_col = f"epistemic_conf_{label}"
    alea_ue_col = f"aleatoric_ue_{label}"
    epis_ue_col = f"epistemic_ue_{label}"
    bce_col = f"crossentropy_loss_{label}"
    incorrect_prob_col = f"incorrect_class_prob_{label}"
    correct_col = f"correct_{label}"
    
    df[alpha_cols] = results["alpha"]
    df[pred_prob_cols] = results["alpha"]/results["alpha"].sum(-1, keepdims=True)
    df[pred_label_col] = results["alpha"].argmax(-1)
    df[alea_conf_col] = df[pred_prob_cols].values.max(axis=-1)
    df[epis_conf_col] = results["alpha"].max(axis=-1)
    df[alea_ue_col] = 1/df[alea_conf_col]
    df[epis_ue_col] = 1/df[epis_conf_col]

    # Add losses
    if not ood:
        df[bce_col] = get_elementwise_crossentropyloss_from_pred_prob(
            true = df[target_col].values, pred_prob=df[pred_prob_cols].values
        )
        df[incorrect_prob_col] = get_incorrect_class_prob(
            true = df[target_col].values, pred_prob=df[pred_prob_cols].values
        )
        df[correct_col] = df[target_col].values == df[pred_label_col].values
    
    return df