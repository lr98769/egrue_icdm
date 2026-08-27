from sklearn.gaussian_process import GaussianProcessClassifier
import time


from src.models.gpc.save_load_model import save_gpc
from src.evaluation.perf_metrics import get_entropy_from_pred_prob, get_bce_from_pred_prob

def train_gpc(split_dict, feat_cols, target_col, fp_model, seed):
    train_df = split_dict["train_df"]
    X = train_df[feat_cols].values
    y = train_df[target_col].values
    print("Fitting model...")
    start = time.time()
    gpc = GaussianProcessClassifier(random_state=seed, n_jobs=-1, copy_X_train=False).fit(X, y)
    print(f"- Took: {time.time()-start}s.")
    print("Saving Model...")
    start = time.time()
    save_gpc(gpc, fp_model)
    print(f"- Took: {time.time()-start}s.")
    print("Predicting...")
    start = time.time()
    all_pred_df = split_dict["test_df"]
    # ue, loss, pred_prob, pred_label
    all_pred_df[f"{target_col}_pred_prob_gpc"] = gpc.predict_proba(all_pred_df[feat_cols].values)[:,1]
    all_pred_df[f"{target_col}_pred_label_gpc"] = round(all_pred_df[f"{target_col}_pred_prob_gpc"])
    all_pred_df[f"entropy_gpc"] = get_entropy_from_pred_prob(all_pred_df[f"{target_col}_pred_prob_gpc"])
    all_pred_df[f"bce_gpc"] = get_bce_from_pred_prob(all_pred_df[target_col], all_pred_df[f"{target_col}_pred_prob_gpc"])
    print(f"- Took: {time.time()-start}s.")
    return all_pred_df