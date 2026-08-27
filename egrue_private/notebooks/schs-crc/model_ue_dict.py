target_col = "colorectal cancer"

pred_model_files = ["egRUE", "pn", "gpc", "de"]
perf_model_files = ["egRUE", "gpc", "MC Dropout", "postNet", "de"]

ue_dict = {
    "RUE":{"ue":"rue", "loss":"bce", "pred_label":f"{target_col}_pred_label"},
    "egRUE":{"ue":"egRUE", "loss":"bce", "pred_label":f"{target_col}_pred_label"},
    "Entropy": {"ue":"entropy", "loss": "bce", "pred_label":f"{target_col}_pred_label"},
    "Deep Ensemble":{"ue":"de_std", "loss":"bce_de", "pred_label":f"{target_col}_pred_label_de"},
    "MC Dropout": {"ue":"ue_mc", "loss": "bce_mc", "pred_label":f"{target_col}_pred_label_mc"},
    "PostNet Aleatoric": {"ue":"aleatoric_ue_pn", "loss":"bce_pn", "pred_label":f"{target_col}_pred_label_pn"},
    "PostNet Epistemic": {"ue":"epistemic_ue_pn", "loss":"bce_pn", "pred_label":f"{target_col}_pred_label_pn"},
    "GPC Entropy": {"ue":"entropy_gpc", "loss":"bce_gpc", "pred_label":f"{target_col}_pred_label_gpc"},
}