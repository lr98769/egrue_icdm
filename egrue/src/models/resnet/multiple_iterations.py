from tqdm.auto import tqdm
from os.path import exists
from IPython.display import display

from src.file_manager.filepath import FilePath
from src.configs.default_configs import fn_pred, fn_pred_perf
from src.training.train import train_model_w_best_param
from src.file_manager.load_save_model import load_model
from src.evaluation.inference import get_all_predictions
from src.file_manager.load_save_df import load_pred_df, save_pred_df, save_pred_perf_df
from src.evaluation.evaluate import get_model_performance

from src.models.resnet.predict import get_resnet_predictions_speedup

def train_pred_resnet_multiple_iterations(
    params, seed_list, 
    data_dict, data_dict_ood_all, data_name, T=10, 
    override_train=False, override_predict=False, override_perf_eval=False, override_ood=False
):
    batch_size = params["batch_size"]
    eval_batch_size = params["eval_batch_size"]
    ModelClass = params["ModelClass"]
    pytorch_split_dict_func = params["pytorch_split_dict_func"]
    with tqdm(seed_list) as pbar:
        for cur_seed in pbar:
            cur_model_name = "tuned"
            fp = FilePath(data_name=data_name, seed=cur_seed)
            # Train if model does not exist
            if override_train or not exists(fp.get_fp_model(ModelClass, cur_model_name)):
                pbar.set_description("Training")
                model = train_model_w_best_param(
                    **params,
                    seed=cur_seed, fp=fp,
                    cur_model_name=cur_model_name
                )
            # Predict if we have not predicted
            if override_predict or not exists(fp.get_fp_df(ModelClass=ModelClass, df_type=fn_pred)):
                pbar.set_description("Predicting")
                model = load_model(fp=fp, ModelClass=ModelClass, cur_model_name=cur_model_name)
                pred_df = get_all_predictions(
                    model=model, 
                    data_dict=data_dict, 
                    batch_size=batch_size, 
                    eval_batch_size=eval_batch_size,
                    pred_func=get_resnet_predictions_speedup,
                    pytorch_split_dict_func=pytorch_split_dict_func,
                    seed=cur_seed,
                ) 
                save_pred_df(pred_df=pred_df, fp=fp, ModelClass=ModelClass)
                
            # Get Model Performance
            if override_perf_eval or not exists(fp.get_fp_df(ModelClass=ModelClass, df_type=fn_pred_perf)):
                pbar.set_description("Performance Evaluation")
                pred_df = load_pred_df(fp=fp, ModelClass=ModelClass)
                perf_df = get_model_performance(all_pred_df=pred_df, data_dict=data_dict, label="resnet")
                save_pred_perf_df(pred_perf_df=perf_df, fp=fp, ModelClass=ModelClass)
                print("ResNet:")
                display(perf_df)
                pred_df = load_pred_df(fp=fp, ModelClass=ModelClass)
                perf_df = get_model_performance(all_pred_df=pred_df, data_dict=data_dict, label="resnet_mc")
                save_pred_perf_df(pred_perf_df=perf_df, fp=fp, ModelClass=ModelClass, optional_label="mc")
                print("ResNet MC:")
                display(perf_df)
                
            # Get OOD Prediction
            all_ood_files_exist = all([exists(
                fp.get_fp_df(ModelClass=ModelClass, df_type=fn_pred, optional_label=f"ood_{key}")) 
                for key in data_dict_ood_all.keys()])
            if override_ood or not all_ood_files_exist:
                pbar.set_description("OOD Detection")
                resnet_model = load_model(fp=fp, ModelClass=ModelClass, cur_model_name="tuned")
                for key, data_dict_ood in data_dict_ood_all.items():
                    pred_df_ood = get_all_predictions(
                        model=resnet_model, 
                        data_dict=data_dict_ood, 
                        batch_size=batch_size, 
                        eval_batch_size=eval_batch_size,
                        pred_func=get_resnet_predictions_speedup,
                        seed=cur_seed,
                        pytorch_split_dict_func=pytorch_split_dict_func,
                        )
                    save_pred_df(pred_df=pred_df_ood, fp=fp, ModelClass=ModelClass, optional_label=f"ood_{key}")