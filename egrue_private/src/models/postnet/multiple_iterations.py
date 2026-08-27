from tqdm.auto import tqdm
from os.path import exists, join
from IPython.display import display
from pickle import load
import pandas as pd

from src.file_manager.filepath import FilePath
from src.configs.default_configs import fn_pred, fn_pred_perf
from src.training.train import train_model_w_best_param
from src.file_manager.load_save_model import load_model
from src.evaluation.inference import get_all_predictions
from src.file_manager.load_save_df import load_pred_df, save_pred_df, save_pred_perf_df
from src.evaluation.evaluate import get_model_performance

from src.configs.default_configs import fn_model
from src.models.postnet.posterior_networks.run import run
from src.models.postnet.posterior_networks.run_eval import run_eval
from src.models.postnet.posterior_networks.PosteriorNetwork import PosteriorNetwork
from src.models.postnet.result_processing import convert_pn_results_to_df


def train_pred_postnet_multiple_iterations(
    params, seed_list, 
    data_dict, fp_data_oods, data_name, 
    override_train=False, override_predict=False, override_perf_eval=False, override_ood=False
):
    ModelClass = PosteriorNetwork
    split = params["split"]
    dataset_name = params["dataset_name"]
    with tqdm(seed_list) as pbar:
        for cur_seed in pbar:
            cur_model_name = "tuned"
            fp = FilePath(data_name=data_name, seed=cur_seed)
            fp_results = f'{fp.get_parent_folder(folder_name=fn_model)}'\
                f'/results-dpn-{cur_seed}-{dataset_name}-{split}-0-1-{cur_seed}.pickle'
            fn_cur_model = f'model-dpn-{cur_seed}-{dataset_name}-{split}-0-1-{cur_seed}'
            directory_dataset = fp.get_preprocessed_folder()
            output_directory = fp.get_parent_folder(folder_name=fn_model)
            # Train if model does not exist
            if override_train or not exists(fp_results):
                pbar.set_description("Training")
                results_metrics = run(
                    # Directory
                    directory_dataset=directory_dataset,
                    directory_model=output_directory,
                    directory_results=output_directory,
                    # Seeds
                    seed_dataset=cur_seed, # No shuffling
                    seed_model=cur_seed,
                    **params
                )
            # Predict if we have not predicted
            if override_predict or not exists(fp.get_fp_df(ModelClass=ModelClass, df_type=fn_pred)):
                pbar.set_description("Predicting")
                with open(fp_results, 'rb') as f:
                    results = load(f)
                fp_data = join(directory_dataset, dataset_name)
                data_df = pd.read_parquet(fp_data).iloc[:-3]
                feat_cols = data_df.columns.to_list()[:-1]
                target_col = data_df.columns.to_list()[-1]
                pred_df = convert_pn_results_to_df(results, feat_cols, target_col, split=split)
                pred_df.index=data_df.index
                save_pred_df(pred_df=pred_df.drop(columns=feat_cols), fp=fp, ModelClass=ModelClass)
            # Get Model Performance
            if override_perf_eval or not exists(fp.get_fp_df(ModelClass=ModelClass, df_type=fn_pred_perf)):
                pbar.set_description("Performance Evaluation")
                pred_df = load_pred_df(fp=fp, ModelClass=ModelClass)
                perf_df = get_model_performance(all_pred_df=pred_df, data_dict=data_dict, label="pn")
                save_pred_perf_df(pred_perf_df=perf_df, fp=fp, ModelClass=PosteriorNetwork)
                print("PostNet:")
                display(perf_df)
            # OOD Detection
            all_ood_files_exist = all([exists(
                fp.get_fp_df(ModelClass=ModelClass, df_type=fn_pred, optional_label=f"ood_{mu}")) 
                for mu in fp_data_oods.keys()])
            if override_ood or not all_ood_files_exist:
                pbar.set_description("OOD Detection")
                for key, fp_data_ood in fp_data_oods.items():
                    # - Prediction
                    ood_postnet_params = params.copy()
                    ood_postnet_params["ood_dataset_names"] = [fp_data_ood]
                    fp_results_ood = run_eval(
                        # Directory
                        directory_dataset=directory_dataset,
                        directory_model=output_directory,
                        directory_results=output_directory,
                        # Seeds
                        seed_dataset=cur_seed, # No shuffling
                        seed_model=cur_seed,
                        **ood_postnet_params,
                        fn_model=fn_cur_model
                    )
                    # Process Result
                    with open(fp_results_ood[0], 'rb') as f:
                        results_ood = load(f)
                    data_df_ood = pd.read_parquet(fp_data_ood).iloc[:-3]
                    feat_cols = data_df_ood.columns.to_list()[:-1]
                    target_col = data_df_ood.columns.to_list()[-1]
                    pred_df_ood = convert_pn_results_to_df(results_ood, feat_cols, target_col, split=[0, 0])
                    save_pred_df(
                        pred_df=pred_df_ood.drop(columns=feat_cols), 
                        fp=fp, ModelClass=PosteriorNetwork, optional_label=f"ood_{key}")
