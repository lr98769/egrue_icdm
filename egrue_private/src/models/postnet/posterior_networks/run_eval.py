import logging
import torch

from os.path import join, basename, exists
from tqdm.auto import tqdm
import pandas as pd

from src.models.postnet.dataset_manager.ClassificationDataset import ClassificationDataset
from src.models.postnet.posterior_networks.PosteriorNetwork import PosteriorNetwork
from src.models.postnet.posterior_networks.train import train, train_sequential
from src.models.postnet.posterior_networks.test import test_on_dataset
from src.configs.default_configs import device
from src.misc import set_seed_pytorch
from src.data_processing.dataloader import get_pytorch_split_dict_image
from src.models.postnet.posterior_networks.misc import get_class_count, get_class_count_from_df

def run_eval(
        # Dataset parameters
        seed_dataset,  # Seed to shuffle dataset. int
        data_dict, 
        ood_data_dicts,
        dataset_name,  # Dataset name. string
        
        # Architecture parameters
        seed_model,  # Seed to init model. int
        directory_model,  # Path to save model. string
        architecture,  # Encoder architecture name. string
        input_dims,  # Input dimension. List of ints
        output_dim,  # Output dimension. int
        hidden_dims,  # Hidden dimensions. list of ints
        kernel_dim,  # Input dimension. int
        latent_dim,  # Latent dimension. int
        no_density,  # Use density estimation or not. boolean
        density_type,  # Density type. string
        n_density,  # Number of density components. int
        k_lipschitz,  # Lipschitz constant. float or None (if no lipschitz)
        budget_function,  # Budget function name applied on class count. name

        # Training parameters
        directory_results,  # Path to save resutls. string
        max_epochs,  # Maximum number of epochs for training
        patience,  # Patience for early stopping. int
        frequency,  # Frequency for early stopping test. int
        batch_size,  # Batch size. int
        lr,  # Learning rate. float
        loss,  # Loss name. string
        training_mode,  # 'joint' or 'sequential' training. string
        regr, # Regularization factor in Bayesian loss. float
        timing,
        fn_model, 
        override=False,
        dl_func = get_pytorch_split_dict_image
):  
    # 1. Load model
    set_seed_pytorch(seed_dataset)

    # Get Class Count
    target_col = data_dict["target_col"]
    if isinstance(data_dict["train_df"], pd.DataFrame):
        N = get_class_count_from_df(data_dict["train_df"], target_col=target_col, output_dim=output_dim)
    else:
        N = get_class_count(ds=data_dict["train_df"], output_dim=output_dim)

    # - Make the model to load into
    model = PosteriorNetwork(
        N=N,
        input_dims=input_dims,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        kernel_dim=kernel_dim,
        latent_dim=latent_dim,
        architecture=architecture,
        k_lipschitz=k_lipschitz,
        no_density=no_density,
        density_type=density_type,
        n_density=n_density,
        budget_function=budget_function,
        batch_size=batch_size,
        lr=lr,
        loss=loss,
        regr=regr,
        seed=seed_model)
    
    # - Load Model
    model.load_state_dict(torch.load(join(directory_model, fn_model))['model_state_dict'])
    model.to(device)

    fp_results = {}

    # - Predict on Train, Valid, Test Set
    print("Predicting Train, Valid, Test...")
    dl_dict = dl_func(data_dict, batch_size, batch_size, shuffle_train=False)
    for dl_name, dl in tqdm(dl_dict.items()):
        if "dl" in dl_name:
            label = dl_name.split("_")[0]
            fp_result = f'{directory_results}/{label}_pn_pred.pickle'
            if override or not exists(fp_result):
                dl.dataset.output_dim = output_dim
                test_on_dataset(model, dl, fp_result)
                fp_results[label] = fp_result

    # Predict on OOD
    print("Predicting OOD...")
    for ood_dataset_name, data_dict in tqdm(ood_data_dicts.items()):
        fp_result = f'{directory_results}/{ood_dataset_name}_pred.pickle'
        dl_dict = get_pytorch_split_dict_image(
            data_dict, batch_size, batch_size, shuffle_train=True)
        ood_dataloader = dl_dict["test_dl"]
        if override or not exists(fp_result):
            test_on_dataset(model, ood_dataloader, fp_result)
            fp_results[ood_dataset_name] = fp_result

    return fp_results


