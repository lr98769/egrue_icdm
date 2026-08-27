import logging
import torch

import sys
import pandas as pd

from src.models.postnet.dataset_manager.ClassificationDataset import ClassificationDataset
from src.models.postnet.posterior_networks.PosteriorNetwork import PosteriorNetwork
from src.models.postnet.posterior_networks.misc import get_class_count, get_class_count_from_df
from src.models.postnet.posterior_networks.train import train, train_sequential
from src.models.postnet.posterior_networks.test import test
from src.data_processing.dataloader import get_pytorch_split_dict_image
from src.misc import set_seed_pytorch

def run(
        # Dataset parameters
        seed_dataset,
        data_dict, 
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
        timing, # Toggle whether the prediction set is just the test set or all
        dl_func = get_pytorch_split_dict_image
):  

    logging.info('Received the following configuration:')
    logging.info(f'DATASET | '
                 f'dataset_name {dataset_name} - ')
    logging.info(f'ARCHITECTURE | '
                 f' seed_model {seed_model} - '
                 f' architecture {architecture} - '
                 f' input_dims {input_dims} - '
                 f' output_dim {output_dim} - '
                 f' hidden_dims {hidden_dims} - '
                 f' kernel_dim {kernel_dim} - '
                 f' latent_dim {latent_dim} - '
                 f' no_density {no_density} - '
                 f' density_type {density_type} - '
                 f' n_density {n_density} - '
                 f' k_lipschitz {k_lipschitz} - '
                 f' budget_function {budget_function}')
    logging.info(f'TRAINING | '
                 f' max_epochs {max_epochs} - '
                 f' patience {patience} - '
                 f' frequency {frequency} - '
                 f' batch_size {batch_size} - '
                 f' lr {lr} - '
                 f' loss {loss} - '
                 f' training_mode {training_mode} - '
                 f' regr {regr}')

    ##################
    ## Load dataset ##
    ##################
    set_seed_pytorch(seed_dataset)
    dl_dict = dl_func(
        data_dict, batch_size, batch_size, shuffle_train=True)
    train_loader, val_loader, test_loader = dl_dict["train_dl"], dl_dict["val_dl"], dl_dict["test_dl"]
    train_loader.dataset.output_dim = output_dim
    val_loader.dataset.output_dim = output_dim
    test_loader.dataset.output_dim = output_dim
    target_col = data_dict["target_col"]
    
    # Get Class Count
    if isinstance(data_dict["train_df"], pd.DataFrame):
        N = get_class_count_from_df(data_dict["train_df"], target_col=target_col, output_dim=output_dim)
    else:
        N = get_class_count(ds=data_dict["train_df"], output_dim=output_dim)

    #################
    ## Train model ##
    #################
    model = PosteriorNetwork(N=N,
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
    print("Number of Parameters:", sum(p.numel() for p in model.parameters()))
    full_config_dict = {'seed_dataset': seed_dataset,
                        'dataset_name': dataset_name,
                        'seed_model': seed_model,
                        'architecture': architecture,
                        # 'N': N,
                        'input_dims': input_dims,
                        'output_dim': output_dim,
                        'hidden_dims': hidden_dims,
                        'kernel_dim': kernel_dim,
                        'latent_dim': latent_dim,
                        'no_density': no_density,
                        'density_type': density_type,
                        'n_density': n_density,
                        'k_lipschitz': k_lipschitz,
                        'budget_function': budget_function,
                        'max_epochs': max_epochs,
                        'patience': patience,
                        'frequency': frequency,
                        'batch_size': batch_size,
                        'lr': lr,
                        'loss': loss,
                        'training_mode': training_mode,
                        'regr': regr}
    full_config_name = ''
    i = 0
    for k, v in full_config_dict.items():
        full_config_name += str(v) + '-'
        i+=1
        if i > 5: # modified to reduce length of model path (long path cause runtime error)
            break
    full_config_name = full_config_name[:-1]
    model_path = f'{directory_model}/model-dpn-{full_config_name}'
    if training_mode == 'joint':
        train_losses, val_losses, train_accuracies, val_accuracies = train(model,
                                                                      train_loader,
                                                                      val_loader,
                                                                      max_epochs=max_epochs,
                                                                      frequency=frequency,
                                                                      patience=patience,
                                                                      model_path=model_path,
                                                                      full_config_dict=full_config_dict)
    elif training_mode == 'sequential':
        assert not no_density
        train_losses, val_losses, train_accuracies, val_accuracies = train_sequential(model,
                                                                                       train_loader,
                                                                                       val_loader,
                                                                                       max_epochs=max_epochs,
                                                                                       frequency=frequency,
                                                                                       patience=patience,
                                                                                       model_path=model_path,
                                                                                       full_config_dict=full_config_dict)
    else:
        raise NotImplementedError

    ################
    ## Test model ##
    ################
    ood_dataset_loaders = {}
    result_path = f'{directory_results}/results-dpn-{full_config_name}'
    model.load_state_dict(torch.load(f'{model_path}')['model_state_dict'])
    metrics = test(model, test_loader, ood_dataset_loaders, result_path)

    results = {
        'model_path': model_path,
        'result_path': result_path,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
    }

    return {**results, **metrics}
    
    
