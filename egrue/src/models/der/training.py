import torch.optim as optim
from torch.nn import Module
from src.data_processing.dataset import TabularDataset


from lightning import LightningDataModule
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from torch_uncertainty.layers.distributions import NormalInverseGammaLayer
from torch_uncertainty.losses import DERLoss
from torch_uncertainty.models.mlp import mlp
from torch_uncertainty.routines import RegressionRoutine


import contextlib
import os

enable_print  = print
disable_print = lambda *x, **y: None


def optim_regression(
    model: Module, learning_rate: float = 0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0,)
    return optimizer


def train_der_w_param(
    n_hidden_layers, hidden_width, train_df, valid_df, inputs, outputs,
    batch_size, seed, max_epochs=500, patience=5, time_training=True):
    seed_everything(seed, workers=True)
    # Data
    train_ds = TabularDataset(df=train_df, feat_cols=inputs, target_cols=outputs)
    valid_ds = TabularDataset(df=valid_df, feat_cols=inputs, target_cols=outputs)
    datamodule = LightningDataModule.from_datasets(
        train_ds, val_dataset=valid_ds, test_dataset=valid_ds, batch_size=batch_size, num_workers=63)
    datamodule.training_task = "regression"
    # print("hidden_dims:", [hidden_width for _ in range(n_hidden_layers)])
    # Model
    model = mlp(
        in_features=len(inputs),
        num_outputs=4*len(outputs),
        hidden_dims=[hidden_width for _ in range(n_hidden_layers)],
        final_layer=NormalInverseGammaLayer,
        final_layer_args={"dim": len(outputs)},
    )

    # Training
    loss = DERLoss(reg_weight=1e-2)
    routine = RegressionRoutine(
        probabilistic=True,
        output_dim=len(outputs),
        model=model,
        loss=loss,
        optim_recipe=optim_regression(model),
    )
    early_stopping = EarlyStopping('val/MSE', patience=patience, verbose=True, mode='min')

    trainer = Trainer(accelerator="gpu", devices=1, max_epochs=max_epochs, enable_progress_bar=True, callbacks=[early_stopping])
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        trainer.fit(model=routine, datamodule=datamodule)
        result = trainer.test(model=routine, datamodule=datamodule)

    return model, result[0]