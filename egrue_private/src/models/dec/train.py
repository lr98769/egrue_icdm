import torch.nn as nn
import torch.optim as optim
from src.models.dec.model import instantiate_dec_model
from src.configs.default_configs import device


from lightning import LightningDataModule
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from torch_uncertainty.losses import DECLoss
from torch_uncertainty.routines import ClassificationRoutine


enable_print  = print
disable_print = lambda *x, **y: None


def optim_classifier(model: nn.Module) -> dict:
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.005)
    # exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    return {"optimizer": optimizer} #, "lr_scheduler": exp_lr_scheduler


def train_dec(
    best_param, data_dict, seed, batch_size,
    max_epochs=500, patience=5, freeze_layers=True, reg_weight=0.001
):
    num_outputs = data_dict["num_classes"]
    seed_everything(seed, workers=True)
    datamodule = LightningDataModule.from_datasets(
        data_dict["train_df"], val_dataset=data_dict["val_df"],
        test_dataset=data_dict["test_df"],
        batch_size=batch_size, num_workers=63)
    datamodule.training_task = "classification"
    # Model
    model = instantiate_dec_model(
        num_outputs=num_outputs, seed=seed, **best_param
    )

    if freeze_layers:
        # Freeze layers before denseblock3
        freeze = True
        for name, module in model[0].named_children():
            if name == "denseblock3":
                freeze=False
            if freeze:
                for param in module.parameters():
                    param.requires_grad = False

    # Training
    loss = DECLoss(reg_weight=reg_weight)
    routine = ClassificationRoutine(
        model=model,
        num_classes=num_outputs,
        loss=loss,
        optim_recipe=optim_classifier(model),
    )
    early_stopping = EarlyStopping('val/cls/NLL', patience=patience, verbose=True, mode='min')

    trainer = Trainer(accelerator="gpu", devices=[device.index], max_epochs=max_epochs, enable_progress_bar=True, callbacks=[early_stopping])
    # with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
    trainer.fit(model=routine, datamodule=datamodule)
    result = trainer.test(model=routine, datamodule=datamodule)

    return model, result[0]