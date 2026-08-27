from src.configs.default_configs import device
from src.evaluation.inference import prediction
from src.training.callbacks import EarlyStoppingCallback, HistoryRecorder, MetricCalculator, \
    LiveHistoryPlotter, ImageStorage
from src.training.display_progress import plot_history, print_split_epoch_metrics
from src.misc import set_seed_pytorch

from torch.nn import L1Loss, MSELoss
from torch.optim import Adam, AdamW
from tqdm.auto import tqdm
import time
import torch


def train_resnet_rue(
    model, train_dl, val_dl, test_dl, seed,
    fp_model, fp_history=None, # Where to store trained model and history of training
    max_epochs=500, lr=0.001, weight_decay=0.1,  # Training parameters
    patience=5, metric_to_monitor = "auc", maximise=True, # For early stopping
    verbose=True, class_weight=None, optimizer="adam"
):
    set_seed_pytorch(seed)
    metric_list = ["rue mae","rue mse", "rue correlation"]
    split_list = ["train", "valid"]

    # loss_fn = 
    loss_fn = L1Loss() # MSELoss()
    if optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Initialise history recorder
    hr = HistoryRecorder(split_list, metric_list)
    
    # Initialise live history plotter
    hp = LiveHistoryPlotter(history_recorder=hr, ncols=3)

    # EarlyStopping callback
    es = EarlyStoppingCallback(patience, metric_to_monitor, fp_model, maximise=maximise)

    # Initialise metric calculator
    mc = MetricCalculator(metric_list)
    
    # Visualise Reconstruction Progress
    img_storage = ImageStorage(fp_history)

    model.train()
    # Train decoder
    for param in model.decoder.parameters():
        param.requires_grad = True
    # Don't train encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    model.encoder.eval()
    # Don't train classifier
    for param in model.classifier.parameters():
        param.requires_grad = False
    model.classifier.eval()
    model.to(device)
    with tqdm(range(max_epochs), leave=False, disable=verbose) as epoch_pbar:
        for epoch in epoch_pbar:
            if epoch > 0:
                epoch_pbar.set_description(f"Valid {metric_to_monitor.capitalize()}: {val_score:.5f}")
            start = time.time()
            all_x, all_y, all_x_logits, all_y_logits = [], [], [], []
            position = 0 if verbose else 1
            with tqdm(train_dl, leave=False, position=position, disable=not verbose) as pbar:
                for x_batch, y_batch in pbar:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    y_pred_logits, x_logits = model(x_batch)
                    loss = loss_fn(x_logits, x_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()

                    loss = loss.detach().cpu()
                    x_batch = x_batch.detach().cpu()
                    y_batch = y_batch.detach().cpu()
                    y_pred_logits = y_pred_logits.detach().cpu()
                    x_logits = x_logits.detach().cpu()

                    all_y.append(y_batch)
                    all_y_logits.append(y_pred_logits)
                    all_x.append(x_batch)
                    all_x_logits.append(x_logits)
                    del x_batch, y_batch, y_pred_logits, x_logits, loss
            
            # Calculate train metrics
            train_y_true, train_y_logits, train_x_true, train_x_logits = (
                torch.cat(all_y), torch.cat(all_y_logits), torch.cat(all_x), torch.cat(all_x_logits)
            )
            train_metric_dict = mc.calculate_metric_dict(
                train_y_true, train_y_logits, train_x_true, train_x_logits)
            del all_y, all_y_logits, all_x, all_x_logits
            del train_y_true, train_y_logits, train_x_true, train_x_logits

            # Calculate validation metrics
            valid_x_true, valid_y_true, valid_y_logits, valid_x_logits = prediction(model, val_dl, verbose)
            valid_metric_dict = mc.calculate_metric_dict(
                valid_y_true, valid_y_logits, valid_x_true, valid_x_logits)
            
            # Image Storage
            if epoch == 0:
                img_storage.store_image(valid_x_true[0])
            img_storage.store_reconstruction(valid_x_logits[0], epoch)
            del valid_x_true, valid_y_true, valid_y_logits, valid_x_logits

            # Print epoch metrics
            epoch_time = time.time() - start
            if verbose:
                print(f"=== Epoch {epoch}, Time Taken: {epoch_time:.1f}s, Time Left: {epoch_time*(max_epochs-epoch-1):.1f}s ===")
                print_split_epoch_metrics(splitname="train", metric_dict=train_metric_dict)
                print_split_epoch_metrics(splitname="valid", metric_dict=valid_metric_dict)

            # Save metrics to visualise training history
            hr.record_epoch_metrics(splitname="train", metric_dict=train_metric_dict)
            hr.record_epoch_metrics(splitname="valid", metric_dict=valid_metric_dict)

            # Update Training History Plot
            hp.update()

            # Early Stopping
            val_score = valid_metric_dict[metric_to_monitor]
            if es.stop_training(val_score, epoch, model):
                break

            

    history = hr.get_history_dict()
    plot_history(history, split_list, metric_list, max_cols = 3, fp_history=fp_history)
    return history

