import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm.auto import tqdm
import time

from src.configs.default_configs import device
from src.training.callbacks import HistoryRecorder, EarlyStoppingCallback, MetricCalculator, ItemStorage, LiveHistoryPlotter
from src.training.display_progress import print_split_epoch_metrics, plot_history
from src.training.params_func import freeze_layers_until, count_parameters, count_trainable_parameters
from src.models.resnet.inference import predict_w_resnet
from src.misc import set_seed_pytorch

def train_resnet(
    model, train_dl, val_dl, test_dl, seed,
    fp_model, fp_history=None, # Where to store trained model and history of training
    max_epochs=500, lr=0.001, weight_decay=0.1,  # Training parameters
    patience=5, metric_to_monitor = "acc", maximise=True, # For early stopping
    verbose=True, class_weights=None
):
    set_seed_pytorch(seed)
    
    metric_list = ["acc", "ce loss", "auc", "f1"]
    split_list = ["train", "valid"]
    n_cols=2
    
    # Set Model to Training
    model.train()
    model.to(device)
    
    # # Freeze weights before layer 2
    # freeze_layers_until(model, layer_name="layer2")
    
    # Set up loss and optimizer
    if class_weights is not None:
        loss_fn = CrossEntropyLoss(weight=class_weights) 
    else:
        loss_fn = CrossEntropyLoss() 
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Check frozen parameters
    print(f"- {count_trainable_parameters(model)} out of {count_parameters(model)} parameters are trainable.")

    # Initialise history recorder, earlystopping callback, metric calculator
    hr = HistoryRecorder(split_list, metric_list)
    es = EarlyStoppingCallback(patience, metric_to_monitor, fp_model, maximise=maximise)
    mc = MetricCalculator(metric_list)
    hp = LiveHistoryPlotter(history_recorder=hr, ncols=n_cols)
    
    with tqdm(range(max_epochs), leave=False, disable=verbose) as epoch_pbar:
        for epoch in epoch_pbar:
            if epoch > 0:
                epoch_pbar.set_description(
                    f"Best Epoch: {es.best_epoch}, Best {es.metric_to_monitor.capitalize()}: {es.best_val_score}, Cur Val {metric_to_monitor.capitalize()}: {val_score:.5f}")
            start = time.time()
            item_storage = ItemStorage()
            position = 0 if verbose else 1 
            with tqdm(train_dl, leave=False, position=position, disable=not verbose) as pbar:
                for input_batch, output_batch in pbar:
                    input_batch, output_batch = input_batch.to(device), output_batch.to(device)
                    y_logits = model(input_batch) 
                    
                    # Backprop
                    loss = loss_fn(y_logits, output_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    
                    # Store Outputs
                    cur_output_batch = [output_batch, y_logits]
                    cur_output_batch = [element.detach().cpu() for element in cur_output_batch]
                    item_storage.add_batch_items(cur_output_batch)
                    
                    del input_batch, output_batch, loss
    
            # Calculate train metrics
            train_outputs = item_storage.get_stored()
            del item_storage
            train_metric_dict = mc.calculate_metric_dict(
                y_true=train_outputs[0], 
                y_logits=train_outputs[1])
            del train_outputs
    
            # Calculate validation metrics
            valid_outputs = predict_w_resnet(model, val_dl, verbose)
            valid_metric_dict = mc.calculate_metric_dict(
                y_true=valid_outputs[0], 
                y_logits=valid_outputs[1])
            del valid_outputs
    
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
    plot_history(history, split_list, metric_list, max_cols = n_cols, fp_history=fp_history)
    return history
    