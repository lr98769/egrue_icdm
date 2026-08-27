import torch
import numpy as np
from functools import partial

from src.training.display_progress import LiveTrainingProgress
from src.evaluation.ue_metrics import *
from src.evaluation.perf_metrics import * 

class EarlyStoppingCallback:
    def __init__(self, patience, metric_to_monitor, fp_model, maximise=True):
        self.patience = patience
        self.metric_to_monitor = metric_to_monitor
        self.fp_model = fp_model
        self.maximise = maximise
        
        self.best_epoch = -1
        self.best_val_score = -np.inf if self.maximise else np.inf
        self.num_non_improving_epochs = 0

    def improved(self, val_score):
        if self.maximise:
            return val_score > self.best_val_score
        else:
            return val_score < self.best_val_score
        
    def stop_training(self, val_score, epoch, model):
        # Output True if we should stop training
        # Output False if we should continue
        # If there is an improvement
        if self.improved(val_score):
            self.best_val_score = val_score
            self.best_epoch = epoch
            self.num_non_improving_epochs = 0
            torch.save(model, self.fp_model)
            return False
        # No improvement
        else:
            self.num_non_improving_epochs += 1
            # If it hasn't improved in a long time
            if (self.num_non_improving_epochs >= self.patience):
                print(
                    f"Early Stopping at Epoch {epoch}, "
                    f"Best Validation {self.metric_to_monitor.capitalize()} ({self.best_val_score:.5f}) at Epoch {self.best_epoch}.")
                # Stop training
                return True
            else:
                # Continue training
                return False

class HistoryRecorder:
    def __init__(self, split_list, metric_list):
        self.split_list = split_list
        self.metric_list = metric_list
        self.epoch_metrics = dict()
        self.history = dict()
        for split_name in self.split_list:
            self.history[split_name] = dict()
            for metric in self.metric_list:
                self.history[split_name][metric] = []

    def record_epoch_metrics(self, splitname, metric_dict):
        for metric in self.metric_list:
            self.history[splitname][metric].append(metric_dict[metric])

    def get_history_dict(self):
        return self.history
    
class LiveHistoryPlotter:
    def __init__(self, history_recorder: HistoryRecorder, ncols=3):
        self.history_recorder = history_recorder
        self.plotter = LiveTrainingProgress(
            metric_names=self.history_recorder.metric_list,
            splits=self.history_recorder.split_list, ncols=ncols
        )
    
    def update(self):
        self.plotter.update_figure(
            value_dict=self.history_recorder.get_history_dict()
        )

class MetricCalculator:
    def __init__(self, metric_list, beta=None, metric_params_dict=dict()):
        self.metric_list = metric_list
        if beta:
            self.beta = beta
        self.metric_functions = {
            "classifier": {
                "acc" : get_accuracy,
                "ce loss" : get_crossentropyloss,
                "auc" : get_auc,
                "f1" : get_f1,
            },
            "rue": {
                "rue mse": get_mse_loss,
                "rue mae": get_mae_loss,
                "rue correlation": get_correlation_w_bce,

            },
            "regressor": {
                "mae": get_reg_mae_loss,
                "correlation": get_correlation,
            },
            "rnet": {
                "ae loss": get_rnet_recon_loss,
                "ae 0 loss": get_rnet_recon_0_loss,
                "ae 1 loss": get_rnet_recon_1_loss,
                "total loss": get_rnet_total_loss,
            },
            "vae": {
                "vae mae": get_vae_mae_loss, 
                "kld": vae_gaussian_kl_loss,
                "kld modified": vae_gaussian_kl_loss_modified,
                "kld modified image": vae_gaussian_kl_loss_modified_image,
                "vae loss modified": get_vae_loss_modified,
                "vae loss modified image": get_vae_loss_modified_image
            },
            "ood": {
                "ood auroc": get_ood_auroc
            }
        }    
        self.selected_metric_functions = {}
        # For loop through the selected metrics
        for metric in self.metric_list:
            found = False
            # For loop throught all categories of metrics
            for metric_category, metric_category_dict in self.metric_functions.items():
                # Find out if this metric is in this category of metrics
                if metric in metric_category_dict:
                    # If it is, add it
                    metric_func = metric_category_dict[metric]
                    # If there are params for func, add to the func
                    if metric in metric_params_dict:
                        metric_func = partial(metric_func, **metric_params_dict[metric])
                    if metric_category not in self.selected_metric_functions:
                        self.selected_metric_functions[metric_category] = {metric: metric_func}
                    else:
                        self.selected_metric_functions[metric_category][metric] = metric_func
                    found = True
                    break
            if not found:
                # If this metric cannot be found, it is not valid
                raise Exception(f"{metric} is not a valid metric!")

    def calculate_metric_dict(
        self, y_true=None, y_logits=None, 
        x_true=None, x_logits=None, 
        samplewise_recon_loss=None,
        z_mean=None, z_log_var=None,
        x_true_ood=None, x_logits_ood=None
    ):
        metric_dict = dict()
        for metric_category, metric_functions in self.selected_metric_functions.items():
            for metric_name, metric_function in metric_functions.items():
                if metric_category == "classifier":
                    metric_dict[metric_name] = metric_function(y_true, y_logits)
                elif metric_category == "rue":
                    metric_dict[metric_name] = metric_function(y_true, y_logits, x_true, x_logits)
                elif metric_category == "regressor":
                    metric_dict[metric_name] = metric_function(y_true, y_logits)
                elif (samplewise_recon_loss is not None) and metric_category == "rnet":
                    if metric_name == "total loss":
                        metric_dict[metric_name] = metric_function(
                            y_true, y_logits, samplewise_recon_loss, self.beta)
                    else:    
                        metric_dict[metric_name] = metric_function(samplewise_recon_loss, y_true)
                elif metric_category == "vae":
                    metric_dict[metric_name] = metric_function(x_true, x_logits, z_mean, z_log_var)
                elif (x_true_ood is not None) and (x_logits_ood is not None) and metric_category == "ood":
                    metric_dict[metric_name] = metric_function(x_true, x_logits, x_true_ood, x_logits_ood)
        return metric_dict
    
class ItemStorage:
    def __init__(self):
        self.storage = []
        self.output = None
    
    def add_batch_items(self, item_list):
        item_list = remove_each_element_from_cpu(item_list)
        if len(self.storage) == 0:
            # Make a [[item1], [item2]]
            for item in item_list:
                self.storage.append([item])
        else:
            assert len(self.storage) == len(item_list)
            # Make a [[item1a, item1b], [item2a, item2b]]
            for i, item in enumerate(item_list):
                self.storage[i].append(item)   
    
    def get_stored(self):
        if self.output is None:
            self.output = [torch.cat(item) for item in self.storage]
        return self.output
        
def remove_each_element_from_cpu(item_list):
    new_list = []
    for item in item_list:
        item = item.detach().cpu()
        new_list.append(item)
        del item
    return new_list

from os.path import join, dirname, exists
from pathlib import Path
from os import makedirs
from src.display.display_image import transform_for_visualisation
from PIL import Image

class ImageStorage:
    def __init__(self, fp_history):
        self.fp_storage_folder = join(dirname(fp_history), Path(fp_history).stem+"_images")
        if not exists(self.fp_storage_folder):
            makedirs(self.fp_storage_folder)
            
    def store_image(self, image, reverse_transform=False):
        transformed = transform_for_visualisation(image, reverse_transform=reverse_transform)
        fp_image = join(self.fp_storage_folder, "image.jpg")
        im = Image.fromarray(transformed)
        im.save(fp_image)
        
    def store_reconstruction(self, reconstruction, epoch, reverse_transform=False):
        transformed = transform_for_visualisation(reconstruction, reverse_transform=reverse_transform)
        fp_image = join(self.fp_storage_folder, f"epoch_{epoch}.jpg")
        im = Image.fromarray(transformed)
        im.save(fp_image)