from src.configs.default_configs import device
from src.models.resnet.model import ResNet18
from src.training.callbacks import torch


import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def predict_w_resnet(model: ResNet18, dl: DataLoader, verbose=True, active_dropout=False):
    all_outputs =  None
    model.eval()
    model.to(device)
    num_batches = len(dl)
    with torch.no_grad():
        with tqdm(dl, leave=False, position=0, total=num_batches, disable=not verbose) as pbar:
            for input_batch, output_batch in pbar:
                input_batch, output_batch = input_batch.to(device), output_batch.to(device)
                output_logits = model(input_batch, active_dropout) #(num_samples, num_outputs)
                cur_batch_output = [output_batch, output_logits]
                cur_batch_output = [element.detach().cpu() for element in cur_batch_output]
                num_elements = len(cur_batch_output)
                # If this is the first batch
                if all_outputs is None:
                    all_outputs = cur_batch_output
                else:
                    for i in range(num_elements):
                        # all_outputs[i] = (num_samples, element_size)
                        # cur_batch_output[i] = (batch_size, element_size)
                        all_outputs[i] = torch.cat((all_outputs[i], cur_batch_output[i]), dim=0)
    return all_outputs #(, , , )

def predict_w_resnet_dropout_speedup(
    model: ResNet18, dl: DataLoader, T, verbose=True):
    all_outputs =  None
    model.eval()
    model.to(device)
    num_batches = len(dl)
    all_outputs = []
    with torch.no_grad():
        with tqdm(dl, leave=False, position=0, total=num_batches, disable=not verbose) as pbar:
            for input_batch, output_batch in pbar:
                input_batch = input_batch.to(device)
                output_logits = model.sample_predictions_dropout(input_batch, T) #(num_samples, num_outputs)
                all_outputs.append(output_logits) # T, batch, pred
    return torch.concatenate(all_outputs, axis=1) # T, num_samples, num_pred
