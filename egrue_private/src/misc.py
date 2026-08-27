import torch
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import os

def set_seed_pytorch(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def set_seed_tf(seed):
    tf.config.experimental.enable_op_determinism()
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
def display_layer_indices(model):
    if hasattr(model, 'model'):
        model = model.model
    layer_list = list(model.children())
    df_list = []
    for i, layer in enumerate(layer_list):
        df_list.append({"Layer Index": i, "Layer": str(layer)})
    return pd.DataFrame(df_list).set_index("Layer Index")

def unsqueeze(tensor1, tensor2):
    if torch.is_tensor(tensor1) and torch.is_tensor(tensor2):
        return unsqueeze_tensor(tensor1, tensor2)
    else:
        return unsqueeze_array(np.array(tensor1), np.array(tensor2))
    
def unsqueeze_array(array1, array2):
    if array1.shape != array2.shape:
        if array2.shape[-1] == 1:
            array1 = np.expand_dims(array1, -1)
        elif array1.shape[-1] == 1:
            array2 = np.expand_dims(array2, -1)
        else:
            raise Exception("Unable to unsqueeze")
    return array1, array2

def unsqueeze_tensor(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        if tensor2.shape[-1] == 1:
            tensor1 = tensor1.unsqueeze(-1)
        elif tensor1.shape[-1] == 1:
            tensor2 = tensor2.unsqueeze(-1)
        else:
            raise Exception("Unable to unsqueeze")
    return tensor1, tensor2
    