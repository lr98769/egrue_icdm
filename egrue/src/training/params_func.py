def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def freeze_layers_until(model, layer_name):
    print("Freezing Weights:")
    for name, param in model.named_parameters():
        param.requires_grad = False
        if layer_name in name:
            break
        print("-", name, "frozen!")