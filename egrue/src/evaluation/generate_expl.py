import matplotlib.image as mpimg
from captum.attr import GradientShap
from torch.utils.data import DataLoader
from torch.nn import Sequential
from torchvision.transforms.v2.functional import grayscale_to_rgb
import numpy as np
import cv2 as cv
from os.path import join, exists
import matplotlib.pyplot as plt
from io import StringIO

from src.misc import set_seed_pytorch
from src.configs.default_configs import fn_expl, device
from src.file_manager.filepath import create_folder
from src.file_manager.load_save_model import load_model
from src.models.resnet_rue.model import RueResNet18
from src.evaluation.inference import get_imp_weights

def get_incorrect_test_predictions(pred_df, ue_dict, ue="ResNet18 egRUE", split="Test"):
    model_name = ue_dict[ue]["model"]
    ue_col = ue_dict[ue]["ue"]
    incorrect_df = pred_df[~pred_df[f"correct_{model_name}"]]
    incorrect_test_df = incorrect_df[incorrect_df["split"]==split]
    incorrect_test_df = incorrect_test_df.sort_values(by=ue_col, ascending=False)
    return incorrect_test_df

def min_max_norm(img):
    return (img-img.min())/(img.max()-img.min())

# def process_expl(img):
#     img = grayscale_to_rgb(min_max_norm(img)).detach().cpu()[0].permute((1, 2, 0)).numpy()
#     img = (img * 255.0).round().astype(np.uint8)
#     return cv.equalizeHist(cv.cvtColor(img, cv.COLOR_BGR2GRAY))

def process_expl(img, gamma=1): # (3, 224, 224)
    # Drop any negative explanations
    img[img<0] = 0
    # min-max scale your explanations
    min_pixel, max_pixel = img.min(), img.max()
    img = (img-min_pixel)/(max_pixel-min_pixel)
    # gamma correction 
    img = np.power(img, gamma)
    # min-max scale your explanations
    min_pixel, max_pixel = img.min(), img.max()
    img = (img-min_pixel)/(max_pixel-min_pixel)
    return img


def save_expl(img, expl, expl_cmap, fp):
    plt.figure(figsize=(6, 6), dpi=300)
    plt.imshow(img)
    plt.imshow(expl, cmap=expl_cmap, vmin=expl.min(), vmax=expl.max())
    plt.axis('off')
    plt.savefig(fp, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def get_explanation(
        data_dict, split, index, fp, seed, 
        gamma=1, color="green", light_color_alpha=0, thres=0.15, alpha=1,
        num_baselines=100, override=False
    ):
    fp_img_folder = join(fp.get_parent_folder(fn_expl), f"{split}_{index}")
    create_folder(fp_img_folder)
    fp_img = join(fp_img_folder, "img.jpg")
    fp_eg = join(fp_img_folder, "eg.jpg")
    fp_rue = join(fp_img_folder, "rue.jpg")
    fp_egrue = join(fp_img_folder, "egrue.jpg")

    if override or (not exists(fp_egrue)):
        # Get model 
        model = load_model(fp=fp, ModelClass=RueResNet18, cur_model_name="tuned")
        
        # Get IMG
        img, _ = data_dict[split][index]
        img = img.to(device).unsqueeze(dim=0)

        # Get baseline instances
        set_seed_pytorch(seed)
        dataloader = DataLoader(data_dict["train_df"], batch_size=num_baselines, shuffle=True)
        batch = next(iter(dataloader))
        baselines, _ = batch # Get a single batch of 100
        baselines = baselines.to(device)

        # Get grad shap explainer
        pred_model = Sequential(model.encoder, model.classifier)
        grad_shap = GradientShap(pred_model)

        # RUE
        pred, recon = model(img)
        recon_error = (recon-img).abs()
        rue = recon_error.mean(axis=1, keepdims=True).detach().cpu().numpy()[0][0] # average along the rgb

        # Get EG
        target = pred.argmax(axis=-1) 
        attr = grad_shap.attribute(inputs=img, baselines=baselines, target=target)
        eg = attr.mean(axis=1, keepdims=True).clip(min=0).cpu().numpy()[0][0] # average along the rgb

        # Get egRUE
        image_dim = [i for i in range(1, len(img.shape))]
        imp_weights = get_imp_weights(attr, image_dim=image_dim)
        egrue = (recon_error*imp_weights).mean(axis=1, keepdims=True).detach().cpu().numpy()[0][0] # average along the rgb

        # Output
        # -img
        img = (img.detach().cpu()[0].permute((1, 2, 0))*255).numpy().round().astype(np.uint8)
        # -eg
        eg = process_expl(eg, gamma=gamma)
        # -rue
        rue = process_expl(rue, gamma=gamma)
        # -egrue
        egrue = process_expl(egrue, gamma=gamma)

        cmap_threshold = get_cmap(color, light_color_alpha, thres)
        plt.imsave(fp_img, img)
        save_expl(img, eg, expl_cmap=cmap_threshold, fp=fp_eg)
        save_expl(img, rue, expl_cmap=cmap_threshold, fp=fp_rue)
        save_expl(img, egrue, expl_cmap=cmap_threshold, fp=fp_egrue)

    output_dict = {
        "Image": mpimg.imread(fp_img), 
        "EG": mpimg.imread(fp_eg), 
        "RUE": mpimg.imread(fp_rue), 
        "egRUE": mpimg.imread(fp_egrue)
    }

    return output_dict # Should be plottable

def process_expl_heatmap(img, gamma=1, c=1): # (3, 224, 224)
    # Drop any negative explanations
    img[img<0] = 0
    # min-max scale your explanations
    # min_pixel, max_pixel = img.min(), img.max()
    # img = (img-min_pixel)/(max_pixel-min_pixel)
    # gamma correction 
    img = c*np.power(img, gamma)
    # min-max scale your explanations
    # min_pixel, max_pixel = img.min(), img.max()
    # img = (img-min_pixel)/(max_pixel-min_pixel)
    return img

def save_expl_heatmap(img, expl, expl_cmap, fp, alpha=1):
    plt.figure(figsize=(6, 6), dpi=300)
    plt.imshow(img)
    print(expl.min(), expl.max())
    plt.imshow(expl, cmap=expl_cmap, vmin=0, vmax=1, alpha=alpha)
    plt.axis('off')
    plt.savefig(fp, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def get_only_explanations(
    data_dict, split, index, 
    fp, seed, 
    gamma=1, num_baselines=100
):
    c=1
    
    # Get model 
    model = load_model(fp=fp, ModelClass=RueResNet18, cur_model_name="tuned")
    
    # Get IMG
    img, _ = data_dict[split][index]
    img = img.to(device).unsqueeze(dim=0)

    # Get baseline instances
    set_seed_pytorch(seed)
    dataloader = DataLoader(data_dict["train_df"], batch_size=num_baselines, shuffle=True)
    batch = next(iter(dataloader))
    baselines, _ = batch # Get a single batch of 100
    baselines = baselines.to(device)

    # Get grad shap explainer
    pred_model = Sequential(model.encoder, model.classifier)
    grad_shap = GradientShap(pred_model)

    # RUE
    pred, recon = model(img)
    recon_error = (recon-img).abs()
    rue = recon_error.mean(axis=1, keepdims=True).detach().cpu().numpy()[0][0] # average along the rgb

    # Get EG
    target = pred.argmax(axis=-1) 
    attr = grad_shap.attribute(inputs=img, baselines=baselines, target=target)
    eg = attr.mean(axis=1, keepdims=True).clip(min=0).cpu().numpy()[0][0] # average along the rgb

    # Get egRUE
    image_dim = [i for i in range(1, len(img.shape))]
    imp_weights = get_imp_weights(attr, image_dim=image_dim)
    egrue = (recon_error*imp_weights).mean(axis=1, keepdims=True).detach().cpu().numpy()[0][0] # average along the rgb

    # Output
    # -img
    img = (img.detach().cpu()[0].permute((1, 2, 0))*255).numpy().round().astype(np.uint8)
    # -eg
    eg = process_expl_heatmap(eg, c=c, gamma=gamma)
    # -rue
    rue = process_expl_heatmap(rue, c=c, gamma=gamma)
    # -egrue
    egrue = process_expl_heatmap(egrue, c=c, gamma=gamma)

    return {
        "Image": img, "EG": eg, "RUE": rue, "egRUE": egrue
    }





def get_explanation_heatmap(
        data_dict, split, index, fp, seed, 
        c=1, gamma=1, alpha=1,
        num_baselines=100, override=False
    ):
    fp_img_folder = join(fp.get_parent_folder(fn_expl), f"{split}_{index}")
    create_folder(fp_img_folder)
    fp_img = join(fp_img_folder, "img.jpg")
    fp_eg = join(fp_img_folder, "eg.jpg")
    fp_rue = join(fp_img_folder, "rue.jpg")
    fp_egrue = join(fp_img_folder, "egrue.jpg")

    if override or (not exists(fp_egrue)):
        # Get model 
        model = load_model(fp=fp, ModelClass=RueResNet18, cur_model_name="tuned")
        
        # Get IMG
        img, _ = data_dict[split][index]
        img = img.to(device).unsqueeze(dim=0)

        # Get baseline instances
        set_seed_pytorch(seed)
        dataloader = DataLoader(data_dict["train_df"], batch_size=num_baselines, shuffle=True)
        batch = next(iter(dataloader))
        baselines, _ = batch # Get a single batch of 100
        baselines = baselines.to(device)

        # Get grad shap explainer
        pred_model = Sequential(model.encoder, model.classifier)
        grad_shap = GradientShap(pred_model)

        # RUE
        pred, recon = model(img)
        recon_error = (recon-img).abs()
        rue = recon_error.mean(axis=1, keepdims=True).detach().cpu().numpy()[0][0] # average along the rgb

        # Get EG
        target = pred.argmax(axis=-1) 
        attr = grad_shap.attribute(inputs=img, baselines=baselines, target=target)
        eg = attr.mean(axis=1, keepdims=True).clip(min=0).cpu().numpy()[0][0] # average along the rgb

        # Get egRUE
        image_dim = [i for i in range(1, len(img.shape))]
        imp_weights = get_imp_weights(attr, image_dim=image_dim)
        egrue = (recon_error*imp_weights).mean(axis=1, keepdims=True).detach().cpu().numpy()[0][0] # average along the rgb

        # Output
        # -img
        img = (img.detach().cpu()[0].permute((1, 2, 0))*255).numpy().round().astype(np.uint8)
        # -eg
        eg = process_expl_heatmap(eg, c=c, gamma=gamma)
        # -rue
        rue = process_expl_heatmap(rue, c=c, gamma=gamma)
        # -egrue
        egrue = process_expl_heatmap(egrue, c=c, gamma=gamma)

        cmap_threshold="hot"# "RdBu_r"
        plt.imsave(fp_img, img)
        save_expl_heatmap(img, eg, expl_cmap=cmap_threshold, alpha=alpha, fp=fp_eg)
        save_expl_heatmap(img, rue, expl_cmap=cmap_threshold, alpha=alpha, fp=fp_rue)
        save_expl_heatmap(img, egrue, expl_cmap=cmap_threshold, alpha=alpha, fp=fp_egrue)

    output_dict = {
        "Image": mpimg.imread(fp_img), 
        "EG": mpimg.imread(fp_eg), 
        "RUE": mpimg.imread(fp_rue), 
        "egRUE": mpimg.imread(fp_egrue)
    }

    return output_dict # Should be plottable

def eg_expl_func(img, target, pred_model, baselines):
    from captum.attr import GradientShap
    eg = GradientShap(pred_model)
    attr = eg.attribute(img, baselines=baselines, target=target)
    return attr

def ig_expl_func(img, target, pred_model, baselines):
    from captum.attr import IntegratedGradients
    ig = IntegratedGradients(pred_model)
    attr = ig.attribute(img, baselines=None, target=target)
    return attr

def saliency_expl_func(img, target, pred_model, baselines):
    from captum.attr import Saliency
    saliency = Saliency(pred_model)
    attr = saliency.attribute(img, target=target)
    return attr

def gc_expl_func(img, target, pred_model, baselines):
    from captum.attr import GuidedGradCam
    gradcam = GuidedGradCam(pred_model, layer=pred_model[0][-1][-1][-1].conv2)
    attr = gradcam.attribute(img, target=target)
    return attr

def gbp_expl_func(img, target, pred_model, baselines):
    from captum.attr import GuidedBackprop
    gbp = GuidedBackprop(pred_model)
    attr = gbp.attribute(img, target=target)
    return attr

def get_explanation_other(
    expl_name, expl_func, data_dict, split, index, fp, seed, 
    gamma=1, color="green", light_color_alpha=0, thres=0.15,
    num_baselines=100, override=False
):
    fp_img_folder = join(fp.get_parent_folder(fn_expl), f"{split}_{index}")
    fp_img = join(fp_img_folder, "img.jpg")
    fp_eg = join(fp_img_folder, f"{expl_name.lower()}.jpg")
    fp_rue = join(fp_img_folder, "rue.jpg")
    fp_explrue = join(fp_img_folder, f"{expl_name.lower()}rue.jpg")

    if override or (not exists(fp_explrue)):
        # If no, generate it
        create_folder(fp_img_folder)
        # Get model 
        model = load_model(fp=fp, ModelClass=RueResNet18, cur_model_name="tuned")
        
        # Get IMG
        img, _ = data_dict[split][index]
        img = img.to(device).unsqueeze(dim=0)

        # Get baseline instances
        set_seed_pytorch(seed)
        dataloader = DataLoader(data_dict["train_df"], batch_size=num_baselines, shuffle=True)
        batch = next(iter(dataloader))
        baselines, _ = batch # Get a single batch of 100
        baselines = baselines.to(device)

        # Get grad shap explainer
        pred_model = Sequential(model.encoder, model.classifier)

        # RUE
        pred, recon = model(img)
        recon_error = (recon-img).abs()
        rue = recon_error.mean(axis=1, keepdims=True).detach().cpu().numpy()[0][0] # average along the rgb

        # Get EG
        target = pred.argmax(axis=-1) 
        attr = expl_func(img, target, pred_model, baselines)
        expl = attr.mean(axis=1, keepdims=True).clip(min=0).detach().cpu().numpy()[0][0] # average along the rgb

        # Get egRUE
        image_dim = [i for i in range(1, len(img.shape))]
        imp_weights = get_imp_weights(attr, image_dim=image_dim)
        explrue = (recon_error*imp_weights).mean(axis=1, keepdims=True).detach().cpu().numpy()[0][0] # average along the rgb

        # Output
        # -img
        img = (img.detach().cpu()[0].permute((1, 2, 0))*255).numpy().round().astype(np.uint8)
        # -eg
        expl = process_expl(expl, gamma=gamma)
        # -rue
        rue = process_expl(rue, gamma=gamma)
        # -egrue
        explrue = process_expl(explrue, gamma=gamma)

        # Save images
        cmap_threshold = get_cmap(color, light_color_alpha, thres)
        plt.imsave(fp_img, img)
        save_expl(img, expl, expl_cmap=cmap_threshold, fp=fp_eg)
        save_expl(img, rue, expl_cmap=cmap_threshold, fp=fp_rue)
        save_expl(img, explrue, expl_cmap=cmap_threshold, fp=fp_explrue)

    output_dict = {
        "Image": mpimg.imread(fp_img), 
        expl_name: mpimg.imread(fp_eg), 
        "RUE": mpimg.imread(fp_rue), 
        expl_name.lower()+"RUE": mpimg.imread(fp_explrue)
    }

    return output_dict # Should be plottable

def get_cmap(color, light_color_alpha, thres):
    from matplotlib.colors import LinearSegmentedColormap
    # CMAP with Threshold
    if color=="red":
        darker_color = (1, 0, 0, 1)
    elif color=="green":
        darker_color = (0, 1, 0, 1)
    elif color=="blue":
        darker_color = (0, 0, 1, 1)
    elif color=="cyan":
        darker_color = (66/255, 249/255, 252/255, 1)
    else:
        raise Exception(f"Invalid colour: {color}") 
    colors = [(0, 0, 0, light_color_alpha), (0, 0, 0, light_color_alpha), darker_color, darker_color]
    nodes = [0.0, thres, thres, 1.0] # Transition at 0.3 (corresponding to data value 3)
    cmap_threshold = LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))
    return cmap_threshold

def show_explanation(img_dict, dpi=300, output_fig_ax=False, img_size = 2):
    num_rows, num_cols = 1, len(img_dict)
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(img_size*num_cols, img_size*num_rows), dpi=dpi)
    for i, (img_label, img) in enumerate(img_dict.items()):
        ax = axes[i]
        ax.imshow(img)
        ax.set_xlabel(img_label)
        ax.set_xticks([])
        ax.set_yticks([])
    if output_fig_ax:
        return fig, axes
    plt.tight_layout()
    # plt.show()

from src.data_processing.dataloader import get_pytorch_split_dict_image
import torch
from tqdm.auto import tqdm

def prediction_egrue_expl(model, dl, expl_func, baselines, squared=True, verbose=True):
    pred_model = Sequential(model.encoder, model.classifier)
    all_outputs = [] # Shape: (T, num_samples, )
    model.eval()
    model.to(device)
    num_batches = len(dl)
    with torch.no_grad():
        with tqdm(dl, leave=False, position=0, total=num_batches, disable=not verbose) as pbar:
            for inputs in pbar:
                x_batch = inputs[0].to(device)
                image_dim = [i for i in range(1, len(x_batch.shape))]
                pred, recon =  model(x_batch) # Shape: (batch, num_classes), (batch, recon...)
                target = pred.argmax(axis=-1) # Shape: (batch, )
                attr = expl_func(x_batch, target, pred_model, baselines) # (batch, recon...)
                imp_weights = get_imp_weights(attr, image_dim, squared)
                recon_errors = torch.abs(recon-x_batch) # (batch, recon...)
                egrue_expl = (recon_errors*imp_weights) #.mean(axis=1, keepdims=True)
                all_outputs.append(egrue_expl.cpu())
    return torch.concatenate(all_outputs, axis=0)  # Shape: (n_samples, recon...)

def get_all_explanations(
        expl_name, expl_func, data_dict, split, fp, seed, eval_batch_size, 
        num_baselines=100, squared=False,   
        pytorch_split_dict_func=get_pytorch_split_dict_image, rerun=False, 
    ):
    # Explanation file
    fp_expl_file = join(fp.get_parent_folder(fn_expl), f"{split}_{expl_name.lower()}.npy")

    # Check if we have cached it
    if exists(fp_expl_file) and not rerun:
        # If yes, retrieve it
        return np.load(fp_expl_file)
    else:
        # Get model 
        model = load_model(fp=fp, ModelClass=RueResNet18, cur_model_name="tuned")

        # Get dataloader
        dls = pytorch_split_dict_func(
            data_dict=data_dict, batch_size=eval_batch_size, eval_batch_size=eval_batch_size, 
            shuffle_train=False
        )
        test_dl = dls[f"{split}_dl"]
        
        # Get baseline instances
        set_seed_pytorch(seed)
        dataloader = DataLoader(data_dict["train_df"], batch_size=num_baselines, shuffle=True)
        batch = next(iter(dataloader))
        baselines, _ = batch # Get a single batch of 100
        baselines = baselines.to(device)

        # Get Uncertainty Explanation
        explrue = prediction_egrue_expl(model, dl=test_dl, expl_func=expl_func, baselines=baselines, squared=squared)

        # Save uncertainty explanation
        explrue = explrue.permute([0, 2, 3, 1]).numpy()
        np.save(fp_expl_file, explrue)

        model.cpu()
        del model

    return explrue


