import numpy as np
import torch
from torch.nn.functional import cross_entropy, softmax,  l1_loss,  mse_loss
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import pearsonr
from torch.nn.functional import binary_cross_entropy_with_logits
from scipy.stats import entropy


# Classification Performance Metrics
def get_crossentropyloss_from_pred_prob(true, pred_prob):
    if len(pred_prob.shape)<2:
        return np.mean(-(
            np.log(pred_prob, where=true==1, out=np.zeros(len(pred_prob))) +
            np.log(1-pred_prob, where=true==0, out=np.zeros(len(pred_prob)))
        ))
    else:
        return np.mean(
            -np.log(pred_prob[range(len(pred_prob)), true])
        )
    
def get_elementwise_crossentropyloss_from_pred_prob(true, pred_prob):
    if len(pred_prob.shape)<2:
        return -(
            np.log(pred_prob, where=true==1, out=np.zeros(len(pred_prob))) +
            np.log(1-pred_prob, where=true==0, out=np.zeros(len(pred_prob)))
        )
    else:
        true = true.astype(int)
        return -np.log(pred_prob[range(len(pred_prob)), true])

def get_crossentropyloss(y_batch, y_logits):
    if len(y_logits.shape) == 1:
        return binary_cross_entropy_with_logits(y_logits, y_batch).item()
    else:
        return cross_entropy(y_logits, y_batch.long()).item()
    
def get_incorrect_class_prob(true, pred_prob):
    if len(pred_prob.shape)<2:
        pred_prob = np.array([1-pred_prob, pred_prob]).transpose()
        return pred_prob[range(len(true)), 1-true]
    else:
        true = (true.numpy() if hasattr(true, "numpy") else true).astype(int)
        return 1-pred_prob[range(len(true)), true]

def get_accuracy(y_batch, y_logits):
    if len(y_logits.shape)>1:
            y_pred_labels = y_logits.argmax(-1)
    else:
        y_pred_labels = torch.sigmoid(y_logits).round() 
    return torch.mean((y_pred_labels==y_batch).float()).item()
    
def get_auc(y_batch, y_logits):
    y_prob = softmax(y_logits, dim=-1)
    if len(y_batch.unique()) == 2: # if binary
        if len(y_prob.shape)>1:
            y_prob = y_prob[:,1]
        return roc_auc_score(y_true=y_batch, y_score=y_prob)
    else:
        return roc_auc_score(y_true=y_batch, y_score=y_prob, multi_class="ovo")

def get_f1(y_batch, y_logits):
    if len(y_batch.unique()) == 2: # if binary
        if len(y_logits.shape)>1:
            y_pred_labels = y_logits.argmax(-1)
        else:
            y_pred_labels = torch.sigmoid(y_logits).round()
        return f1_score(y_batch, y_pred_labels)
    else:
        y_pred_labels = y_logits.argmax(-1)
        return f1_score(y_batch, y_pred_labels, average="micro")

# Reconstruction Performance Metric
def get_mae_loss(y_batch, y_logits, x_batch=None, x_logits=None):
    return l1_loss(x_logits, x_batch).detach().cpu().item()

def get_reg_mae_loss(y_batch, y_logits):
    return l1_loss(y_logits, y_batch).detach().cpu().item()

def get_mse_loss(y_batch, y_logits, x_batch=None, x_logits=None):
    return mse_loss(x_logits, x_batch).detach().cpu().item()

# Regression Performance Metric
def get_correlation(y_batch, y_logits):
    corr, _ = pearsonr(y_logits, y_batch)
    return corr

# Reconstruction Net Metrics
def get_rnet_recon_loss(samplewise_recon_loss, y_true):
    return torch.mean(samplewise_recon_loss).detach().cpu().item()

def get_rnet_recon_0_loss(samplewise_recon_loss, y_true):
    return torch.masked_select(samplewise_recon_loss, y_true==0).mean().detach().cpu().item()

def get_rnet_recon_1_loss(samplewise_recon_loss, y_true):
    return torch.masked_select(samplewise_recon_loss, y_true==1).mean().detach().cpu().item()

def get_rnet_total_loss(y_batch, y_logits, samplewise_recon_loss, beta):
    classifier_loss = get_crossentropyloss(y_batch, y_logits)
    ae_loss = get_rnet_recon_loss(samplewise_recon_loss, y_batch)
    return classifier_loss + beta * ae_loss

# VAE Metrics
def get_vae_mae_loss(x_batch, x_logits, z_mean=None, z_log_var=None):
    return l1_loss(x_logits, x_batch).detach().cpu().item()

def vae_gaussian_kl_loss(x_batch=None, x_logits=None, z_mean=None, z_log_var=None):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    KLD = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1)
    return KLD.mean()

def vae_gaussian_kl_loss_modified(x_batch=None, x_logits=None, z_mean=None, z_log_var=None):
    # removed - mu.pow(2)  since it is not trained (derived from the mlp itself)
    KLD = -0.5 * torch.sum(1 + z_log_var - z_log_var.exp(), dim=1)
    return KLD.mean()

def get_vae_loss_modified(x_batch, x_logits, z_mean=None, z_log_var=None, kld_weight=1):
    recon_loss = get_vae_mae_loss(x_batch=x_batch, x_logits=x_logits)
    kld_loss = vae_gaussian_kl_loss_modified(z_mean = z_mean, z_log_var= z_log_var)
    return recon_loss + kld_weight * kld_loss
    
def vae_gaussian_kl_loss_modified_image(x_batch=None, x_logits=None, z_mean=None, z_log_var=None):
    # removed - mu.pow(2)  since it is not trained (derived from the mlp itself)
    KLD = -0.5 * torch.mean(1 + z_log_var - z_log_var.exp(), dim=[i for i in range(1, len(z_log_var.shape))])
    return KLD.mean()

def get_vae_loss_modified_image(x_batch, x_logits, z_mean=None, z_log_var=None, kld_weight=1):
    recon_loss = get_vae_mae_loss(x_batch=x_batch, x_logits=x_logits)
    kld_loss = vae_gaussian_kl_loss_modified_image(z_mean = z_mean, z_log_var= z_log_var)
    return recon_loss + kld_weight * kld_loss
    
def get_pred_prob(y_logits):
    return torch.sigmoid(y_logits)

def get_pred_labels(y_logits):
    return torch.round(get_pred_prob(y_logits))

def get_bce_from_pred_prob(true, pred_prob):
    return -(
        np.log(pred_prob, where=true==1, out=np.zeros(len(pred_prob))) +
        np.log(1-pred_prob, where=true==0, out=np.zeros(len(pred_prob)))
    )

def get_entropy_from_pred_prob(pred_prob):
    base = 2  # work in units of bits
    if len(pred_prob.shape)<2:
        pred_prob = [1-pred_prob, pred_prob]
        return entropy(pred_prob, base=base, axis=0)
    return entropy(pred_prob, base=base, axis=1)
    
def get_rue(x_true, x_logits):
    return torch.mean(torch.abs(x_true-x_logits), axis=-1)

def get_rue_featwise(x_true, x_logits):
    return torch.abs(x_true-x_logits)
    
