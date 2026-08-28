import torch
import pickle
from src.models.postnet.results_manager.metrics_prior import accuracy, confidence, brier_score, anomaly_detection
import time
from tqdm.auto import tqdm

from src.configs.default_configs import device
from src.evaluation.timer import Timer

use_cuda = torch.cuda.is_available()
torch.backends.cudnn.benchmark = True


def compute_X_Y_alpha(model, loader, alpha_only=False):
    for batch_index, (X, Y) in tqdm(enumerate(loader), total=len(loader)):
        X, Y = X.to(device), Y.to(device)
        alpha_pred = model(X, None, return_output='alpha', compute_loss=False)
        if batch_index == 0:
            X_duplicate_all = X.to("cpu")
            orig_Y_all = Y.to("cpu")
            alpha_pred_all = alpha_pred.to("cpu")
        else:
            X_duplicate_all = torch.cat([X_duplicate_all, X.to("cpu")], dim=0)
            orig_Y_all = torch.cat([orig_Y_all, Y.to("cpu")], dim=0)
            alpha_pred_all = torch.cat([alpha_pred_all, alpha_pred.to("cpu")], dim=0)
    if alpha_only:
        return alpha_pred_all
    else:
        return orig_Y_all, X_duplicate_all, alpha_pred_all
    
def compute_alpha(model, loader, alpha_only=False):
    alpha_pred_all, Y_all = [], []
    for batch_index, (X, Y) in tqdm(enumerate(loader), total=len(loader)):
        X, Y = X.type(torch.float64).to(device), Y.to(device)
        alpha_pred = model(X, None, return_output='alpha', compute_loss=False)
        alpha_pred_all.append(alpha_pred.to("cpu"))
        Y_all.append(Y.to("cpu"))
    return torch.cat(alpha_pred_all, dim=0), torch.cat(Y_all, dim=0)


def test(model, test_loader, ood_dataset_loaders, result_path='saved_results'):
    model.to(device)
    model.eval()

    with torch.no_grad():
        start = time.time()
        orig_Y_all, X_duplicate_all, alpha_pred_all = compute_X_Y_alpha(model, test_loader)
        print(f"Prediction with PostNet took {time.time()-start}s")
        
        # Save each data result
        n_test_samples = orig_Y_all.size(0)
        full_results_dict = {'Y': orig_Y_all.cpu().detach().numpy(),
                            #  'X': X_duplicate_all.view(n_test_samples, -1).cpu().detach().numpy(),
                             'alpha': alpha_pred_all.cpu().detach().numpy()}
        with open(f'{result_path}.pickle', 'wb') as handle:
            pickle.dump(full_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Save metrics
        metrics = {}
        metrics['accuracy'] = accuracy(Y=orig_Y_all, alpha=alpha_pred_all)
        metrics['confidence_aleatoric'] = confidence(Y= orig_Y_all, alpha=alpha_pred_all, score_type='APR', uncertainty_type='aleatoric')
        metrics['confidence_epistemic'] = confidence(Y= orig_Y_all, alpha=alpha_pred_all, score_type='APR', uncertainty_type='epistemic')
        metrics['brier_score'] = brier_score(Y= orig_Y_all, alpha=alpha_pred_all)
        for ood_dataset_name, ood_loader in ood_dataset_loaders.items():
            ood_alpha_pred_all = compute_X_Y_alpha(model, ood_loader, alpha_only=True)
            metrics[f'anomaly_detection_aleatoric_{ood_dataset_name}'] = anomaly_detection(alpha=alpha_pred_all, ood_alpha=ood_alpha_pred_all, score_type='APR', uncertainty_type='aleatoric')
            metrics[f'anomaly_detection_epistemic_{ood_dataset_name}'] = anomaly_detection(alpha=alpha_pred_all, ood_alpha=ood_alpha_pred_all, score_type='APR', uncertainty_type='epistemic')

    return metrics

def test_on_dataset(model, dataloader, fp_result):
    model.to(device)
    model.eval()

    with torch.no_grad():
        timer = Timer("PN")
        alpha_pred_all, Y_all = compute_alpha(model, dataloader)
        timer.end()
        
        # Save each data result
        full_results_dict = {
            'alpha': alpha_pred_all.cpu().detach().numpy(), "Y": Y_all.cpu().detach().numpy()}
        with open(fp_result, 'wb') as handle:
            pickle.dump(full_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


