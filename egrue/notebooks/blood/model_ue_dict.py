from src.models.resnet.model import ResNet18
from src.models.resnet_rue.model import RueResNet18
from src.models.resnet_egrue.model import egRUEResNet18
from src.models.postnet.posterior_networks.PosteriorNetwork import PosteriorNetwork
from src.models.bnn.model import BNN
from src.models.dec.model import DEC
from src.models.de.model import DeepEnsemble

ModelClass_dict = {
    ResNet18:[], DeepEnsemble: [], RueResNet18:[], egRUEResNet18:[], BNN: [], DEC: [], PosteriorNetwork: []}
# 
ue_dict = {
    "ResNet18 Entropy": {"ue": f"entropy_resnet", "model": "resnet"},
    "ResNet18 MC Dropout" : {"ue": f"pred_std_mc", "model": "resnet_mc"},
    "Deep Ensemble": {"ue": "pred_std_de", "model": "de"},
    "ResNet18 RUE": {"ue": "rue", "model": "resnet"},
    "ResNet18 egRUE": {"ue": "egrue", "model": "resnet"},
    "BNN": {"ue": "bnn_uncertainty", "model": "bnn"},
    "DEC": {"ue": "dec_uncertainty", "model": "dec"},
    "PostNet Aleatoric": {"ue": "aleatoric_ue_pn", "model": "pn"},
    "PostNet Epistemic": {"ue": "epistemic_ue_pn", "model": "pn"},
}