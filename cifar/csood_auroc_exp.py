# -*- coding: utf-8 -*-
"""
csOOD AUROC experiment: compare prototype-based csOOD scores
between UniEnt+ (static prototypes) and RUniEnt2 (dynamic prototypes).
Validates Fix C effectiveness directly.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn import metrics
from robustbench.data import load_cifar10c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from data import load_svhn_c
import tent
import runient2 as r2

DATA_DIR = './data'
CKPT_DIR = './ckpt'
ARCH     = 'Hendrycks2020AugMix_WRN'
DATASET  = 'cifar10'
BS       = 100
N_EX     = 10000

CORRUPTIONS = [
    "gaussian_noise", "defocus_blur", "snow",
    "brightness", "contrast", "jpeg_compression"
]

def setup_optimizer(params):
    return optim.Adam(params, lr=1e-3)

def get_csood_auroc(model, x_ind, x_ood, batch_size=100):
    """Compute AUROC using the model's csOOD score (1 - max_cos_sim)."""
    device = next(model.parameters()).device

    # Get reference model (frozen, head replaced with Identity)
    if hasattr(model, '_ref_model'):
        ref_model = model._ref_model
        prototype = model.prototype
    else:
        # UniEnt+: use static source weights
        ref = type(model.model)(num_classes=10) if False else None
        # Build ref model from model.model0
        import copy
        ref_model = copy.deepcopy(model.model0)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        if hasattr(ref_model, 'fc') and isinstance(ref_model.fc, nn.Linear):
            prototype = ref_model.fc.weight.data.clone()
            ref_model.fc = nn.Identity()
        elif hasattr(ref_model, 'classifier') and isinstance(ref_model.classifier, nn.Linear):
            prototype = ref_model.classifier.weight.data.clone()
            ref_model.classifier = nn.Identity()

    def csood_score(x):
        with torch.no_grad():
            feat = ref_model(x.to(device))
            feat = F.normalize(feat, dim=1)
            proto = F.normalize(prototype, dim=1)
            cos_sim = torch.matmul(feat, proto.T)
            max_sim = cos_sim.max(dim=1)[0]
            return (1.0 - max_sim).clamp(0, 1).cpu().numpy()

    scores_ind, scores_ood = [], []
    n = math.ceil(x_ind.shape[0] / batch_size)
    for i in range(n):
        xb_ind = x_ind[i*batch_size:(i+1)*batch_size]
        xb_ood = x_ood[i*batch_size:(i+1)*batch_size]
        scores_ind.append(csood_score(xb_ind))
        scores_ood.append(csood_score(xb_ood))

    scores_ind = np.concatenate(scores_ind)
    scores_ood = np.concatenate(scores_ood)
    y_true  = np.concatenate([np.zeros(len(scores_ind)), np.ones(len(scores_ood))])
    y_score = np.concatenate([scores_ind, scores_ood])
    auroc = metrics.roc_auc_score(y_true, y_score)
    return auroc, scores_ind.mean(), scores_ood.mean()

def run_and_collect(model_name, corruption):
    base = load_model(ARCH, CKPT_DIR, DATASET, ThreatModel.corruptions).cuda()
    if model_name == 'unient':
        base = tent.configure_model(base)
        params, _ = tent.collect_params(base)
        opt = setup_optimizer(params)
        model = tent.Tent(base, opt, steps=1, episodic=False,
                          alpha=[0.5, 0.5], criterion='ent_ind_ood')
    else:
        base = tent.configure_model(base)
        params, _ = tent.collect_params(base)
        opt = setup_optimizer(params)
        model = r2.RUniEnt2(base, opt, steps=1, episodic=False,
                             use_ema=True, use_ood_marg=True, use_proto=True)

    x_ind, y_ind = load_cifar10c(N_EX, 5, DATA_DIR, False, [corruption])
    x_ood, _     = load_svhn_c(N_EX, 5, DATA_DIR, False, [corruption])
    x_ind, y_ind, x_ood = x_ind.cuda(), y_ind.cuda(), x_ood.cuda()

    # Run adaptation (forward pass updates model)
    n = math.ceil(x_ind.shape[0] / BS)
    for i in range(n):
        xb = torch.cat([x_ind[i*BS:(i+1)*BS], x_ood[i*BS:(i+1)*BS]], dim=0)
        _ = model(xb)

    # Measure csOOD AUROC after adaptation
    auroc, mu_ind, mu_ood = get_csood_auroc(model, x_ind, x_ood, BS)
    return auroc, mu_ind, mu_ood

def main():
    print(f"{'Corruption':<22} {'UniEnt+ AUROC':<16} {'RUniEnt2 AUROC':<16} {'Delta':<8}")
    print('-' * 66)
    aurocs_u, aurocs_r = [], []
    for corr in CORRUPTIONS:
        au, _, _ = run_and_collect('unient',   corr)
        ar, _, _ = run_and_collect('runient2', corr)
        aurocs_u.append(au)
        aurocs_r.append(ar)
        print(f"{corr:<22} {au:<16.4f} {ar:<16.4f} {ar-au:+.4f}")
    print('-' * 66)
    print(f"{'Mean':<22} {np.mean(aurocs_u):<16.4f} {np.mean(aurocs_r):<16.4f} {np.mean(aurocs_r)-np.mean(aurocs_u):+.4f}")

if __name__ == '__main__':
    main()
