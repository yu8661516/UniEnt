# -*- coding: utf-8 -*-
"""
Measure per-batch inference time for Source, UniEnt+, RUniEnt2
at batch sizes 32, 64, 100, 200.
"""
import time
import math
import torch
import torch.optim as optim
import numpy as np
from robustbench.data import load_cifar10c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
import tent
import runient2 as r2

DATA_DIR  = './data'
CKPT_DIR  = './ckpt'
ARCH      = 'Hendrycks2020AugMix_WRN'
DATASET   = 'cifar10'
N_WARMUP  = 5
N_MEASURE = 20

def setup_optimizer(params, lr=1e-3):
    return optim.Adam(params, lr=lr)

def load_data(n=5000):
    x, y = load_cifar10c(n, 5, DATA_DIR, False, ['gaussian_noise'])
    return x.cuda(), y.cuda()

def time_method(method_name, batch_size, x):
    base = load_model(ARCH, CKPT_DIR, DATASET, ThreatModel.corruptions).cuda()
    if method_name == 'source':
        base.eval()
        model = base
        use_grad = False
    elif method_name == 'unient':
        base = tent.configure_model(base)
        params, _ = tent.collect_params(base)
        opt = setup_optimizer(params)
        model = tent.Tent(base, opt, steps=1, episodic=False,
                          alpha=[0.5, 0.5], criterion='ent_ind_ood')
        use_grad = True
    elif method_name == 'runient2':
        base = tent.configure_model(base)
        params, _ = tent.collect_params(base)
        opt = setup_optimizer(params)
        model = r2.RUniEnt2(base, opt, steps=1, episodic=False,
                             use_ema=True, use_ood_marg=True, use_proto=True)
        use_grad = True

    n_batches = math.ceil(x.shape[0] / batch_size)
    times = []
    for i in range(min(N_WARMUP + N_MEASURE, n_batches)):
        xb = x[i*batch_size:(i+1)*batch_size]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if use_grad:
            out = model(xb)
        else:
            with torch.no_grad():
                out = model(xb)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if i >= N_WARMUP:
            times.append((t1 - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))

def main():
    x, _ = load_data(n=5000)
    print(f"{'Method':<12} {'BS':<6} {'Mean(ms)':<12} {'Std(ms)':<10}")
    print('-' * 44)
    for bs in [32, 64, 100, 200]:
        for method in ['source', 'unient', 'runient2']:
            mean_t, std_t = time_method(method, bs, x)
            print(f"{method:<12} {bs:<6} {mean_t:<12.2f} {std_t:<10.2f}")

if __name__ == '__main__':
    main()
