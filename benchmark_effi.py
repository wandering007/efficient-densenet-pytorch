import numpy as np
import sys
import time
import torch
import torch.backends.cudnn as cudnn
from densenet import DenseNet
import torch.nn as nn


def measure(model, x, y):
    # synchronize gpu time and measure fp
    torch.cuda.synchronize()
    t0 = time.time()
    y_pred = model(x)
    torch.cuda.synchronize()
    elapsed_fp = time.time() - t0

    # zero gradients, synchronize time and measure
    model.zero_grad()
    t0 = time.time()
    y_pred.backward(y)
    torch.cuda.synchronize()
    elapsed_bp = time.time() - t0
    return elapsed_fp, elapsed_bp


def benchmark(model, x, y):

    # DRY RUNS
    for i in range(5):
        _, _ = measure(model, x, y)

    print('DONE WITH DRY RUNS, NOW BENCHMARKING')

    # START BENCHMARKING
    t_forward = []
    t_backward = []
    for i in range(10):
        t_fp, t_bp = measure(model, x, y)
        t_forward.append(t_fp)
        t_backward.append(t_bp)

    # free memory
    del model

    return t_forward, t_backward


use_cuda = True
multigpus = True

# set cudnn backend to benchmark config
cudnn.benchmark = True

# instantiate the models
densenet = DenseNet(efficient=False)
densnet_effi = DenseNet(efficient=True)
# build dummy variables to input and output
x = torch.randn(128, 3, 32, 32)
y = torch.randn(128, 100)
if use_cuda:
    densenet = densenet.cuda()
    densnet_effi = densnet_effi.cuda()
    x = x.cuda()
    y = y.cuda()
    if multigpus:
        densenet = nn.DataParallel(densenet, device_ids=[0, 1])
        densnet_effi = nn.DataParallel(densnet_effi, device_ids=[0, 1])
# build the dict to iterate over
architectures = {'densenet': densenet,
                 'densenet-effi': densnet_effi
                 }


# loop over architectures and measure them
for deep_net in architectures:
    print(deep_net)
    t_fp, t_bp = benchmark(architectures[deep_net], x, y)
    # print results
    print('FORWARD PASS: ', np.mean(np.asarray(t_fp) * 1e3), '+/-', np.std(np.asarray(t_fp) * 1e3))
    print('BACKWARD PASS: ', np.mean(np.asarray(t_bp) * 1e3), '+/-', np.std(np.asarray(t_bp) * 1e3))
    print('RATIO BP/FP:', np.mean(np.asarray(t_bp)) / np.mean(np.asarray(t_fp)))
