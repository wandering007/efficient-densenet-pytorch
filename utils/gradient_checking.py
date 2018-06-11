import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from models import DenseNet
from collections import OrderedDict

# run it with python -m utils.test_densenet.py
print('please remove dropout first')
use_cuda = True
bn_size = None
multigpus = False
is_eval = False
model = DenseNet(input_size=32, bn_size=bn_size, efficient=False)
model_effi = DenseNet(input_size=32, bn_size=bn_size, efficient=True)
# for stronger test
model.features.denseblock2.denselayer12._modules['norm1'].running_mean.fill_(1)
model.features.denseblock2.denselayer12._modules['norm1'].running_var.fill_(2)
state = model.state_dict()
state = OrderedDict((k.replace('.norm1.', '.bottleneck.norm_'), v) for k, v in state.items())
state = OrderedDict((k.replace('.conv1.', '.bottleneck.conv_'), v) for k, v in state.items())

model_effi.load_state_dict(state)
if use_cuda:
    model.cuda()
    model_effi.cuda()
    cudnn.deterministic = True
    if multigpus:
        model = nn.DataParallel(model, device_ids=[0, 1])
        model_effi = nn.DataParallel(model_effi, device_ids=[0, 1])
if is_eval:
    model.eval()
    model_effi.eval()
# create the model inputs
input_var = torch.randn(8, 3, 32, 32)
if use_cuda:
    input_var = input_var.cuda()

out = model(input_var)
model.zero_grad()
out.sum().backward()
param_grads = OrderedDict()
if multigpus:
    model = model.module
for name, param in model.named_parameters():
    assert param.grad is not None, name
    param_grads[name] = param.grad.data
out_effi = model_effi(input_var)

model_effi.zero_grad()
out_effi.sum().backward()
param_grads_effi = OrderedDict()
if multigpus:
    model_effi = model_effi.module
for name, param in model_effi.named_parameters():
    assert param.grad is not None, name
    param_grads_effi[name] = param.grad.data


def almost_equal(a, b, eps=1e-5):
    res = torch.max(torch.abs(a - b))
    if res >= eps:
        print(a, b, res)
    return res < eps


# compare the output and parameters gradients
assert torch.equal(out.data, out_effi.data)
param_grads = OrderedDict(reversed(list(param_grads.items())))
print('------gradient checking------')
for name in param_grads:
    print(name)
    name_effi = name.replace('.conv1.', '.bottleneck.conv_').replace('.norm1.', '.bottleneck.norm_')
    assert almost_equal(param_grads[name], param_grads_effi[name_effi])

print('----weight & buffer checking-----')
d1 = model.state_dict()
d2 = model_effi.state_dict()
for key in d1:
    print(key)
    key_effi = key.replace('.conv1.', '.bottleneck.conv_').replace('.norm1.', '.bottleneck.norm_')
    assert torch.equal(d1[key], d2[key_effi])

print('succeed.')
