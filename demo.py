from __future__ import print_function
import configargparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchnet as tnt
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from tqdm import tqdm
from models import DenseNet
from datasets import ImageNet, CIFAR10, CIFAR100
import os
import copy
import math
import random
import numpy as np

parser = configargparse.ArgParser(default_config_files=[])
parser.add('--config', required=True, is_config_file=True, help='config file path')
parser.add('--batch-size', type=int, default=256, metavar='N',
           help='input batch size for training (default: 256)')
parser.add('--num-batch-splits', type=int, default=1, metavar='split',
           help='split batch size for training (default: 1)')
parser.add('--dataset', type=str, required=True, metavar='dataset',
           help="dataset name: ImageNet | CIFAR10 | CIFAR100 (default: '')")
parser.add('--data', type=str, default='datasets', metavar='data_root_path',
           help="data root: /path/to/dataset (default: 'datasets')")
parser.add('--test-batch-size', type=int, default=1024, metavar='N',
           help='input batch size for testing (default: 1024)')
parser.add('--bn-size', default=None, type=int,
           metavar='bn_size', help='bottleneck size')
parser.add('--num-init-features', type=int, default=None,
           metavar='num_init_features', help='num_init_features')
parser.add('--compression', type=float, default=1.,
           metavar='compression', help='compression at transition')
parser.add('--block-config', type=int, default=None, nargs='+', metavar='model_config',
           help='model block config')
parser.add('--epochs', type=int, default=90, metavar='N',
           help='number of epochs to train (default: 90)')
parser.add('--lr', type=float, default=0.1, metavar='LR',
           help='learning rate (default: 0.1)')
parser.add('--lr-type', default='multistep', type=str, metavar='T',
           help='learning rate strategy (default: multistep)',
           choices=['multistep', 'cosine', 'triangle'])
parser.add('--momentum', type=float, default=0.9, metavar='M',
           help='SGD momentum (default: 0.9)')
parser.add('--clip', type=float, default=4,
           help='gradient clipping')
parser.add('--weight-decay', '--wd', default=1e-4, type=float,
           metavar='W', help='weight decay (default: 1e-4)')
parser.add('--gpus', type=int, default=None, nargs='*', metavar='--gpus 0 1 2 ...',
           help='gpu ids for CUDA training')
parser.add('--seed', type=int, default=1, metavar='S',
           help='random seed (default: 1)')
parser.add('--resume', default='', type=str, metavar='PATH',
           help='path to latest checkpoint (default: none)')
parser.add('--start-epoch', default=0, type=int, metavar='N',
           help='manual epoch number (useful on restarts)')
parser.add('--checkpoints', default='checkpoints', type=str, metavar='checkpoints',
           help='checkpoints path')
parser.add('--test-only', action='store_true', default=False,
           help='only test model')
parser.add('--visdom', action='store_true', default=False,
           help='visualize the process')
parser.add('--log-name', type=str, default='', metavar='LOG_NAME',
           help='log name for clarifying')
parser.add('--save-interval', type=int, default=5,
           metavar='model_checkpoint_interval', help='model checkpoint save interval')

args = parser.parse_args()

if not args.gpus or (len(args.gpus) > 0 and (args.gpus[0] < 0 or not torch.cuda.is_available())):
    args.gpus = []

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed + 1)
random.seed(args.seed + 2)
np.random.seed(args.seed + 3)

kwargs = {'num_workers': 20, 'pin_memory': True} if len(args.gpus) > 0 else {}

train_transform = test_transform = None
if 'CIFAR' in args.dataset:
    from torchvision import transforms

    if args.dataset == 'CIFAR10':
        mean = [125.3 / 255, 123.0 / 255, 113.9 / 255]
        std = [63.0 / 255, 62.1 / 255, 66.7 / 255]
    else:
        mean = [129.3 / 255, 124.1 / 255, 112.4 / 255]
        std = [68.2 / 255, 65.4 / 255, 70.4 / 255]
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Pad(padding=4),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

train_loader = torch.utils.data.DataLoader(
    globals()[args.dataset](root=args.data, transform=train_transform, train=True),
    batch_size=args.batch_size, shuffle=True, drop_last=True, worker_init_fn=None, **kwargs)
test_loader = torch.utils.data.DataLoader(
    globals()[args.dataset](root=args.data, transform=test_transform, train=False),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)
num_classes = {"CIFAR10": 10, "CIFAR100": 100, "ImageNet": 1000}
input_size = args.dataset == 'ImageNet' and 224 or 32
model = DenseNet(num_init_features=args.num_init_features, block_config=args.block_config, compression=args.compression,
                 input_size=input_size, bn_size=args.bn_size, num_classes=num_classes[args.dataset], efficient=True)
print(model)

if not os.path.isdir(args.checkpoints):
    os.mkdir(args.checkpoints)

# optionally resume from a checkpoint
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        if 'epoch' in checkpoint:
            args.start_epoch = checkpoint['epoch'] + 1
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict=state_dict, strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, args.start_epoch - 1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
print(args)

if len(args.gpus) > 0:
    model.cuda()
    cudnn.benchmark = True
    if len(args.gpus) > 1:
        model = nn.DataParallel(model, device_ids=args.gpus).cuda()

engine = Engine()
meter_loss = tnt.meter.AverageValueMeter()
topk = [1, 5]
classerr = tnt.meter.ClassErrorMeter(topk=topk, accuracy=False)  # default is also False
confusion_meter = tnt.meter.ConfusionMeter(num_classes[args.dataset], normalized=True)

if args.visdom:
    if args.log_name == '':
        args.log_name = args.build_type

    train_loss_logger = VisdomPlotLogger('line', opts={'title': '[{}] Train Loss'.format(args.log_name)})
    train_err_logger = VisdomPlotLogger('line', opts={'title': '[{}] Train Class Error'.format(args.log_name)})
    test_loss_logger = VisdomPlotLogger('line', opts={'title': '[{}] Test Loss'.format(args.log_name)})
    test_err_logger = VisdomPlotLogger('line', opts={'title': '[{}] Test Class Error'.format(args.log_name)})
    confusion_logger = VisdomLogger('heatmap', opts={'title': '[{}] Confusion matrix'.format(args.log_name),
                                                     'columnnames': list(range(num_classes[args.dataset])),
                                                     'rownames': list(range(num_classes[args.dataset]))})

criterion = nn.CrossEntropyLoss()


def network(sample):
    if sample[2]:  # train mode
        model.train()
    else:
        model.eval()
    inputs, targets = sample[0], sample[1]
    if len(args.gpus) > 0:
        inputs, targets = inputs.cuda(), targets.cuda()
    with torch.set_grad_enabled(sample[2]):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    return loss, outputs


def network_split_batch(sample):
    outputs = Variable()
    if len(args.gpus) > 0:
        outputs = outputs.cuda()

    if args.num_batch_splits >= len(sample[1]):
        return network((sample[0], sample[1], sample[2]))

    d = (len(sample[1]) + args.num_batch_splits - 1) // args.num_batch_splits

    for i in range(args.num_batch_splits):
        start = i * d
        end = min((i + 1) * d, len(sample[1]))
        with torch.set_grad_enabled(sample[2]):
            loss, split_outputs = network((sample[0][start:end], sample[1][start:end], sample[2]))
            if sample[2] and i < args.num_batch_splits - 1:
                loss.backward()
        outputs = torch.cat([outputs, split_outputs], dim=0) if len(outputs) > 0 else split_outputs
    return loss, outputs


network_forward = network if args.num_batch_splits == 1 else network_split_batch


def on_start(state):
    state['epoch'] = args.start_epoch
    state['t'] = args.start_epoch * len(state['iterator'])


def on_sample(state):
    state['sample'].append(state['train'])  # sample[2] is mode
    if state['train']:
        T_total = state['maxepoch'] * len(state['iterator'])
        if args.lr_type == 'multistep':
            lr_decay = state['epoch'] // 30
            lr = args.lr * 0.1 ** lr_decay
        elif args.lr_type == 'cosine':
            num_cycles = 4
            cycle_len = T_total / num_cycles
            lr = 0.5 * args.lr * (1 + math.cos(math.pi * (state['t'] % cycle_len) / cycle_len))
        elif args.lr_type == 'triangle':
            num_cycles = 4
            min_lr = min(1e-3, args.lr)
            max_lr = args.lr
            cycle_len = int(T_total * 0.9) // num_cycles
            if state['t'] < cycle_len * num_cycles:
                p = state['t'] % cycle_len
                if p < cycle_len / 2:
                    lr = min_lr + (max_lr - min_lr) * p * 2 / cycle_len
                else:
                    lr = max_lr - (max_lr - min_lr) * (p - cycle_len / 2) * 2 / cycle_len
            else:
                lr = min_lr * (T_total - state['t']) / (T_total - cycle_len * num_cycles)
        # change lr
        for group in state['optimizer'].param_groups:
            group['lr'] = lr
        if state['t'] == state['epoch'] * len(state['iterator']):
            for i, p in enumerate(state['optimizer'].param_groups):
                print(str(i) + ':', p['lr'])


def reset_meters():
    classerr.reset()
    meter_loss.reset()
    confusion_meter.reset()


def on_start_epoch(state):
    reset_meters()
    state['iterator'] = tqdm(state['iterator'])


def on_forward(state):
    nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    classerr.add(state['output'].data, torch.LongTensor(state['sample'][1]))  # doubt on the LongTensor
    confusion_meter.add(state['output'].data, torch.LongTensor(state['sample'][1]))
    meter_loss.add(state['loss'].item())


def on_end_epoch(state):
    # state['epoch'] += 1 is before this function
    print('[Epoch {:03d}] Training loss: {:.4f}\tTop 1: {:.2f}\tTop 5: {:.2f}'.format(
        state['epoch'] - 1, meter_loss.value()[0], classerr.value(k=1), classerr.value(k=5)))

    if args.visdom:
        train_loss_logger.log(state['epoch'] - 1, meter_loss.value()[0])
        train_err_logger.log(state['epoch'] - 1, classerr.value(k=1))

    if state['epoch'] % args.save_interval == 0:
        saved_model = model.module if len(args.gpus) > 1 else model
        copied_model = copy.deepcopy(saved_model).cpu()
        torch.save(obj={'epoch': state['epoch'] - 1, 'state_dict': copied_model.state_dict()},
                   f=os.path.join(args.checkpoints, 'ImageNet_{:03d}.tar'.format(state['epoch'] - 1)))

    # do validation at the end of each epoch
    reset_meters()
    engine.test(network=network_forward, iterator=test_loader)
    if args.visdom:
        test_loss_logger.log(state['epoch'] - 1, meter_loss.value()[0])
        test_err_logger.log(state['epoch'] - 1, classerr.value()[0])
        confusion_logger.log(confusion_meter.value())

    print('[Epoch {:03d}] Test loss: {:.4f}\tTop 1: {:.2f}\tTop 5: {:.2f}'.format(
        state['epoch'] - 1, meter_loss.value()[0], classerr.value(k=1), classerr.value(k=5)))


if args.test_only:
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.test(network=network_forward, iterator=test_loader)
    print('Test loss: {:.4f}\tTop 1: {:.2f}\tTop 5: {:.2f}'.format(
        meter_loss.value()[0], classerr.value(k=1), classerr.value(k=5)))
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)
    engine.hooks['on_start'] = on_start
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.train(network=network_forward, iterator=train_loader, maxepoch=args.epochs, optimizer=optimizer)
