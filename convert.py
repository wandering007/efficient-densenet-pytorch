import torch
from collections import OrderedDict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--to', type=str, default='efficient', choices=['efficient', 'normal'], metavar='CONVERT_TYPE',
                    help='convert type: efficient | normal')
parser.add_argument('--checkpoint', type=str, metavar='CHECKPOINT', help='path for densenet checkpoint')
parser.add_argument('--output', type=str, default='densenet.pth', metavar='OUTPUT', help='path for output file')

args = parser.parse_args()
state = torch.load(args.checkpoint)
state = OrderedDict((k.replace('.norm.1.', '.norm1.'), v) for k, v in state.items())
state = OrderedDict((k.replace('.conv.1.', '.conv1.'), v) for k, v in state.items())
state = OrderedDict((k.replace('.norm.2.', '.norm2.'), v) for k, v in state.items())
state = OrderedDict((k.replace('.conv.2.', '.conv2.'), v) for k, v in state.items())
if args.to == 'efficient':
    state = OrderedDict((k.replace('.norm1.', '.bottleneck.norm_'), v) for k, v in state.items())
    state = OrderedDict((k.replace('.conv1.', '.bottleneck.conv_'), v) for k, v in state.items())
else:
    state = OrderedDict((k.replace('.bottleneck.norm_', '.norm1.'), v) for k, v in state.items())
    state = OrderedDict((k.replace('.bottleneck.conv_', '.conv1.'), v) for k, v in state.items())

torch.save(state, args.output)
print('Covert ' + args.checkpoint + ' to the ' + args.to + ' version '
                                                           'and save as ' + args.output + ' successfully!')
