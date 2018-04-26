# efficient-densenet-pytorch
Memory-Efficient Implementation of DenseNets by PyTorch v0.4

The current implementation passes the forward/backward checking under CPU/single GPU cases, but fails under the multi-GPU case.  
Check the `test_densenet.py` and [the issue](https://github.com/wandering007/efficient-densenet-pytorch/issues/1) for details, help needed.

