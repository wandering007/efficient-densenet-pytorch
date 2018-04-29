# efficient-densenet-pytorch
Memory-Efficient Implementation of DenseNets by PyTorch v0.4

The current implementation passes the forward/backward checking under CPU/GPU cases without any bugs. However, the multi-GPU backward time (`nn.DataParallel`) is long for now, I'm thinking to figure it out...

