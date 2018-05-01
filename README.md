# efficient-densenet-pytorch
Memory-Efficient Implementation of DenseNets, **support both DenseNet and DeseNet-BC series.**

Environments: Linux CPU/GPU, Python 3, PyTorch 0.4 or higher

Check the implementation correctness by `python test_densenet.py` with different settings in `test_densenet.py` (CPU, single GPU, multiple GPUs), some asset errors may be caused by occasional random states, you can relax the error tolerance (default: `1e-5`) or try it several times.

Benchmark the forward/backward of efficient densenet / non-efficient denseness by `python benckmark_effi.py` (CPU, single GPU, multiple GPUs). The following table is reported on the Linux system equipped with 40 Intel(R) Xeon(R) CPUs (E5-2630 v4 @ 2.20GHz) and NVIDIA GTX TiTan 1080Ti. 

```
num_init_features=24, block_config=(12, 12, 12), compression=1, input_size=32, bn_size=None, batch size=128. 
F means average forward time (ms), B means average backward time (ms), R=B/F.
```

|     Model     |           CPU           |          1 GPU          |         2 GPUs         |         4 GPUs         |
| :-----------: | :---------------------: | :---------------------: | :--------------------: | :--------------------: |
|   Efficient   | F=15849, B=36738, R=2.3 | F=36.3, B=103.8, R=2.86 | F=43.2, B=66.9, R=1.55 | F=69.6, B=81.7, R=1.17 |
| Non-efficient | F=24889, B=36732, R=1.5 | F=36.1, B=74.9, R=2.07  | F=37.9, B=40.7, R=1.07 | F=62.0, B=33.1, R=0.53 |

The efficient version can process up to 1450 batches in a single GPU (~12GB), compared with 350 batches of the non-efficient version. That is, the efficient version is **~4x memory-efficient** than the non-efficient version.

# References

Pleiss, Geoff, et al. "Memory-efficient implementation of densenets." *arXiv preprint arXiv:1707.06990* (2017).

