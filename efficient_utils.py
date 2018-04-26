# This implementation is a new efficient implementation of Densenet-BC,
# as described in "Memory-Efficient Implementation of DenseNets"
# The code is based on https://github.com/gpleiss/efficient_densenet_pytorch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _single, _pair, _triple


class _EfficientDensenetBottleneck(nn.Module):
    """
    A optimized layer which encapsulates the batch normalization, ReLU, and
    convolution operations within the bottleneck of a DenseNet layer.

    This layer usage shared memory allocations to store the outputs of the
    concatenation and batch normalization features. Because the shared memory
    is not perminant, these features are recomputed during the backward pass.
    """

    def __init__(self, shared_alloc, num_input_channels, num_output_channels,
                 kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, dims=2,
                 momentum=0.1, eps=1e-5):
        super(_EfficientDensenetBottleneck, self).__init__()
        assert dims in [1, 2, 3]
        self.shared_alloc = shared_alloc
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.dims = dims
        _repeat = dims == 1 and _single or dims == 2 and _pair or dims == 3 and _triple
        self.kernel_size = _repeat(kernel_size)
        self.stride = _repeat(stride)
        self.padding = _repeat(padding)
        self.dilation = _repeat(dilation)
        self.groups = groups
        self.bias = bias
        self.momentum = momentum
        self.eps = eps
        self.conv = self.dims == 1 and F.conv1d or self.dims == 2 and F.conv2d or self.dims == 3 and F.conv3d
        self.register_parameter('norm_weight', nn.Parameter(torch.Tensor(num_input_channels)))
        self.register_parameter('norm_bias', nn.Parameter(torch.Tensor(num_input_channels)))
        self.register_buffer('norm_running_mean', torch.zeros(num_input_channels))
        self.register_buffer('norm_running_var', torch.ones(num_input_channels))
        self.register_parameter('conv_weight', nn.Parameter(torch.Tensor(num_output_channels,
                                                                         num_input_channels, *self.kernel_size)))
        if bias:
            self.register_parameter('conv_bias', nn.Parameter(torch.Tensor(num_input_channels)))
        else:
            self.register_parameter('conv_bias', None)
        self._reset_parameters()

    def _reset_parameters(self):
        self._buffers['norm_running_mean'].zero_()
        self._buffers['norm_running_var'].fill_(1)
        self._parameters['norm_weight'].data.uniform_()
        self._parameters['norm_bias'].data.zero_()
        stdv = 1. / math.sqrt(self.num_input_channels)
        self._parameters['conv_weight'].data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]

        # The EfficientDensenetBottleneckFn performs the concatenation, batch norm, and ReLU.
        # It does not create any new storage
        # Rather, it uses a shared memory allocation to store the intermediate feature maps
        # These intermediate feature maps have to be re-populated before the backward pass
        fn = _EfficientDensenetBottleneckFn(self.shared_alloc,
                                            self._parameters['norm_weight'], self._parameters['norm_bias'],
                                            self._buffers['norm_running_mean'], self._buffers['norm_running_var'],
                                            training=self.training, momentum=self.momentum, eps=self.eps)
        relu_output = fn(*inputs)

        # The convolutional output - using relu_output which is stored in shared memory allocation
        conv_output = self.conv(relu_output, self._parameters['conv_weight'], bias=self._parameters['conv_bias'],
                                stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

        # Register a hook to re-populate the storages (relu_output and concat) on backward pass
        # To do this, we need a dummy function
        dummy_fn = _DummyBackwardHookFn(fn)
        output = dummy_fn(conv_output)

        # Return the convolution output
        return output

    def __repr__(self):
        s = ('{name}(Concat'
             '\n\t--> BatchNorm({num_input_channels}, momentum={momentum}, eps={eps}) --> ReLU(inplace=True)'
             '\n\t--> Conv{dims}d({num_input_channels}, {num_output_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += '))'
        return s.format(name=self.__class__.__name__, **self.__dict__)


# Here's where we define the internals of the efficient bottleneck layer
class _SharedAllocation(object):
    """
    A helper class which maintains a shared memory allocation.
    Used for concatenation and batch normalization.
    """

    def __init__(self, size=1024):
        self._cpu_storage = torch.Storage(size)
        self._gpu_storages = []
        if torch.cuda.is_available():
            for device_idx in range(torch.cuda.device_count()):
                with torch.cuda.device(device_idx):
                    self._gpu_storages.append(torch.Storage(size).cuda())

    def type(self, t):
        if not t.is_cuda:
            self._cpu_storage = self._cpu_storage.type(t)
        else:
            for device_idx, storage in enumerate(self._gpu_storages):
                with torch.cuda.device(device_idx):
                    self._gpu_storages[device_idx] = storage.type(t)

    def type_as(self, obj):
        if isinstance(obj, torch.Tensor):
            if not obj.is_cuda:
                self._cpu_storage = self._cpu_storage.type(obj.storage().type())
            else:
                for device_idx, storage in enumerate(self._gpu_storages):
                    with torch.cuda.device(device_idx):
                        self._gpu_storages[device_idx] = storage.type(obj.storage().type())
        else:
            if not obj.is_cuda:
                self._cpu_storage = self._cpu_storage.type(obj.storage().type())
            else:
                for device_idx, storage in enumerate(self._gpu_storages):
                    with torch.cuda.device(device_idx):
                        self._gpu_storages[device_idx] = storage.type(obj.type())

    def resize_(self, size):
        if self._cpu_storage.size() < size:
            self._cpu_storage.resize_(size)
        for device_idx, storage in enumerate(self._gpu_storages):
            if storage.size() < size:
                with torch.cuda.device(device_idx):
                    self._gpu_storages[device_idx].resize_(size)
        return self

    def storage_for(self, val):
        if val.is_cuda:
            with torch.cuda.device_of(val):
                curr_device_id = torch.cuda.current_device()
                return self._gpu_storages[curr_device_id]
        else:
            return self._cpu_storage


class _EfficientDensenetBottleneckFn(Function):
    """
    The autograd function which performs the efficient bottlenck operations:
    --
    1) concatenation
    2) Batch Normalization
    3) ReLU
    --
    Convolution is taken care of in a separate function

    NOTE:
    The output of the function (ReLU) is written on a temporary memory allocation.
    If the output is not used IMMEDIATELY after calling forward, it is not guarenteed
    to be the ReLU output
    """

    def __init__(self, shared_alloc,
                 bn_weight, bn_bias,
                 running_mean, running_var,
                 training=True, momentum=0.1, eps=1e-5):
        assert isinstance(shared_alloc, tuple) and len(shared_alloc) == 2
        self.shared_allocation = shared_alloc
        self.bn_weight = bn_weight
        self.bn_bias = bn_bias
        self.running_mean = running_mean
        self.running_var = running_var
        self.training = training
        self.momentum = momentum
        self.eps = eps

        # Buffers to store old versions of bn statistics
        self.prev_running_mean = self.running_mean.new(self.running_mean.size())
        self.prev_running_var = self.running_var.new(self.running_var.size())

    def forward(self, *inputs):
        if self.training:
            # Save the current BN statistics for later
            self.prev_running_mean.copy_(self.running_mean)
            self.prev_running_var.copy_(self.running_var)

        # Create tensors that use shared allocations
        # One for the concatenation output (bn_input)
        # One for the ReLU output (relu_output)
        all_num_channels = [input.size(1) for input in inputs]
        size = list(inputs[0].size())
        for num_channels in all_num_channels[1:]:
            size[1] += num_channels
        storage1 = self.shared_allocation[0].storage_for(inputs[0])
        storage2 = self.shared_allocation[1].storage_for(inputs[0])
        bn_input = inputs[0].new(storage1).resize_(size)
        relu_output = inputs[0].new(storage2).resize_(size)
        with torch.no_grad():
            torch.cat(inputs, dim=1, out=bn_input)
            bn_output = F.batch_norm(bn_input, self.running_mean, self.running_var,
                                     self.bn_weight, self.bn_bias, training=self.training,
                                     momentum=self.momentum, eps=self.eps)

            # Do ReLU - and have the output be in the intermediate storage
            torch.clamp(bn_output, min=0, out=relu_output)

        self.save_for_backward(bn_input, relu_output, *inputs)
        if self.training:
            # restore the BN statistics for later
            self.running_mean.copy_(self.prev_running_mean)
            self.running_var.copy_(self.prev_running_var)
        return relu_output

    def prepare_backward(self):
        bn_input, relu_output = self.saved_tensors[:2]
        inputs = self.saved_tensors[2:]
        # Create variable, using existing storage
        torch.cat(inputs, dim=1, out=bn_input)
        self.bn_input = bn_input.requires_grad_()
        with torch.enable_grad():
            # Do batch norm
            self.bn_output = F.batch_norm(self.bn_input, self.running_mean, self.running_var,
                                          self.bn_weight, self.bn_bias, training=self.training,
                                          momentum=self.momentum, eps=self.eps)

        # Do ReLU
        torch.clamp(self.bn_output, min=0, out=relu_output)

    def backward(self, grad_output):
        """
        Precondition: must call prepare_backward before calling backward
        """

        grads = [None] * len(self.saved_tensors)
        relu_output = self.saved_tensors[1]
        inputs = self.saved_tensors[2:]

        # If we don't need gradients, don't run backwards
        if not any(self.needs_input_grad):
            return grads

        # BN weight/bias grad
        # With the shared allocations re-populated, compute ReLU/BN backward
        relu_grad_input = grad_output.masked_fill_(relu_output <= 0, 0)
        self.bn_output.backward(gradient=relu_grad_input)

        # Input grad (if needed)
        # Run backwards through the concatenation operation
        if any(self.needs_input_grad):
            all_num_channels = [input.size(1) for input in inputs]
            index = 0
            for i, num_channels in enumerate(all_num_channels):
                new_index = num_channels + index
                grads[i] = self.bn_input.grad.data[:, index:new_index]
                index = new_index

        # Delete all intermediate variables
        del self.bn_input
        del self.bn_output
        del self.shared_allocation
        del self.prev_running_mean
        del self.prev_running_var

        return tuple(grads)


class _DummyBackwardHookFn(Function):
    """
    A dummy function, which is just designed to run a backward hook
    This allows us to re-populate the shared storages before running the backward
    pass on the bottleneck layer
    The function itself is just an identity function
    """

    def __init__(self, fn):
        """
        fn: function to call "prepare_backward" on
        """
        self.fn = fn

    def forward(self, input):
        """
        Though this function is just an identity function, we have to return a new
        tensor object in order to trigger the autograd.
        """
        size = input.size()
        res = input.new(input.storage()).view(*size)
        return res

    def backward(self, grad_output):
        self.fn.prepare_backward()
        return grad_output