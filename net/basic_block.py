import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import AlexNet
from torchviz import make_dot
import torch
from torchvision.models import AlexNet

from tensorboardX import SummaryWriter

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    def __init__(self, inplace=True):
        super(Mish, self).__init__()
        inplace = True

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def get_activation(name="mish", inplace=True):
    if name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == 'mish':
        module = Mish(inplace=inplace)
    elif name == 'silu':
        module = nn.SiLU(inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> mish/leaky relu block"""

    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=True, act="relu", bn=True):
        super(BaseConv, self).__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)
        self.use_bn = bn
    def forward(self, x):
        if self.use_bn:
            return self.act(self.bn(self.conv(x)))
        else:
            return self.act(self.conv(x))
