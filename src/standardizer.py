import torch
import torch.nn as nn


class ChannelStandardize(nn.Module):
    def __init__(self, mean, std, eps=1e-6, learnable=False):
        super().__init__()
        C = mean.numel()
        self.register_buffer("mean", mean.view(1,C,1,1))
        self.register_buffer("std",  std.view(1,C,1,1))
        self.eps = eps
        if learnable:
            self.weight = nn.Parameter(torch.ones(1,C,1,1))
            self.bias   = nn.Parameter(torch.zeros(1,C,1,1))
        else:
            self.weight = self.bias = None
            
    def forward(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        if self.weight is not None: x = x * self.weight + self.bias
        return x

