from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import random
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=4,
                 padding=0, dilation=1, heads=2, squeeze_rate=16, gate_factor=0.75):
        super(DFC, self).__init__()
        self.norm = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.squeeze_rate = squeeze_rate
        self.gate_factor = gate_factor
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.is_pruned = True
        self.register_buffer('_inactive_channels', torch.zeros(1))
        self.head1 = nn.Conv1d(in_channels, out_channels // self.heads, kernel_size, stride, dilation, groups=1, bias=False)
        self.head2 = HeadConv(in_channels, out_channels // self.heads, squeeze_rate, kernel_size, stride, padding,
                              dilation, 1, gate_factor)

    def forward(self, x):
        """
        The code here is just a coarse implementation.
        The forward process can be quite slow and memory consuming, need to be optimized.
        """
        self.inactive_channels = round(self.in_channels * (1 - self.gate_factor))

        _lasso_loss = 0.0

        x_averaged = x.mean([2])
        x_mask = []
        weight = []
        x_mask.append(x)
        weight.append(self.head1.weight)
        i_x, i_lasso_loss = self.head2(x, x_averaged, self.inactive_channels)
        x_mask.append(i_x)
        weight.append(self.head2.conv.weight)
        _lasso_loss = _lasso_loss + i_lasso_loss
        x_mask = torch.cat(x_mask, dim=1)  # batch_size, 4 x C_in, H
        weight = torch.cat(weight, dim=0)  # C_out, C_in, k

        out = F.conv1d(x_mask, weight, None, self.stride, self.padding, self.dilation, self.heads)
        b, c, h = out.size()
        out = out.view(b, self.heads, c // self.heads, h)
        out = out.transpose(1, 2).contiguous().view(b, c, h)
        return out, _lasso_loss

    @property
    def inactive_channels(self):
        return int(self._inactive_channels[0])

    @inactive_channels.setter
    def inactive_channels(self, val):
        self._inactive_channels.fill_(val)


class HeadConv(nn.Module):
    def __init__(self, in_channels, out_channels, squeeze_rate, kernel_size, stride=2, padding=0, dilation=1, groups=1,
                 gate_factor=0.75):
        super(HeadConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, dilation, groups=1, bias=False)
        self.target_pruning_rate = gate_factor
    def forward(self, x, x_averaged, inactive_channels):
        b, c, _ = x.size()
        mask = x_averaged.view(b, c)
        _lasso_loss = 0.0
        mask_d = mask.detach()
        mask_c = mask
        if inactive_channels > 0:
            mask_c = mask.clone()
            topk_maxmum, _ = mask_d.topk(inactive_channels, dim=1, largest=False, sorted=False)
            clamp_max, _ = topk_maxmum.max(dim=1, keepdim=True)
            mask_index1 = mask_d.le(clamp_max)
            mask_c[mask_index1] = 0

        mask_c = mask_c.view(b, c, 1)
        x = x * mask_c.expand_as(x)

        return x, _lasso_loss


if __name__ == '__main__':
    x = np.random.randint(0, 255, (64, 16, 32))
    a = torch.from_numpy(x).float()
    gate = DFC(16, 8, 4)
    b = gate(a)
    print(b)