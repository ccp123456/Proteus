import logging
import random
from typing import List, Callable
import torch
import time
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import sys
sys.path.append("..")
from dyn_slim_1.models.dyn_slim_blocks import DSInvertedResidual, DSDepthwiseSeparable, set_exist_attr
from dyn_slim_1.models.dyn_slim_ops import DSConv2d, DSpwConv2d, DSBatchNorm2d, DSLinear, DSAdaptiveAvgPool2d
from dyn_slim_1.models.dyn_slim_stages import DSStage
from timm.models.layers import sigmoid
from DFC import DFC
from dyn_slim_1.utils import efficientnet_init_weights


def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def gumbel_softmax(logits, tau=1, hard=False, dim=1, training=True):
    """ See `torch.nn.functional.gumbel_softmax()` """
    gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)
    with torch.no_grad():
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return y_soft, ret, index

class DSBatchNorm1d(nn.Module):
    def __init__(self, num_features_list, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(DSBatchNorm1d, self).__init__()
        self.out_channels_list = num_features_list
        self.aux_bn = nn.ModuleList([
            nn.BatchNorm1d(channel, affine=False) for channel in
            self.out_channels_list[:-1]])
        self.aux_bn.append(nn.BatchNorm1d(self.out_channels_list[-1], eps=eps, momentum=momentum, affine=affine))
        self.affine = affine
        self.channel_choice = -1
        self.mode = 'dynamic'

    def set_zero_weight(self):
        if self.affine:
            nn.init.zeros_(self.aux_bn[-1].weight)

    def forward(self, x):
        self.running_inc = x.size(1)

        if self.mode == 'dynamic' and isinstance(self.channel_choice, tuple):
            self.channel_choice, idx = self.channel_choice
            running_mean = torch.zeros_like(self.aux_bn[-1].running_mean).repeat(len(self.out_channels_list), 1)
            running_var = torch.zeros_like(self.aux_bn[-1].running_var).repeat(len(self.out_channels_list), 1)
            for i in range(len(self.out_channels_list)):
                running_mean[i, :self.out_channels_list[i]] += self.aux_bn[i].running_mean
                running_var[i, :self.out_channels_list[i]] += self.aux_bn[i].running_var
            running_mean = torch.matmul(self.channel_choice, running_mean)[..., None].expand_as(x)
            running_var = torch.matmul(self.channel_choice, running_var)[..., None].expand_as(x)
            weight = self.aux_bn[-1].weight[:self.running_inc] if self.affine else None
            bias = self.aux_bn[-1].bias[:self.running_inc] if self.affine else None
            x = (x - running_mean) / torch.sqrt(running_var + self.aux_bn[-1].eps)
            x = x * weight[..., None].expand_as(x) + bias[..., None].expand_as(x)
            return apply_differentiable_gate_channel(x, self.channel_choice, self.out_channels_list)
        else:
            idx = self.out_channels_list.index(self.running_inc)
            running_mean = self.aux_bn[idx].running_mean
            running_var = self.aux_bn[idx].running_var
            weight = self.aux_bn[-1].weight[:self.running_inc] if self.affine else None
            bias = self.aux_bn[-1].bias[:self.running_inc] if self.affine else None
            return F.batch_norm(x, running_mean, running_var, weight, bias, self.training, self.aux_bn[-1].momentum, self.aux_bn[-1].eps)

def apply_differentiable_gate_channel(x, channel_gate, channel_list):
    ret = torch.zeros_like(x)
    if not isinstance(channel_gate, torch.Tensor):
        ret[:, :channel_list[channel_gate]] += x[:, :channel_list[channel_gate]]
    else:
        for idx in range(len(channel_list)):
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
            ret[:, :channel_list[idx]] += x[:, :channel_list[idx]] * (channel_gate[:, idx, None, None].expand_as(x[:, :channel_list[idx]]))
    return ret

class DSLinear(nn.Linear):
    def __init__(self, in_features_list, out_features, bias=True):
        super(DSLinear, self).__init__(
            in_features=in_features_list[-1],
            out_features=out_features,
            bias=bias)
        self.in_channels_list = in_features_list
        self.out_channels_list = [out_features]
        self.channel_choice = -1
        self.mode = 'largest'
        self.in_channels_list_tensor = torch.from_numpy(
            np.array(self.in_channels_list)).float().cuda()

    def forward(self, x):
        if self.mode == 'dynamic':
            if isinstance(self.channel_choice, tuple):
                self.channel_choice = self.channel_choice[0]
                self.running_inc = torch.matmul(self.channel_choice, self.in_channels_list_tensor)
            else:
                self.running_inc = self.in_channels_list[self.channel_choice]
            self.running_outc = self.out_features
            return F.linear(x, self.weight, self.bias)
        else:
            self.running_inc = x.size(1)
            self.running_outc = self.out_features
            weight = self.weight[:, :self.running_inc]
            return F.linear(x, weight, self.bias)

class DDNN(nn.Module):

    def __init__(self, num_classes, d_dim=16, stages_out_channels=64, se_ratio=0.25, AE_train = False, AE_test = False):
        super(DDNN, self).__init__()
        self.has_head = True
        self.AE_train = AE_train
        self.AE_test = AE_test
        self._stage_out_channels = stages_out_channels
        self.byteembedding = nn.Embedding(num_embeddings=50000, embedding_dim=d_dim)
        output_channels = self._stage_out_channels
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, groups=1),
            nn.ReLU(inplace=True)
        )
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.DFC = DFC(16, 32, 4)
        self.relu_fc2 = nn.ReLU(inplace=True)
        output_channels = 32
        self.in_channels_list = [int(output_channels*0.25), int(output_channels*0.5), int(output_channels*0.75), output_channels]
        self.out_channels_list = [output_channels*0.25]
        self.channel_choice = -1
        self.gate = MultiHeadGate(self.in_channels_list, se_ratio=se_ratio, channel_gate_num=4)
        self.conv_head = DSConv1d(self.in_channels_list, 16, 1)
        self.bn = nn.BatchNorm1d(16)
        self.bn1 = DSBatchNorm1d(self.in_channels_list)
        self.act = nn.ReLU(inplace=True)
        self.fc = DSLinear([1024], 15)
        self.mode = 'largest'
        self.AutoEncoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.loss_fn = torch.nn.MSELoss()
        self.threshold = 400

    def forward(self, x):
        AE_result = self.AutoEncoder(x.to(torch.float32))
        if self.AE_train:
            return AE_result, x.to(torch.float32)
        MSEloss = self.loss_fn(AE_result, x.to(torch.float32))
        if self.AE_test:
            if MSEloss > self.threshold:
                return torch.tensor([0])
            else:
                return torch.tensor([1])
        out = self.byteembedding(x)
        out = out.transpose(-2,-1)
        out, lasso_loss = self.DFC(out)
        out = self.act(out)
        out = self.gate(out)
        self.prev_channel_choice = None
        self.channel_choice = self._new_gate()
        self._set_gate()
        if isinstance(self.channel_choice, int):
            out = out[:, :self.in_channels_list[self.channel_choice]]
        out = self.bn1(out)
        out = self.conv_head(out)
        out = out.view(-1, 1024)
        out = self.fc(out)
        return out, lasso_loss

    def _set_gate(self):
        set_exist_attr(self.conv_head, 'channel_choice', self.channel_choice)
        set_exist_attr(self.bn1, 'channel_choice', self.channel_choice)
        set_exist_attr(self.conv_head, 'prev_channel_choice', self.prev_channel_choice)

    def _new_gate(self):
        if self.mode == 'largest':
            return -1
        elif self.mode == 'smallest':
            return 0
        elif self.mode == 'uniform':
            return self.random_choice
        elif self.mode == 'random':
            return random.randint(0, len(self.out_channels_list))
        elif self.mode == 'dynamic':
            if self.gate.has_gate:
                return self.gate.get_gate()
            else:
                return 0
    def get_gate(self):
        gate = nn.ModuleList()
        for n, m in self.named_modules():
            if isinstance(m, MultiHeadGate) and m.has_gate:
                gate += [m.gate]
        return gate
    def set_mode(self, mode, seed=None, choice=None):
        self.mode = mode
        if seed is not None:
            random.seed(seed)
            seed += 1
        assert mode in ['largest', 'smallest', 'dynamic', 'uniform']
        for m in self.modules():
            set_exist_attr(m, 'mode', mode)
        if mode == 'largest':
            self.channel_choice = -1
            if self.has_head:
                self.set_module_choice(self.conv_head)
        elif mode == 'smallest' or mode == 'dynamic':
            self.channel_choice = 0
            if self.has_head:
                self.set_module_choice(self.conv_head)
        elif mode == 'uniform':
            self.channel_choice = 0
            if self.has_head:
                self.set_module_choice(self.conv_head)
            if choice is not None:
                self.random_choice = choice
            else:
                self.random_choice = random.randint(1, 2)

    def set_module_choice(self, m):
        set_exist_attr(m, 'channel_choice', self.channel_choice)

class DSConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels_list,
                 out_channels_list,
                 kernel_size,
                 stride=1,
                 dilation=1,  
                 groups=1,  
                 bias=True,
                 padding_mode='zeros'):
        if not isinstance(in_channels_list, (list, tuple)):
            in_channels_list = [in_channels_list]
        if not isinstance(out_channels_list, (list, tuple)):
            out_channels_list = [out_channels_list]
        super(DSConv1d, self).__init__(
            in_channels=in_channels_list[-1],
            out_channels=out_channels_list[-1],
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode)
        self.running_stride = stride
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.channel_choice = -1 
        self.in_chn_static = len(in_channels_list) == 1
        self.out_chn_static = len(out_channels_list) == 1
        self.running_inc = self.in_channels if self.in_chn_static else None
        self.running_outc = self.out_channels if self.out_chn_static else None
        self.running_kernel_size = self.kernel_size[0]
        self.running_groups = self.groups
        self.mode = 'largest'
        self.prev_channel_choice = None
        self.in_channels_list_tensor = torch.from_numpy(
            np.array(self.in_channels_list)).float().cuda()
        self.out_channels_list_tensor = torch.from_numpy(
            np.array(self.out_channels_list)).float().cuda()

    def forward(self, x):
        if self.prev_channel_choice is None:  
            self.prev_channel_choice = self.channel_choice
        if self.mode == 'dynamic' and isinstance(self.prev_channel_choice, tuple):  
            weight = self.weight
            if not self.in_chn_static:  
                if isinstance(self.prev_channel_choice, int):
                    self.running_inc = self.in_channels_list[self.channel_choice]
                    weight = self.weight[:, :self.running_inc]
                else:
                    self.running_inc = torch.matmul(self.prev_channel_choice[0], self.in_channels_list_tensor)
            if not self.out_chn_static:
                self.running_outc = torch.matmul(self.channel_choice[0], self.out_channels_list_tensor)
            output = F.conv1d(x,
                              weight,
                              self.bias,
                              self.stride,
                              self.padding,
                              self.dilation,
                              self.groups)
            if not self.out_chn_static:
                output = apply_differentiable_gate_channel(output,
                                                           self.channel_choice[0],
                                                           self.out_channels_list)
            self.prev_channel_choice = None
            self.channel_choice = -1
            return output
        else:
            if not self.in_chn_static:
                if isinstance(self.channel_choice, int):
                    self.running_inc = self.in_channels_list[self.channel_choice]
            if not self.out_chn_static:
                self.running_outc = self.out_channels_list[self.channel_choice]
            weight = self.weight[:self.running_outc, :self.running_inc]
            bias = self.bias[:self.running_outc] if self.bias is not None else None
            x = x[:, :self.running_inc]
            self.running_groups = 1 if self.groups == 1 else self.groups
            self.prev_channel_choice = None
            self.channel_choice = -1
            return F.conv1d(x,
                            weight,
                            bias,
                            self.stride,
                            self.padding,
                            self.dilation,
                            self.running_groups)

class DSpwConv1d(DSConv1d):
    def __init__(self,
                 in_channels_list,
                 out_channels_list,
                 bias=True):
        super(DSpwConv1d, self).__init__(
            in_channels_list=in_channels_list,
            out_channels_list=out_channels_list,
            kernel_size=1,
            stride=1,
            dilation=1,
            groups=1,
            bias=bias,
            padding_mode='zeros')


class DSAdaptiveAvgPool1d(nn.AdaptiveAvgPool1d):
    def __init__(self, output_size, channel_list):
        super(DSAdaptiveAvgPool1d, self).__init__(output_size=output_size)
        self.in_channels_list = channel_list
        self.channel_choice = -1
        self.mode = 'largest'
        self.in_channels_list_tensor = torch.from_numpy(
            np.array(self.in_channels_list)).float().cuda()
            # np.array(self.in_channels_list)).float()

    def forward(self, x):
        if self.mode == 'dynamic':
            if isinstance(self.channel_choice, tuple):
                self.channel_choice = self.channel_choice[0]
                self.running_inc = torch.matmul(self.channel_choice, self.in_channels_list_tensor)
            else:
                self.running_inc = self.in_channels_list[self.channel_choice]
        else:
            self.running_inc = x.size(1)
        return super(DSAdaptiveAvgPool1d, self).forward(input=x)


class MultiHeadGate(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None, act_layer=nn.ReLU,
                 attn_act_fn=sigmoid, divisor=1, channel_gate_num=None, gate_num_features=1024):
        super(MultiHeadGate, self).__init__()
        self.attn_act_fn = attn_act_fn
        self.channel_gate_num = channel_gate_num
        self.conv_reduce = DSpwConv1d(in_chs, in_chs, bias=True)
        self.act1 = act_layer(inplace=True)

        self.has_gate = False
        if channel_gate_num > 1:
            self.has_gate = True
            self.gate = nn.Sequential(DSpwConv1d([in_chs[-1]], [channel_gate_num], bias=False))

        self.mode = 'largest'
        self.keep_gate, self.print_gate, self.print_idx = None, None, None
        self.channel_choice = None

    def forward(self, x):
        x_pool = x.mean([-1])  # pool
        b, c = x_pool.size()
        x_reduced = x_pool.view(b, c, 1)
        x = x

        if self.mode == 'dynamic' and self.has_gate: 
            channel_choice = self.gate(x_reduced).squeeze(-1) # pointwise
            self.keep_gate, self.print_gate, self.print_idx = gumbel_softmax(channel_choice, dim=1, training=self.training)
            self.channel_choice = self.print_gate, self.print_idx
        else:
            self.channel_choice = None
        return x

    def get_gate(self):
        return self.channel_choice
class DSpw(nn.Module):

    def __init__(self, in_channels_list, out_channels,act_layer=nn.ReLU,
                 se_ratio=0.25, norm_layer=nn.BatchNorm1d, has_gate=True):
        super(DSpw, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.has_gate = has_gate
        self.downsample = None
        # Channel attention and gating
        self.gate = MultiHeadGate(in_channels_list,
                                  se_ratio=se_ratio,
                                  channel_gate_num=4 if has_gate else 0)

        # Point-wise convolution
        self.conv_pw = DSpwConv1d(self.in_channels_list,16)
        self.bn2 = norm_layer(out_channels)
        self.act2 = act_layer(inplace=True)
        self.channel_choice = -1
        self.mode = 'largest'
        self.next_channel_choice = None
        self.random_choice = 0

    def forward(self, x):
        self._set_gate()
        residual = x
        out = self.gate(x) 
        if self.has_gate:
            self.prev_channel_choice = self.channel_choice
            self.channel_choice = self._new_gate()
            self._set_gate(set_pw=True)
        x = self.conv_pw(x)
        x = self.bn2(x)
        x = self.act2(x)

        return x

    def _set_gate(self, set_pw=False):
        for n, m in self.named_modules():
            set_exist_attr(m, 'channel_choice', self.channel_choice)
        if set_pw:
            self.conv_pw.prev_channel_choice = self.prev_channel_choice
            if self.downsample is not None:
                for n, m in self.downsample.named_modules():
                    set_exist_attr(m, 'prev_channel_choice', self.prev_channel_choice)

    def _new_gate(self):
        if self.mode == 'largest':
            return -1
        elif self.mode == 'smallest':
            return 0
        elif self.mode == 'uniform':
            return self.random_choice
        elif self.mode == 'random':
            return random.randint(0, len(self.out_channels) - 1)
        elif self.mode == 'dynamic':
            if self.gate.has_gate:
                return self.gate.get_gate()
            else:
                return 0

    def get_gate(self):
        return self.channel_choice


if __name__ == '__main__':
    x = np.random.randint(0, 255, (64, 258))
    y = np.random.randint(0, 9, (64))
    a = torch.from_numpy(x).long()
    y = torch.from_numpy(y).long()
    gate = DDNN(num_classes=10)
    b,c = gate(a)
    gate.train()
    loss = torch.nn.CrossEntropyLoss()
    learnstep = 0.01
    optim = torch.optim.SGD(gate.parameters(), lr=learnstep)
    outputs,_ = gate(a)
    result_loss = loss(outputs, y)
    optim.zero_grad()
    result_loss.backward()
    optim.step()
    print(b)
