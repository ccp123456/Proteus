from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import os
import shutil
import math
import time

from functools import reduce
import operator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


count_ops = 0
count_params = 0

def get_num_gen(gen):
    return sum(1 for x in gen)

def is_pruned(layer):
    if hasattr(layer, 'mask'):
        return True
    elif hasattr(layer, 'is_pruned'):
        return True
    else:
        return False

def is_leaf(model):
    return get_num_gen(model.children()) == 0


def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name


def get_layer_param(model):
    return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])



def measure_layer(layer, x):
    global count_ops, count_params
    delta_ops = 0
    delta_params = 0
    multi_add = 1
    type_name = get_layer_info(layer)

    if type_name in ['Conv1d', 'Conv1d_lasso']:
        out_h = int((x.size()[-1] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] * out_h / layer.groups * multi_add
        delta_params = get_layer_param(layer)

    elif type_name in ['HeadConv']:
        x_ori = x
        x = F.adaptive_avg_pool1d(x, 1)
        b, c, _ = x.size()
        x = x.view(b, c)
        measure_layer(layer.fc1, x)
        x = layer.fc1(x)
        measure_layer(layer.relu_fc1, x)
        x = layer.relu_fc1(x)
        measure_layer(layer.fc2, x)
        x = layer.fc2(x)
        measure_layer(layer.relu_fc2, x)
        delta_ops = reduce(operator.mul, x.size(), 1)
        delta_params = 0

        x = x_ori
        conv = layer.conv
        out_h = int((x.size()[-1] + 2 * conv.padding[0] - conv.kernel_size[0]) /
                    conv.stride[0] + 1)
        delta_ops += conv.in_channels * conv.out_channels * conv.kernel_size[0] * \
                        out_h * layer.target_pruning_rate * multi_add
        delta_params += get_layer_param(conv)

    elif type_name in ['DynamicMultiHeadConv']:
        measure_layer(layer.relu, x)
        measure_layer(layer.norm, x)
        measure_layer(layer.avg_pool, x)
        for i in range(layer.heads):
            measure_layer(layer.__getattr__('headconv_%1d' % i), x)
        delta_ops = 0
        delta_params = 0

    elif type_name in ['Embedding']:
        delta_ops = x.numel()
        delta_params = get_layer_param(layer)

    elif type_name in ['ReLU', 'ReLU6', 'Sigmoid']:
        delta_ops = x.numel()
        delta_params = get_layer_param(layer)

    elif type_name in ['AvgPool1d', 'MaxPool1d']:
        in_w = x.size()[2]
        kernel_ops = layer.kernel_size * 1
        out_w = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        delta_ops = x.size()[0] * x.size()[1] * out_w * kernel_ops
        delta_params = get_layer_param(layer)

    elif type_name in ['AdaptiveAvgPool1d']:
        in_w = x.size()[2]
        kernel_size = in_w
        padding = 0
        kernel_ops = kernel_size * 1
        out_w = int((in_w + 2 * padding - kernel_size) / 1 + 1)
        delta_ops = x.size()[0] * x.size()[1] * out_w * kernel_ops
        delta_params = get_layer_param(layer)

    elif type_name in ['AdaptiveAvgPool1d']:
        delta_ops = x.size()[0] * x.size()[1] * x.size()[2]
        delta_params = get_layer_param(layer)

    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        try:
            bias_ops = layer.bias.numel()
        except AttributeError:
            bias_ops = 0
        delta_ops = x.size()[0] * (weight_ops + bias_ops)
        delta_params = get_layer_param(layer)

    elif type_name in ['BatchNorm1d', 'Dropout1d', 'DropChannel', 'Dropout']:
        delta_params = get_layer_param(layer)

    else:
        raise TypeError('unknown layer type: %s' % type_name)

    count_ops += delta_ops
    count_params += delta_params
    return


def measure_model(model, H):
    global count_ops, count_params
    count_ops = 0
    count_params = 0
    data = torch.zeros(1, H).long()

    def should_measure(x):
        return is_leaf(x) or is_pruned(x)

    def modify_forward(model):
        for child in model.children():
            if should_measure(child):
                def new_forward(m):
                    def lambda_forward(x):
                        measure_layer(m, x)
                        return m.old_forward(x)
                    return lambda_forward
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        for child in model.children():
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model)
    model.forward(data)
    restore_forward(model)

    return count_ops, count_params

def load_checkpoint(args):

    model_dir = os.path.join(args.savedir, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    model_filename = ''

    if args.evaluate is not None:
        model_filename = args.evaluate
    else:
        if os.path.exists(latest_filename):
            with open(latest_filename, 'r') as fin:
                model_filename = fin.readlines()[0].strip()
    loadinfo = "=> loading checkpoint from '{}'".format(model_filename)
    print(loadinfo)

    state = None
    if os.path.exists(model_filename):
        state = torch.load(model_filename, map_location='cpu')
        loadinfo2 = "=> loaded checkpoint '{}' successfully".format(model_filename)
    else:
        loadinfo2 = "no checkpoint loaded"
    print(loadinfo2)

    return state


def save_checkpoint(state, epoch, root, is_best, saveID, keep_freq=10):

    filename = 'checkpoint_%03d.pth.tar' % epoch
    model_dir = os.path.join(root, 'save_models')
    model_filename = os.path.join(model_dir, filename)
    latest_filename = os.path.join(model_dir, 'latest.txt')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    torch.save(state, model_filename)
    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    print("=> saved checkpoint '{}'".format(model_filename))

    if is_best:
        best_filename = os.path.join(model_dir, 'model_best.pth.tar')
        shutil.copyfile(model_filename, best_filename)

    if saveID is not None and saveID % keep_freq != 0:
        filename = 'checkpoint_%03d.pth.tar' % saveID
        model_filename = os.path.join(model_dir, filename)
        if os.path.exists(model_filename):
            os.remove(model_filename)
            print('=> removed checkpoint %s' % model_filename)

    print('##########Time##########', time.strftime('%Y-%m-%d %H:%M:%S'))
    return epoch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, args, batch=None,
                         nBatch=None, method='cosine'):
    if method == 'cosine':
        T_total = args.epochs * nBatch
        T_cur = (epoch % args.epochs) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def accuracy(output, target):
    output = F.softmax(output, dim=-1).max(1)[1]
    n_correct = output.eq(target)
    acc = n_correct.sum().item() / n_correct.shape[0]

    return acc * 100