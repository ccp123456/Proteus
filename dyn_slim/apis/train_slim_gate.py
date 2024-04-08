import logging
import time
from collections import OrderedDict

from DDNN_demo_1 import MultiHeadGate
from dyn_slim.utils import add_flops
from tqdm import tqdm, trange

from timm.utils import AverageMeter, reduce_tensor
from utils_DFC import *
import numpy as np
import torch
import torch.nn as nn
import random

model_mac_hooks = []

seed = 2022
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def accuracy(output, target):
    output = F.softmax(output, dim=-1).max(1)[1]
    n_correct = output.eq(target)
    acc = n_correct.sum().item() / n_correct.shape[0]

    return acc * 100

def accuracy_gate(output, target, no_reduce=False):
    """Computes the precision@k for the specified values of k"""
    # output = F.softmax(output[0], dim=-1).max(1)[1]
    batch_size = target.size(0)

    conf, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    if no_reduce:
        return conf, correct
    return correct[:1].reshape(-1).float().sum(0) * 100. / batch_size

def train_epoch_slim_gate(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir='', use_amp=False, model_ema=None,
        optimizer_step=1):
    start_chn_idx = args.start_chn_idx
    num_gate = 1

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    # lasso_losses_m = AverageMeter()
    acc_m = AverageMeter()
    flops_m = AverageMeter()
    ce_loss_m = AverageMeter()
    flops_loss_m = AverageMeter()
    acc_gate_m_l = [AverageMeter() for i in range(num_gate)]
    gate_loss_m_l = [AverageMeter() for i in range(num_gate)]
    model.train()
    for n, m in model.named_modules():  # Freeze bn
        if isinstance(m, nn.BatchNorm1d):
            m.eval()

    for n, m in model.named_modules():
        if len(getattr(m, 'in_channels_list', [])) > 4:
            m.in_channels_list = m.in_channels_list[start_chn_idx:4]
            m.in_channels_list_tensor = torch.from_numpy(
                np.array(m.in_channels_list)).float().cuda()
        if len(getattr(m, 'out_channels_list', [])) > 4:
            m.out_channels_list = m.out_channels_list[start_chn_idx:4]
            m.out_channels_list_tensor = torch.from_numpy(
                np.array(m.out_channels_list)).float().cuda()

    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    model.apply(lambda m: add_mac_hooks(m))
    batch_idx = 1
    for batch in tqdm(loader, mininterval=2, desc='  - (Training)   ', leave=False):
    # for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        input, target = batch
        input, target = input.cuda(), target.cuda()
        target = target.contiguous().view(-1)
        ## Adjust learning rate
        # lr = adjust_learning_rate(optimizer, epoch, args, batch=batch_idx, nBatch=len(loader), method=args.lr_type)

        if last_batch or (batch_idx + 1) % optimizer_step == 0:
            optimizer.zero_grad()
        # generate online labels
        with torch.no_grad():
            set_model_mode(model, 'smallest')
            output,_ = model(input)
            conf_s, correct_s = accuracy_gate(output, target, no_reduce=True)
            gate_target = [torch.LongTensor([0]) if correct_s[0][idx] else torch.LongTensor([3])
                           for idx in range(correct_s[0].size(0))]
            # print(gate_target)
            gate_target = torch.stack(gate_target).squeeze(-1).cuda()
        # =============
        set_model_mode(model, 'dynamic')
        begin = time.time()
        output,_ = model(input)
        end = time.time()

        if hasattr(model, 'module'):
            model_ = model.module
        else:
            model_ = model

        #  SGS Loss
        gate_loss = 0
        gate_num = 0
        gate_loss_l = []
        gate_acc_l = []
        for n, m in model_.named_modules():
            if isinstance(m, MultiHeadGate):
                if getattr(m, 'keep_gate', None) is not None:
                    gate_num += 1
                    g_loss = loss_fn(m.keep_gate, gate_target)
                    gate_loss += g_loss
                    gate_loss_l.append(g_loss)
                    gate_acc_l.append(torch.tensor(accuracy(m.keep_gate, gate_target)))
                else :
                    gate_num = 1
        gate_loss /= gate_num

        #  MAdds Loss
        running_flops = add_flops(model)
        if isinstance(running_flops, torch.Tensor):
            running_flops = running_flops.float().mean().cuda()
        else:
            running_flops = torch.FloatTensor([running_flops]).cuda()
        flops_loss = (running_flops / 1e5) ** 2

        #  Target Loss, back-propagate through gumbel-softmax
        ce_loss = loss_fn(output, target)

        # ## Add group lasso loss,DGC
        # lasso_loss = 0
        # if args.group_lasso_lambda > 0:
        #     for lasso_m in _lasso_list:
        #         lasso_loss = lasso_loss + lasso_m.mean()

        loss = gate_loss + ce_loss
        # loss = ce_loss
        acc1 = torch.tensor(accuracy(output, target))

        loss.backward()
        if last_batch or (batch_idx + 1) % optimizer_step == 0:
            optimizer.step()

        torch.cuda.synchronize()
        num_updates += 1

        losses_m.update(loss.item(), input.size(0))
        # lasso_losses_m.update(lasso_loss.item())
        acc_m.update(acc1.item(), input.size(0))
        flops_m.update(running_flops.item(), input.size(0))
        ce_loss_m.update(ce_loss.item(), input.size(0))
        flops_loss_m.update(flops_loss.item(), input.size(0))
        reduced_acc_gate_l = gate_acc_l[0]
        reduced_gate_loss_l = gate_loss_l[0]
        acc_gate_m_l[0].update(reduced_acc_gate_l.item(), input.size(0))
        gate_loss_m_l[0].update(reduced_gate_loss_l.item(), input.size(0))
        batch_time_m.update(end - begin)
        if (last_batch or batch_idx % args.log_interval == 0) and args.local_rank == 0 and batch_idx != 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            print_gate_stats(model)
            logging.info(
                'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                'CELoss: {celoss.val:>9.6f} ({celoss.avg:>6.4f})  '
                'GateLoss: {gate_loss[0].val:>6.4f} ({gate_loss[0].avg:>6.4f})  '
                'FlopsLoss: {flopsloss.val:>9.6f} ({flopsloss.avg:>6.4f})  '
                'TrainAcc: {acc.val:>9.6f} ({acc.avg:>6.4f})  '
                'GateAcc: {acc_gate[0].val:>6.4f}({acc_gate[0].avg:>6.4f})  '
                'Flops: {flops.val:>6.0f} ({flops.avg:>6.0f})  '
                'LR: {lr:.3e}  '
                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                'DataTime: {data_time.val:.3f} ({data_time.avg:.3f})\n'.format(
                    epoch,
                    batch_idx, last_idx,
                    100. * batch_idx / last_idx,
                    loss=losses_m,
                    flopsloss=flops_loss_m,
                    # lasso_loss=lasso_losses_m,
                    acc=acc_m,
                    flops=flops_m,
                    celoss=ce_loss_m,
                    batch_time=batch_time_m,
                    rate=input.size(0) / batch_time_m.val,
                    rate_avg=input.size(0) / batch_time_m.avg,
                    lr=lr,
                    data_time=data_time_m,
                    gate_loss=gate_loss_m_l,
                    acc_gate=acc_gate_m_l
                )
            )
        batch_idx = batch_idx + 1

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(
                model, optimizer, args, epoch, model_ema=model_ema, use_amp=use_amp,
                batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return acc_m.avg, losses_m.avg, lr, OrderedDict([('loss', losses_m.avg)])


@torch.no_grad()
def validate_gate(model, loader, loss_fn, args, log_suffix=''):
    start_chn_idx = args.start_chn_idx
    num_gate = 1

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    flops_m = AverageMeter()
    prec_m = AverageMeter()
    acc_gate_m_l = [AverageMeter() for i in range(num_gate)]
    total_pred = []
    ## Switch to evaluate mode
    model.eval()

    for n, m in model.named_modules():
        if len(getattr(m, 'in_channels_list', [])) > 4:
            m.in_channels_list = m.in_channels_list[start_chn_idx:4]
            m.in_channels_list_tensor = torch.from_numpy(
                np.array(m.in_channels_list)).float().cuda()
        if len(getattr(m, 'out_channels_list', [])) > 4:
            m.out_channels_list = m.out_channels_list[start_chn_idx:4]
            m.out_channels_list_tensor = torch.from_numpy(
                np.array(m.out_channels_list)).float().cuda()

    last_idx = len(loader) - 1
    model.apply(lambda m: add_mac_hooks(m))
    batch_idx = 1
    for batch in tqdm(loader, mininterval=2, desc='  - (Testing)   ', leave=False):
    # for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        input, target = batch
        input, target = input.cuda(), target.cuda()
        target = target.contiguous().view(-1)
        # generate online labels
        with torch.no_grad():
            set_model_mode(model, 'smallest')
            output,_ = model(input)
            conf_s, correct_s = accuracy_gate(output, target, no_reduce=True)
            gate_target = [torch.LongTensor([0]) if correct_s[0][idx] else torch.LongTensor([3])
                           for idx in range(correct_s[0].size(0))]
            gate_target = torch.stack(gate_target).squeeze(-1).cuda()
        # =============
        
        set_model_mode(model, 'dynamic')
        begin = time.time()
        output,_ = model(input)
        end = time.time()
        data_time_m.update(end - begin)

        if hasattr(model, 'module'):
            model_ = model.module
        else:
            model_ = model

        gate_acc_l = []
        for n, m in model_.named_modules():
            if isinstance(m, MultiHeadGate):
                if getattr(m, 'keep_gate', None) is not None:
                    gate_acc_l.append(torch.tensor(accuracy(m.keep_gate, gate_target)))

        running_flops = add_flops(model)
        if isinstance(running_flops, torch.Tensor):
            running_flops = running_flops.float().mean().cuda()
        else:
            running_flops = torch.FloatTensor([running_flops]).cuda()

        loss = loss_fn(output, target)
        prec = torch.tensor(accuracy(output, target))

        losses_m.update(loss.item(), input.size(0))
        prec_m.update(prec.item(), input.size(0))
        flops_m.update(running_flops.item(), input.size(0))
        output = F.softmax(output, dim=-1).max(1)[1]
        total_pred.extend(output.long().tolist())

        batch_time_m.update(time.time() - end)
        if (last_batch or batch_idx % args.log_interval == 0) and args.local_rank == 0 and batch_idx != 0:
            print_gate_stats(model)
            log_name = 'Test' + log_suffix
            logging.info(
                '{}: [{:>4d}/{} ({:>3.0f}%)]  '
                'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                'Acc@: {prec.val:>9.6f} ({prec.avg:>6.4f})  '
                'GateAcc: {acc_gate[0].val:>6.4f}({acc_gate[0].avg:>6.4f})  '
                'Flops: {flops.val:>6.0f} ({flops.avg:>6.0f})  '
                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                'DataTime: {data_time.val:.3f} ({data_time.avg:.3f})\n'.format(
                    log_name,
                    batch_idx, last_idx,
                    100. * batch_idx / last_idx,
                    loss=losses_m,
                    prec=prec_m,
                    flops=flops_m,
                    batch_time=batch_time_m,
                    rate=input.size(0) / batch_time_m.val,
                    rate_avg=input.size(0) / batch_time_m.avg,
                    data_time=data_time_m,
                    acc_gate=acc_gate_m_l
                )
            )

        end = time.time()
        metrics = OrderedDict(
            [('prec', prec_m.avg), ('loss', losses_m.avg), ('flops', flops_m.avg)])
    return prec_m.avg, total_pred, batch_time_m, losses_m.avg, flops_m.avg, metrics


def reduce_list_tensor(tensor_l, world_size):
    ret_l = []
    for tensor in tensor_l:
        ret_l.append(reduce_tensor(tensor, world_size))
    return ret_l


def set_gate(m, gate=None):
    if gate is not None:
        gate = gate.cuda()
    if hasattr(m, 'gate'):
        setattr(m, 'gate', gate)

def module_mac(self, input, output):
    if isinstance(input[0], tuple):
        if isinstance(input[0][0], list):
            ins = input[0][0][3].size()
        else:
            ins = input[0][0].size()
    else:
        ins = input[0].size()
    if isinstance(output, tuple):
        if isinstance(output[0], list):
            outs = output[0][3].size()
        else:
            outs = output[0].size()
    elif isinstance(output, list):
        outs = output[0].size()
    else:
        outs = output.size()
    if isinstance(self, (nn.Conv1d, nn.ConvTranspose1d)):
        # print(type(self.running_inc), type(self.running_outc), type(self.running_kernel_size), type(outs[2]), type(self.running_groups))
        if hasattr(self, 'running_inc'):
            self.running_flops = (self.running_inc * self.running_outc * self.running_kernel_size * outs[2] / self.running_groups)
        else:
            self.running_flops = self.in_channels * self.out_channels * self.kernel_size[0] * outs[2] / self.groups
        # print(type(self), self.running_flops.mean().item() if isinstance(self.running_flops, torch.Tensor) else self.running_flops)
    elif isinstance(self, nn.Linear):
        if hasattr(self, 'running_inc'):
            self.running_flops = self.running_inc * self.running_outc
        else:
            self.running_flops = self.in_features * self.out_features
        # print(type(self), self.running_flops.mean().item() if isinstance(self.running_flops, torch.Tensor) else self.running_flops)
    elif isinstance(self, (nn.AvgPool1d, nn.AdaptiveAvgPool1d)):
        # NOTE: this function is correct only when stride == kernel size
        if hasattr(self, 'running_inc'):
            self.running_flops = self.running_inc * ins[2]
        else:
            self.running_flops = 0
        # print(type(self), self.running_flops.mean().item() if isinstance(self.running_flops, torch.Tensor) else self.running_flops)
    return

def add_mac_hooks(m):
    global model_mac_hooks
    model_mac_hooks.append(
        m.register_forward_hook(lambda m, input, output: module_mac(
            m, input, output)))


def remove_mac_hooks():
    global model_mac_hooks
    for h in model_mac_hooks:
        h.remove()
    model_mac_hooks = []


def set_model_mode(model, mode):
    if hasattr(model, 'module'):
        model.module.set_mode(mode)
    else:
        model.set_mode(mode)


def print_gate_stats(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    for n, m in model.named_modules():
        if isinstance(m, MultiHeadGate) and getattr(m, 'print_gate', None) is not None:
            logging.info('{}: {}'.format(n, m.print_gate.sum(0)))