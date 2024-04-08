from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import argparse
import os
import time
import torch.optim as optim
import pickle
import yaml

import argparse
import logging
import random
import torch.backends.cudnn as cudnn
import numpy as np
from utils_DFC import *
from timm.utils import get_outdir, distribute_bn, update_summary
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.scheduler import create_scheduler
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
from DDNN_demo_1 import DDNN
from dyn_slim.apis.train_slim_gate import validate_gate
from dyn_slim.utils import model_profiling, setup_default_logging, CheckpointSaver, resume_checkpoint, ModelEma
from dyn_slim.apis import train_epoch_slim, validate_slim, train_epoch_slim_gate

config_parser = parser = argparse.ArgumentParser(description='PyTorch train code for Double Dynamic Neural Network')
parser.add_argument('--model', type=str, default='DDNN',
                    help='model to train the dataset')
parser.add_argument('-b', '--batch-size', type=int, default=64,
                    help='mini-batch size')
parser.add_argument('--lr-type', type=str, default='cosine',
                    help='learning rate strategy',
                    choices=['cosine', 'multistep']) 
parser.add_argument('--group-lasso-lambda', type=float, default=1e-5,
                    help='group lasso loss weight')  
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum for sgd')
parser.add_argument('--weight-decay', '--wd', type=float, default=1e-4,
                    help='weight decay')
parser.add_argument('--seed', type=int, default=2022,
                    help='manual seed')
parser.add_argument('--gpu', type=str, default='',
                    help='gpu available')

parser.add_argument('--stages', type=str,
                    help='per layer depth')
parser.add_argument('--squeeze-rate', type=int, default=16,
                    help='squeeze rate in SE head')  
parser.add_argument('--heads', type=int, default=4,
                    help='number of heads for 1x1 convolution')
parser.add_argument('--group-3x3', type=int, default=4,
                    help='3x3 group convolution') 
parser.add_argument('--gate-factor', type=float, default=0.5,
                    help='gate factor')
parser.add_argument('--growth', type=str,
                    help='per layer growth')
parser.add_argument('--bottleneck', type=int, default=4,
                    help='bottleneck in densenet')  

parser.add_argument('--print-freq', type=int, default=10,
                    help='print frequency')
parser.add_argument('--save-freq', type=int, default=10,
                    help='save frequency')
parser.add_argument('--evaluate', type=str, default=None,
                    help="full path to checkpoint to be evaluated")

### DS参数

parser.add_argument('-c', '--config', default='config.yml', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

# Dataset / Model parameters
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='./model_best.pth.tar', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--gp', default='avg', type=str, metavar='POOL',
                    help='Type of global pool, "avg", "max", "avgmax", "avgmaxc" (default: "avg")')
parser.add_argument('--img-size', type=int, default=None, metavar='10',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N',
                    help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=4,
                    metavar='N',
                    help='ratio of validation batch size to training batch size (default: 4)')
parser.add_argument('--drop', type=float, default=0.0, metavar='DROP',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-path', type=float, default=0.0, metavar='DROP',
                    help='Drop connect rate (default: 0.)')
parser.add_argument('--jsd', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
# Optimizer parameters
parser.add_argument('--opt', default='Adam', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
# Learning rate schedule parameters
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--min-step-lr', type=float, default=0, metavar='LR',
                    help='lower lr bound for step schedulers (0)')
parser.add_argument('--epochs', type=int, default=1, metavar='N')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')

parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')  
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')

parser.add_argument('--decay-rate', '--dr', type=float, default=0.0, metavar='RATE',
                    help='LR decay rate (default: 0.1)')
# Augmentation parameters
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='const',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')

parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')
parser.add_argument('--model_ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model_ema_force_cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model_ema_decay', type=float, default=0.997,
                    help='decay factor for model weights moving average (default: 0.9998)')
# Misc
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery_interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N')
parser.add_argument('--num_gpu', type=int, default=4,
                    help='Number of GPUS to use')
parser.add_argument('--save_images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA amp for mixed precision training')
parser.add_argument('--pin_mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no_prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--eval_metric', default='', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "prec1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)

parser.add_argument('--slim_train', action='store_true', default=False)
parser.add_argument('--gate_train', action='store_true', default=False)  
parser.add_argument('--start_chn_idx', type=int, default=0, help='Modify this to change the dynamic routing space.')
parser.add_argument('--inplace_bootstrap', action='store_true', default=False)  
parser.add_argument('--ensemble_ib', action='store_true', default=False)
parser.add_argument('--test_mode', action='store_true', default=False) 
parser.add_argument('--train_mode', action='store_true', default='') 
parser.add_argument('--optimizer_step', action='store_true', default=1)
parser.add_argument('--distributed', action='store_true', default=False)
parser.add_argument('--reset_bn', action='store_true', default=False) 
parser.add_argument('--num_choice', action='store_true', default=4)

args = parser.parse_args()

attack_cat = ['FTP-Patator', 'SSH-Patator', 'DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest', 'Heartbleed',
              'Web Attack - Brute Force', 'Web Attack - Sql Injection', 'Web Attack - XSS', 'Infiltration', 'Bot', 'PortScan', 'DDoS', 'benign']



def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    args = parser.parse_args(remaining)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


class Dataset(torch.utils.data.Dataset):
    """docstring for Dataset"""

    def __init__(self, x, label):
        super(Dataset, self).__init__()
        self.x = np.reshape(x, (-1, 256))
        self.label = label

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.label[idx]


def paired_collate_fn(insts):
    x, label = list(zip(*insts))
    return torch.LongTensor(x), torch.LongTensor(label)


def load_epoch_data(flow_dict, train='train'):
    flow_dict = flow_dict[train]
    x, label = [], []
    for p in attack_cat:
        pkts = flow_dict[p]
        for byte in pkts:
            x.append(byte)
            label.append(attack_cat.index(p))
    a = np.array(label)[:, np.newaxis]

    return np.array(x), np.array(label)[:, np.newaxis]


def main(i, flow_dict):
    global args, best_prec1
    args, args_text = _parse_args()
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = ''
    if args.local_rank == 0:
        output_base = args.output if args.output else './output'
        exp_name = 'train_IDS'
        if args.gate_train:
            exp_name += '-dynamic'
        if args.slim_train:
            exp_name += '-slimmable'
        exp_name += '-{}'.format(args.model)
        exp_info = '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            args.model])
        output_dir = get_outdir(output_base, exp_name, exp_info)
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(checkpoint_dir=output_dir, decreasing=decreasing)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)
    if i == 0:
        setup_default_logging(outdir=output_dir, local_rank=args.local_rank)

    cudnn.benchmark = True
    if args.seed is None:
        args.seed = int(time.time())
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    with open('./results_%d.txt' % i, 'w') as f:
        f.write('Train Loss Time Test\n')
        f.flush()
        ### Create model
        model = DDNN(num_classes=15)
        model = model.cuda()
        model_ema = None
        # ------------- loss_fn --------------
        if args.jsd:
            validate_loss_fn = nn.CrossEntropyLoss().cuda()
        elif args.smoothing:
            validate_loss_fn = nn.CrossEntropyLoss().cuda()
        else:
            train_loss_fn = nn.CrossEntropyLoss().cuda()
            validate_loss_fn = train_loss_fn
        if args.train_mode == 'gate':
            optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.get_gate().parameters()))
        else:
            optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()))
        lr_scheduler, num_epochs = create_scheduler(args, optimizer)
        ### DS代码
        test_x, test_label = load_epoch_data(flow_dict, 'test')
        val_loader = torch.utils.data.DataLoader(
            Dataset(x=test_x, label=test_label),
            num_workers=0,
            collate_fn=paired_collate_fn,
            batch_size=128,
            shuffle=False
        )
        best_prec1 = 0
        ### Optionally resume from a checkpoint

        if args.resume:
            resume_epoch = resume_checkpoint(model, checkpoint_path=args.resume,
                                             optimizer=optimizer if not args.no_resume_opt else None,
                                             log_info=args.local_rank == 0, strict=False)

        args.start_epoch = 0
        start_epoch = 0
        if args.start_epoch is not None:
            # a specified start_epoch will always override the resume epoch
            start_epoch = args.start_epoch
        elif resume_epoch is not None:
            start_epoch = resume_epoch
        if lr_scheduler is not None and start_epoch > 0:
            lr_scheduler.step(start_epoch)

        if args.local_rank == 0:
            logging.info('Scheduled epochs: {}'.format(num_epochs))
        ### Evaluate directly if required
        print(args)

        saveID = None
        for epoch in range(args.start_epoch, args.epochs):
            ### Evaluate on validation set
            eval_sample_list = ['dynamic']
            eval_metrics = [validate_slim(model, val_loader, validate_loss_fn, args, model_mode=model_mode)
                            for model_mode in eval_sample_list]
            test_acc, pred, test_time, losses, flops, eval_metrics = eval_metrics[0]
            p, r, fscore, num = precision_recall_fscore_support(test_label, pred, average='weighted')

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, losses)

            # end training
            if best_metric is not None:
                logging.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))
            # *************************** #
            ### Remember best prec@1 and save checkpoint
            is_best = test_acc >= best_prec1
            best_prec1 = max(test_acc, best_prec1)

            log = ("Epoch %03d/%03d: ACC %.4f" + "| PRE %.4f | REC %.4f | F1 %.4f | Time %s\n") \
                  % (epoch, args.epochs, test_acc, p, r, fscore, time.strftime('%Y-%m-%d %H:%M:%S'))
            print(log)

    return

if __name__ == '__main__':
    for i in range(0, 1):
        with open('./flows_%d_fold_cut.pkl' % i, 'rb') as f:
            flow_dict = pickle.load(f)
        print('====', i, ' fold validation ====')
        main(i, flow_dict)