#!/usr/bin/env python
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import byol.builder
import byol.loader
import wandb

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning_rate', default=0.5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save_freq', default=1000, type=int,
                    metavar='N', help='save frequency (default: 20)')
parser.add_argument('--input_size', default=224, type=int, help='input resolution (default 224x224)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to pretrained checkpoint (default: none) used for finetune')
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--save_dir', type=str, default='checkpoints', help='where to save models')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--byol_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--byol_hidden_dim', default=512, type=int,
                    help='hidden dimension (default: 512)')

# options for moco v2
parser.add_argument('--aug_plus', action='store_true',
                    help='')
# parser.add_argument('--cos', action='store_true',
#                     help='use cosine lr schedule')
parser.add_argument('--cos', type=int, default=1)


parser.add_argument('--syncBN', action='store_true',
                    help='using synchronized batch normalization')
parser.add_argument('--warm', action='store_true',
                    help='warm-up for large batch training')
parser.add_argument('--load_pretrain', type=str, default='no')

def main():
    args = parser.parse_args()
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    args.save_dir = os.path.join(args.save_dir, 'byol_finetune', str(args.input_size), args.arch)
    if args.pretrained:
        args.save_dir = os.path.join(args.save_dir, 'pretrained_from_{}'.format(args.pretrained.split('/')[-1]))
    args.save_dir = os.path.join(args.save_dir, 'gpus_{}_lr_{}_bs_{}_dim_{}_hdim_{}_epochs_{}_path_{}'.format(len(args.gpus.split(',')),
                                                                                     args.lr,
                                                                                     args.batch_size,
                                                                                     args.byol_dim,
                                                                                     args.byol_hidden_dim,
                                                                                     args.epochs,
                                                                                     args.data.split('/')[-1]))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    wandb_name = 'isize_{}_lr_{}_bs_{}_dim_{}_hdim_{}_aug_plus_{}_epochs_{}_path_{}'.format(args.input_size,
                                                                                args.lr,
                                                                                args.batch_size,
                                                                                args.byol_dim,
                                                                                args.byol_hidden_dim,
                                                                                args.aug_plus,
                                                                                args.epochs,
                                                                                args.data.split('/')[-1])
    wandb.init(
            config=args,
            name = wandb_name,
            project='selfsup',
            anonymous='allow'
        )

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = byol.builder.BYOL(models.__dict__[args.arch](pretrained=False), image_size=args.input_size, projection_hidden_size=args.byol_hidden_dim, projection_size=args.byol_dim)
    
    args.warmup_epochs = 0
    # warm-up for large-batch training,
    if args.batch_size > 256:
        args.warm = True
    if args.warm:
        args.warmup_from = 0.01
        args.warmup_epochs = 10

    # AllGather implementation (batch shuffle, queue update, etc.) in
    # this code only supports DistributedDataParallel.
    model = torch.nn.DataParallel(model)
    model = model.cuda(args.gpu)

    criterion = None
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading pretrain weight '{}'".format(args.pretrained))
            if args.gpu is None:
                checkpoint = torch.load(args.pretrained)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.pretrained, map_location=loc)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded pretrained weight '{}'".format(args.pretrained))
        else:
            print("=> no pretrained weight found at '{}'".format(args.pretrained))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([byol.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
        
    # for celebA
    else:
        augmentation = [
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0, 0)  # not strengthened
            ], p=0.8),
            transforms.RandomApply([byol.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    train_dataset = datasets.ImageFolder(
        traindir,
        byol.loader.TwoCropsTransform(transforms.Compose(augmentation)))

    train_sampler = None
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    train_start = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        #adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            if epoch % args.save_freq == 0 or epoch==args.epochs-1:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.module.state_dict(), #model.module.online_encoder.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, is_best=False, root=args.save_dir, filename='checkpoint_{:04d}.pth.tar'.format(epoch))
    train_end = time.time()

    print('total training time elapses {} hours'.format((train_end-train_start)/3600.0))

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, i, len(train_loader), args)
        # measure data loading time
        data_time.update(time.time() - end)

        #bsz = images[0].size(0)
        #images = torch.cat([images[0], images[1]], dim=0)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
        # compute output
        loss = model(image_one=images[0], image_two=images[1])
        #loss = loss.mean()

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        losses.update(loss.item(), images[0].size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.module.update_moving_average()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
    progress.display(i)
    metrics = {
        'train_loss': loss.item(),
        'train_loss_avg': losses.avg,
    }
    wandb.log(metrics)
    # wandb.log({'epoch_loss_avg':losses.avg})
    
def save_checkpoint(state, is_best, root, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(root, filename))
    if is_best:
        shutil.copyfile(os.path.join(root, filename), os.path.join(root, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return 0.5 * (1. + math.cos(math.pi * current / rampdown_length))
    #return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch, args):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    real_epoch = epoch + step_in_epoch / total_steps_in_epoch

    if args.warm:
    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
        lr = linear_rampup(real_epoch, args.warmup_epochs) * (args.lr - args.warmup_from) + args.warmup_from

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    #"""
    if args.cos:  # cosine lr schedule
        if epoch >= args.warmup_epochs:
            lr *= cosine_rampdown(epoch-args.warmup_epochs, args.epochs-args.warmup_epochs)
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    #"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
