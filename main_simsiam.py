#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import os
import random
from re import A
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from data.crops import CropsDataSet
from utils import yaml_config_hook, save_model
from modules import one_class_loss
import os.path as osp
import numpy as np

import pdb

import simsiam.loader
import simsiam.builder

def main():
    # pdb.set_trace()
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    root = osp.join(os.path.expanduser("~"), "Crops_Dataset")
    list_path = osp.join(root, 'file_lists')
    human_list_path = osp.join(list_path, 'human_lists')
    human_pool_path = os.path.join(human_list_path, args.human_list_filename)
    human_pool = np.loadtxt(human_pool_path, dtype=str)
    human_pool_list = []

    for name in human_pool:
        img_name = name.split('/')[-1]
        human_pool_list.append(img_name)

    args.run_name = "bs_" + str(args.batch_size) + "_projdim_" + str(args.pred_dim) + "_ep_" + str(args.epochs) + "_hlf_" + str(args.human_list_filename)
    if args.semi_supervised_weight != 0:
        args.run_name += "_ss_" + str(args.semi_supervised_weight)
    main_worker(args, human_pool_list)


def main_worker(args, human_pool_list):

    torch.cuda.empty_cache()

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = simsiam.builder.SimSiam(
        models.__dict__[args.arch],
        args.dim, args.pred_dim)

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    model = model.to('cuda')
    #print(model) # print model

    # define loss function (criterion) and optimizer
    criterion = nn.CosineSimilarity(dim=1).cuda(None)#TODO: Check the end of the line

    if args.fix_pred_lr:
        optim_params = [{'params': model.encoder.parameters(), 'fix_lr': False},
                        {'params': model.predictor.parameters(), 'fix_lr': True}]#model.module for parallel computing
    else:
        optim_params = model.parameters()

    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.model_path:
        if os.path.isfile(args.model_path):
            print("=> loading checkpoint '{}'".format(args.model_path))
            checkpoint = torch.load(args.model_path)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.model_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.model_path))

    cudnn.benchmark = True

    # # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    augmentation = [
        transforms.RandomResizedCrop(args.image_size, scale=(0.5, 1.0)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        #transforms.ToTensor(),
        #normalize
    ]
    transformations = torchvision.transforms.Compose(augmentation)

    #initialize dataset
    if args.dataset == "crops":
        train_dataset = CropsDataSet(args.dataset_dir, 
                                    args.list_path, 
                                    args.mean,
                                    transforms=transformations)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=(train_sampler is None),
                                                   drop_last=True,
                                                   num_workers=args.workers,
                                                   sampler=train_sampler,
                                                )#removed pin_memory

    writer = SummaryWriter('runs/' + str(args.run_name))
    writer_simsiam = SummaryWriter('runs/' + str(args.run_name) + '_simsiam_loss')
    writer_one_class = SummaryWriter('runs/' + str(args.run_name) + '_one_class_loss')

    last_best_epoch = -1
    last_best_loss = 0
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        avg_train_loss, avg_simsiam_train_loss, avg_one_class_train_loss = train(train_loader, model, criterion, optimizer, epoch, args, human_pool_list)

        #tensorboard
        writer.add_scalar('training loss', avg_train_loss, epoch + 1)
        writer_simsiam.add_scalar('training loss', avg_simsiam_train_loss, epoch + 1)
        writer_one_class.add_scalar('training loss', avg_one_class_train_loss, epoch + 1)

        #save if best run or multiple of 200
        if last_best_epoch == -1 or avg_train_loss < last_best_loss:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            },True,last_best_epoch,args)
            last_best_epoch = epoch + 1
            last_best_loss = avg_train_loss
        if (epoch + 1) % 200 == 0 and args.epochs - epoch >= 200:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            },False,-1,args)
    save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        },False,-1,args)


def train(train_loader, model, criterion, optimizer, epoch, args, human_pool_list):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    losses_simsiam = AverageMeter('Loss', ':.4f')
    losses_one_class = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    loss_device = torch.device("cuda")
    criterion_one_class = one_class_loss.OneClassLoss(loss_device)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (x_og, x_i, x_j, _, _, names, _) in enumerate(train_loader):
        x_og = x_og.to('cuda')
        x_i = x_i.to('cuda')
        x_j = x_j.to('cuda')
        images = [x_i, x_j]

        #Find which samples in the current batch belong to the human pool
        indexes = list()
        for index, name in enumerate(names):
            if name in human_pool_list:
                indexes.append(index)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output and loss (NOTE: both losses kept negative)
        p1, p2, z0, z1, z2 = model(x_og, x_i, x_j)
        selected_embeddings = z0[indexes]
        loss_one_class = -criterion_one_class(selected_embeddings)
        loss_simsiam = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
        loss = loss_simsiam * (1 - args.semi_supervised_weight) + loss_one_class * args.semi_supervised_weight

        losses.update(loss.item(), images[0].size(0))
        losses_simsiam.update(loss_simsiam.item(), images[0].size(0))
        losses_one_class.update(loss_one_class.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return losses.avg, losses_simsiam.avg, losses_one_class.avg#return avg loss for the epoch


def save_checkpoint(state, is_best, last_best_epoch, args):
    if is_best:
        #save new file, remove last file
        filename = os.path.join(args.model_path, args.dataset, args.run_name + "_best_checkpoint_" + str(state['epoch']) +".tar")
        torch.save(state, filename)
        if last_best_epoch != -1:
            filename = os.path.join(args.model_path, args.dataset, args.run_name + "_best_checkpoint_" + str(last_best_epoch) +".tar")
            os.remove(filename)
    else:
        filename = os.path.join(args.model_path, args.dataset, args.run_name + "_checkpoint_" + str(state['epoch']) +".tar")
        torch.save(state, filename)


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


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


if __name__ == '__main__':
    main()
