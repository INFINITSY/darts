import os
import sys
import time
import glob
import numpy as np
import pandas as pd
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from copy import deepcopy
from numpy import linalg as LA
import json
import codecs

from torch.autograd import Variable
from model_search import Network
from architect import Architect
from metrics import Metrics
from sgd import SGDVec
from AdaS import AdaS
from analyze import Analyzer
from adaptive_stop import StopChecker

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.05, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=32, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--scheduler_beta', type=float, default=0.8, help='beta for lr scheduler')
parser.add_argument('--scheduler_p', type=int, default=1, help='p for lr scheduler')
parser.add_argument('--learnable_bn', action='store_true', default=False, help='learnable parameters in batch normalization')
parser.add_argument('--cifar100', action='store_true', default=False, help='search with cifar100 dataset')
parser.add_argument('--compute_hessian', action='store_true', default=False, help='compute or not Hessian')
parser.add_argument('--gumbel', action='store_true', default=False, help='use or not Gumbel-softmax trick')
parser.add_argument('--adaptive_stop', action='store_true', default=False, help='adaptive stopping criterion')

args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# CIFAR_CLASSES = 100
"""From https://github.com/chenxin061/pdarts/"""
if args.cifar100:
    CIFAR_CLASSES = 100
    data_folder = 'cifar-100-python'
else:
    CIFAR_CLASSES = 10
    data_folder = 'cifar-10-batches-py'

device = torch.device("cpu")

def main():
    # if not torch.cuda.is_available():
    #     logging.info('no gpu device available')
    #     sys.exit(1)

    np.random.seed(args.seed)
    # torch.cuda.set_device(args.gpu)
    # cudnn.benchmark = True
    # torch.manual_seed(args.seed)
    # cudnn.enabled = True
    # torch.cuda.manual_seed(args.seed)
    # logging.info('gpu device = %d' % args.gpu)
    # logging.info("args = %s", args)

    torch.manual_seed(args.seed)
    logging.info('use cpu')
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    # criterion = criterion.cuda()
    criterion.to(device)
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, learnable_bn=args.learnable_bn)
    # model = model.cuda()
    model.to(device)
    a = list(model.parameters())

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     args.learning_rate,
    #     momentum=args.momentum,
    #     weight_decay=args.weight_decay)

    ################################################################################
    # AdaS: optimizer and scheduler
    optimizer = SGDVec(
        params=model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    scheduler = AdaS(
        parameters=list(model.parameters()),
        init_lr=args.learning_rate,
        # min_lr=kwargs['min_lr'],
        # zeta=kwargs['zeta'],
        p=args.scheduler_p,
        beta=args.scheduler_beta)
    ################################################################################

    # train_transform, valid_transform = utils._data_transforms_cifar100(args)
    # train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    """From https://github.com/chenxin061/pdarts/"""
    if args.cifar100:
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
    else:
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
    if args.cifar100:
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    else:
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=2)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)
    """Hessian"""
    analyser = Analyzer(model, args)
    """adaptive stopping"""
    stop_checker = StopChecker()

    METRICS = Metrics(list(model.parameters()), p=1)

    PERFORMANCE_STATISTICS = {}
    ARCH_STATISTICS = {}
    GENOTYPE_STATISTICS = {}
    metrics_path = './metrics_stat_test_adas.xlsx'
    weights_path = './weights_stat_test_adas.xlsx'
    genotypes_path = './genotypes_stat_test_adas.xlsx'

    for epoch in range(args.epochs):
        # scheduler.step()
        # lr = scheduler.get_lr()[0]
        # logging.info

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)
        if epoch % 5 == 0 or epoch == args.epochs-1:
            GENOTYPE_STATISTICS[f'epoch_{epoch}'] = [genotype]
            genotypes_df = pd.DataFrame(data=GENOTYPE_STATISTICS)
            genotypes_df.to_excel(genotypes_path)

        print(F.softmax(model.alphas_normal, dim=-1))
        print(F.softmax(model.alphas_reduce, dim=-1))

        # training
        train_acc, train_obj = train(epoch, train_queue, valid_queue,
                                     model, architect, criterion,
                                     optimizer, METRICS, scheduler, analyser)
        logging.info('train_acc %f', train_acc)

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)

        # metrics
        io_metrics = METRICS.evaluate(epoch)
        PERFORMANCE_STATISTICS[f'in_S_epoch_{epoch}'] = io_metrics.input_channel_S
        PERFORMANCE_STATISTICS[f'out_S_epoch_{epoch}'] = io_metrics.output_channel_S
        PERFORMANCE_STATISTICS[f'fc_S_epoch_{epoch}'] = io_metrics.fc_S
        PERFORMANCE_STATISTICS[f'in_rank_epoch_{epoch}'] = io_metrics.input_channel_rank
        PERFORMANCE_STATISTICS[f'out_rank_epoch_{epoch}'] = io_metrics.output_channel_rank
        PERFORMANCE_STATISTICS[f'fc_rank_epoch_{epoch}'] = io_metrics.fc_rank
        PERFORMANCE_STATISTICS[f'in_condition_epoch_{epoch}'] = io_metrics.input_channel_condition
        PERFORMANCE_STATISTICS[f'out_condition_epoch_{epoch}'] = io_metrics.output_channel_condition
        ################################################################################
        # AdaS: update learning rates
        lr_metrics = scheduler.step(epoch, METRICS)
        PERFORMANCE_STATISTICS[f'rank_velocity_epoch_{epoch}'] = lr_metrics.rank_velocity
        PERFORMANCE_STATISTICS[f'learning_rate_epoch_{epoch}'] = lr_metrics.r_conv
        ################################################################################
        # write metrics data to xls file
        metrics_df = pd.DataFrame(data=PERFORMANCE_STATISTICS)
        metrics_df.to_excel(metrics_path)

        # weights
        weights_normal = F.softmax(model.alphas_normal, dim=-1).detach().cpu().numpy()
        weights_reduce = F.softmax(model.alphas_reduce, dim=-1).detach().cpu().numpy()
        # normal
        ARCH_STATISTICS[f'normal_none_epoch{epoch}'] = weights_normal[:, 0]
        ARCH_STATISTICS[f'normal_max_epoch{epoch}'] = weights_normal[:, 1]
        ARCH_STATISTICS[f'normal_avg_epoch{epoch}'] = weights_normal[:, 2]
        ARCH_STATISTICS[f'normal_skip_epoch{epoch}'] = weights_normal[:, 3]
        ARCH_STATISTICS[f'normal_sep_3_epoch{epoch}'] = weights_normal[:, 4]
        ARCH_STATISTICS[f'normal_sep_5_epoch{epoch}'] = weights_normal[:, 5]
        ARCH_STATISTICS[f'normal_dil_3_epoch{epoch}'] = weights_normal[:, 6]
        ARCH_STATISTICS[f'normal_dil_5_epoch{epoch}'] = weights_normal[:, 7]
        # reduce
        ARCH_STATISTICS[f'reduce_none_epoch{epoch}'] = weights_reduce[:, 0]
        ARCH_STATISTICS[f'reduce_max_epoch{epoch}'] = weights_reduce[:, 1]
        ARCH_STATISTICS[f'reduce_avg_epoch{epoch}'] = weights_reduce[:, 2]
        ARCH_STATISTICS[f'reduce_skip_epoch{epoch}'] = weights_reduce[:, 3]
        ARCH_STATISTICS[f'reduce_sep_3_epoch{epoch}'] = weights_reduce[:, 4]
        ARCH_STATISTICS[f'reduce_sep_5_epoch{epoch}'] = weights_reduce[:, 5]
        ARCH_STATISTICS[f'reduce_dil_3_epoch{epoch}'] = weights_reduce[:, 6]
        ARCH_STATISTICS[f'reduce_dil_5_epoch{epoch}'] = weights_reduce[:, 7]
        # write weights data to xls file
        weights_df = pd.DataFrame(data=ARCH_STATISTICS)
        weights_df.to_excel(weights_path)

        # adaptive stopping criterion
        if args.adaptive_stop and epoch >= 10:
            # apply local stopping criterion
            stop_checker.local_stop(METRICS, epoch)
            # freeze some edges based on their knowledge gains
            iteration_p = 0
            for p in model.parameters():
                if ~METRICS.layers_index_todo[iteration_p]:
                    p.requires_grad = False
                    p.grad = None
                iteration_p += 1

        utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(epoch, train_queue, valid_queue, model, architect, criterion, optimizer, metrics, scheduler, analyser):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    lr = scheduler.lr_vector
    layers_todo = metrics.layers_index_todo

    for step, (input, target) in enumerate(train_queue):
        # one mini-batch
        logging.info('train mini batch %03d', step)
        model.train()
        n = input.size(0)

        # input = Variable(input, requires_grad=False).cuda()
        # target = Variable(target, requires_grad=False).cuda(async=True)
        input = Variable(input, requires_grad=False).to(device)
        target = Variable(target, requires_grad=False).to(device)

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        # input_search = Variable(input_search, requires_grad=False).cuda()
        # target_search = Variable(target_search, requires_grad=False).cuda(async=True)
        input_search = Variable(input_search, requires_grad=False).to(device)
        target_search = Variable(target_search, requires_grad=False).to(device)

        logging.info('update arch...')
        architect.step(input, target, input_search, target_search, lr, layers_todo, optimizer, unrolled=args.unrolled)

        logging.info('update weights...')
        optimizer.zero_grad()
        """gdas"""
        logits = model.forward(input, gumbel=args.gumbel)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # optimizer.step()
        ################################################################################
        # AdaS: update optimizer
        optimizer.step(layers_todo, scheduler.lr_vector)
        ################################################################################

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    if args.compute_hessian:
        _data_loader = deepcopy(train_queue)
        input, target = next(iter(_data_loader))

        # input = Variable(input, requires_grad=False).cuda()
        # target = Variable(target, requires_grad=False).cuda(async=True)
        input = Variable(input, requires_grad=False).to(device)
        target = Variable(target, requires_grad=False).to(device)

        # get gradient information
        # param_grads = [p.grad for p in model.parameters() if p.grad is not None]
        # param_grads = torch.cat([x.view(-1) for x in param_grads])
        # param_grads = param_grads.cpu().data.numpy()
        # grad_norm = np.linalg.norm(param_grads)

        # gradient_vector = torch.cat([x.view(-1) for x in gradient_vector])
        # grad_norm = LA.norm(gradient_vector.cpu())
        # logging.info('\nCurrent grad norm based on Train Dataset: %.4f',
        #             grad_norm)
        # logging.info('Compute Hessian start')
        H = analyser.compute_Hw(input, target, input_search, target_search,
                                lr, layers_todo, optimizer, unrolled=False)
        # g = analyser.compute_dw(input, target, input_search, target_search,
        #                         lr, layers_todo, optimizer, unrolled=False)
        # g = torch.cat([x.view(-1) for x in g])

        del _data_loader
        # logging.info('Compute Hessian finished')
        # HESSIAN_STATISTICS[f'hessian_epoch{epoch}'] = weights_normal[:, 0]
        hessian_file = "../save_data/hessian_adas_c100_{0}_epoch_{1}".format(args.file_name, epoch)
        np.save(hessian_file, H.cpu().data.numpy())
        # logging.info('Writing Hessian finished')

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            logging.info('infer mini batch %03d', step)

            # input = Variable(input).cuda()
            # target = Variable(target).cuda(async=True)
            input = Variable(input, volatile=True).to(device)
            target = Variable(target, volatile=True).to(device)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
