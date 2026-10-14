import argparse
import os
import random
import time
import warnings
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
import shutil

from sklearn.metrics import confusion_matrix
from utils import *
from losses import LDAMLoss, FocalLoss
from medical_dataset import MedicalImageDataset

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Medical Image Training')
parser.add_argument('--data_path', type=str,
                    default='/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/LDAM-DRW-master/data/食管_1query',
                    help='path to data root (contains train/val/test subdirs)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--loss_type', default="LDAM", type=str,
                    help='loss type: CE | LDAM | Focal')
parser.add_argument('--train_rule', default='DRW', type=str,
                    help='data sampling strategy: None | Resample | Reweight | DRW')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay', dest='weight_decay')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    help='print frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    default=True, help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int, help='seed for training')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use')
parser.add_argument('--root_log', type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint_tsy_shiguan')
parser.add_argument('--img_size', type=int, default=224, help='input image size')
parser.add_argument('--drw_epoch', type=int, default=80,
                    help='epoch to switch to DRW (epoch // drw_epoch)')
parser.add_argument('--dataset_name', type=str, default='',
                    help='organ/dataset name used as prefix in store_name, '
                         'e.g. 胃, 食管, 十二指肠')

best_acc1 = 0


def main():
    args = parser.parse_args()
    args.store_name = '_'.join([
        args.dataset_name, 'medical', args.arch, args.loss_type, args.train_rule,
        'ep' + str(args.epochs), 'bs' + str(args.batch_size),
        'lr' + str(args.lr).replace('.', '')
    ])
    if not args.dataset_name:
        args.store_name = '_'.join([
            'medical', args.arch, args.loss_type, args.train_rule,
            'ep' + str(args.epochs), 'bs' + str(args.batch_size),
            'lr' + str(args.lr).replace('.', '')
        ])
    prepare_folders(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('Seeding training. This will slow down training.')

    if args.gpu is not None:
        warnings.warn('Using specific GPU.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    num_classes = len([d for d in os.listdir(os.path.join(args.data_path, 'train'))
                       if os.path.isdir(os.path.join(args.data_path, 'train', d))])
    print(f"=> number of classes: {num_classes}")

    use_norm = True if args.loss_type == 'LDAM' else False
    model = models.__dict__[args.arch](pretrained=args.pretrained)

    if hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'classifier'):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                  weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0', weights_only=False)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    img_size = args.img_size

    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = MedicalImageDataset(
        os.path.join(args.data_path, 'train'), transform=transform_train)
    val_dataset = MedicalImageDataset(
        os.path.join(args.data_path, 'val'), transform=transform_val)
    test_dataset = MedicalImageDataset(
        os.path.join(args.data_path, 'test'), transform=transform_val)

    cls_num_list = train_dataset.get_cls_num_list()
    print('cls num list:', cls_num_list)
    args.cls_num_list = cls_num_list

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if not os.path.exists(os.path.join(args.root_log, args.store_name)):
        os.makedirs(os.path.join(args.root_log, args.store_name), exist_ok=True)
    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
    log_val = open(os.path.join(args.root_log, args.store_name, 'log_val.csv'), 'w')
    log_test = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))

    try:
        from tensorboardX import SummaryWriter
        tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    except ImportError:
        tf_writer = None
        warnings.warn('tensorboardX not installed, skipping tensorboard logging')

    start_epoch = getattr(args, 'start_epoch', 0)

    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        if args.train_rule == 'None':
            train_sampler = None
            per_cls_weights = None
        elif args.train_rule == 'Resample':
            train_sampler = ImbalancedDatasetSampler(train_dataset)
            per_cls_weights = None
        elif args.train_rule == 'Reweight':
            train_sampler = None
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        elif args.train_rule == 'DRW':
            train_sampler = None
            betas = [0, 0.9999]
            idx = min(epoch // args.drw_epoch, len(betas) - 1)
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        else:
            per_cls_weights = None
            warnings.warn('Unknown train rule')

        if args.loss_type == 'CE':
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'LDAM':
            criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'Focal':
            criterion = FocalLoss(weight=per_cls_weights, gamma=1).cuda(args.gpu)
        else:
            criterion = nn.CrossEntropyLoss().cuda(args.gpu)

        train(train_loader, model, criterion, optimizer, epoch, args, log_training, tf_writer)

        acc1 = validate(val_loader, model, criterion, epoch, args, log_val, tf_writer, flag='val')

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        print('Best Prec@1: %.3f' % best_acc1)
        log_val.write('Best Prec@1: %.3f\n' % best_acc1)
        log_val.flush()

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best)

    # evaluate on test set with best model
    print("\n=== Final Test Evaluation ===")
    best_ckpt = os.path.join(args.root_model, args.store_name, 'ckpt.best.pth.tar')
    if os.path.exists(best_ckpt):
        checkpoint = torch.load(best_ckpt, map_location='cuda:0', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded best model (epoch {checkpoint['epoch']}, acc {checkpoint['best_acc1']:.3f})")
    else:
        print("Best checkpoint not found, using current model")

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    acc_test = validate(test_loader, model, criterion, args.epochs - 1, args, log_test, tf_writer, flag='test')

    print("\n=== Results ===")
    print(f"Best Val Acc: {best_acc1:.3f}")
    print(f"Test Acc: {acc_test:.3f}")

    log_training.close()
    log_val.close()
    log_test.close()


def train(train_loader, model, criterion, optimizer, epoch, args, log, tf_writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    model.train()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        output = model(input)
        loss = criterion(output, target)

        acc1, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output_str = ('Epoch: [{0}][{1}/{2}], lr: {lr:.6f}\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                              epoch, i, len(train_loader),
                              batch_time=batch_time, data_time=data_time,
                              loss=losses, top1=top1,
                              lr=optimizer.param_groups[0]['lr']))
            print(output_str)
            log.write(output_str + '\n')
            log.flush()

    if tf_writer:
        tf_writer.add_scalar('loss/train', losses.avg, epoch)
        tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
        tf_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)


def validate(val_loader, model, criterion, epoch, args, log=None, tf_writer=None, flag='val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            output = model(input)
            loss = criterion(output, target)

            acc1, _ = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                print_str = ('Test: [{0}/{1}] Time {batch_time.val:.3f} '
                             'Loss {loss.val:.4f} Prec@1 {top1.val:.3f}'.format(
                                 i, len(val_loader), batch_time=batch_time,
                                 loss=losses, top1=top1))
                print(print_str)

    cf = confusion_matrix(all_targets, all_preds).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit / cls_cnt

    result_str = ('{flag} Results: Prec@1 {top1.avg:.3f} Loss {loss.avg:.5f}'.format(
        flag=flag, top1=top1, loss=losses))
    cls_acc_str = '{flag} Class Accuracy: {acc}'.format(
        flag=flag,
        acc=','.join(['%.3f' % x for x in cls_acc]))
    print(result_str)
    print(cls_acc_str)

    if log:
        log.write(result_str + '\n')
        log.write(cls_acc_str + '\n')
        log.flush()

    if tf_writer:
        tf_writer.add_scalar('loss/test_' + flag, losses.avg, epoch)
        tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg, epoch)

    return top1.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Cosine annealing learning rate"""
    lr = args.lr * 0.5 * (1 + np.cos(np.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
