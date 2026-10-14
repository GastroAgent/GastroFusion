import argparse
import json
import builtins
import math
import os

import numpy as np
os.environ['OPENBLAS_NUM_THREADS'] = '7'
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
import torchvision.models as models
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

from models import resnet_imagenet
from randaugment import rand_augment_transform, GaussianBlur
import moco.loader
import moco.builder
from dataset.imagenet import ImageNetLT
from dataset.inat import INaturalist
from dataset.imagenet_moco import ImageNetLT_moco
from dataset.inat_moco import INaturalist_moco
from dataset.gastric import Gastric_moco
from losses import PaCoLoss, BalSCL, BPaCoLoss
from utils import shot_acc
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import timm
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
model_names += ['resnext101_32x4d']

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='gastric', choices=['inat', 'imagenet', 'ISIC2018', 'gastric'])
parser.add_argument('--data', metavar='DIR', default='./data')
parser.add_argument('--root_path', type=str, default='./data')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.02, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:9996', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=1314, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', default=False, action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--no-distributed', action='store_true',
                    help='Disable distributed training (use single GPU)')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=8192, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.2, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', default=True, type=bool,
                    help='use mlp head')
parser.add_argument('--aug-plus', default=True, type=bool,
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', default=True, type=bool,
                    help='use cosine lr schedule')
parser.add_argument('--normalize', default=False, type=bool,
                    help='use cosine lr schedule')

# options for paco
parser.add_argument('--mark', default=None, type=str,
                    help='log dir')
parser.add_argument('--reload', default=None, type=str,
                    help='load supervised model')
parser.add_argument('--warmup_epochs', default=10, type=int,
                    help='warmup epochs')
parser.add_argument('--alpha', default=1.0, type=float,
                    help='contrast weight among samples')
parser.add_argument('--beta', default=1.0, type=float,
                    help='contrast weight between centers and samples')
parser.add_argument('--gamma', default=1.0, type=float,
                    help='paco loss')
parser.add_argument('--aug', default=None, type=str,
                    help='aug strategy')
parser.add_argument('--randaug_m', default=10, type=int, help='randaug-m')
parser.add_argument('--randaug_n', default=2, type=int, help='randaug-n')
parser.add_argument('--num_classes', default=7, type=int, help='num classes in dataset')
parser.add_argument('--feat_dim', default=2048, type=int,
                    help='last feature dim of backbone')

# fp16
parser.add_argument('--fp16', action='store_true', help=' fp16 training')

# Test-Test episodic evaluation
parser.add_argument('--eval-protocol', default='test_test',
                    choices=['test_test', 'full'],
                    help='evaluation protocol; test_test = query/support both sampled from test')
parser.add_argument('--n-way', default=7, type=int,
                    help='N-way for episodic Test-Test evaluation')
parser.add_argument('--n-shot', default=5, type=int,
                    help='K-shot support per class for episodic Test-Test evaluation')
parser.add_argument('--n-query', default=1, type=int,
                    help='query images per class per episode')
parser.add_argument('--n-episodes', default=100, type=int,
                    help='number of Test-Test episodes')
parser.add_argument('--organ', default=None, type=str,
                    help='organ name for gastric txt files, e.g. 胃/食管/十二指肠; '
                         'default: infer from basename of --data')
parser.add_argument('--test-txt', default=None, type=str,
                    help='optional explicit test txt path')
parser.add_argument('--train-txt', default=None, type=str,
                    help='optional explicit train txt path (training/full-eval only)')
parser.add_argument('--val-txt', default=None, type=str,
                    help='optional explicit val txt path (training validation only)')


best_acc = 0
def main():
    args = parser.parse_args()
    args.root_model = f'{args.root_path}/{args.dataset}/{args.mark}'
    os.makedirs(args.root_model, exist_ok=True)

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

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 and args.multiprocessing_distributed and not args.no_distributed

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
    global best_acc
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # Check if we should use distributed or single-GPU mode
    use_single_gpu = args.no_distributed or args.world_size == 1 or not args.multiprocessing_distributed
    args.distributed = args.world_size > 1 and args.multiprocessing_distributed and not args.no_distributed

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    # num_classes = 8142 if args.dataset == 'inat' else 1000
    print("=> creating model '{}'".format(args.arch))
    model = moco.builder.MoCo(
        models.__dict__[args.arch] if args.arch != 'resnext101_32x4d' else getattr(resnet_imagenet, args.arch),
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, args.feat_dim, num_classes=args.num_classes)
    print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

            filename=f'{args.root_model}/moco_ckpt.best.pth.tar'
            if (not args.resume) and os.path.exists(filename):
               args.resume = filename

            if args.reload:
               state_dict = model.state_dict()
               state_dict_ssp = torch.load(args.reload)['state_dict']

               print(state_dict_ssp.keys())

               for key in state_dict.keys():
                      print(key)
                      if key in state_dict_ssp.keys() and state_dict[key].shape == state_dict_ssp[key].shape:
                         state_dict[key]=state_dict_ssp[key]
                         print(key+" ****loaded******* ")
                      else:
                         print(key+" ****unloaded******* ")
               model.load_state_dict(state_dict)

        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # Single-GPU / CPU mode - no DistributedDataParallel needed
        pass

    # define loss function (criterion) and optimizer
    criterion_ce = nn.CrossEntropyLoss().cuda(args.gpu)
    # criterion = PaCoLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma, temperature=args.moco_t, K=args.moco_k, num_classes=args.num_classes).cuda(args.gpu)
    criterion = BPaCoLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma, temperature=args.moco_t, K=args.moco_k, num_classes=args.num_classes).cuda(args.gpu)
    # criterion = AbPaCoLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma, temperature=args.moco_t, K=args.moco_k, num_classes=args.num_classes).cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Check for existing checkpoint in single-GPU mode
    if not args.distributed:
        filename = f'{args.root_model}/moco_ckpt.best.pth.tar'
        # Respect an explicitly supplied --resume; only auto-discover when empty.
        if (not args.resume) and os.path.exists(filename):
            args.resume = filename

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')

    # Set up dataset-specific paths
    if args.dataset == 'gastric':
        # Infer organ from data directory, e.g. 食管_1query -> 食管.
        inferred_organ = os.path.basename(os.path.normpath(args.data)).replace('_1query', '')
        organ = args.organ if args.organ is not None else inferred_organ

        txt_train = args.train_txt if args.train_txt is not None else os.path.join(
            args.data, f'{organ}_train.txt'
        )
        txt_val = args.val_txt if args.val_txt is not None else os.path.join(
            args.data, f'{organ}_val.txt'
        )
        txt_test = args.test_txt if args.test_txt is not None else os.path.join(
            args.data, f'{organ}_test.txt'
        )

        # Training: train -> txt_train, validation -> txt_val.
        # Evaluation: Test-Test uses txt_test.
        for split_name, split_path in (
            ('train', txt_train),
            ('val', txt_val),
            ('test', txt_test),
        ):
            if not os.path.isfile(split_path):
                raise FileNotFoundError(
                    f"{split_name} txt not found: {split_path}\n"
                    f"Expected files like {organ}_train.txt / {organ}_val.txt / {organ}_test.txt "
                    f"under {args.data}"
                )

        print('=' * 80)
        print('[GASTRIC DATA SPLITS]')
        print(f'Organ     : {organ}')
        print(f'Train txt : {txt_train}')
        print(f'Val txt   : {txt_val}')
        print(f'Test txt  : {txt_test}')
        print('=' * 80)

        # Use gastric-specific normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    elif args.dataset == 'ISIC2018':
        txt_train = './data/ISIC2018/ISIC2018_train.txt'
        txt_val = './data/ISIC2018/ISIC2018_val.txt'
        txt_test = txt_val
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif args.dataset == 'inat':
        txt_train = os.path.join(args.data, 'train.txt')
        txt_val = os.path.join(args.data, 'val.txt')
        txt_test = txt_val
        normalize = transforms.Normalize(mean=[0.466, 0.471, 0.380], std=[0.195, 0.194, 0.192])
    else:  # imagenet
        txt_train = os.path.join(args.data, 'train.txt')
        txt_val = os.path.join(args.data, 'val.txt')
        txt_test = txt_val
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    augmentation_sim = [
            transforms.RandomResizedCrop(224),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
            ], p=1.0),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
    ]

    rgb_mean = (0.485, 0.456, 0.406)
    ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),)
    augmentation_randnclsstack = [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.randaug_n, args.randaug_m), ra_params),
            transforms.ToTensor(),
            normalize,
    ]

    augmentation_randncls = [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.randaug_n, args.randaug_m), ra_params),
            transforms.ToTensor(),
            normalize,
    ]

    val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

    # During training, select best checkpoint on the validation split.
    # During explicit evaluation, use the test split.
    eval_txt = txt_test if args.evaluate else txt_val
    eval_split_name = 'test' if args.evaluate else 'val'
    print(f'[DATA] evaluation loader split = {eval_split_name}: {eval_txt}')

    val_dataset = INaturalist(
        root=args.data,
        txt=eval_txt,
        transform=val_transform
    ) if args.dataset == 'inat' else ImageNetLT(
        root=args.data,
        txt=eval_txt,
        transform=val_transform)

    # ==============================================================
    # Test-Test episodic evaluation:
    #   Query   <- test
    #   Support <- test
    # Query/support are disjoint within each episode.
    #
    # BPaCo is a fixed parametric classifier at evaluation time:
    # support is sampled to match the few-shot protocol, but is NOT
    # used to update/recompute the classifier prediction.
    # ==============================================================
    if args.evaluate and args.eval_protocol == 'test_test':
        print(" start Test-Test episodic evaluation **** ")
        test_test_matrix(val_dataset, model, args)
        return

    if args.aug == 'sim_sim':
       transform_train = [transforms.Compose(augmentation_sim), transforms.Compose(augmentation_sim)]
    elif args.aug == 'randcls_sim':
         transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_sim)]
    elif args.aug == 'randclsstack_sim':
         transform_train = [transforms.Compose(augmentation_randnclsstack), transforms.Compose(augmentation_sim)]
    elif args.aug == 'randcls_randclsstack':
         transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_randnclsstack)]
    else:
         transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_sim)]

    # Select dataset class based on dataset type
    if args.dataset == 'gastric':
        train_dataset = Gastric_moco(
            root=args.data,
            txt=txt_train,
            num_classes=args.num_classes,
            transform=transform_train
        )
    elif args.dataset == 'inat':
        train_dataset = INaturalist_moco(
            root=args.data,
            txt=txt_train,
            transform=transform_train
        )
    else:  # ImageNetLT, ISIC2018, etc.
        train_dataset = ImageNetLT_moco(
            root=args.data,
            txt=txt_train,
            num_classes=args.num_classes,
            transform=transform_train
        )
    print(f'===> Training data length {len(train_dataset)}')
    # criterion_bcl = BalSCL(cls_num_list=train_dataset.cls_num_list, temperature=args.moco_t).cuda(args.gpu)
    criterion_bpaco = BPaCoLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma, temperature=args.moco_t, K=args.moco_k, num_classes=args.num_classes, cls_num_list1=train_dataset.cls_num_list).cuda(args.gpu)
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    criterion.cal_weight_for_classes(train_dataset.cls_num_list)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        print(" start full-set evaluation **** ")
        test_matrix(val_loader, model, args, train_loader)
        return

    # mixed precision 
    scaler = GradScaler()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, criterion_bpaco, criterion_ce, optimizer, epoch, scaler, args)
        acc, many, med, few, f1 = validate(val_loader, train_loader, model, criterion, args, train_dataset)
        # remember best acc@1 and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        if is_best:
            best_many = many
            best_med = med
            best_few = few
            best_f1=f1
        

        if not args.distributed or \
                (args.distributed and args.rank % ngpus_per_node == 0):
            print("训练次数:{},{}".format(epoch, best_acc))
            print('Best Prec@1: {:.4f}, Many Prec@1: {:.4f}, Med Prec@1: {:.4f}, Few Prec@1: {:.4f}, F1{:.4f}'.format(best_acc,
                                                                                                        best_many,
                                                                                                        best_med,
                                                                                                        best_few,best_f1))
            save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'acc': acc,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=is_best, filename=f'{args.root_model}/moco_ckpt.pth.tar')

# def saliency_score(img_saliency):
#     B=img_saliency.shape[0]
#     patch_score=F.avg_pool2d(img_saliency,16,16).reshape(B,196)
#     min_i, _ = torch.min(patch_score, dim=1, keepdim=True)
#     max_i, _ = torch.max(patch_score, dim=1, keepdim=True)
#     patch_score = (patch_score - min_i) / (max_i-min_i)
#     patch_score=patch_score.reshape(B,14,14)
#     return patch_score


def _extract_dataset_targets(dataset):
    """
    Return one integer label per item without assuming a specific dataset class.
    Prefer cached attributes; fall back to reading dataset items once.
    """
    for attr in ("targets", "labels"):
        if hasattr(dataset, attr):
            values = getattr(dataset, attr)
            try:
                arr = np.asarray(values, dtype=np.int64).reshape(-1)
                if len(arr) == len(dataset):
                    return arr
            except Exception:
                pass

    targets = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            raise RuntimeError(
                "Cannot infer labels from val_dataset: dataset[idx] must return (image, target)."
            )
        target = sample[1]
        if torch.is_tensor(target):
            target = int(target.item())
        else:
            target = int(target)
        targets.append(target)

    return np.asarray(targets, dtype=np.int64)


def test_test_matrix(test_dataset, model, args):
    """
    Test-Test episodic evaluation.

    Protocol:
        Query   <- test
        Support <- test

    For each selected class:
        1) sample query first;
        2) exclude those query indices from the support candidate pool;
        3) sample K-shot support from the remaining test images;
        4) if remaining support candidates < K, sample support WITH replacement.

    Important:
        BPaCo's inference is a fixed parametric classifier. Support is sampled
        only to enforce/report the same N-way K-shot Test-Test episode protocol.
        The support images do not modify the classifier logits.
    """
    model.eval()
    device = torch.device(
        'cuda:{}'.format(args.gpu)
        if args.gpu is not None
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )

    n_way = int(args.n_way)
    n_shot = int(args.n_shot)
    n_query = int(args.n_query)
    n_episodes = int(args.n_episodes)

    if n_way < 2:
        raise ValueError("--n-way must be >= 2")
    if n_shot < 1:
        raise ValueError("--n-shot must be >= 1")
    if n_query < 1:
        raise ValueError("--n-query must be >= 1")
    if n_episodes < 1:
        raise ValueError("--n-episodes must be >= 1")

    targets = _extract_dataset_targets(test_dataset)

    class_to_indices = {}
    for idx, label in enumerate(targets.tolist()):
        class_to_indices.setdefault(int(label), []).append(int(idx))

    # Need at least n_query query samples + one different support candidate.
    valid_classes = [
        cls for cls, indices in sorted(class_to_indices.items())
        if len(indices) >= n_query + 1
    ]

    effective_way = min(n_way, len(valid_classes))
    if effective_way < 2:
        raise ValueError(
            "Not enough valid test classes for Test-Test evaluation. "
            f"Need >=2 classes with at least n_query+1 images; got {len(valid_classes)}."
        )

    if effective_way < n_way:
        print(
            f"[BPaCo Test-Test] requested {n_way}-way, "
            f"but only {len(valid_classes)} valid classes exist; "
            f"use effective {effective_way}-way."
        )

    # For formal 7-way / 7-class BPaCo this should normally be equal.
    print("\n" + "=" * 100)
    print("[BPaCo Test-Test Episodic Evaluation]")
    print(f"Test dataset size : {len(test_dataset)}")
    print(f"Requested way     : {n_way}")
    print(f"Effective way     : {effective_way}")
    print(f"Shot              : {n_shot}")
    print(f"Query per class   : {n_query}")
    print(f"Episodes          : {n_episodes}")
    print(f"Model classes     : {args.num_classes}")
    print("Valid classes:")
    for cls in valid_classes:
        print(f"  Class {cls}: test={len(class_to_indices[cls])}")
    print("Support usage     : protocol-only (fixed BPaCo classifier; no support adaptation)")
    print("=" * 100 + "\n")

    rng = np.random.RandomState(args.seed)

    episode_acc1 = []
    episode_acc5 = []
    all_preds = []
    all_targets = []
    all_probs = []
    latency_list = []
    episode_records = []
    replacement_class_events = 0

    with torch.no_grad():
        for ep in range(n_episodes):
            selected_classes = rng.choice(
                valid_classes,
                size=effective_way,
                replace=False
            ).astype(np.int64).tolist()

            query_indices = []
            query_targets = []
            episode_support = {}
            episode_query = {}
            episode_replacement = {}

            for cls in selected_classes:
                cls_indices = np.asarray(class_to_indices[int(cls)], dtype=np.int64)

                # Query: no replacement.
                q_idx = rng.choice(
                    cls_indices,
                    size=n_query,
                    replace=False
                ).astype(np.int64).tolist()

                q_set = set(q_idx)
                support_candidates = np.asarray(
                    [idx for idx in cls_indices.tolist() if idx not in q_set],
                    dtype=np.int64
                )

                if len(support_candidates) < 1:
                    raise RuntimeError(
                        f"Episode {ep}: Class {cls} has no support candidate "
                        "after excluding query."
                    )

                support_replace = len(support_candidates) < n_shot
                if support_replace:
                    replacement_class_events += 1

                s_idx = rng.choice(
                    support_candidates,
                    size=n_shot,
                    replace=support_replace
                ).astype(np.int64).tolist()

                query_indices.extend(q_idx)
                query_targets.extend([int(cls)] * n_query)

                episode_query[str(int(cls))] = q_idx
                episode_support[str(int(cls))] = s_idx
                episode_replacement[str(int(cls))] = bool(support_replace)

                # Hard safety check: query/support must not share an item.
                if set(q_idx) & set(s_idx):
                    raise RuntimeError(
                        f"Episode {ep}, class {cls}: query/support overlap detected."
                    )

            # Load query images only. Support is not used by fixed BPaCo inference.
            query_images = []
            observed_targets = []
            for idx in query_indices:
                image, target = test_dataset[int(idx)]
                query_images.append(image)
                if torch.is_tensor(target):
                    observed_targets.append(int(target.item()))
                else:
                    observed_targets.append(int(target))

            if observed_targets != query_targets:
                raise RuntimeError(
                    f"Episode {ep}: dataset target mismatch. "
                    f"expected={query_targets}, observed={observed_targets}"
                )

            images = torch.stack(query_images, dim=0).to(device, non_blocking=True)
            target_tensor = torch.tensor(query_targets, dtype=torch.long, device=device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            output = model(images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latency_list.append(t1 - t0)

            # N-way classification space = selected episode classes.
            selected_tensor = torch.tensor(
                selected_classes,
                dtype=torch.long,
                device=output.device
            )
            selected_logits = output.index_select(1, selected_tensor)

            probs_selected = F.softmax(selected_logits, dim=1)
            pred_local = selected_logits.argmax(dim=1)
            pred_global = selected_tensor[pred_local]

            topk = min(5, effective_way)
            topk_local = selected_logits.topk(topk, dim=1).indices
            topk_global = selected_tensor[topk_local]

            correct1 = pred_global.eq(target_tensor)
            correct5 = topk_global.eq(target_tensor.unsqueeze(1)).any(dim=1)

            ep_acc1 = correct1.float().mean().item() * 100.0
            ep_acc5 = correct5.float().mean().item() * 100.0

            episode_acc1.append(ep_acc1)
            episode_acc5.append(ep_acc5)

            # Expand episode probabilities back to num_classes for aggregate metrics.
            full_probs = torch.zeros(
                (images.size(0), args.num_classes),
                dtype=probs_selected.dtype,
                device=probs_selected.device
            )
            full_probs[:, selected_tensor] = probs_selected

            all_preds.extend(pred_global.cpu().numpy().astype(int).tolist())
            all_targets.extend(target_tensor.cpu().numpy().astype(int).tolist())
            all_probs.extend(full_probs.cpu().numpy().tolist())

            episode_records.append({
                "episode": ep,
                "classes": [int(x) for x in selected_classes],
                "query_indices": episode_query,
                "support_indices": episode_support,
                "support_with_replacement": episode_replacement,
                "acc1_percent": ep_acc1,
                "acc5_percent": ep_acc5,
            })

            if ep % 10 == 0 or ep == n_episodes - 1:
                print(
                    f"Test-Test: [{ep:3d}/{n_episodes}] "
                    f"n_ways: {effective_way} "
                    f"n_imgs: {effective_way * (n_shot + n_query)} "
                    f"acc1: {ep_acc1:.4f} ({np.mean(episode_acc1):.4f}) "
                    f"acc5: {ep_acc5:.4f} ({np.mean(episode_acc5):.4f})"
                )

    y_pre = np.asarray(all_preds, dtype=np.int64)
    y_target = np.asarray(all_targets, dtype=np.int64)
    all_probs = np.asarray(all_probs, dtype=np.float64)

    acc_mean = float(np.mean(episode_acc1))
    acc_std = float(np.std(episode_acc1, ddof=0))
    ci95 = float(1.96 * acc_std / np.sqrt(n_episodes))
    acc5_mean = float(np.mean(episode_acc5))

    print("\n" + "=" * 100)
    print("[BPaCo Test-Test Result]")
    print(f"Acc@1: {acc_mean:.2f}% +- {acc_std:.2f}%")
    print(f"95% CI: +/- {ci95:.2f}%")
    print(f"Acc@5: {acc5_mean:.2f}%")
    print(f"Total query predictions: {len(y_target)}")
    print(
        "Support replacement events: "
        f"{replacement_class_events}/{n_episodes * effective_way} class-episodes"
    )
    print("=" * 100)

    # Classification report over all episode query predictions.
    print("\n========== Precision / Recall / F1 ==========")
    print(classification_report(
        y_target,
        y_pre,
        labels=np.arange(args.num_classes),
        digits=4,
        zero_division=0
    ))

    # Confusion matrix.
    cm = confusion_matrix(
        y_target,
        y_pre,
        labels=np.arange(args.num_classes)
    )
    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = cm_norm / (row_sums + 1e-12)
    per_class_acc = np.diag(cm_norm) * 100.0

    save_dir = args.root_model
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_title('BPaCo Test-Test Normalised Confusion Matrix')
    for r in range(args.num_classes):
        for c in range(args.num_classes):
            ax.text(
                c, r, '{:.2f}'.format(cm_norm[r, c]),
                ha='center', va='center',
                color='white' if cm_norm[r, c] > 0.6 else 'black',
                fontsize=8
            )
    plt.tight_layout()
    cm_path = os.path.join(save_dir, 'confusion_matrix_test_test.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print("Confusion matrix saved -> {}".format(cm_path))

    # Stable distance-quality metric: clip probabilities before log.
    temperature = float(getattr(args, 'moco_t', 0.2))
    safe_probs = np.clip(all_probs, 1e-12, 1.0)
    dist_matrix = -np.log(safe_probs) / temperature

    intra_dist, inter_dist = [], []
    for q_idx in range(len(y_target)):
        true_cls = int(y_target[q_idx])
        for cls in range(args.num_classes):
            d = float(dist_matrix[q_idx, cls])
            if cls == true_cls:
                intra_dist.append(d)
            else:
                inter_dist.append(d)

    intra = np.asarray(intra_dist, dtype=np.float64)
    inter = np.asarray(inter_dist, dtype=np.float64)
    intra = intra[np.isfinite(intra)]
    inter = inter[np.isfinite(inter)]

    fisher_ratio = float('nan')
    kl_div = float('nan')
    if len(intra) > 0 and len(inter) > 0:
        mu_intra, std_intra = intra.mean(), intra.std()
        mu_inter, std_inter = inter.mean(), inter.std()
        fisher_ratio = (
            (mu_inter - mu_intra) ** 2
            / (std_intra ** 2 + std_inter ** 2 + 1e-8)
        )

        all_d = np.concatenate([intra, inter])
        if np.isfinite(all_d).all() and all_d.max() > all_d.min():
            bins = np.linspace(all_d.min(), all_d.max(), 60)
            p_intra, _ = np.histogram(intra, bins=bins, density=True)
            p_inter, _ = np.histogram(inter, bins=bins, density=True)
            p_intra = p_intra + 1e-8
            p_inter = p_inter + 1e-8
            p_intra /= p_intra.sum()
            p_inter /= p_inter.sum()
            kl_div = float(stats.entropy(p_inter, p_intra))

    print("\n========== Distance Quality ==========")
    print("Fisher Ratio: {:.4f}".format(fisher_ratio))
    print("KL Divergence: {:.4f}".format(kl_div))

    # AUC over aggregate episode queries.
    auc = float('nan')
    try:
        auc = float(
            roc_auc_score(
                y_target,
                all_probs,
                multi_class='ovr',
                average='macro'
            )
        )
        print("AUC (macro, OvR): {:.4f}".format(auc))
    except Exception as e:
        print("AUC computation skipped: {}".format(e))

    # Latency.
    latencies = np.asarray(latency_list, dtype=np.float64) * 1000.0
    mean_lat = float(latencies.mean()) if len(latencies) else float('nan')

    # Save full Test-Test summary.
    log_path = os.path.join(
        args.root_model,
        (args.mark or 'bpaco') + '_test_test_metrics.log'
    )
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("Evaluation setting: test_test\n")
        f.write("Query pool: test\n")
        f.write("Support pool: test\n")
        f.write("Query/support overlap: forbidden within episode\n")
        f.write(
            "Support usage: protocol-only; fixed BPaCo classifier does not adapt to support\n"
        )
        f.write(f"Requested N-way: {n_way}\n")
        f.write(f"Effective N-way: {effective_way}\n")
        f.write(f"N-shot: {n_shot}\n")
        f.write(f"N-query: {n_query}\n")
        f.write(f"Num episodes: {n_episodes}\n")
        f.write(f"Candidate classes: {valid_classes}\n")
        f.write(f"Acc@1: {acc_mean:.2f}% +- {acc_std:.2f}%\n")
        f.write(f"Acc@1 mean: {acc_mean:.2f}%\n")
        f.write(f"Acc@1 std: {acc_std:.2f}%\n")
        f.write(f"95% CI: +/- {ci95:.2f}%\n")
        f.write(f"Acc@5: {acc5_mean:.2f}%\n")
        f.write(f"Total query predictions: {len(y_target)}\n")
        f.write(
            "Support replacement events: "
            f"{replacement_class_events}/{n_episodes * effective_way}\n"
        )
        f.write(f"Fisher Ratio: {fisher_ratio:.4f}\n")
        f.write(f"KL Divergence: {kl_div:.4f}\n")
        f.write(f"AUC (macro, OvR): {auc:.4f}\n")
        f.write(f"Mean Latency: {mean_lat:.2f} ms / episode-query batch\n")
        f.write("\nPer-class accuracy (%):\n")
        for cls in range(args.num_classes):
            f.write(f"  Class {cls}: {per_class_acc[cls]:.2f}%\n")

    episodes_path = os.path.join(
        args.root_model,
        (args.mark or 'bpaco') + '_test_test_episodes.json'
    )
    with open(episodes_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                "evaluation_setting": "test_test",
                "seed": int(args.seed),
                "requested_n_way": n_way,
                "effective_n_way": effective_way,
                "n_shot": n_shot,
                "n_query": n_query,
                "n_episodes": n_episodes,
                "valid_classes": [int(x) for x in valid_classes],
                "episodes": episode_records,
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print("Metrics log saved -> {}".format(log_path))
    print("Episodes saved    -> {}".format(episodes_path))


def test_matrix(test_loader, model, args, train_loader):
    y_pre = []
    y_target = []
    all_probs = []
    latency_list = []

    model.eval()
    device = torch.device('cuda:{}'.format(args.gpu) if args.gpu is not None else 'cuda')

    with torch.no_grad():
        for i, (images, target) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            t0 = time.perf_counter()
            output = model(images)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latency_list.append(t1 - t0)

            prob = F.softmax(output, dim=1)
            _, y = output.max(1)
            y_pre.extend(list(y.cpu().numpy()))
            y_target.extend(list(target.cpu().numpy()))
            all_probs.extend(prob.cpu().numpy())

    y_pre = np.array(y_pre)
    y_target = np.array(y_target)
    all_probs = np.array(all_probs)

    # --- baseline metrics ---
    many_acc_top1, median_acc_top1, low_acc_top1, cls_accs, f1 = shot_acc(
        y_pre, y_target, train_loader, acc_per_cls=True)
    print('Many_acc: {:.5f}, Medium_acc: {:.5f}, Low_acc: {:.5f}, F1: {:.5f}'.format(
        many_acc_top1, median_acc_top1, low_acc_top1, f1))
    print(classification_report(y_target, y_pre, digits=4))

    # ================================================================== #
    #  METRIC 1 — Confusion Boundary Analysis
    # ================================================================== #
    num_classes = args.num_classes
    #cm = confusion_matrix(y_target, y_pre)
    
    cm = confusion_matrix(
        y_target,
        y_pre,
        labels=np.arange(args.num_classes)
)

    print("\n========== Metric 1: Confusion Boundary Analysis ==========")

    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = cm_norm / (row_sums + 1e-8)
    per_class_acc = np.diag(cm_norm) * 100

    off_diag = []
    for r in range(num_classes):
        for c in range(num_classes):
            if r != c:
                off_diag.append((cm[r, c], r, c))
    off_diag.sort(reverse=True)

    print("Top-5 most confused class pairs (true -> predicted, count):")
    for count, r, c in off_diag[:5]:
        print("  Class {} -> Class {}: {} misclassifications ({:.1f}% of class {} queries)".format(
            r, c, count, count / (row_sums[r, 0] + 1e-8) * 100, r))

    print("\nPer-class accuracy (%):")
    for cls in range(num_classes):
        print("  Class {}: {:.2f}%".format(cls, per_class_acc[cls]))

    if len(off_diag) > 0:
        _, hard_r, hard_c = off_diag[0]
        hard_correct = cm[hard_r, hard_r] + cm[hard_c, hard_c]
        hard_total = cm[hard_r, :].sum() + cm[hard_c, :].sum()
        print("\nHard-pair accuracy (Class {} vs Class {}): {}/{} = {:.2f}%".format(
            hard_r, hard_c, hard_correct, hard_total,
            hard_correct / (hard_total + 1e-8) * 100))

    save_dir = args.root_model
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_title('Normalised Confusion Matrix')
    for r in range(num_classes):
        for c in range(num_classes):
            ax.text(c, r, '{:.2f}'.format(cm_norm[r, c]),
                    ha='center', va='center',
                    color='white' if cm_norm[r, c] > 0.6 else 'black', fontsize=8)
    plt.tight_layout()
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print("Confusion matrix saved -> {}".format(cm_path))

    # ================================================================== #
    #  METRIC 2 — Distance Quality Analysis (Fisher Ratio)
    # ================================================================== #
    print("\n========== Metric 2: Distance Quality Analysis ==========")

    temperature = getattr(args, 'moco_t', 0.2)
    dist_matrix = (-torch.tensor(all_probs).log() / temperature).numpy()

    intra_dist, inter_dist = [], []
    for q_idx in range(len(y_target)):
        true_cls = y_target[q_idx]
        for cls in range(num_classes):
            d = dist_matrix[q_idx, cls]
            if cls == true_cls:
                intra_dist.append(d)
            else:
                inter_dist.append(d)

    intra = np.array(intra_dist)
    inter = np.array(inter_dist)

    mu_intra, std_intra = intra.mean(), intra.std()
    mu_inter, std_inter = inter.mean(), inter.std()

    fisher_ratio = (mu_inter - mu_intra) ** 2 / (std_intra ** 2 + std_inter ** 2 + 1e-8)

    print("Intra-class distances  -- mean: {:.4f}, std: {:.4f}".format(mu_intra, std_intra))
    print("Inter-class distances  -- mean: {:.4f}, std: {:.4f}".format(mu_inter, std_inter))
    print("Fisher Ratio (class-sep): {:.4f}".format(fisher_ratio))

    all_d = np.concatenate([intra, inter])
    bins = np.linspace(all_d.min(), all_d.max(), 60)
    p_intra, _ = np.histogram(intra, bins=bins, density=True)
    p_inter, _ = np.histogram(inter, bins=bins, density=True)
    p_intra = p_intra + 1e-8
    p_inter = p_inter + 1e-8
    p_intra /= p_intra.sum()
    p_inter /= p_inter.sum()
    kl_div = stats.entropy(p_inter, p_intra)
    print("KL Divergence (inter||intra): {:.4f}".format(kl_div))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(intra, bins=50, alpha=0.6, color='steelblue', label='Intra-class', density=True)
    ax.hist(inter, bins=50, alpha=0.6, color='tomato',    label='Inter-class', density=True)
    ax.axvline(mu_intra, color='steelblue', linestyle='--', linewidth=1.5,
               label='mu_intra={:.3f}'.format(mu_intra))
    ax.axvline(mu_inter, color='tomato',    linestyle='--', linewidth=1.5,
               label='mu_inter={:.3f}'.format(mu_inter))
    ax.set_xlabel('Distance (-log(prob) / temperature)')
    ax.set_ylabel('Density')
    ax.set_title('Distance Distribution  |  Fisher Ratio = {:.3f}'.format(fisher_ratio))
    ax.legend()
    plt.tight_layout()
    dist_path = os.path.join(save_dir, 'distance_distribution.png')
    plt.savefig(dist_path, dpi=150)
    plt.close()
    print("Distance distribution plot saved -> {}".format(dist_path))

    # ================================================================== #
    #  METRIC 3 — Complexity-Performance Trade-off
    # ================================================================== #
    print("\n========== Metric 3: Complexity-Performance Trade-off ==========")

    latencies = np.array(latency_list) * 1000
    mean_lat = latencies.mean()
    std_lat = latencies.std()
    median_lat = np.median(latencies)
    overall_acc = (y_pre == y_target).mean() * 100

    print("Inference latency per batch:")
    print("  Mean   : {:.2f} ms".format(mean_lat))
    print("  Std    : {:.2f} ms".format(std_lat))
    print("  Median : {:.2f} ms".format(median_lat))
    print("  Min    : {:.2f} ms".format(latencies.min()))
    print("  Max    : {:.2f} ms".format(latencies.max()))
    print("Overall accuracy: {:.4f}%".format(overall_acc))

    # ================================================================== #
    #  METRIC 4 — AUC / ROC
    # ================================================================== #
    print("\n========== Metric 4: AUC / ROC ==========")
    try:
        auc = roc_auc_score(y_target, all_probs, multi_class='ovr', average='macro')
        print("AUC (macro, OvR): {:.4f}".format(auc))
    except Exception as e:
        print("AUC computation skipped: {}".format(e))

    # ================================================================== #
    #  Summary
    # ================================================================== #
    print("\n========== Summary ==========")
    print("Overall Accuracy : {:.4f}%".format(overall_acc))
    print("Many-shot Acc    : {:.4f}%".format(many_acc_top1 * 100))
    print("Medium-shot Acc  : {:.4f}%".format(median_acc_top1 * 100))
    print("Low-shot Acc     : {:.4f}%".format(low_acc_top1 * 100))
    print("Fisher Ratio     : {:.4f}  (higher = better class separation)".format(fisher_ratio))
    print("KL Divergence    : {:.4f}  (higher = better separation)".format(kl_div))
    print("Mean Latency     : {:.2f} ms / batch".format(mean_lat))
    try:
        print("AUC (macro, OvR) : {:.4f}".format(auc))
    except:
        pass

    log_path = os.path.join(args.root_model, args.mark + '_test_metrics.log')
    with open(log_path, 'w') as f:
        f.write("Overall Accuracy : {:.4f}%\n".format(overall_acc))
        f.write("Many-shot Acc   : {:.4f}%\n".format(many_acc_top1 * 100))
        f.write("Medium-shot Acc : {:.4f}%\n".format(median_acc_top1 * 100))
        f.write("Low-shot Acc    : {:.4f}%\n".format(low_acc_top1 * 100))
        f.write("Fisher Ratio    : {:.4f}\n".format(fisher_ratio))
        f.write("KL Divergence   : {:.4f}\n".format(kl_div))
        f.write("Mean Latency    : {:.2f} ms\n".format(mean_lat))
        try:
            f.write("AUC (macro, OvR): {:.4f}\n".format(auc))
        except:
            pass
        f.write("\nPer-class accuracy (%):\n")
        for cls in range(num_classes):
            f.write("  Class {}: {:.2f}%\n".format(cls, per_class_acc[cls]))
    print("Metrics log saved -> {}".format(log_path))


def saliency_score(img_saliency):
    B=img_saliency.shape[0]
    noise=torch.randint(-5000,5000,img_saliency.shape).cuda()
    img_saliency+=noise*0.0001
    patch_score=F.avg_pool2d(img_saliency,16,16).reshape(B,196)
    min_i, _ = torch.min(patch_score, dim=1, keepdim=True)
    max_i, _ = torch.max(patch_score, dim=1, keepdim=True)
    patch_score = (patch_score - min_i) / (max_i-min_i)
    # patch_score =patch_score/2+0.5
    return patch_score.reshape(B,14,14)

def train(train_loader, model, criterion, criterion1, criterion2, optimizer, epoch, scaler, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    print('[DEBUG] Epoch {}: {} batches (total {} samples), batch_size={}'.format(
        epoch, len(train_loader), len(train_loader.dataset), train_loader.batch_size))
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        if not args.fp16:
            features, labels, logits, center_logits, bcl_features = model(im_q=images[0], im_k=images[1], labels=target)
            center_logits = center_logits[:args.num_classes]
            actual_bs = target.size(0)
            ce_loss = criterion2(logits[:actual_bs], labels[:actual_bs])# batch_size / gpus * 2
            bpaco_loss = criterion1(features, labels, logits, center_logits)
            loss = ce_loss + 0.25 * bpaco_loss

        else:
            with autocast():
                features, labels, logits, center_logits, bcl_features = model(im_q=images[0], im_k=images[1], labels=target)
                center_logits = center_logits[:args.num_classes]
                actual_bs = target.size(0)
                ce_loss = criterion2(logits[:actual_bs], labels[:actual_bs])# batch_size / gpus * 2
                bpaco_loss = criterion1(features, labels, logits, center_logits)
                loss = ce_loss + 0.25 * bpaco_loss

        output = logits[:target.size(0)]
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), output.size(0))
        top1.update(acc1[0], logits.size(0))
        top5.update(acc5[0], logits.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if not args.fp16:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, args)


def validate(val_loader, train_loader, model, criterion, args, train_dataset=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    device = torch.device('cuda:{}'.format(args.gpu) if args.gpu is not None else 'cuda')
    total_logits = torch.empty((0, args.num_classes), device=device)
    total_labels = torch.empty(0, dtype=torch.long, device=device)

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            output = model(images)
            # loss = criterion(output, target)

            total_logits = torch.cat((total_logits, output))
            total_labels = torch.cat((total_labels, target))

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i, args)

        # TODO: this should also be done with the ProgressMeter
        open(args.root_model+"/"+args.mark+".log","a+").write(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}\n'
              .format(top1=top1, top5=top5))

        probs, preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        # Use dataset-adaptive thresholds based on training set size
        many_shot_thr = max(20, int(sum(train_dataset.cls_num_list) * 0.7 / len(train_dataset.cls_num_list)))
        low_shot_thr = max(5, int(sum(train_dataset.cls_num_list) * 0.2 / len(train_dataset.cls_num_list)))
        many_acc_top1, median_acc_top1, low_acc_top1, cls_accs, f1 = shot_acc(
            preds, total_labels, train_dataset,
            many_shot_thr=many_shot_thr, low_shot_thr=low_shot_thr, acc_per_cls=True)
        open(args.root_model+"/"+args.mark+".log","a+").write('Many_acc: %.5f, Medium_acc: %.5f Low_acc: %.5f\n'%(many_acc_top1, median_acc_top1, low_acc_top1))

    return top1.avg,many_acc_top1, median_acc_top1, low_acc_top1,f1


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


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

    def display(self, batch, args):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        open(args.root_model+"/"+args.mark+".log","a+").write('\t'.join(entries)+"\n")

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if epoch < args.warmup_epochs:
       lr = lr / args.warmup_epochs * (epoch + 1 )
    elif args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs + 1 ) / (args.epochs - args.warmup_epochs + 1 )))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
