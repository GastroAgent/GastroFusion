import os
import numpy as np
import time
import random
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path
from tabulate import tabulate
import torchvision.transforms as transforms

from engine import evaluate
import utils.deit_util as utils
from utils.args import get_args_parser
from models import get_model
from datasets.episodic_dataset import EpisodeDataset


def dataset_setting(nSupport, img_size=224):
    mean = [x/255.0 for x in [129.37731888, 124.10583864, 112.47758569]]
    std = [x/255.0 for x in [68.20947949, 65.43124043, 70.45866994]]
    normalize = transforms.Normalize(mean=mean, std=std)
    valTransform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    inputW, inputH, nbCls = img_size, img_size, 8
    trainDir = '/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/pmf_cvpr22/data/胃_1query/train/'
    valDir = '/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/pmf_cvpr22/data/胃_1query/val/'
    testDir = '/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/pmf_cvpr22/data/胃_1query/test/'
    return valTransform, inputW, inputH, testDir, nbCls


def get_test_loader(args):
    _, _, dataset_val = get_sets(args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    generator = torch.Generator()
    generator.manual_seed(args.seed + 10000)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        generator=generator
    )
    return data_loader_val


def get_sets(args):
    valTransform, inputW, inputH, testDir, nbCls = dataset_setting(args.nSupport, args.img_size)
    testSet = EpisodeDataset(
        imgDir=testDir,
        nCls=nbCls,
        nSupport=args.nSupport,
        nQuery=args.nQuery,
        transform=valTransform,
        inputW=inputW,
        inputH=inputH,
        nEpisode=args.nEpisode
    )
    return None, None, testSet


def main(args):
    utils.init_distributed_mode(args)
    args.eval = True

    print(args)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    args.seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    print(f"Creating model: {args.deploy} {args.arch}")
    model = get_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=args.unused_params)
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        print(f'Load ckpt from {args.resume}')

    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    criterion = torch.nn.CrossEntropyLoss()
    data_loader_val = get_test_loader(args)

    best_lr = args.ada_lr
    if args.deploy == 'finetune':
        print("Start selecting the best lr...")
        best_acc = 0
        for lr in [0, 0.0001, 0.001, 0.01]:
            model_without_ddp.lr = lr
            test_stats = evaluate(data_loader_val, model, criterion, device, seed=1234, ep=5)
            acc = test_stats['acc1']
            print(f"*lr = {lr}: acc1 = {acc}")
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
        model_without_ddp.lr = best_lr
        print(f"### Selected lr = {best_lr}")

    data_loader_val.generator.manual_seed(args.seed + 10000)
    test_stats = evaluate(data_loader_val, model, criterion, device)

    print(f"Test acc1: {test_stats['acc1']:.1f}% +- {test_stats['acc_std']:.1f}%")
    print(f"Acc@5: {test_stats['acc5']:.1f}%")

    if args.output_dir and utils.is_main_process():
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"test_results_胃_test.txt", "w") as f:
            f.write(f"Model: {args.resume}\n")
            f.write(f"Test dataset: {args.data_path}/test\n")
            f.write(f"Acc@1: {test_stats['acc1']:.2f}%\n")
            f.write(f"Acc@1 std: {test_stats['acc_std']:.2f}%\n")
            f.write(f"Acc@5: {test_stats['acc5']:.2f}%\n")
            f.write(f"95% CI: +/- {(1.96 * test_stats['acc_std']) / np.sqrt(args.nEpisode):.2f}%\n")
            f.write(f"Num episodes: {args.nEpisode}\n")
            f.write(f"LR: {best_lr}\n")
            f.write(f"Deploy: {args.deploy}\n")
        print(f"\nResults saved to {output_dir / 'test_results_胃_test.txt'}")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    args.resume = '/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/pmf_cvpr22/outputs/胃_5shot/best.pth'
    args.data_path = '/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/pmf_cvpr22/data/胃_1query/'
    args.output_dir = '/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/pmf_cvpr22/outputs/胃_5shot/'
    args.img_size = 224
    args.nClsEpisode = 8
    args.nSupport = 5
    args.nQuery = 1
    args.nEpisode = 100
    args.device = 'cuda:0'
    args.num_workers = 4
    args.pin_mem = True
    args.deploy = 'vanilla'
    args.arch = 'dino_base_patch16'
    args.ada_lr = 0.05
    args.ada_steps = 40
    args.aug_prob = 0.9

    main(args)
