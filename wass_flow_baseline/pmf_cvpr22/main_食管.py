import sys
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from engine import train_one_epoch, evaluate
import utils.deit_util as utils
from datasets import get_loaders
from utils.args import get_args_parser
from models import get_model


import os

# ============================================================
# 单卡运行时自动补齐 distributed 环境变量
# ============================================================

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29501")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")




# ============================================================
#                 固定输出目录
# ============================================================

OUTPUT_DIR = (
    "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/pmf_cvpr22/outputs_tsy/食管_8way_5shot"
)
ARCH = "dino_base_patch16"


def main(args):

    # ========================================================
    # 强制使用顶部指定的输出目录
    # ========================================================
    args.output_dir = OUTPUT_DIR
    args.arch = ARCH

    utils.init_distributed_mode(args)

    print(args)
    print(f"Output directory: {args.output_dir}")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    args.seed = seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    # ========================================================
    # Output dir
    # ========================================================
    output_dir = Path(args.output_dir)

    if utils.is_main_process():
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        with (output_dir / "log.txt").open("a") as f:
            f.write(" ".join(sys.argv) + "\n")

    ##############################################
    # Data loaders
    ##############################################

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    data_loader_train, data_loader_val = get_loaders(
        args,
        num_tasks,
        global_rank
    )

    ##############################################
    # Mixup regularization (by default OFF)
    ##############################################

    mixup_fn = None

    mixup_active = (
        args.mixup > 0
        or args.cutmix > 0.
        or args.cutmix_minmax is not None
    )

    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nClsEpisode
        )

    ##############################################
    # Model
    ##############################################

    print(
        f"Creating model: ProtoNet {args.arch}"
    )

    model = get_model(args)
    model.to(device)

    model_ema = None

    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu'
            if args.model_ema_force_cpu
            else '',
            resume=''
        )

    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=args.unused_params
        )

        model_without_ddp = model.module

    n_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        'number of params:',
        n_parameters
    )

    ##############################################
    # Optimizer & scheduler & criterion
    ##############################################

    if args.fp16:

        scale = 1 / 8

        linear_scaled_lr = (
            args.lr
            * utils.get_world_size()
            * scale
        )

        args.lr = linear_scaled_lr

    loss_scaler = (
        NativeScaler()
        if args.fp16
        else None
    )

    optimizer = torch.optim.SGD(
        [
            p
            for p in model_without_ddp.parameters()
            if p.requires_grad
        ],
        args.lr,
        momentum=args.momentum,
        weight_decay=0,
    )

    lr_scheduler, _ = create_scheduler(
        args,
        optimizer
    )

    if args.mixup > 0.:

        criterion = (
            SoftTargetCrossEntropy()
        )

    elif args.smoothing:

        criterion = (
            LabelSmoothingCrossEntropy(
                smoothing=args.smoothing
            )
        )

    else:

        criterion = (
            torch.nn.CrossEntropyLoss()
        )

    ##############################################
    # Resume training from ckpt
    ##############################################

    if args.resume:

        if args.resume.startswith(
            'https'
        ):

            checkpoint = (
                torch.hub
                .load_state_dict_from_url(
                    args.resume,
                    map_location='cpu',
                    check_hash=True
                )
            )

        else:

            checkpoint = torch.load(
                args.resume,
                map_location='cpu'
            )

        model_without_ddp.load_state_dict(
            checkpoint['model']
        )

        if (
            not args.eval
            and 'optimizer' in checkpoint
            and 'lr_scheduler' in checkpoint
            and 'epoch' in checkpoint
        ):

            optimizer.load_state_dict(
                checkpoint['optimizer']
            )

            lr_scheduler.load_state_dict(
                checkpoint['lr_scheduler']
            )

            args.start_epoch = (
                checkpoint['epoch']
                + 1
            )

            if args.model_ema:
                utils._load_checkpoint_for_ema(
                    model_ema,
                    checkpoint['model_ema']
                )

            if (
                'scaler' in checkpoint
                and loss_scaler is not None
            ):
                loss_scaler.load_state_dict(
                    checkpoint['scaler']
                )

        print(
            f'Resume from {args.resume} '
            f'at epoch {args.start_epoch}.'
        )

    ##############################################
    # Initial validation
    ##############################################

    test_stats = evaluate(
        data_loader_val,
        model,
        criterion,
        device,
        args.seed + 10000
    )

    print(
        "Accuracy of the network "
        f"on dataset_val: "
        f"{test_stats['acc1']:.1f}%"
    )

    if (
        args.output_dir
        and utils.is_main_process()
    ):

        test_stats['epoch'] = -1

        with (
            output_dir / "log.txt"
        ).open("a") as f:

            f.write(
                json.dumps(
                    test_stats
                )
                + "\n"
            )

    if args.eval:
        return

    ##############################################
    # Training
    ##############################################

    if utils.is_main_process():

        writer = SummaryWriter(
            log_dir=str(output_dir)
        )

    else:

        writer = None

    print(
        f"Start training for "
        f"{args.epochs} epochs"
    )

    start_time = time.time()

    max_accuracy = (
        test_stats['acc1']
    )

    for epoch in range(
        args.start_epoch,
        args.epochs
    ):

        train_stats = train_one_epoch(
            data_loader_train,
            model,
            criterion,
            optimizer,
            epoch,
            device,
            loss_scaler,
            args.fp16,
            args.clip_grad,
            model_ema,
            mixup_fn,
            writer,
            set_training_mode=False
        )

        lr_scheduler.step(epoch)

        test_stats = evaluate(
            data_loader_val,
            model,
            criterion,
            device,
            args.seed + 10000
        )

        log_stats = {
            **{
                f'train_{k}': v
                for k, v
                in train_stats.items()
            },
            **{
                f'test_{k}': v
                for k, v
                in test_stats.items()
            },
            'epoch': epoch,
            'n_parameters': n_parameters
        }

        ##############################################
        # Save checkpoint
        ##############################################

        if args.output_dir:

            checkpoint_paths = [
                output_dir / 'checkpoint.pth',
                output_dir / 'best.pth'
            ]

            for checkpoint_path in checkpoint_paths:

                state_dict = {
                    'model':
                        model_without_ddp.state_dict(),

                    'optimizer':
                        optimizer.state_dict(),

                    'lr_scheduler':
                        lr_scheduler.state_dict(),

                    'epoch':
                        epoch,

                    'model_ema':
                        get_state_dict(model_ema)
                        if args.model_ema
                        else None,

                    'args':
                        args,
                }

                if loss_scaler is not None:
                    state_dict[
                        'scalar'
                    ] = loss_scaler.state_dict()

                utils.save_on_master(
                    state_dict,
                    checkpoint_path
                )

                if (
                    test_stats["acc1"]
                    <= max_accuracy
                ):
                    break

        print(
            "Accuracy of the network "
            f"on dataset_val: "
            f"{test_stats['acc1']:.1f}%"
        )

        max_accuracy = max(
            max_accuracy,
            test_stats["acc1"]
        )

        print(
            f'Max accuracy: '
            f'{max_accuracy:.2f}%'
        )

        if (
            args.output_dir
            and utils.is_main_process()
        ):

            log_stats[
                'best_test_acc'
            ] = max_accuracy

            with (
                output_dir / "log.txt"
            ).open("a") as f:

                f.write(
                    json.dumps(
                        log_stats
                    )
                    + "\n"
                )

    total_time = (
        time.time()
        - start_time
    )

    total_time_str = str(
        datetime.timedelta(
            seconds=int(total_time)
        )
    )

    print(
        'Training time {}'
        .format(
            total_time_str
        )
    )

    if utils.is_main_process():

        writer.close()

        import tables

        tables.file._open_files.close_all()


if __name__ == '__main__':

    parser = get_args_parser()

    args = parser.parse_args()

    main(args)