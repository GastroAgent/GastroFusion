import os
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path
import torchvision.transforms as transforms

from engine import evaluate
import utils.deit_util as utils
from utils.args import get_args_parser
from models import get_model

from datasets.episodic_dataset1 import EpisodeDataset


# ============================================================
# Dataset setting
# ============================================================

def dataset_setting(
    data_path,
    nCls,
    img_size=224,
):
    """
    Test-Test protocol:

        Query:
            <data_path>/test/

        Support:
            <data_path>/test/

    Query and support are sampled from the same test class pool,
    while EpisodeDataset guarantees that query images are excluded
    from the support candidate pool within each episode.
    """

    mean = [
        x / 255.0
        for x in [
            129.37731888,
            124.10583864,
            112.47758569,
        ]
    ]

    std = [
        x / 255.0
        for x in [
            68.20947949,
            65.43124043,
            70.45866994,
        ]
    ]

    normalize = transforms.Normalize(
        mean=mean,
        std=std,
    )

    valTransform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    inputW = img_size
    inputH = img_size

    testDir = os.path.join(data_path, "test")

    if not os.path.isdir(testDir):
        raise FileNotFoundError(
            f"Test directory does not exist: {testDir}"
        )

    return (
        valTransform,
        inputW,
        inputH,
        testDir,
        nCls,
    )


# ============================================================
# Dataset
# ============================================================

def get_sets(args):

    (
        valTransform,
        inputW,
        inputH,
        testDir,
        nbCls,
    ) = dataset_setting(
        data_path=args.data_path,
        nCls=args.nClsEpisode,
        img_size=args.img_size,
    )

    # ========================================================
    # Test-Test episodic dataset
    # Query and support are both sampled from testDir.
    # ========================================================

    testSet = EpisodeDataset(
        queryDir=testDir,
        supportDir=testDir,
        nCls=nbCls,
        nSupport=args.nSupport,
        nQuery=args.nQuery,
        transform=valTransform,
        inputW=inputW,
        inputH=inputH,
        nEpisode=args.nEpisode,
    )

    print("\n" + "=" * 100)
    print("[PMF Test-Test Evaluation]")
    print(f"Data root      : {args.data_path}")
    print(f"Query root     : {testDir}")
    print(f"Support root   : {testDir}")
    print(f"Requested way  : {nbCls}")
    print(f"Effective way  : {testSet.nCls}")
    print(f"Shot           : {args.nSupport}")
    print(f"Query          : {args.nQuery}")
    print(f"Episodes       : {args.nEpisode}")
    print(f"Classes        : {testSet.clsList}")
    print("=" * 100 + "\n")

    return None, None, testSet


# ============================================================
# DataLoader
# ============================================================

def get_test_loader(args):

    _, _, dataset_val = get_sets(args)

    sampler_val = torch.utils.data.SequentialSampler(
        dataset_val
    )

    generator = torch.Generator()
    generator.manual_seed(
        args.seed + 10000
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,

        sampler=sampler_val,

        batch_size=1,

        num_workers=args.num_workers,

        pin_memory=args.pin_mem,

        drop_last=False,

        generator=generator,
    )

    return data_loader_val


# ============================================================
# Main
# ============================================================

def main(args):

    # --------------------------------------------------------
    # Distributed initialization
    # --------------------------------------------------------

    utils.init_distributed_mode(args)

    args.eval = True

    print(args)

    device = torch.device(
        args.device
    )

    # --------------------------------------------------------
    # Random seed
    # --------------------------------------------------------

    seed = (
        args.seed
        + utils.get_rank()
    )

    args.seed = seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = True

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print(
        f"Creating model: "
        f"{args.deploy} {args.arch}"
    )

    model = get_model(args)

    model.to(device)

    model_without_ddp = model

    if args.distributed:

        model = torch.nn.parallel.DistributedDataParallel(
            model,

            device_ids=[
                args.gpu
            ],

            find_unused_parameters=(
                args.unused_params
            ),
        )

        model_without_ddp = model.module

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    if args.resume:

        checkpoint = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False,
        )

        model_without_ddp.load_state_dict(
            checkpoint["model"],
            strict=True,
        )

        print(
            f"Load ckpt from "
            f"{args.resume}"
        )

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    n_parameters = sum(
        p.numel()
        for p in model_without_ddp.parameters()
        if p.requires_grad
    )

    print(
        "number of params:",
        n_parameters,
    )

    # --------------------------------------------------------
    # Criterion
    # --------------------------------------------------------

    criterion = torch.nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Build episodic test loader
    # --------------------------------------------------------

    data_loader_val = get_test_loader(
        args
    )

    # ========================================================
    # Optional LR selection for finetune deployment
    # ========================================================

    best_lr = args.ada_lr

    if args.deploy == "finetune":

        print(
            "Start selecting the best lr..."
        )

        best_acc = 0.0

        for lr in [
            0,
            0.0001,
            0.001,
            0.01,
        ]:

            model_without_ddp.lr = lr

            # Reset generator before each comparison
            data_loader_val.generator.manual_seed(
                args.seed + 10000
            )

            test_stats = evaluate(
                data_loader_val,
                model,
                criterion,
                device,
                seed=1234,
                ep=5,
            )

            acc = test_stats["acc1"]

            print(
                f"* lr={lr}: "
                f"acc1={acc}"
            )

            if acc > best_acc:

                best_acc = acc
                best_lr = lr

        model_without_ddp.lr = best_lr

        print(
            f"### Selected lr = "
            f"{best_lr}"
        )

    # ========================================================
    # Final evaluation
    # ========================================================

    data_loader_val.generator.manual_seed(
        args.seed + 10000
    )

    test_stats = evaluate(
        data_loader_val,
        model,
        criterion,
        device,
    )

    print(
        f"Test acc1: "
        f"{test_stats['acc1']:.1f}% "
        f"+- {test_stats['acc_std']:.1f}%"
    )

    print(
        f"Acc@5: "
        f"{test_stats['acc5']:.1f}%"
    )

    # ========================================================
    # Save
    # ========================================================

    if (
        args.output_dir
        and utils.is_main_process()
    ):

        output_dir = Path(
            args.output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        result_file = (
            output_dir
            / "test_results_test_test.txt"
        )

        # Effective way from actual dataset
        effective_way = (
            data_loader_val.dataset.nCls
        )

        valid_classes = (
            data_loader_val.dataset.clsList
        )

        query_root = (
            data_loader_val.dataset.queryDir
        )

        support_root = (
            data_loader_val.dataset.supportDir
        )

        ci95 = (
            1.96
            * test_stats["acc_std"]
            / np.sqrt(args.nEpisode)
        )

        with open(
            result_file,
            "w",
            encoding="utf-8",
        ) as f:

            f.write(
                f"Model: {args.resume}\n"
            )

            f.write(
                "Evaluation setting: "
                "test_test\n"
            )

            f.write(
                f"Data root: "
                f"{args.data_path}\n"
            )

            f.write(
                f"Query root: "
                f"{query_root}\n"
            )

            f.write(
                f"Support root: "
                f"{support_root}\n"
            )

            f.write(
                f"Requested N-way: "
                f"{args.nClsEpisode}\n"
            )

            f.write(
                f"Effective N-way: "
                f"{effective_way}\n"
            )

            f.write(
                f"N-shot: "
                f"{args.nSupport}\n"
            )

            f.write(
                f"N-query: "
                f"{args.nQuery}\n"
            )

            f.write(
                f"Candidate classes: "
                f"{valid_classes}\n"
            )

            f.write(
                f"Acc@1: "
                f"{test_stats['acc1']:.2f}%\n"
            )

            f.write(
                f"Acc@1 std: "
                f"{test_stats['acc_std']:.2f}%\n"
            )

            f.write(
                f"Acc@5: "
                f"{test_stats['acc5']:.2f}%\n"
            )

            f.write(
                f"95% CI: "
                f"+/- {ci95:.2f}%\n"
            )

            f.write(
                f"Num episodes: "
                f"{args.nEpisode}\n"
            )

            f.write(
                f"LR: "
                f"{best_lr}\n"
            )

            f.write(
                f"Deploy: "
                f"{args.deploy}\n"
            )

            f.write(
                "Rare support policy: "
                "sample_with_replacement_if_less_than_K\n"
            )

        print(
            f"\nResults saved to "
            f"{result_file}"
        )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    parser = get_args_parser()

    args = parser.parse_args()

    # ========================================================
    # Current experiment: Esophagus
    # ========================================================

    args.resume = (
        "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/pmf_cvpr22/outputs_tsy/食管_7way_5shot/best.pth"
    )

    # IMPORTANT:
    #
    # data_path points to the parent directory containing:
    #
    #   test/
    #
    args.data_path = (
        "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/pmf_cvpr22/data_tsy_827_doctor_test_split_test_test/"
        "食管_1query/"
    )

    args.output_dir = (
        "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/tsy/pmf_cvpr22_new_100eposide/"
        "outputs_test_test/食管_5shot/"
    )

    # --------------------------------------------------------
    # Episode settings
    # --------------------------------------------------------

    args.img_size = 224

    # Maximum requested way.
    #
    # If test has fewer classes than requested,
    # EpisodeDataset automatically reduces N-way.
    args.nClsEpisode = 7

    args.nSupport = 5
    args.nQuery = 1
    args.nEpisode = 100

    # --------------------------------------------------------
    # Runtime
    # --------------------------------------------------------

    args.device = "cuda:0"

    args.num_workers = 4
    args.pin_mem = True

    # --------------------------------------------------------
    # PMF
    # --------------------------------------------------------

    args.deploy = "vanilla"

    args.arch = "dino_base_patch16"

    args.ada_lr = 0.05
    args.ada_steps = 40

    args.aug_prob = 0.9

    main(args)