import time
from collections import defaultdict

import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
from tqdm import tqdm
from pathlib import Path

from core.backbone.get_model import get_model
from core.classifiers import ODC
from core.manifolds import Oblique, Euclidean
from core.optimizers import RiemannianAdam
from core.utils import setup, get_meta_data, clean_up, compute_loss_acc, set_seed, reduce_tensor, smooth_one_hot
from dataloader.data_loader_tsy import get_loader

from core.utils import (
    setup,
    get_meta_data,
    clean_up,
    compute_loss_acc,
    set_seed,
    reduce_tensor,
    smooth_one_hot,
    run_model,
    set_gpu,
    ensure_path
)



import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--backbone', type=str, default='resnet18')
parser.add_argument('--dataset', type=str, default='medical')
parser.add_argument('--medical_subset', type=str, default='食管_7way_1query')
parser.add_argument('--eval_mode', type=str, default='test-test')
parser.add_argument('--val_set', type=str, default='test')

parser.add_argument('--n_way', type=int, default=7)
parser.add_argument('--k_shot', type=int, default=5)
parser.add_argument('--k_query', type=int, default=1)
parser.add_argument('--val_episodes', type=int, default=100)

parser.add_argument('--train_meta_save_path', type=str,
                    default='./results_tsy/test-test')

parser.add_argument('--gpu', default='0')
parser.add_argument('--port', type=int, default=23333)
parser.add_argument('--seed', type=int, default=12667)

parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--use_dali', type=bool, default=False)

parser.add_argument('--T', type=int, default=14)
parser.add_argument('--kernels', type=int, default=11)

parser.add_argument('--lr_weights', type=float, default=0.1)
parser.add_argument('--lr_anchors', type=float, default=0.1)

parser.add_argument('--train_meta_epochs', type=int, default=20)
parser.add_argument('--opt_level', type=str, default='O0')

args = parser.parse_args()

print("================ DEBUG ================")
print("medical_subset =", args.medical_subset)
print("n_way =", args.n_way)
print("eval_mode =", args.eval_mode)
print("========================================")

args.orig_imsize = -1
args.out_dim = 512
args.batch_size = 32
args.num_class = 8 if "8way" in args.medical_subset else 7

args.num_support = args.n_way * args.k_shot
args.num_query = args.n_way * args.k_query
args.num_samples = args.num_support + args.num_query


args.train_episodes = 200
args.train_pretrain_epochs = 30
args.train_meta_epochs = 20
args.train_meta_step = 50


args.w1 = 1
args.w2 = 10
args.w3 = 1
args.scale_factor = 15

checkpoint_root = (
    "/mnt/inaisfs/data/home/tansy_criait/"
    "qsz/qcjw/fsl_scoreboard/FSL-OM-main/"
    "checkpoints/medical"
)

args.train_pretrain_best_model = str(
    Path(checkpoint_root)
    / args.medical_subset
    / "softmax"
    / args.backbone
    / "model_best.pth.tar"
)

print("Using checkpoint:", args.train_pretrain_best_model)

if not Path(args.train_pretrain_best_model).exists():
    raise FileNotFoundError(args.train_pretrain_best_model)

set_gpu(args.gpu)


def fine_tune(rank, world_size, cfg):
    set_seed(cfg.seed, rank)
    print_rank = 0

    torch.cuda.set_device(rank)
    print(f"Running basic DDP example on rank {rank}.")
    cfg.device = f'cuda:{rank}'
    setup(rank, world_size, cfg.port)

    train_loader, val_loader = get_loader(cfg)

    # labels 用于为每个 episode 生成 0~n_way-1 的局部标签序列
    labels = torch.arange(cfg.n_way, dtype=torch.long, device=cfg.device).repeat(
        cfg.k_shot + cfg.k_query)  # shape[75]:012340123401234...

    #################################################################
    # backbone and load backbone state_dict
    #################################################################
    backbone = get_model(cfg.backbone, cfg.num_class).cuda()
    backbone = DistributedDataParallel(backbone, device_ids=[rank])

    # Try to load pretrained weights; if not found and dataset is medical, use ImageNet pretrained weights
    if os.path.exists(cfg.train_pretrain_best_model):
        print(f'Loading Parameters from pretrain model: {cfg.train_pretrain_best_model}')
        model_dict = backbone.state_dict()
        map_location = 'cpu'
        pretrained_dict = torch.load(cfg.train_pretrain_best_model,
                                     map_location=map_location
                                     # map_location=torch.device('cpu')
                                     )['state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        backbone.load_state_dict(model_dict)
    else:
        if cfg.dataset == 'medical':
            # Use ImageNet pretrained weights for medical data when no local pretrain model exists
            import torchvision.models as models
            print(f'No pretrain model found at {cfg.train_pretrain_best_model}, loading ImageNet pretrained weights...')
            if cfg.backbone == 'resnet18':
                pretrained_backbone = models.resnet18(pretrained=True)
            else:
                pretrained_backbone = models.wide_resnet50_2(pretrained=True)
            pretrained_dict = pretrained_backbone.state_dict()
            model_dict = backbone.state_dict()
            # Backbone is wrapped in DistributedDataParallel, so its state_dict has 'module.' prefix.
            # Strip 'module.' from model_dict keys to match torchvision keys.
            # FSL-OM's ResNet18 first layer is 3x3/stride=1 which differs from torchvision's 7x7/stride=2,
            # so we skip mismatched shapes and load everything else.
            model_dict_stripped = {k.replace('module.', ''): v for k, v in model_dict.items()}
            pretrained_dict_clean = {}
            for k, v in pretrained_dict.items():
                if k in model_dict_stripped and v.shape == model_dict_stripped[k].shape:
                    pretrained_dict_clean['module.' + k] = v
            model_dict.update(pretrained_dict_clean)
            backbone.load_state_dict(model_dict)
            print(f'Loaded {len(pretrained_dict_clean)} ImageNet pretrained layers')
        else:
            raise FileNotFoundError(f'Pretrain model not found at {cfg.train_pretrain_best_model}')

    # Train and validation
    best_acc_list = []
    loss_float = defaultdict(float)
    if rank == print_rank:
        valid_pbar = tqdm(val_loader, position=0,
                          # bar_format="{desc:9.9s}: {percentage:0.1f}%|{bar}{r_bar}"
                          )  # only show for device_rank=0
    else:
        valid_pbar = val_loader

    val_acc = 0
    pm = 0

    for step, data in enumerate(valid_pbar):
        if rank == print_rank:
            description = ["{}:{:.6f}".format(key, loss_float[key]) for key in loss_float]
            valid_pbar.set_description(
                f"val: rank:{rank}, {time.strftime('%H:%M:%S')} " + ','.join(description))

        # Support both original episodic test-test and cross-domain test-train.
        # test-test:
        #     data -> normal CategoriesSampler output
        # test-train:
        #     data -> (support_images, query_images, labels information)
        if getattr(cfg, 'eval_mode', 'test-test') == 'test-train':

            # CrossDomainEpisodeDataset + DataLoader(batch_size=1)
            # output:
            # support_images: [1, n_support, 3, H, W]
            # support_labels: [1, n_support]
            # query_images:   [1, n_query, 3, H, W]
            # query_labels:   [1, n_query]

            support_images, support_labels, query_images, query_labels = data


            # remove DataLoader batch dimension
            support_images = support_images.squeeze(0)
            support_labels = support_labels.squeeze(0)

            query_images = query_images.squeeze(0)
            query_labels = query_labels.squeeze(0)


            support_images = support_images.to(cfg.device)
            support_labels = support_labels.to(cfg.device)

            query_images = query_images.to(cfg.device)
            query_labels = query_labels.to(cfg.device)


            # concatenate support + query
            images = torch.cat(
                [support_images, query_images],
                dim=0
            )

            _labels = torch.cat(
                [support_labels, query_labels],
                dim=0
            )
        else:
            if cfg.use_dali:
                data = [data[0]['data'], data[0]['label']]
            data = get_meta_data(data, labels, cfg)
            _data = [x.to(cfg.device) for x in data]
            images, _labels = _data
        backbone.eval()

        feas = []
        anchor = []
        with torch.no_grad():
            fea = backbone(images)

            # RSMA
            f_mean = F.adaptive_avg_pool2d(fea, (1, 1)).squeeze().unsqueeze(1)  # num_sample, out_dim
            feas.append(f_mean)
            for i_num in range(1, cfg.kernels + 1):
                f_max = F.adaptive_max_pool2d(fea, (i_num, i_num))  # num_sample, i, i out_dim
                key = f_max.permute(0, 2, 3, 1)
                value = f_max.permute(0, 2, 3, 1)
                query = f_mean.unsqueeze(1)
                f_m = F.softmax(key * query / np.sqrt(key.size(-1)), -1) * value + value
                f_mean_ = F.adaptive_avg_pool2d(f_m.permute(0, 3, 2, 1), (1, 1)).squeeze().unsqueeze(1)  # num_sample, out_dim
                feas.append(f_mean_)

            fea = torch.cat(feas, 1)
            # fea shape: (B, kernels+1, out_dim)

            # -------------------------------------------------------------------------
            # Remap dataloader's real labels (which may be arbitrary integers like
            # [4,2,0,6,5,1,3, 4,2,0,6,5,1,3]) to consecutive 0..n_way-1 so that they
            # align with the episode-local labels generated by cfg.labels above.
            # The support and query segments each have their own global class ordering,
            # but they are in the SAME order (CategoriesSampler picks classes once, then
            # samples n_per images per class). So the k-th support and k-th query image
            # belong to the same global class. We remap each segment independently so that
            # remapped_sup[k] == remapped_que[k] for all k.
            # -------------------------------------------------------------------------
            sup_glo = _labels[:cfg.num_support].cpu().numpy()
            que_glo = _labels[cfg.num_support:].cpu().numpy()

            sup_unique = sorted(set(sup_glo.tolist()))
            que_unique = sorted(set(que_glo.tolist()))
            sup_to_local = {c: i for i, c in enumerate(sup_unique)}
            que_to_local = {c: i for i, c in enumerate(que_unique)}

            sup_local = torch.tensor(
                [sup_to_local[c] for c in sup_glo], dtype=torch.long, device=cfg.device
            )
            que_local = torch.tensor(
                [que_to_local[c] for c in que_glo], dtype=torch.long, device=cfg.device
            )
            # After independent remapping, sup_local[i] == que_local[i] for all i
            # because both segments are in the same class order

            # For medical (1-shot 1-query): reorder fea so that classes are consecutive:
            # class0_sup, class0_que, class1_sup, class1_que, ...
            # inv_sup[local_label] gives the position in fea[:n_way] whose remapped label is local_label.
            if cfg.dataset == 'medical' and cfg.k_shot == 1 and cfg.k_query == 1:
                n_way = cfg.n_way
                # Build inverse maps: local_label_value -> position index in each segment
                inv_sup = {sup_local[i].item(): i for i in range(n_way)}
                inv_que = {que_local[i].item(): i for i in range(n_way)}
                # Collect reorder indices
                reorder_sup = [inv_sup[i] for i in range(n_way)]           # positions in fea[:n_way]
                reorder_que = [inv_que[i] + n_way for i in range(n_way)]  # positions in fea[n_way:]
                fea = torch.cat([fea[reorder_sup], fea[reorder_que]], dim=0)
                _labels = torch.arange(n_way, dtype=torch.long, device=cfg.device).repeat(2)
            else:
                # Non-medical: just use the independently-remapped labels
                _labels = torch.cat([sup_local, que_local], dim=0)
            # -------------------------------------------------------------------------

            # initialization for weights
            sup_prototype = fea[:cfg.num_support, ...] \
                .contiguous().view(cfg.k_shot, cfg.n_way, fea.size(-2), fea.size(-1)).transpose(0, 1).mean(
                1)  # n_way, num, out_dim//num

            # sup_prototype = torch.randn_like(sup_prototype)

            # initialization for anchors
            anchor.append(fea[cfg.num_support:, ...].sum(0, keepdim=True) / cfg.num_support)  # T=0
            for t in range(1, cfg.T + 1):
                anchor.append(
                    ((cfg.T - t) * fea[cfg.num_support:, ...].sum(0, keepdim=True) +
                     t * fea[:cfg.num_support, ...].sum(0, keepdim=True)) /
                    ((cfg.T - t) * cfg.num_support + t * cfg.num_query))

            if len(anchor) > 1:
                anchor = torch.cat(anchor, 0)
            else:
                anchor = anchor[0]

        mani = "Euclidean"
        # mani = "oblique"
        if mani == "oblique":
            manifold = Oblique()
        else:
            manifold = Euclidean()

        projection = manifold.proj(fea)  # project to manifold
        sup_prototype = manifold.proj(sup_prototype)  # project to manifold
        anchor = manifold.proj(anchor)

        que_proj = projection[cfg.num_support:, ...].contiguous()  # (num_query, out_dim, out_dim )
        sup_proj = projection[:cfg.num_support, ...].contiguous()  # (num_support, out_dim, out_dim)

        sup_fea = sup_proj.clone().detach()
        que_fea = que_proj.clone().detach()
        sup_labels = _labels[:cfg.num_support].contiguous().clone().detach()
        que_labels = _labels[cfg.num_support:].contiguous().clone().detach()

        # fineTune
        #################################################################

        # init weights and layers
        fc = ODC(cfg, manifold=manifold, anchor=anchor, proto=sup_prototype).cuda()

        acc = []
        # optimizer = RiemannianAdam(
        #     [{'params': fc.parameters(), 'lr': cfg.train_meta_lr}],
        #     # weight_decay=cfg.weight_decay,
        # )
        # scheduler = StepLR(optimizer, cfg.train_meta_step, gamma=0.1, last_epoch=-1)

        optimizer = RiemannianAdam(
            [{'params': fc.proto, 'lr': cfg.lr_weights}, {'params': fc.anchor, 'lr': cfg.lr_anchors}],
            # weight_decay=1e-4,
        )
        # scheduler = StepLR(optimizer, 400, gamma=0.5, last_epoch=-1)
        scheduler = CosineAnnealingLR(optimizer, 0.1 * cfg.train_meta_epochs, eta_min=1e-9)

        # optimizer = RiemannianSGD(
        #     fc.parameters(), lr=1e-3,
        #     weight_decay=cfg.weight_decay, momentum=0.99)
        # scheduler = StepLR(optimizer, 40, gamma=0.1, last_epoch=-1)
        # scheduler = MultiStepLR(optimizer,
        #                         milestones=[int(0.5 * cfg.train_meta_epochs),
        #                                     int(0.75 * cfg.train_meta_epochs)],
        #                         # milestones=[50, 75],
        #                         last_epoch=-1)

        # amp
        opt_level = cfg.opt_level

        # fc, optimizer = amp.initialize(fc, optimizer, opt_level=opt_level, verbosity=0)
        loss_weight = [cfg.w1, cfg.w2, cfg.w3]
        acc_max = 0
        # for ep in range(0, cfg.train_meta_epochs):
        for ep in range(0, cfg.train_meta_epochs):
            ep += 1
            fc.train()

            logits = fc(torch.cat([sup_fea, que_fea], 0))

            sup_logits = logits[:cfg.num_support, ...] - 0 * smooth_one_hot(sup_labels, classes=cfg.n_way, smoothing=0)
            que_logits = logits[cfg.num_support:, ...]

            # compute loss and acc
            sup_loss, sup_acc = compute_loss_acc(sup_logits, sup_labels, num_class=cfg.n_way, smoothing=0, )

            if cfg.T == 0:
                loss = loss_weight[0] * sup_loss
            else:
                que_probs = que_logits.softmax(-1)
                que_cond_ent = -(que_probs * torch.log(que_probs + 1e-12)).sum(-1).mean(0)
                que_ent = -(que_probs.mean(0) * torch.log(que_probs.mean(0))).sum(-1)
                loss = loss_weight[0] * sup_loss - (loss_weight[1] * que_ent - loss_weight[2] * que_cond_ent)

            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # fc.eval()
            # logits = fc(torch.cat([sup_fea, que_fea], 0))
            # que_logits = logits[cfg.num_support:, ...]
            que_loss, que_acc = compute_loss_acc(que_logits, que_labels, num_class=cfg.n_way, smoothing=0)
            if rank == print_rank and ep % 1090 == 0:
                print(ep, f"{optimizer.param_groups[0]['lr']:.5f}", sup_loss.item(), sup_acc.item(),
                      (loss_weight[1] * que_ent - loss_weight[2] * que_cond_ent).item(), que_loss.item(),
                      que_acc.item())
            acc.append(que_acc)



        acc = torch.stack(acc, 0)[-1]
        gather_t = [torch.ones_like(acc) for _ in range(world_size)]
        torch.distributed.all_gather(gather_t, acc.clone().detach())

        best_acc_list_ = [acc_item.item() for acc_item in gather_t]
        best_acc_list = best_acc_list + best_acc_list_

        cur_acc = np.mean(best_acc_list_) + 0
        val_acc = np.mean(best_acc_list) + 0
        std = np.std(best_acc_list)
        pm = 1.96 * (std / np.sqrt(len(best_acc_list)))

        loss_float['cur_acc'] = cur_acc
        loss_float['val_acc'] = val_acc
        loss_float['pm'] = pm


    if rank == print_rank:
        print(f'val_acc:{val_acc * 100:.3f}, pm:{pm * 100:.3f}')
        if hasattr(cfg, 'trial'):
            print(f'w1:{cfg.w1:.6f}, w2{cfg.w2:.6f}, w3{cfg.w3:.6f}, scale_factor:{cfg.scale_factor:.6f}')
        # Save evaluation result to a JSON file for easy access
        import json
        os.makedirs(
            cfg.train_meta_save_path,
            exist_ok=True
        )

        result_path = os.path.join(
            cfg.train_meta_save_path,
            'eval_result.json'
        )

        # with open(result_path, 'w') as f:
        #     json.dump(
        #         result,
        #         f,
        #         indent=2
        #     )
        result = {
            'dataset': cfg.dataset,
            'medical_subset': getattr(cfg, 'medical_subset', None),
            'n_way': cfg.n_way,
            'k_shot': cfg.k_shot,
            'k_query': cfg.k_query,
            'backbone': cfg.backbone,
            'val_acc': float(val_acc * 100),
            'pm': float(pm * 100),
        }
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'Result saved to {result_path}')
    clean_up()


# ===========================
# FSL-OM evaluation launcher
# ===========================


# ===========================
# TEST-TEST evaluation launcher
# ===========================

def main(rank, world_size, cfg):
    cfg.world_size = world_size

    print(
        f"Start FSL-OM TEST-TEST evaluation: "
        f"{cfg.medical_subset}"
    )

    cfg.val_set = "test"

    fine_tune(
        rank,
        world_size,
        cfg
    )


if __name__ == "__main__":

    world_size_ = 1

    run_model(
        main,
        world_size_,
        args
    )
