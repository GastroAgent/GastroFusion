import torch
from torch.utils.data import DataLoader

from dataloader.samplers_tsy import CategoriesSampler


def data_loader_without_dali(cfg):
    print("init data loader")
    from dataloader.dataset import FSLDateset as Dataset
    return Dataset


def get_loader_without_dali(cfg, pretrain):
    Dataset = data_loader_without_dali(cfg)

    # Pretrain: use DistributedSampler for standard datasets, or full-data loader for medical
    if pretrain:
        if cfg.dataset == 'mini2cub':
            cfg.dataset = 'mini'
            train_set = Dataset('train', cfg, augment=True)
            cfg.dataset = 'mini2cub'
        elif cfg.dataset == 'medical':
            # For medical: use all train data with a standard DataLoader
            train_set = Dataset('train', cfg, augment=True)
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
            train_loader = DataLoader(dataset=train_set, batch_size=cfg.batch_size,
                                    sampler=train_sampler,
                                    num_workers=cfg.num_workers, pin_memory=True)
            return train_loader, None
        else:
            train_set = Dataset('train', cfg, augment=True)
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
            train_loader = DataLoader(dataset=train_set, batch_size=cfg.batch_size,
                                    sampler=train_sampler,
                                    num_workers=cfg.num_workers, pin_memory=True)
            return train_loader, None

    # Meta/eval: use CategoriesSampler for episodic loading
    if cfg.dataset == 'mini2cub':
        cfg.dataset = 'mini'
        train_set = Dataset('train', cfg, augment=True)
        cfg.dataset = 'mini2cub'
    elif cfg.dataset == 'medical':
        train_set = Dataset('train', cfg, augment=True)
    else:
        train_set = Dataset('train', cfg, augment=True)

    train_sampler = CategoriesSampler(train_set.label, cfg.train_episodes // cfg.world_size, cfg.n_way,
                                     cfg.k_shot + cfg.k_query)
    train_loader = DataLoader(dataset=train_set, batch_sampler=train_sampler, num_workers=cfg.num_workers,
                             pin_memory=True)

    if cfg.dataset == 'mini2cub':
        cfg.dataset = 'mini'
        val_set = Dataset(cfg.val_set, cfg)
        cfg.dataset = 'mini2cub'
    elif cfg.dataset == 'medical':
        val_set = Dataset(cfg.val_set, cfg)
    else:
        val_set = Dataset(cfg.val_set, cfg)

    val_sampler = CategoriesSampler(val_set.label, cfg.val_episodes // cfg.world_size, cfg.n_way,
                                   cfg.k_shot + cfg.k_query)
    val_loader = DataLoader(dataset=val_set, batch_sampler=val_sampler,
                           num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader


def get_loader(cfg, pretrain=False):
    return get_loader_without_dali(cfg, pretrain)
