
from torch.utils.data import DataLoader

from dataloader.dataset import FSLDateset as Dataset
from dataloader.samplers_tsy import (
    CategoriesSampler,
    CrossDomainCategoriesSampler
)
from dataloader.dataset_tsy import CrossDomainEpisodeDataset



def get_loader(cfg, pretrain=False):

    if pretrain:
        train_set = Dataset(
            "train",
            cfg,
            augment=True
        )

        loader = DataLoader(
            train_set,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers
        )

        return loader, None



    train_set = Dataset(
        "train",
        cfg,
        augment=True
    )

    train_sampler = CategoriesSampler(
        train_set.label,
        cfg.train_episodes,
        cfg.n_way,
        cfg.k_shot + cfg.k_query
    )


    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler
    )


    if getattr(cfg,"eval_mode","test-test") == "test-train":

        support_set = Dataset(
            "train",
            cfg
        )

        query_set = Dataset(
            "test",
            cfg
        )


        sampler = CrossDomainCategoriesSampler(
            support_set.label,
            query_set.label,
            cfg.val_episodes,
            cfg.n_way,
            cfg.k_shot,
            cfg.k_query
        )


        val_set = CrossDomainEpisodeDataset(
            support_set,
            query_set,
            sampler
        )


        val_loader = DataLoader(
            val_set,
            batch_size=1
        )


    else:

        val_set = Dataset(
            "test",
            cfg
        )

        sampler = CategoriesSampler(
            val_set.label,
            cfg.val_episodes,
            cfg.n_way,
            cfg.k_shot + cfg.k_query
        )


        val_loader = DataLoader(
            val_set,
            batch_sampler=sampler
        )


    return train_loader, val_loader
