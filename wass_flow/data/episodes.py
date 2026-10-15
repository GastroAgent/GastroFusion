from __future__ import annotations

import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from PIL import Image

from .class_pool import ImageRecord, build_rae_transform


class RandomEpisodeDataset(Dataset):
    """On-the-fly N-way K-shot Q-query episodes from a labelled image pool."""

    def __init__(
        self,
        records: list[ImageRecord],
        n_way: int,
        n_shot: int,
        n_query: int,
        episodes_per_epoch: int,
        transform=None,
        seed: int = 0,
    ) -> None:
        self.n_way = int(n_way)
        self.n_shot = int(n_shot)
        self.n_query = int(n_query)
        self.episodes_per_epoch = int(episodes_per_epoch)
        self.transform = transform or build_rae_transform()
        self.seed = int(seed)
        pool: dict[str, list[ImageRecord]] = defaultdict(list)
        for record in records:
            pool[record.label].append(record)
        min_count = self.n_shot + self.n_query
        self.pool = {k: v for k, v in pool.items() if len(v) >= min_count}
        self.classes = sorted(self.pool)
        if len(self.classes) < self.n_way:
            raise ValueError(
                f"Only {len(self.classes)} classes have >= {min_count} images; n_way={self.n_way}"
            )
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.episodes_per_epoch

    def _load(self, record: ImageRecord) -> torch.Tensor:
        with Image.open(record.path) as image:
            return self.transform(image.convert("RGB"))

    def __getitem__(self, index: int):
        rng = random.Random(self.seed + self.epoch * self.episodes_per_epoch + index)
        selected = rng.sample(self.classes, self.n_way)
        support, query, query_labels = [], [], []
        support_paths, query_paths = [], []
        for local_id, label in enumerate(selected):
            chosen = rng.sample(self.pool[label], self.n_shot + self.n_query)
            s_records = chosen[: self.n_shot]
            q_records = chosen[self.n_shot :]
            support.append(torch.stack([self._load(r) for r in s_records]))
            query.extend(self._load(r) for r in q_records)
            query_labels.extend([local_id] * self.n_query)
            support_paths.append([r.path for r in s_records])
            query_paths.extend(r.path for r in q_records)
        return {
            "support_images": torch.stack(support),          # (N,K,3,H,W)
            "query_images": torch.stack(query),              # (N*Q,3,H,W)
            "query_labels": torch.tensor(query_labels, dtype=torch.long),
            "class_names": selected,
            "support_paths": support_paths,
            "query_paths": query_paths,
        }
