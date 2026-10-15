import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class FolderEpisodeDataset(Dataset):
    def __init__(self, root, ways=5, shots=5, query_shots=1,
                 episodes=100000, transform=None, split=None):
        self.root = root
        self.ways = ways
        self.shots = shots
        self.query_shots = query_shots
        self.episodes = episodes
        self.transform = transform

        self.classes = sorted([
            c for c in os.listdir(root)
            if os.path.isdir(os.path.join(root, c))
        ])

        self.data = {
            c: [
                os.path.join(root, c, f)
                for f in os.listdir(os.path.join(root, c))
                if os.path.isfile(os.path.join(root, c, f))
            ]
            for c in self.classes
        }

    def __len__(self):
        return self.episodes

    def load_image(self, path):
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        rng = random.Random(idx)

        valid = [
            c for c in self.classes
            if len(self.data[c]) > 0
        ]

        chosen = rng.sample(valid, self.ways)

        sx, sy = [], []
        qx, qy = [], []

        for y, cls in enumerate(chosen):
            imgs = self.data[cls]

            support = rng.choices(imgs, k=self.shots)
            query = rng.choices(imgs, k=self.query_shots)

            sx.extend([self.load_image(p) for p in support])
            sy.extend([y] * self.shots)

            qx.extend([self.load_image(p) for p in query])
            qy.extend([y] * self.query_shots)

        return {
            "train": (
                torch.stack(sx),
                torch.tensor(sy, dtype=torch.long)
            ),
            "test": (
                torch.stack(qx),
                torch.tensor(qy, dtype=torch.long)
            )
        }


def get_dataset(dataset, folder, ways, shots, query_shots,
                download=False, train_transform=None,
                test_transform=None):

    train_set = FolderEpisodeDataset(
        folder,
        ways,
        shots,
        query_shots,
        transform=train_transform,
        split="train"
    )

    val_set = FolderEpisodeDataset(
        folder,
        ways,
        shots,
        query_shots,
        transform=test_transform,
        split="val"
    )

    test_set = FolderEpisodeDataset(
        folder,
        ways,
        shots,
        query_shots,
        transform=test_transform,
        split="test"
    )

    return train_set, val_set, test_set
