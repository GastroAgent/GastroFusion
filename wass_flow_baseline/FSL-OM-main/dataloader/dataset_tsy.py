
import torch
from torch.utils.data import Dataset


class CrossDomainEpisodeDataset(Dataset):

    """
    Generate episodes where support/query
    are from different datasets.
    """

    def __init__(
        self,
        support_dataset,
        query_dataset,
        sampler
    ):
        self.support_dataset = support_dataset
        self.query_dataset = query_dataset
        self.sampler = sampler


    def __len__(self):
        return len(self.sampler)


    def __getitem__(self, index):

        sampler_iter = iter(self.sampler)

        for _ in range(index + 1):
            support_idx, query_idx = next(sampler_iter)


        support_images = []
        support_labels = []

        query_images = []
        query_labels = []


        for i in support_idx:

            img, label = self.support_dataset[
                i.item()
            ]

            support_images.append(img)
            support_labels.append(label)


        for i in query_idx:

            img, label = self.query_dataset[
                i.item()
            ]

            query_images.append(img)
            query_labels.append(label)


        return (
            torch.stack(support_images),
            torch.tensor(support_labels),
            torch.stack(query_images),
            torch.tensor(query_labels)
        )
