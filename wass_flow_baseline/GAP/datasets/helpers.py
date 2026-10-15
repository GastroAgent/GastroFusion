# Deprecated torchmeta helper replacement.
from .episode_dataset import FolderEpisodeDataset

def miniimagenet(folder, shots, ways, test_shots, meta_train=False, meta_val=False, meta_test=False, transform=None, **kwargs):
    split = "train" if meta_train else ("val" if meta_val else "test")
    return FolderEpisodeDataset(folder, ways, shots, test_shots, transform=transform, split=split)
