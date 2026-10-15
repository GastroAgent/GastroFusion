# Torch 2.x compatible GAP dataset package
#
# No torchmeta dependency.
# Provides the same dataset entry points required by GAP.

from .episode_dataset import FolderEpisodeDataset
from .miniimagenet import MiniImagenet
from .helpers import miniimagenet

__all__ = [
    "FolderEpisodeDataset",
    "MiniImagenet",
    "miniimagenet",
]
