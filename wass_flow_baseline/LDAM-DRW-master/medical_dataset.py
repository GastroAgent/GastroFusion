import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image


class MedicalImageDataset(torch.utils.data.Dataset):
    """适用于 ImageFolder 风格目录结构的 Dataset（train/val/test 下每层子文件夹为一类）"""

    def __init__(self, data_dir, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

        self.classes, self.class_to_idx = self._find_classes()
        self.samples = self._make_dataset()
        self.targets = [s[1] for s in self.samples]

        self.num_per_cls_dict = dict()
        for _, label in self.samples:
            self.num_per_cls_dict[label] = self.num_per_cls_dict.get(label, 0) + 1

    def _find_classes(self):
        classes = sorted([d for d in os.listdir(self.data_dir)
                          if os.path.isdir(os.path.join(self.data_dir, d))])
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self):
        samples = []
        for target_class in sorted(self.class_to_idx.keys()):
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(self.data_dir, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        path = os.path.join(root, fname)
                        samples.append((path, class_index))
        return samples

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        return len(self.samples)

    def get_cls_num_list(self):
        cls_num_list = [self.num_per_cls_dict.get(i, 0) for i in range(len(self.classes))]
        return cls_num_list
