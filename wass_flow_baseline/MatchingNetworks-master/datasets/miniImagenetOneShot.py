##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os.path
import csv
import math
import collections
from tqdm import tqdm

import numpy as np

# LAMBDA FUNCTIONS
filenameToPILImage = lambda x: Image.open(x).convert('RGB')
PiLImageResize = lambda x: x.resize((84, 84))


class miniImagenetOneShotDataset(data.Dataset):
    """
    Supports both:
      - miniImageNet (miniimagenet): expects <dataroot>/{train,val,test}.csv and <dataroot>/images/<file>
      - medical: expects <dataroot>/<medical_subset>/<split>/<class>/<img>
                 split can be 'train' / 'val' / 'test', e.g.
                 <dataroot>/胃_1query/train/class_a/img_001.jpg
    """

    def __init__(self, dataroot='/home/aberenguel/Dataset/miniImagenet',
                 type='train', nEpisodes=1000, classes_per_set=10,
                 samples_per_class=1, k_query=5,
                 medical_subset='胃_1query', dataset='miniimagenet',
                 channels=3):
        self.nEpisodes = nEpisodes
        self.classes_per_set = classes_per_set
        self.samples_per_class = samples_per_class
        self.k_query = k_query
        self.n_samples = self.samples_per_class * self.classes_per_set
        self.n_query_total = self.k_query * self.classes_per_set
        self.dataset = dataset
        self.channels = channels

        if channels == 1:
            self.transform = transforms.Compose([
                lambda x: x.convert('L'),
                PiLImageResize,
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                filenameToPILImage,
                PiLImageResize,
                transforms.ToTensor(),
            ])

        if dataset == 'miniimagenet':
            self.miniImagenetImagesDir = os.path.join(dataroot, 'images')
            self.data = self._load_csv_split(os.path.join(dataroot, type + '.csv'))
            self.classes_dict = {k: i for i, k in enumerate(self.data.keys())}
            self.imagesDir = self.miniImagenetImagesDir
        elif dataset == 'medical':
            # dataroot = <project_root>/data, medical_subset = 胃_1query, type = train/val/test
            self.imagesDir = os.path.join(dataroot, medical_subset, type)
            self.data = self._load_medical_split(self.imagesDir)
            self.classes_dict = {k: i for i, k in enumerate(self.data.keys())}
        else:
            raise ValueError(f'Unsupported dataset: {dataset}')

        self.create_episodes(self.nEpisodes)

    def _load_csv_split(self, split_file):
        dictLabels = collections.OrderedDict()
        with open(split_file) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)
            for row in csvreader:
                if len(row) < 2:
                    continue
                filename = row[0]
                label = row[1]
                if label in dictLabels:
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels

    def _load_medical_split(self, split_dir):
        """Load <split_dir>/<class>/*.jpg|*.png|*.jpeg structure."""
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f'Medical split dir not found: {split_dir}')
        dictLabels = collections.OrderedDict()
        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        classes = sorted([d for d in os.listdir(split_dir)
                          if os.path.isdir(os.path.join(split_dir, d))])
        for cls in classes:
            cls_dir = os.path.join(split_dir, cls)
            files = sorted([f for f in os.listdir(cls_dir)
                            if os.path.splitext(f)[1].lower() in valid_ext])
            if not files:
                continue
            rel_files = [os.path.join(cls, f) for f in files]
            dictLabels[cls] = rel_files
        return dictLabels

    def create_episodes(self, episodes):
        self.support_set_x_batch = []
        self.target_x_batch = []
        # need at least k_shot + k_query samples per selected class
        needed = self.samples_per_class + self.k_query
        # classes that have enough samples
        valid_classes = [c for c, files in self.data.items() if len(files) >= needed]
        if len(valid_classes) < self.classes_per_set:
            raise RuntimeError(
                f'Only {len(valid_classes)} classes have >= {needed} samples '
                f'in split. Need at least {self.classes_per_set}.'
            )
        rng = np.random.RandomState(2191)  # for reproducibility
        self.valid_classes = valid_classes
        for b in range(episodes):
            selected = rng.choice(len(valid_classes), self.classes_per_set, replace=False)
            support_x = []
            target_x = []
            for c in selected:
                cls_name = valid_classes[c]
                n = len(self.data[cls_name])
                # support indices + query indices, without overlap
                picks = rng.choice(n, needed, replace=False)
                support_x.append([self.data[cls_name][i] for i in picks[:self.samples_per_class]])
                target_x.append([self.data[cls_name][i] for i in picks[self.samples_per_class:]])
            self.support_set_x_batch.append(support_x)
            self.target_x_batch.append(target_x)

    def __getitem__(self, index):
        support_set_x = torch.FloatTensor(self.n_samples, self.channels, 84, 84)
        support_set_y = np.zeros((self.n_samples,), dtype=np.int64)
        target_x = torch.FloatTensor(self.n_query_total, self.channels, 84, 84)
        target_y = np.zeros((self.n_query_total,), dtype=np.int64)

        flat_support = [os.path.join(self.imagesDir, item)
                        for sublist in self.support_set_x_batch[index]
                        for item in sublist]
        support_set_y = np.array(
            [self.classes_dict[item.split('/')[0]]
             for sublist in self.support_set_x_batch[index]
             for item in sublist], dtype=np.int64)

        flat_target = [os.path.join(self.imagesDir, item)
                       for sublist in self.target_x_batch[index]
                       for item in sublist]
        target_y = np.array(
            [self.classes_dict[item.split('/')[0]]
             for sublist in self.target_x_batch[index]
             for item in sublist], dtype=np.int64)

        for i, path in enumerate(flat_support):
            if self.transform is not None:
                support_set_x[i] = self.transform(path)
        for i, path in enumerate(flat_target):
            if self.transform is not None:
                target_x[i] = self.transform(path)

        # remap labels to [0, classes_per_set)
        unique = np.unique(support_set_y)
        classes_dict_temp = {int(u): i for i, u in enumerate(unique)}
        support_set_y = np.array([classes_dict_temp[int(v)] for v in support_set_y], dtype=np.int64)
        target_y = np.array([classes_dict_temp[int(v)] for v in target_y], dtype=np.int64)

        return support_set_x, torch.from_numpy(support_set_y), target_x, torch.from_numpy(target_y)

    def __len__(self):
        return self.nEpisodes