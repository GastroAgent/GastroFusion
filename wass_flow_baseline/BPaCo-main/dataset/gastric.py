import os
import copy
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from .randaugment import rand_augment_transform
import moco.loader


class Gastric_moco(Dataset):
    """Dataset for gastric (stomach) medical images with MoCo augmentation.

    This dataset reads images from a txt file (which lists both original and
    _masked_crop images), then creates pairs: each original image is paired
    with its _masked_crop counterpart (or a second copy if no masked version exists).
    """

    def __init__(self, root, txt, num_classes, transform=None):
        self.img_path = []
        self.labels = []
        self.root = root
        self.num_classes = num_classes

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.origin_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        self.resized_crop = transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.))
        self.resized_crop_random = transforms.RandomResizedCrop((224, 224))

        with open(txt, encoding='utf-8') as f:
            for line in f:
                parts = line.rsplit(None, 1)
                if len(parts) >= 2:
                    self.img_path.append(os.path.join(root, parts[0]))
                    self.labels.append(int(parts[1]))

        self.class_data = [[] for _ in range(self.num_classes)]
        for i, y in enumerate(self.labels):
            self.class_data[y].append(i)

        self.cls_num_list = [len(self.class_data[i]) for i in range(self.num_classes)]

    def __len__(self):
        return len(self.labels)

    def _get_masked_path(self, path):
        """Get the corresponding masked_crop image path. If it doesn't exist, return original path."""
        if "_masked_crop" not in path:
            base, ext = os.path.splitext(path)
            masked = base + "_masked_crop" + ext
            if os.path.exists(masked):
                return masked
        return path

    def __getitem__(self, index):
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(
            translate_const=int(224 * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
        )

        path = self.img_path[index]
        target = self.labels[index]
        masked_path = self._get_masked_path(path)

        with open(path, 'rb') as f:
            img_origin = Image.open(f).convert('RGB')
        with open(masked_path, 'rb') as f:
            img_pre = Image.open(f).convert('RGB')

        # View 1: strong augmentation on origin + weak on masked
        img_origin1, img_pre1 = copy.copy(img_origin), copy.copy(img_pre)
        origin_sample1, img_sample1 = self._resized_crop_with_preknowledge(
            self.resized_crop, img_origin1, img_pre1)
        origin_sample1, img_sample1 = self._random_horizontal_flip(
            origin_sample1, img_sample1, p=0.5)
        origin_sample1 = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0)(origin_sample1)
        origin_sample1, img_sample1 = rand_augment_transform(
            'rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params,
            origin_sample1, img_sample1)
        origin_sample1 = self.origin_transform(origin_sample1)
        img_sample1 = self.origin_transform(img_sample1)

        # View 2: weak augmentation on origin + weak on masked
        img_origin2, img_pre2 = copy.copy(img_origin), copy.copy(img_pre)
        origin_sample2, img_sample2 = self._resized_crop_random_with_preknowledge(
            self.resized_crop_random, img_origin2, img_pre2)
        origin_sample2, img_sample2 = self._random_horizontal_flip(
            origin_sample2, img_sample2, p=0.5)
        origin_sample2 = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0)(origin_sample2)
        origin_sample2 = transforms.RandomGrayscale(p=0.2)(origin_sample2)
        origin_sample2 = transforms.RandomApply(
            [moco.loader.GaussianBlur([.1, 2.])], p=0.5)(origin_sample2)
        origin_sample2, img_sample2 = rand_augment_transform(
            'rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params,
            origin_sample2, img_sample2)
        origin_sample2 = self.origin_transform(origin_sample2)
        img_sample2 = self.origin_transform(img_sample2)

        return [origin_sample1, origin_sample2, img_sample1, img_sample2], target

    def _resized_crop_with_preknowledge(self, tf, origin, img):
        from torchvision.transforms import functional as F
        assert isinstance(tf, transforms.RandomResizedCrop)
        i, j, h, w = tf.get_params(origin, tf.scale, tf.ratio)
        img = F.resized_crop(img, i, j, h, w, tf.size, tf.interpolation)
        origin = F.resized_crop(origin, i, j, h, w, tf.size, tf.interpolation)
        return origin, img

    def _resized_crop_random_with_preknowledge(self, tf, origin, img):
        from torchvision.transforms import functional as F
        assert isinstance(tf, transforms.RandomResizedCrop)
        i, j, h, w = tf.get_params(origin, tf.scale, tf.ratio)
        img = F.resized_crop(img, i, j, h, w, tf.size, tf.interpolation)
        origin = F.resized_crop(origin, i, j, h, w, tf.size, tf.interpolation)
        return origin, img

    def _random_horizontal_flip(self, origin, img, p):
        from torchvision.transforms import functional as F
        if random.random() < p:
            img = F.hflip(img)
            origin = F.hflip(origin)
        return origin, img
