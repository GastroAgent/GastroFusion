##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## Modified to support:
##   1. test-test:
##      support comes from test, query comes from test
##   2. test-train:
##      support comes from train, query comes from test
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import csv
import collections

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


VALID_IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'
}


def filename_to_pil_image(path):
    """Open an image path and convert it to RGB."""
    with Image.open(path) as image:
        return image.convert('RGB')


def pil_image_resize(image):
    """Resize PIL image to 84 x 84."""
    return image.resize((84, 84))


class miniImagenetOneShotDataset(data.Dataset):
    """
    Episodic dataset supporting MiniImageNet and medical datasets.

    MiniImageNet directory format:
        <dataroot>/
            train.csv
            val.csv
            test.csv
            images/
                image_1.jpg
                image_2.jpg
                ...

    Medical directory format:
        <dataroot>/
            <medical_subset>/
                train/
                    <class_name>/
                        image_1.jpg
                        ...
                val/
                    <class_name>/
                        ...
                test/
                    <class_name>/
                        ...

    Evaluation protocols
    --------------------
    test-test:
        support_split = 'test'
        query_split = 'test'

        Support and query images are sampled from the same test class pool.
        Query images are sampled without replacement. Query images are then
        removed from the support candidate pool. Support is sampled without
        replacement when enough candidates exist; otherwise only support is
        sampled with replacement. Support and query never overlap.

    test-train:
        support_split = 'train'
        query_split = 'test'

        Support images are sampled from train.
        Query images are sampled from test.

    If support_type and query_type are not provided, both default to `type`.
    Therefore:

        type='train' -> train-train
        type='val'   -> val-val
        type='test'  -> test-test
    """

    def __init__(
        self,
        dataroot='/home/aberenguel/Dataset/miniImagenet',
        type='train',
        nEpisodes=1000,
        classes_per_set=10,
        samples_per_class=1,
        k_query=5,
        medical_subset='胃_1query',
        dataset='miniimagenet',
        channels=3,
        support_type=None,
        query_type=None,
        seed=2191,
    ):
        super().__init__()

        self.dataroot = dataroot
        self.type = type
        self.nEpisodes = nEpisodes
        self.classes_per_set = classes_per_set
        self.samples_per_class = samples_per_class
        self.k_query = k_query
        self.medical_subset = medical_subset
        self.dataset = dataset
        self.channels = channels
        self.seed = seed

        # If no separate support/query split is specified,
        # preserve the original same-split behavior.
        self.support_type = support_type if support_type is not None else type
        self.query_type = query_type if query_type is not None else type

        self.n_samples = self.samples_per_class * self.classes_per_set
        self.n_query_total = self.k_query * self.classes_per_set

        self._validate_arguments()
        self._build_transform()
        self._load_support_and_query_data()

        self.create_episodes(self.nEpisodes)

        print(
            '[Dataset] '
            f'dataset={self.dataset}, '
            f'type={self.type}, '
            f'support_split={self.support_type}, '
            f'query_split={self.query_type}, '
            f'n_way={self.classes_per_set}, '
            f'k_shot={self.samples_per_class}, '
            f'k_query={self.k_query}, '
            f'episodes={self.nEpisodes}, '
            f'valid_classes={len(self.valid_classes)}'
        )

    def _validate_arguments(self):
        """Validate dataset configuration."""
        if self.dataset not in {'miniimagenet', 'medical'}:
            raise ValueError(
                f'Unsupported dataset: {self.dataset}. '
                f'Expected "miniimagenet" or "medical".'
            )

        valid_splits = {'train', 'val', 'test'}

        if self.support_type not in valid_splits:
            raise ValueError(
                f'Invalid support_type: {self.support_type}. '
                f'Expected one of {sorted(valid_splits)}.'
            )

        if self.query_type not in valid_splits:
            raise ValueError(
                f'Invalid query_type: {self.query_type}. '
                f'Expected one of {sorted(valid_splits)}.'
            )

        if self.classes_per_set <= 0:
            raise ValueError('classes_per_set must be greater than 0.')

        if self.samples_per_class <= 0:
            raise ValueError('samples_per_class must be greater than 0.')

        if self.k_query <= 0:
            raise ValueError('k_query must be greater than 0.')

        if self.nEpisodes <= 0:
            raise ValueError('nEpisodes must be greater than 0.')

        if self.channels not in {1, 3}:
            raise ValueError(
                f'Unsupported channels={self.channels}. Expected 1 or 3.'
            )

    def _build_transform(self):
        """Create the image preprocessing transform."""
        transform_steps = [
            filename_to_pil_image,
        ]

        if self.channels == 1:
            transform_steps.append(
                transforms.Lambda(lambda image: image.convert('L'))
            )

        transform_steps.extend([
            transforms.Lambda(pil_image_resize),
            transforms.ToTensor(),
        ])

        self.transform = transforms.Compose(transform_steps)

    def _load_support_and_query_data(self):
        """Load the support split and query split separately."""
        if self.dataset == 'miniimagenet':
            self.support_images_dir = os.path.join(
                self.dataroot,
                'images'
            )
            self.query_images_dir = self.support_images_dir

            support_csv = os.path.join(
                self.dataroot,
                self.support_type + '.csv'
            )
            query_csv = os.path.join(
                self.dataroot,
                self.query_type + '.csv'
            )

            self.support_data = self._load_csv_split(support_csv)
            self.query_data = self._load_csv_split(query_csv)

        elif self.dataset == 'medical':
            subset_root = os.path.join(
                self.dataroot,
                self.medical_subset
            )

            self.support_images_dir = os.path.join(
                subset_root,
                self.support_type
            )
            self.query_images_dir = os.path.join(
                subset_root,
                self.query_type
            )

            self.support_data = self._load_medical_split(
                self.support_images_dir
            )
            self.query_data = self._load_medical_split(
                self.query_images_dir
            )

        self.same_source_split = (
            self.support_type == self.query_type
        )

    def _load_csv_split(self, split_file):
        """Load one MiniImageNet CSV split."""
        if not os.path.isfile(split_file):
            raise FileNotFoundError(
                f'MiniImageNet split file not found: {split_file}'
            )

        labels_to_files = collections.OrderedDict()

        with open(split_file, newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader, None)

            for row in csv_reader:
                if len(row) < 2:
                    continue

                filename = row[0].strip()
                label = row[1].strip()

                if not filename or not label:
                    continue

                if label not in labels_to_files:
                    labels_to_files[label] = []

                labels_to_files[label].append(filename)

        if not labels_to_files:
            raise RuntimeError(
                f'No valid data found in CSV file: {split_file}'
            )

        return labels_to_files

    def _load_medical_split(self, split_dir):
        """
        Load a medical split organized as:

            <split_dir>/<class_name>/<image>
        """
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f'Medical split directory not found: {split_dir}'
            )

        labels_to_files = collections.OrderedDict()

        class_names = sorted([
            directory_name
            for directory_name in os.listdir(split_dir)
            if os.path.isdir(
                os.path.join(split_dir, directory_name)
            )
        ])

        for class_name in class_names:
            class_dir = os.path.join(
                split_dir,
                class_name
            )

            filenames = sorted([
                filename
                for filename in os.listdir(class_dir)
                if (
                    os.path.isfile(os.path.join(class_dir, filename))
                    and os.path.splitext(filename)[1].lower()
                    in VALID_IMAGE_EXTENSIONS
                )
            ])

            if not filenames:
                continue

            # Store paths relative to the split directory.
            relative_paths = [
                os.path.join(class_name, filename)
                for filename in filenames
            ]

            labels_to_files[class_name] = relative_paths

        if not labels_to_files:
            raise RuntimeError(
                f'No valid medical images found in: {split_dir}'
            )

        return labels_to_files

    def _find_valid_classes(self):
        """
        Find classes shared between support and query splits that have
        enough samples for the requested episode configuration.
        """
        support_classes = set(self.support_data.keys())
        query_classes = set(self.query_data.keys())

        common_classes = sorted(
            support_classes.intersection(query_classes)
        )

        valid_classes = []

        for class_name in common_classes:
            num_support_images = len(
                self.support_data[class_name]
            )
            num_query_images = len(
                self.query_data[class_name]
            )

            if self.same_source_split:
                # Same-source protocol (including test-test): reserve k_query
                # distinct images for Query and require at least one different
                # image for Support. When fewer than k_shot support candidates
                # remain, Support alone will be sampled with replacement.
                required_unique_images = self.k_query + 1

                if num_support_images >= required_unique_images:
                    valid_classes.append(class_name)

            else:
                # test-train:
                # support and query come from different directories.
                if (
                    num_support_images >= self.samples_per_class
                    and num_query_images >= self.k_query
                ):
                    valid_classes.append(class_name)

        if len(valid_classes) < self.classes_per_set:
            support_only = sorted(
                support_classes - query_classes
            )
            query_only = sorted(
                query_classes - support_classes
            )

            raise RuntimeError(
                '\nUnable to create episodes.\n'
                f'Support split: {self.support_type}\n'
                f'Query split: {self.query_type}\n'
                f'Requested n-way: {self.classes_per_set}\n'
                f'Number of valid shared classes: '
                f'{len(valid_classes)}\n'
                f'Valid shared classes: {valid_classes}\n'
                f'Classes only in support split: {support_only}\n'
                f'Classes only in query split: {query_only}\n'
                f'Each support class needs at least '
                f'{self.samples_per_class} images.\n'
                f'Each query class needs at least '
                f'{self.k_query} images.\n'
                f'When both use the same split with support replacement, '
                f'each class needs at least {self.k_query + 1} different '
                f'images: {self.k_query} for Query and at least 1 separate '
                f'image for Support.'
            )

        return valid_classes

    def create_episodes(self, episodes):
        """Pre-generate all few-shot episodes."""
        self.support_set_x_batch = []
        self.target_x_batch = []
        self.episode_classes_batch = []

        self.valid_classes = self._find_valid_classes()

        rng = np.random.RandomState(self.seed)

        for _ in range(episodes):
            selected_classes = rng.choice(
                self.valid_classes,
                size=self.classes_per_set,
                replace=False
            ).tolist()

            episode_support = []
            episode_query = []

            for class_name in selected_classes:
                if self.same_source_split:
                    # Same-pool protocol, including test-test.
                    #
                    # 1. Select Query without replacement.
                    # 2. Completely remove Query images from the Support pool.
                    # 3. Sample Support without replacement when possible;
                    #    otherwise sample Support with replacement.
                    # This permits repeated Support images but guarantees that
                    # Support and Query never contain the same image.
                    class_files = self.support_data[class_name]
                    num_images = len(class_files)

                    query_indices = rng.choice(
                        num_images,
                        size=self.k_query,
                        replace=False
                    )

                    query_index_set = {
                        int(index) for index in query_indices
                    }

                    support_candidate_indices = [
                        index
                        for index in range(num_images)
                        if index not in query_index_set
                    ]

                    if not support_candidate_indices:
                        raise RuntimeError(
                            f'Class "{class_name}" has no support image '
                            f'left after selecting {self.k_query} query '
                            f'image(s).'
                        )

                    support_replace = (
                        len(support_candidate_indices)
                        < self.samples_per_class
                    )

                    support_indices = rng.choice(
                        support_candidate_indices,
                        size=self.samples_per_class,
                        replace=support_replace
                    )

                    support_files = [
                        class_files[int(index)]
                        for index in support_indices
                    ]
                    query_files = [
                        class_files[int(index)]
                        for index in query_indices
                    ]

                    overlap = set(support_files).intersection(query_files)
                    if overlap:
                        raise RuntimeError(
                            f'Support/query overlap detected in class '
                            f'"{class_name}": {sorted(overlap)}'
                        )

                else:
                    # Cross-pool protocol, including test-train.
                    #
                    # Support and query are sampled independently because
                    # they come from different physical split directories.
                    support_pool = self.support_data[class_name]
                    query_pool = self.query_data[class_name]

                    support_indices = rng.choice(
                        len(support_pool),
                        size=self.samples_per_class,
                        replace=False
                    )
                    query_indices = rng.choice(
                        len(query_pool),
                        size=self.k_query,
                        replace=False
                    )

                    support_files = [
                        support_pool[index]
                        for index in support_indices
                    ]
                    query_files = [
                        query_pool[index]
                        for index in query_indices
                    ]

                episode_support.append(support_files)
                episode_query.append(query_files)

            self.support_set_x_batch.append(
                episode_support
            )
            self.target_x_batch.append(
                episode_query
            )
            self.episode_classes_batch.append(
                selected_classes
            )

    def __getitem__(self, index):
        """Return one complete few-shot episode."""
        if index < 0 or index >= self.nEpisodes:
            raise IndexError(
                f'Episode index {index} is outside '
                f'[0, {self.nEpisodes}).'
            )

        support_set_x = torch.empty(
            self.n_samples,
            self.channels,
            84,
            84,
            dtype=torch.float32
        )

        target_x = torch.empty(
            self.n_query_total,
            self.channels,
            84,
            84,
            dtype=torch.float32
        )

        support_labels = []
        query_labels = []

        flat_support_paths = []
        flat_query_paths = []

        episode_support = self.support_set_x_batch[index]
        episode_query = self.target_x_batch[index]

        # The position of a selected class in the episode is its
        # episode-local label: 0, 1, ..., n_way - 1.
        for episode_label in range(self.classes_per_set):
            support_files = episode_support[episode_label]
            query_files = episode_query[episode_label]

            for relative_path in support_files:
                full_path = os.path.join(
                    self.support_images_dir,
                    relative_path
                )
                flat_support_paths.append(full_path)
                support_labels.append(episode_label)

            for relative_path in query_files:
                full_path = os.path.join(
                    self.query_images_dir,
                    relative_path
                )
                flat_query_paths.append(full_path)
                query_labels.append(episode_label)

        for image_index, image_path in enumerate(
            flat_support_paths
        ):
            try:
                support_set_x[image_index] = self.transform(
                    image_path
                )
            except Exception as error:
                raise RuntimeError(
                    f'Failed to load support image: {image_path}'
                ) from error

        for image_index, image_path in enumerate(
            flat_query_paths
        ):
            try:
                target_x[image_index] = self.transform(
                    image_path
                )
            except Exception as error:
                raise RuntimeError(
                    f'Failed to load query image: {image_path}'
                ) from error

        support_set_y = torch.tensor(
            support_labels,
            dtype=torch.long
        )
        target_y = torch.tensor(
            query_labels,
            dtype=torch.long
        )

        return (
            support_set_x,
            support_set_y,
            target_x,
            target_y
        )

    def __len__(self):
        return self.nEpisodes
