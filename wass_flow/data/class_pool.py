from __future__ import annotations

import glob
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    path: str
    label: str
    class_id: int


def build_rae_transform(image_size: int = 256, augment: bool = False):
    ops: list = [transforms.Resize((image_size, image_size))]
    if augment:
        ops.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1)
            ], p=0.2),
        ])
    # RAE encode in the supplied code expects [0,1].
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


def _first_present(item: dict, keys: Iterable[str]):
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return value
    return None


def _iter_json_items(obj):
    if isinstance(obj, list):
        yield from obj
    elif isinstance(obj, dict):
        for key in ("data", "items", "records", "samples"):
            if isinstance(obj.get(key), list):
                yield from obj[key]
                return
        # A class-name -> list/dict mapping is also accepted.
        for label, values in obj.items():
            if isinstance(values, list):
                for value in values:
                    if isinstance(value, str):
                        yield {"path": value, "label": label}
                    elif isinstance(value, dict):
                        value = dict(value)
                        value.setdefault("label", label)
                        yield value
            elif isinstance(values, dict):
                value = dict(values)
                value.setdefault("label", label)
                yield value
    else:
        raise ValueError(f"Unsupported JSON top level: {type(obj).__name__}")


def records_from_json_glob(pattern: str) -> list[ImageRecord]:
    paths = sorted(glob.glob(f"{pattern}/*.json"))
    if not paths:
        raise FileNotFoundError(f"No JSON matched: {pattern}")
    raw: list[tuple[str, str, int | None]] = []
    for json_path in tqdm(paths):
        with open(json_path, "r") as f:
            data = json.load(f)
        fallback_label = Path(json_path).stem
        for item in _iter_json_items(data):
            if not isinstance(item, dict):
                continue
            image_path = _first_present(item, (
                "path", "image", "image_path", "img_path", "x1_path", "x0_path", "file", "x"
            ))
            label = _first_present(item, (
                "label", "label_B", "class_name", "category", "caption"
            )) or fallback_label
            class_id = _first_present(item, ("class_id", "label_id", "label_B_id"))
            if image_path is None:
                continue
            image_path = os.path.expanduser(str(image_path))
            if not os.path.isabs(image_path):
                image_path = os.path.join(os.path.dirname(json_path), image_path)
            if os.path.isfile(image_path):
                raw.append((os.path.abspath(image_path), str(label), int(class_id) if class_id is not None else None))
    if not raw:
        raise RuntimeError(f"No readable images found in JSONs: {pattern}")
    label_names = sorted({label for _, label, _ in raw})
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    return [ImageRecord(p, label, cid if cid is not None else label_to_id[label]) for p, label, cid in raw]


def records_from_imagefolder(root: str) -> list[ImageRecord]:
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(root)
    class_dirs = sorted(p for p in root_path.iterdir() if p.is_dir())
    records: list[ImageRecord] = []
    for cid, class_dir in tqdm(enumerate(class_dirs)):
        for path in class_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                records.append(ImageRecord(str(path.resolve()), class_dir.name, cid))
    if not records:
        raise RuntimeError(f"No class-folder images found under {root}")
    return records


def load_records(data_root: str | None = None, json_glob: str | None = None) -> list[ImageRecord]:
    if bool(data_root) == bool(json_glob):
        raise ValueError("Specify exactly one of data_root or json_glob")
    return records_from_imagefolder(data_root) if data_root else records_from_json_glob(json_glob)


class RecordDataset(Dataset):
    def __init__(
        self,
        records: list[ImageRecord],
        transform=None,
        label_to_index: dict[str, int] | None = None,
    ) -> None:
        self.records = records
        self.transform = transform or build_rae_transform()
        # Reindex arbitrary label IDs into contiguous training targets while
        # preserving the human-readable labels. Validation can reuse the exact
        # training mapping via label_to_index.
        if label_to_index is None:
            names = sorted({r.label for r in records})
            self.label_to_index = {name.replace(" ", "_"): i for i, name in enumerate(names)}
        else:
            self.label_to_index = dict(label_to_index)
            unknown = sorted({r.label for r in records} - set(self.label_to_index))
            if unknown:
                raise ValueError(f"Validation contains unknown classes: {unknown}")
        self.targets = [self.label_to_index[r.label.replace(" ", "_")] for r in records]
        self.classes = [name for name, _ in sorted(self.label_to_index.items(), key=lambda item: item[1])]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, self.targets[index], record.path


class PKBatchSampler(Sampler[list[int]]):
    """Class-balanced P classes x K images batch sampler."""

    def __init__(
        self,
        targets: list[int],
        classes_per_batch: int,
        samples_per_class: int,
        batches_per_epoch: int,
        seed: int = 0,
    ) -> None:
        self.class_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, target in enumerate(targets):
            self.class_to_indices[int(target)].append(idx)
        self.p = int(classes_per_batch)
        self.k = int(samples_per_class)
        self.classes = sorted(
            cls for cls, indices in self.class_to_indices.items()
            if len(indices) >= self.k
        )
        if len(self.classes) < self.p:
            raise ValueError(
                f"Need P={self.p} classes with at least K={self.k} distinct images; "
                f"only {len(self.classes)} classes qualify"
            )
        self.batches_per_epoch = int(batches_per_epoch)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.batches_per_epoch):
            selected = rng.sample(self.classes, self.p)
            batch: list[int] = []
            for cls in selected:
                pool = self.class_to_indices[cls]
                batch.extend(rng.sample(pool, self.k))
            yield batch
