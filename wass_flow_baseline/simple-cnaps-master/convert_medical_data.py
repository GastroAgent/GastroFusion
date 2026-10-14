"""
Medical dataset converter for Simple CNAPS / Meta-Dataset format.

Converts the medical image datasets from fsl_scoreboard/data/ into Meta-Dataset
TFRecord records so that run_simple_cnaps.py can evaluate them.

Usage:
  cd fsl_scoreboard/simple-cnaps-master
  python convert_medical_data.py --data_root ../data --records_root ../records
"""
import argparse
import io
import json
import os

import tensorflow.compat.v1 as tf
from PIL import Image


DATASETS = [
    '胃_7way_1query',
    '十二指肠_7way_1query',
    '食管_8way_1query',
]


def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list(values)))


def write_tfrecord(class_files, class_label, output_path):
    writer = tf.io.TFRecordWriter(output_path)
    written = 0
    for path in class_files:
        with open(path, 'rb') as f:
            image_bytes = f.read()
        try:
            img = Image.open(io.BytesIO(image_bytes))
        except Exception as exc:
            raise RuntimeError(f'Failed to open image {path}: {exc}') from exc
        if img.mode != 'RGB':
            img = img.convert('RGB')
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        buf.seek(0)
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': _bytes_feature([buf.getvalue()]),
            'label': _int64_feature([class_label]),
        }))
        writer.write(example.SerializeToString())
        written += 1
    writer.close()
    return written


def convert_dataset(data_root, records_root, subset):
    subset_path = os.path.join(data_root, subset)
    records_path = os.path.join(records_root, subset)
    os.makedirs(records_path, exist_ok=True)

    split_dirs = {
        'train': 'train',
        'valid': 'val',
        'test': 'test',
    }

    class_names = {}
    images_per_class = {}
    classes_per_split = {split: 0 for split in split_dirs}

    class_id = 0
    for split, folder in split_dirs.items():
        split_path = os.path.join(subset_path, folder)
        class_dirs = sorted([
            d for d in os.listdir(split_path)
            if os.path.isdir(os.path.join(split_path, d))
        ])
        classes_per_split[split] = len(class_dirs)
        for class_name in class_dirs:
            class_names[class_id] = class_name
            class_dir = os.path.join(split_path, class_name)
            class_files = sorted([
                os.path.join(class_dir, f)
                for f in os.listdir(class_dir)
                if os.path.isfile(os.path.join(class_dir, f))
                and f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            output_path = os.path.join(records_path, f'{class_id}.tfrecords')
            n = write_tfrecord(class_files, class_id, output_path)
            images_per_class[class_id] = n
            class_id += 1

    spec = {
        '__class__': 'DatasetSpecification',
        'name': subset,
        'classes_per_split': {
            'TRAIN': classes_per_split['train'],
            'VALID': classes_per_split['valid'],
            'TEST': classes_per_split['test'],
        },
        'images_per_class': {int(k): int(v) for k, v in images_per_class.items()},
        'class_names': {int(k): v for k, v in class_names.items()},
        'path': records_path,
        'file_pattern': '{}.tfrecords',
    }

    spec_path = os.path.join(records_path, 'dataset_spec.json')
    with tf.io.gfile.GFile(spec_path, 'w') as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)

    train_ids = list(range(0, classes_per_split['train']))
    valid_ids = list(range(classes_per_split['train'],
                           classes_per_split['train'] + classes_per_split['valid']))
    test_ids = list(range(classes_per_split['train'] + classes_per_split['valid'], class_id))

    print(f'Converted {subset}:')
    for split, ids in [('train', train_ids), ('valid', valid_ids), ('test', test_ids)]:
        print(f'  {split}: {classes_per_split[split]} classes (IDs {ids[0]}-{ids[-1]})')
    print(f'  Records written to: {records_path}')


def main():
    parser = argparse.ArgumentParser(description='Convert medical data to Meta-Dataset TFRecords')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing medical subsets (e.g. ../../data)')
    parser.add_argument('--records_root', type=str, required=True,
                        help='Root directory to write TFRecords to')
    parser.add_argument('--subset', type=str, default=None,
                        help='Specific subset to convert (default: all)')
    args = parser.parse_args()

    subsets = [args.subset] if args.subset else DATASETS
    for subset in subsets:
        if subset not in DATASETS:
            raise ValueError(f'Unknown subset {subset}. Choose from: {DATASETS}')
        convert_dataset(args.data_root, args.records_root, subset)


if __name__ == '__main__':
    main()