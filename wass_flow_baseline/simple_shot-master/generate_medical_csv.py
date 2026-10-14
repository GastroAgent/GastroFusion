#!/usr/bin/env python3
"""
Generate SimpleShot-style CSV split files from PT-MAP JSON filelists.

The PT-MAP filelists (base.json / val.json / novel.json) use absolute paths
and a shared global label space across splits.  This script converts them to
SimpleShot's CSV format (relative paths from the subset directory, split-local labels).

SimpleShot CSV format (e.g. split/mini/test.csv):
    filename,label
    test/n01532829/n0153282900000005.jpg,n01532829

Here <subset_dir> = /path/to/data/<subset>/, so CSV paths are relative to it:
    test/Esophageal Glycogenic Acanthosis/01.2012022100263_11.jpg

Usage:
    python generate_medical_csv.py --subset 食管_1query \
        --filelist_dir ../PT-MAP-master/filelists/medical/ \
        --subset_dir   ../data/食管_1query/ \
        --out_dir      ./split/medical/

    Then run SimpleShot with:
        --data      <subset_dir>
        --split-dir ./split/medical/食管_1query/
"""
import argparse
import csv
import json
import os
from pathlib import Path


def json_to_csv(json_path, subset_dir, out_csv, split_tag):
    """
    Convert a PT-MAP JSON filelist to SimpleShot CSV.

    PT-MAP JSON paths are absolute, e.g.:
      /mnt/.../data/食管_1query/test/Esophageal.../img.jpg

    subset_dir = /mnt/.../data/食管_1query/
    We want CSV rows where filename is relative to subset_dir:
      test/Esophageal Glycogenic Acanthosis/img.jpg
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    label_names = data['label_names']
    image_names = data['image_names']
    image_labels = data['image_labels']

    subset_dir_str = str(Path(subset_dir).resolve()).rstrip('/')

    rows = []
    for img_abs, label_id in zip(image_names, image_labels):
        img_str = str(img_abs)
        # Strip the subset_dir prefix to get relative path
        if img_str.startswith(subset_dir_str + '/'):
            rel = img_str[len(subset_dir_str) + 1:]
        else:
            # Fallback: try to extract split/class/file from the path
            parts = img_str.split('/')
            # Find the split part (train/val/test) and keep everything from there
            rel_parts = []
            for p in parts:
                if p in ('train', 'val', 'test'):
                    rel_parts = parts[parts.index(p):]
                    break
            rel = '/'.join(rel_parts)

        rows.append((rel, int(label_id)))

    # Sort by label then filename for reproducibility
    rows.sort(key=lambda x: (x[1], x[0]))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])
        writer.writerows(rows)

    print(f'  [{split_tag}] -> {out_csv}  ({len(rows)} images, {len(label_names)} classes)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', required=True,
                        help='Medical subset name, e.g. 食管_1query')
    parser.add_argument('--filelist_dir', required=True,
                        help='Directory containing PT-MAP JSON filelists')
    parser.add_argument('--subset_dir', required=True,
                        help='Full path to the subset directory (e.g. ../data/食管_1query/)')
    parser.add_argument('--out_dir', required=True,
                        help='Output directory for CSV files')
    args = parser.parse_args()

    subset_dir = Path(args.subset_dir)
    out_dir = Path(args.out_dir) / args.subset

    splits = [
        ('train', Path(args.filelist_dir) / args.subset / 'base.json'),
        ('val',   Path(args.filelist_dir) / args.subset / 'val.json'),
        ('test',  Path(args.filelist_dir) / args.subset / 'novel.json'),
    ]

    for split_tag, json_path in splits:
        if not json_path.exists():
            print(f'  [WARN] {json_path} not found, skipping {split_tag}')
            continue
        out_csv = out_dir / f'{split_tag}.csv'
        json_to_csv(json_path, subset_dir, out_csv, split_tag)

    print(f'\nDone. CSV files written to: {out_dir}')
    print('\nTo use with SimpleShot:')
    print(f'  --data      {subset_dir.resolve()}')
    print(f'  --split-dir {out_dir.resolve()}')


if __name__ == '__main__':
    main()
