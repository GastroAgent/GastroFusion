#!/usr/bin/env python3
"""
SimpleShot Medical Evaluation Script

Replicates the SimpleShot few-shot evaluation pipeline (UN / L2N / CL2N nearest-neighbor
classifiers) on medical datasets using PT-MAP's pretrained backbone.

The evaluation runs in two modes:
  --mode extract:  Extract features using a PT-MAP pretrained model and cache them.
  --mode evaluate: Load cached features and run SimpleShot evaluation.
  --mode full:     Extract + evaluate (default).

Reference PT-MAP result for 食管_1query:
  n_ways=8, n_shot=5, n_queries=1, n_runs=10000
  test_acc=27.60%, pm=0.26, lam=10, alpha=0.2, beta=0.5

Usage (full pipeline):
  python eval_medical.py --subset 食管_1query --model ResNet18 --method S2M2_R

  python eval_medical.py --subset 食管_1query --mode evaluate \
      --feature_dir ../PT-MAP-master/checkpoints/medical/食管_1query/ResNet18_S2M2_R/last/

For a quick sanity check (fewer episodes, no tqdm):
  python eval_medical.py --subset 食管_1query --n_runs 100 --disable_tqdm
"""
import argparse
import collections
import json
import os
import pickle
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from numpy import linalg as LA
from scipy.stats import mode
from tqdm import tqdm

# ---- backbone model (PT-MAP's res_mixup_model.py) ----
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()
        self.L = nn.Linear(indim, outdim, bias=False)
        self.scale_factor = 10

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-6)
        return self.scale_factor * self.L(x_normalized)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=200):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = distLinear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x, self.fc(x)


def resnet18(num_classes=200):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


# ---- torchvision for image transforms ----
import torchvision.transforms as transforms
from PIL import Image

try:
    from torch.utils.data import Dataset
except ImportError:
    from torch.utils.data import Dataset


# ---- ImageNet normalisation ----
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class MedicalDataset(Dataset):
    """Simple dataset that reads CSV files in SimpleShot's format."""

    def __init__(self, csv_path, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform

        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines()[1:] if l.strip()]
        self.paths = []
        self.labels = []
        for line in lines:
            path, label = line.split(',', 1)
            self.paths.append(path)
            self.labels.append(int(label))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_root, self.paths[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def get_transform(img_size=84, enlarge=False):
    if enlarge:
        resize = int(img_size * 256.0 / 224)
    else:
        resize = img_size
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_model(checkpoint_path, num_classes, device='cuda'):
    model = resnet18(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = checkpoint.get('state', checkpoint)

    state_keys = list(state.keys())
    if state_keys and state_keys[0].startswith('module.'):
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state.items():
            new_state[k.replace('module.', '')] = v
        state = new_state

    model_dict = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_dict}
    model_dict.update(filtered)
    model.load_state_dict(model_dict)
    model = model.to(device)
    model.eval()
    return model


def extract_features(data_loader, model, device, desc='Extracting'):
    """Extract features for all images in data_loader. Returns dict {label: [features]}."""
    output_dict = collections.defaultdict(list)
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc=desc, disable=args.disable_tqdm):
            inputs = inputs.to(device)
            features, _ = model(inputs)
            features = features.cpu().numpy()
            for feat, label in zip(features, labels.numpy()):
                output_dict[int(label)].append(feat)
    return output_dict


def compute_confidence_interval(data):
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


def sample_case(ld_dict, n_way, shot, n_query):
    """Sample a few-shot episode: sample n_way classes, then shot+n_query images each."""
    sample_class = random.sample(list(ld_dict.keys()), n_way)
    train_input, test_input = [], []
    train_label, test_label = [], []
    for label_id, cls in enumerate(sample_class):
        samples = random.sample(ld_dict[cls], shot + n_query)
        train_label += [label_id] * shot
        test_label  += [label_id] * n_query
        train_input += samples[:shot]
        test_input  += samples[shot:]
    train_input = np.array(train_input, dtype=np.float32)
    test_input  = np.array(test_input, dtype=np.float32)
    return train_input, test_input, train_label, test_label


def metric_class_type(gallery, query, train_label, test_label, n_way, shot,
                      train_mean=None, norm_type='CL2N'):
    """
    SimpleShot nearest-neighbor classification with three normalization strategies.

    gallery:  (n_way * shot, n_feat) support features
    query:    (n_way * n_query, n_feat) query features
    train_mean: mean of all training features for CL2N centering
    norm_type: 'CL2N' (center+normalize), 'L2N' (normalize only), 'UN' (no transform)
    """
    if norm_type == 'CL2N':
        gallery = gallery - train_mean
        gallery = gallery / (LA.norm(gallery, 2, 1, keepdims=True) + 1e-10)
        query   = query   - train_mean
        query   = query   / (LA.norm(query,   2, 1, keepdims=True) + 1e-10)
    elif norm_type == 'L2N':
        gallery = gallery / (LA.norm(gallery, 2, 1, keepdims=True) + 1e-10)
        query   = query   / (LA.norm(query,   2, 1, keepdims=True) + 1e-10)

    # class prototypes: average over shots
    gallery = gallery.reshape(n_way, shot, -1).mean(axis=1)
    train_label = np.array(train_label[::shot])

    # nearest-centroid
    subtract = gallery[:, None, :] - query
    distance = LA.norm(subtract, 2, axis=-1)
    predict  = np.argmin(distance, axis=0)
    test_label = np.array(test_label)
    acc = (predict == test_label).mean()
    return acc


def meta_evaluate(feature_dict, n_way, shot, n_query, n_iter, train_mean=None):
    """Run SimpleShot evaluation over n_iter episodes for all three norms."""
    un_list, l2n_list, cl2n_list = [], [], []

    iterator = range(n_iter)
    if not args.disable_tqdm:
        iterator = tqdm(iterator, desc='Evaluating', leave=False)

    for _ in iterator:
        train_data, test_data, train_label, test_label = sample_case(
            feature_dict, n_way, shot, n_query)

        cl2n_list.append(metric_class_type(
            train_data, test_data, train_label, test_label, n_way, shot,
            train_mean=train_mean, norm_type='CL2N'))
        l2n_list.append(metric_class_type(
            train_data, test_data, train_label, test_label, n_way, shot,
            train_mean=train_mean, norm_type='L2N'))
        un_list.append(metric_class_type(
            train_data, test_data, train_label, test_label, n_way, shot,
            train_mean=train_mean, norm_type='UN'))

    un_mean,   un_pm   = compute_confidence_interval(un_list)
    l2n_mean,  l2n_pm  = compute_confidence_interval(l2n_list)
    cl2n_mean, cl2n_pm = compute_confidence_interval(cl2n_list)

    return {
        'UN':   (un_mean,   un_pm),
        'L2N':  (l2n_mean,  l2n_pm),
        'CL2N': (cl2n_mean, cl2n_pm),
    }


def main():
    global args
    parser = argparse.ArgumentParser(description='SimpleShot Medical Evaluation')
    # Dataset
    parser.add_argument('--subset', type=str, default='食管_1query',
                        help='Medical subset name (e.g. 食管_1query)')
    parser.add_argument('--data_root', type=str,
                        default='/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/data',
                        help='Root directory of the image data')
    parser.add_argument('--csv_dir', type=str,
                        default='/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/simple_shot-master/split',
                        help='Directory containing CSV split files')
    # PT-MAP backbone
    parser.add_argument('--model', type=str, default='ResNet18',
                        choices=['ResNet18', 'WideResNet28_10'],
                        help='PT-MAP backbone model')
    parser.add_argument('--method', type=str, default='S2M2_R',
                        choices=['S2M2_R', 'rotation'],
                        help='PT-MAP training method')
    parser.add_argument('--ptmap_root', type=str,
                        default='/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/PT-MAP-master',
                        help='PT-MAP root directory')
    # Evaluation settings
    parser.add_argument('--n_ways', type=int, default=8)
    parser.add_argument('--n_shot', type=int, default=5)
    parser.add_argument('--n_queries', type=int, default=1)
    parser.add_argument('--n_runs', type=int, default=10000)
    parser.add_argument('--num_classes', type=int, default=8,
                        help='Total number of classes in the dataset')
    # Feature caching
    parser.add_argument('--feature_dir', type=str, default=None,
                        help='Directory containing cached features (if None, extract first)')
    # General
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_json', type=str, default='',
                        help='Path to save results as JSON')
    args = parser.parse_args()

    # ---- Seed ----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- Paths ----
    subset_dir = os.path.join(args.csv_dir, args.subset)
    train_csv  = os.path.join(subset_dir, 'train.csv')
    test_csv   = os.path.join(subset_dir, 'test.csv')

    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        print(f'ERROR: CSV files not found in {subset_dir}')
        print('Run generate_medical_csv.py first to create them.')
        sys.exit(1)

    data_root = os.path.join(args.data_root, args.subset)

    # Feature cache path
    if args.feature_dir:
        feat_cache = args.feature_dir
    else:
        feat_cache = os.path.join(
            args.ptmap_root, 'checkpoints', 'medical', args.subset,
            f'{args.model}_{args.method}', 'last')

    train_feat_cache = os.path.join(feat_cache, 'train_features.pkl')
    test_feat_cache  = os.path.join(feat_cache, 'test_features.pkl')
    os.makedirs(feat_cache, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print(f'Subset: {args.subset}  |  {args.n_ways}-way  {args.n_shot}-shot  {args.n_queries}-query')
    print(f'CSV dir: {subset_dir}')
    print(f'Feature cache: {feat_cache}')

    # ============================================================
    #  STEP 1: Extract or load features
    # ============================================================
    if os.path.exists(train_feat_cache) and os.path.exists(test_feat_cache):
        print('\n[Step 1] Loading cached features...')
        with open(train_feat_cache, 'rb') as f:
            train_dict = pickle.load(f)
        with open(test_feat_cache, 'rb') as f:
            test_dict  = pickle.load(f)
        print(f'  train: {sum(len(v) for v in train_dict.values())} images, '
              f'{len(train_dict)} classes')
        print(f'  test:  {sum(len(v) for v in test_dict.values())} images, '
              f'{len(test_dict)} classes')
    else:
        print('\n[Step 1] Extracting features with PT-MAP backbone...')

        # Load PT-MAP checkpoint
        ckpt_dir = os.path.join(
            args.ptmap_root, 'checkpoints', 'medical', args.subset,
            f'{args.model}_{args.method}')

        import glob as _glob
        checkpoints = sorted(_glob.glob(os.path.join(ckpt_dir, '*.tar')))
        checkpoints = [c for c in checkpoints if not os.path.basename(c).startswith('best')]
        if not checkpoints:
            print(f'ERROR: No checkpoint found in {ckpt_dir}')
            print('Run PT-MAP training first: python ../PT-MAP-master/train.py ...')
            sys.exit(1)
        ckpt_path = checkpoints[-1]
        print(f'  Using checkpoint: {ckpt_path}')

        model = load_model(ckpt_path, args.num_classes, device)

        transform = get_transform(img_size=84, enlarge=True)

        # Extract train features
        print('  Extracting train features...')
        train_dataset = MedicalDataset(train_csv, data_root, transform=transform)
        train_loader  = torch.utils.data.DataLoader(
            train_dataset, batch_size=32, shuffle=False,
            num_workers=4, pin_memory=True)
        train_dict = extract_features(train_loader, model, device, 'Train')

        # Extract test features
        print('  Extracting test features...')
        test_dataset = MedicalDataset(test_csv, data_root, transform=transform)
        test_loader  = torch.utils.data.DataLoader(
            test_dataset, batch_size=32, shuffle=False,
            num_workers=4, pin_memory=True)
        test_dict  = extract_features(test_loader, model, device, 'Test')

        with open(train_feat_cache, 'wb') as f:
            pickle.dump(train_dict, f)
        with open(test_feat_cache, 'wb') as f:
            pickle.dump(test_dict, f)
        print(f'  Cached to {feat_cache}')

    # Verify we have enough samples
    for c, feats in train_dict.items():
        if len(feats) < args.n_shot + args.n_queries:
            print(f'WARNING: class {c} has only {len(feats)} samples '
                  f'(need {args.n_shot + args.n_queries})')

    # ============================================================
    #  STEP 2: Compute training mean (for CL2N centering)
    # ============================================================
    print('\n[Step 2] Computing training mean...')
    all_train = np.concatenate(list(train_dict.values()), axis=0)
    train_mean = all_train.mean(axis=0, keepdims=True)
    print(f'  train_mean shape: {train_mean.shape}')
    print(f'  feature dim: {all_train.shape[1]}')

    # ============================================================
    #  STEP 3: SimpleShot Evaluation
    # ============================================================
    print('\n[Step 3] Running SimpleShot evaluation...')
    results = meta_evaluate(
        test_dict,
        n_way=args.n_ways,
        shot=args.n_shot,
        n_query=args.n_queries,
        n_iter=args.n_runs,
        train_mean=train_mean,
    )

    print('\n' + '=' * 60)
    print(f'SimpleShot Results  |  {args.n_ways}-way  {args.n_shot}-shot  {args.n_queries}-query')
    print('=' * 60)
    print(f'{"Method":<8}  {"Accuracy":>10}  {"95% CI":>10}')
    print('-' * 40)
    for method, (mean, pm) in results.items():
        print(f'{method:<8}  {mean*100:>9.2f}%  {pm*100:>9.2f}%')
    print('=' * 60)

    best_method = max(results, key=lambda k: results[k][0])
    best_mean, best_pm = results[best_method]
    print(f'\nBest: {best_method} = {best_mean*100:.2f}% ± {best_pm*100:.2f}%')

    # Save JSON
    if args.output_json:
        out = {
            'dataset':    args.subset,
            'method':     'SimpleShot',
            'backbone':   f'{args.model}_{args.method}',
            'n_ways':     args.n_ways,
            'n_shot':     args.n_shot,
            'n_queries':  args.n_queries,
            'n_runs':     args.n_runs,
            'results': {
                k: {'acc': float(v[0]*100), 'pm': float(v[1]*100)}
                for k, v in results.items()
            },
            'best': {
                'method': best_method,
                'acc':    float(best_mean*100),
                'pm':     float(best_pm*100),
            }
        }
        with open(args.output_json, 'w') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f'\nResults saved to: {args.output_json}')


if __name__ == '__main__':
    main()
