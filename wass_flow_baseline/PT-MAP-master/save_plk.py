from __future__ import print_function

import os
import collections
import pickle

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from io_utils import parse_args
from data.datamgr import SimpleDataManager
import configs
x
import wrn_mixup_model
import res_mixup_model

from io_utils import parse_args, get_resume_file


use_gpu = torch.cuda.is_available()


class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def extract_feature(val_loader, model, checkpoint_dir, tag='last'):
    save_dir = os.path.join(checkpoint_dir, tag)
    plk_path = os.path.join(save_dir, 'output.plk')
    if os.path.isfile(plk_path):
        print(f'  [save_plk] {plk_path} already exists, skipping extraction')
        return load_pickle(plk_path)

    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        output_dict = collections.defaultdict(list)
        device = next(model.parameters()).device
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs, _ = model(inputs)
            outputs = outputs.cpu().data.numpy()
            for out, label in zip(outputs, labels):
                output_dict[label.item()].append(out)

    save_pickle(plk_path, output_dict)
    return output_dict


def resolve_load_path(params):
    if params.dataset == 'medical':
        ckpt_dir = os.path.join(
            configs.save_dir, 'checkpoints', params.dataset,
            params.medical_subset, f'{params.model}_{params.method}')
        loadfile = os.path.join(
            configs.data_dir['medical'], params.medical_subset, 'novel.json')
    else:
        ckpt_dir = os.path.join(
            configs.save_dir, 'checkpoints', params.dataset,
            f'{params.model}_{params.method}')
        loadfile = configs.data_dir[params.dataset] + 'novel.json'
    return ckpt_dir, loadfile


if __name__ == '__main__':
    params = parse_args('test')

    checkpoint_dir, loadfile = resolve_load_path(params)

    image_size = 32 if params.dataset == 'cifar' else 84
    datamgr = SimpleDataManager(image_size, batch_size=256)
    novel_loader = datamgr.get_data_loader(loadfile, aug=False)

    # Auto-detect num_classes for medical subsets so the model's fc matches
    # the checkpoint state. Without this, --num_classes defaults to 200 and
    # the saved checkpoint (e.g. fc shape [8, 512]) won't load.
    if params.dataset == 'medical':
        base_file = os.path.join(
            configs.data_dir['medical'], params.medical_subset, 'base.json')
        with open(base_file) as _f:
            _j = __import__('json').load(_f)
        params.num_classes = len(_j['label_names'])
        print(f'  [save_plk] num_classes auto-set to {params.num_classes} '
              f'(from {base_file})')

    modelfile = get_resume_file(checkpoint_dir)
    if modelfile is None:
        raise FileNotFoundError(
            f'No checkpoint found in {checkpoint_dir}.\n'
            f'Train a model first with: python train.py --dataset {params.dataset} '
            f'--method {params.method} --model {params.model} ...')

    if params.model == 'WideResNet28_10':
        model = wrn_mixup_model.wrn28_10(num_classes=params.num_classes)
    elif params.model == 'ResNet18':
        model = res_mixup_model.resnet18(num_classes=params.num_classes)

    if use_gpu:
        model = model.cuda()
        cudnn.benchmark = True

    print(f'  Loading checkpoint: {modelfile}')
    checkpoint = torch.load(modelfile, map_location='cpu')
    state = checkpoint['state']
    state_keys = list(state.keys())

    callwrap = bool(state_keys and 'module' in state_keys[0])
    if callwrap:
        model = WrappedModel(model)
    model_dict_load = model.state_dict()
    model_dict_load.update(state)
    model.load_state_dict(model_dict_load)
    model.eval()

    print(f'  Extracting features from novel.json into:')
    output_dict = extract_feature(novel_loader, model, checkpoint_dir, tag='last')
    n_classes = len(output_dict)
    n_total = sum(len(v) for v in output_dict.values())
    print(f'  features saved! ({n_classes} classes, {n_total} total feature vectors)')
