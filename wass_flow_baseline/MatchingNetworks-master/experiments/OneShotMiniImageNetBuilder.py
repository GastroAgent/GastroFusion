##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from models.MatchingNetwork import MatchingNetwork


class miniImageNetBuilder:

    def __init__(self, dataTrain, dataVal, dataTest, save_dir='./checkpoints',
                 dataset='medical', medical_subset='胃_1query',
                 save_best=True):
        self.dataTrain = dataTrain
        self.dataVal = dataVal
        self.dataTest = dataTest
        self.save_dir = save_dir
        self.dataset = dataset
        self.medical_subset = medical_subset
        self.save_best = save_best

    def build_experiment(self, batch_size, classes_per_set, samples_per_class,
                         channels, fce, lr=1e-3, wd=1e-4, load_path=''):
        # 允许通过环境变量覆盖 num_workers（默认 0，避免 Bus error）
        _nw = int(os.environ.get('MATCHING_NUM_WORKERS', '0'))
        pin_mem = bool(int(os.environ.get('MATCHING_PIN_MEMORY', '0')))
        self.train_loader = torch.utils.data.DataLoader(
            self.dataTrain, batch_size=batch_size, shuffle=True,
            num_workers=_nw, pin_memory=pin_mem)
        self.val_loader = torch.utils.data.DataLoader(
            self.dataVal, batch_size=batch_size, shuffle=True,
            num_workers=_nw, pin_memory=pin_mem)
        self.test_loader = torch.utils.data.DataLoader(
            self.dataTest, batch_size=batch_size, shuffle=True,
            num_workers=_nw, pin_memory=pin_mem)

        self.classes_per_set = classes_per_set
        self.samples_per_class = samples_per_class
        self.keep_prob = torch.FloatTensor(1)

        self.matchingNet = MatchingNetwork(
            batch_size=batch_size,
            keep_prob=self.keep_prob, num_channels=channels,
            fce=fce,
            num_classes_per_set=classes_per_set,
            num_samples_per_class=samples_per_class,
            nClasses=0, image_size=84)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            cudnn.benchmark = True
            torch.cuda.manual_seed_all(0)
        self.matchingNet.to(self.device)

        self.optimizer_kind = 'adam'
        self.lr = lr
        self.lr_decay = 1e-6
        self.wd = wd
        self.total_train_iter = 0
        self.current_best_val = -1.0

        if load_path and os.path.isfile(load_path):
            ckpt = torch.load(load_path, map_location=self.device)
            if 'state_dict' in ckpt:
                self.matchingNet.load_state_dict(ckpt['state_dict'])
            else:
                self.matchingNet.load_state_dict(ckpt)
            if 'best_val' in ckpt:
                self.current_best_val = ckpt['best_val']
            print(f'Loaded checkpoint from {load_path}, best_val={self.current_best_val}')

        # Make save dir
        subset_key = self.medical_subset if self.dataset == 'medical' else self.dataset
        self.save_subdir = os.path.join(self.save_dir, self.dataset,
                                        subset_key, 'matching')
        os.makedirs(self.save_subdir, exist_ok=True)

    def __create_optimizer(self, model, lr):
        if self.optimizer_kind == 'adam':
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=self.wd)
        else:
            return torch.optim.SGD(model.parameters(), lr=lr,
                                   momentum=0.9, dampening=0.9, weight_decay=self.wd)

    def __adjust_learning_rate(self, optimizer):
        for group in optimizer.param_groups:
            if 'step' not in group:
                group['step'] = 0
            group['step'] += 1
            group['lr'] = self.lr / (1 + group['step'] * self.lr_decay)

    def _move_to_device(self, *tensors):
        return [t.to(self.device, non_blocking=True) for t in tensors]

    def _one_pass(self, loader, train_mode, total_batches=None):
        total_c_loss = 0.0
        total_accuracy = 0.0
        n_batches = total_batches if total_batches is not None else len(loader)
        optimizer = self.__create_optimizer(self.matchingNet, self.lr) if train_mode else None
        if train_mode:
            self.matchingNet.train()
        else:
            self.matchingNet.eval()

        pbar = tqdm(loader, total=n_batches)
        for batch_idx, (x_support_set, y_support_set, x_target, target_y) in enumerate(pbar):
            if batch_idx >= n_batches:
                break
            x_support_set = x_support_set.float()
            y_support_set = y_support_set.long()
            x_target = x_target.float()
            y_target = target_y.long()

            y_support_set = y_support_set.unsqueeze(2)
            B, S = y_support_set.size(0), y_support_set.size(1)
            y_support_set_one_hot = torch.FloatTensor(B, S, self.classes_per_set).zero_()
            y_support_set_one_hot.scatter_(2, y_support_set, 1)

            (x_support_set, y_support_set_one_hot, x_target, y_target) = \
                self._move_to_device(x_support_set, y_support_set_one_hot, x_target, y_target)

            if train_mode:
                acc, c_loss_value = self.matchingNet(x_support_set, y_support_set_one_hot,
                                                     x_target, y_target)
                optimizer.zero_grad()
                c_loss_value.backward()
                optimizer.step()
                self.__adjust_learning_rate(optimizer)
                self.total_train_iter += 1
                if self.total_train_iter % 2000 == 0:
                    self.lr /= 2
                    print('change learning rate', self.lr)
            else:
                with torch.no_grad():
                    acc, c_loss_value = self.matchingNet(x_support_set, y_support_set_one_hot,
                                                         x_target, y_target)

            total_c_loss += c_loss_value.item()
            total_accuracy += acc.item()
            mode_str = 'tr' if train_mode else 'val'
            pbar.set_description(f'{mode_str}_loss: {c_loss_value.item():.4f}, '
                                 f'{mode_str}_acc: {acc.item():.4f}')

        return total_c_loss / n_batches, total_accuracy / n_batches

    def run_training_epoch(self):
        return self._one_pass(self.train_loader, train_mode=True)

    def run_validation_epoch(self):
        return self._one_pass(self.val_loader, train_mode=False)

    def run_testing_epoch(self):
        return self._one_pass(self.test_loader, train_mode=False)

    def save_checkpoint(self, val_acc, tag='best'):
        ckpt = {
            'state_dict': self.matchingNet.state_dict(),
            'best_val': val_acc,
        }
        path = os.path.join(self.save_subdir, f'{tag}.pt')
        torch.save(ckpt, path)
        print(f'Saved checkpoint -> {path} (val_acc={val_acc:.4f})')

    def save_result(self, result):
        path = os.path.join(self.save_subdir, 'eval_result.json')
        with open(path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'Saved result -> {path}')